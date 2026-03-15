"""CodeMemo benchmark evaluation for synapt recall.

A coding-assistant memory benchmark: ingest multi-session Claude Code
transcripts, retrieve context for coding questions, generate + judge answers.

Usage:
    # Full run on one project (requires OPENAI_API_KEY):
    python evaluation/codememo/eval.py --project project_01_cli_tool

    # RecallDB mode (FTS5 + embeddings):
    python evaluation/codememo/eval.py --project project_01_cli_tool --recalldb

    # Full pipeline (RecallDB + enrich + consolidate + knowledge graph):
    python evaluation/codememo/eval.py --project project_01_cli_tool --full-pipeline

    # Retrieval-only (no API key needed):
    python evaluation/codememo/eval.py --retrieval-only

    # All projects:
    python evaluation/codememo/eval.py
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Protocol, runtime_checkable

from evaluation.codememo.schema import CATEGORY_NAMES, ALL_CATEGORIES

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------

CODEMEMO_DIR = Path(__file__).parent
DATA_DIR = CODEMEMO_DIR / "data"


# ---------------------------------------------------------------------------
# SystemUnderTest protocol — plug in any memory system
# ---------------------------------------------------------------------------

@runtime_checkable
class SystemUnderTest(Protocol):
    """Interface for a memory system that can be benchmarked."""

    def ingest(self, session_paths: list[Path]) -> None:
        """Ingest one project's session transcripts."""
        ...

    def query(self, question: str, max_chunks: int = 20) -> str:
        """Retrieve context for a question. Returns formatted text."""
        ...


# ---------------------------------------------------------------------------
# SynaptSUT — synapt recall implementation
# ---------------------------------------------------------------------------

class SynaptSUT:
    """SystemUnderTest backed by synapt recall."""

    def __init__(
        self,
        mode: str = "baseline",
        enrich_model: str = "",
        max_knowledge: int | None = None,
        knowledge_boost: float | None = None,
    ):
        self.mode = mode  # "baseline", "recalldb", "full-pipeline"
        self.enrich_model = enrich_model
        self.max_knowledge = max_knowledge
        self.knowledge_boost = knowledge_boost
        self._index = None
        self._db = None
        self._work_dir: Path | None = None

    def ingest(self, session_paths: list[Path]) -> None:
        """Build a synapt recall index from session transcripts."""
        from synapt.recall.core import build_index, parse_transcript
        from synapt.recall.storage import RecallDB

        self._work_dir = Path(tempfile.mkdtemp(prefix="codememo_"))
        index_dir = self._work_dir / "index"
        index_dir.mkdir()

        # Sessions are already in Claude Code JSONL format — no conversion
        # needed. Just point build_index at a directory containing them.
        transcript_dir = self._work_dir / "transcripts"
        transcript_dir.mkdir()

        # Symlink sessions into the transcript dir
        for sp in session_paths:
            dst = transcript_dir / sp.name
            if not dst.exists():
                dst.symlink_to(sp.resolve())

        if self.mode in ("recalldb", "full-pipeline"):
            db = RecallDB(index_dir / "recall.db")
            all_chunks = []
            for sp in session_paths:
                chunks = parse_transcript(transcript_dir / sp.name)
                all_chunks.extend(chunks)
            db.save_chunks(all_chunks)
            self._db = db
        else:
            db = None

        if self.mode == "full-pipeline":
            self._run_enrichment(transcript_dir, index_dir)

        self._index = build_index(
            source_dir=transcript_dir,
            use_embeddings=True,
            cache_dir=index_dir,
            db=self._db,
        )

    def _run_enrichment(self, transcript_dir: Path, index_dir: Path) -> None:
        """Run enrich + consolidate for full pipeline mode."""
        from synapt.recall.enrich import enrich_all
        from synapt.recall.consolidate import consolidate
        from synapt.recall.knowledge import read_nodes
        from synapt.recall.journal import JournalEntry, append_entry

        # Set up directory structure that enrich_all expects
        project_dir = self._work_dir / "project"
        project_dir.mkdir()
        data_dir = project_dir / ".synapt" / "recall"
        # _worktree_name(project_dir) resolves to project_dir.name = "project"
        wt_dir = data_dir / "worktrees" / "project"
        archive_dir = wt_dir / "transcripts"
        archive_dir.mkdir(parents=True)

        # Copy transcripts to archive
        for tp in sorted(transcript_dir.glob("*.jsonl")):
            dst = archive_dir / tp.name
            if not dst.exists():
                shutil.copy2(tp, dst)

        # Create journal stubs
        journal_path = wt_dir / "journal.jsonl"
        for tp in sorted(archive_dir.glob("*.jsonl")):
            session_id = tp.stem
            timestamp = ""
            try:
                with open(tp, encoding="utf-8") as tf:
                    first_line = tf.readline().strip()
                timestamp = json.loads(first_line).get("timestamp", "")
            except (json.JSONDecodeError, IndexError):
                pass
            stub = JournalEntry(
                timestamp=timestamp or "2025-01-01T00:00:00+00:00",
                session_id=session_id,
                auto=True,
                enriched=False,
            )
            append_entry(stub, journal_path)

        # Enrich
        enrich_kwargs = {"project_dir": project_dir, "max_entries": 0}
        if self.enrich_model:
            enrich_kwargs["model"] = self.enrich_model
        enriched_count = enrich_all(**enrich_kwargs)
        print(f"    Enriched {enriched_count} sessions")

        # Detect content profile for adaptive consolidation filters
        from synapt.recall.content_profile import detect_content_profile
        from synapt.recall.core import parse_transcript
        enrich_chunks = []
        for tp in sorted(archive_dir.glob("*.jsonl")):
            enrich_chunks.extend(parse_transcript(tp))
        c_profile = detect_content_profile(enrich_chunks)

        # Consolidate
        consolidate_kwargs = {
            "project_dir": project_dir,
            "force": True,
            "min_entries": 2,
            "content_profile": c_profile,
        }
        if self.enrich_model:
            consolidate_kwargs["model"] = self.enrich_model
        con_result = consolidate(**consolidate_kwargs)
        print(f"    Knowledge: {con_result.nodes_created} created, "
              f"{con_result.nodes_corroborated} corroborated, "
              f"{con_result.clusters_found} clusters")

        # Sync knowledge nodes into DB
        kn_path = data_dir / "knowledge.jsonl"
        if kn_path.exists() and self._db is not None:
            kn_nodes = read_nodes(kn_path)
            if kn_nodes:
                self._db.save_knowledge_nodes([n.to_dict() for n in kn_nodes])

        # Release MLX resources
        import gc
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass
        gc.collect()

    def query(self, question: str, max_chunks: int = 20) -> str:
        """Retrieve context via synapt recall."""
        if self._index is None:
            raise RuntimeError("Must call ingest() before query()")
        return self._index.lookup(
            query=question,
            max_chunks=max_chunks,
            max_tokens=2000,
            half_life=0,
            threshold_ratio=0.0,
            depth="full",
            knowledge_boost=self.knowledge_boost,
            max_knowledge=self.max_knowledge,
        )

    def stats(self) -> dict:
        """Return index statistics."""
        if self._index is None:
            return {}
        return self._index.stats()

    def close(self) -> None:
        """Clean up resources."""
        if self._db is not None:
            try:
                self._db.close()
            except Exception:
                pass
        if self._work_dir and self._work_dir.exists():
            shutil.rmtree(self._work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """\
You are an intelligent coding assistant with access to memories from past \
development sessions. Your task is to answer questions about a software \
project based on retrieved session transcripts.

# CONTEXT:
You have access to conversation transcripts from coding sessions on a \
software project. These contain discussions about code changes, debugging \
sessions, architecture decisions, and development conventions.

# INSTRUCTIONS:
1. Carefully analyze ALL provided transcript excerpts
2. Pay attention to file paths, function names, config values, library \
versions, and error messages — be precise with technical details
3. If the question asks about a specific decision or change, look for the \
reasoning and outcome in the transcripts
4. For code-related answers, include the exact values (file paths, config \
keys, version numbers) from the transcripts
5. If transcripts contain contradictory information (e.g., a value was \
changed), report the most recent state unless the question asks about history
6. Normalize file paths — treat "./src/foo.py" and "src/foo.py" as the same
7. NEVER say "not mentioned" if any relevant detail appears in the context
8. Answer with specific technical details. Be precise and complete but \
concise — use 2-3 sentences if needed to cover all relevant details.

Retrieved context:
{context}

Question: {question}
Answer:"""

JUDGE_PROMPT = """\
Your task is to label an answer to a coding question as "CORRECT" or "WRONG".

You will be given: (1) a question about a software project's development \
history, (2) a gold (ground truth) answer, (3) a generated answer to score.

The question tests whether a coding assistant remembers details from past \
development sessions — things like library versions, file paths, config \
values, bug fixes, architecture decisions, and coding conventions.

Grading guidelines:
- Be generous: if the generated answer captures the essential fact from the \
gold answer, mark CORRECT even if it includes extra detail or different wording
- IMPORTANT: If the generated answer says it cannot find the information, \
doesn't know, says the evidence is not in the transcripts, or otherwise \
fails to answer the question, mark WRONG — even if the question asks about \
something that was "never implemented" or "never changed"
- For file paths: treat "src/foo.py" and "./src/foo.py" as equivalent
- For version numbers: "2.1" and "v2.1" and "2.1.0" are equivalent
- For code: if the logic is equivalent even with different variable names or \
formatting, mark CORRECT
- For temporal questions: if the answer identifies the correct time period \
(same week/month/session), mark CORRECT even if the exact date differs slightly
- For "why" questions: the answer must capture the core reasoning, not just \
restate the decision

Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, \
then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG \
in your response.

Return the label in JSON format with the key "label"."""


# ---------------------------------------------------------------------------
# Utility functions (shared with locomo_eval.py)
# ---------------------------------------------------------------------------

def _is_anthropic_model(model: str) -> bool:
    return model.startswith("claude-")


def _api_call_with_retry(client, messages, max_tokens=100, retries=3, model="gpt-4o-mini"):
    """Call OpenAI or Anthropic API with exponential backoff retry."""
    for attempt in range(retries):
        try:
            if _is_anthropic_model(model):
                response = client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                return response.content[0].text.strip()
            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"API call failed after {retries} retries: {e}") from e


def normalize_answer(s: str) -> str:
    """Normalize answer for F1 computation."""
    s = str(s).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = " ".join(s.split())
    return s


def token_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def generate_answer(question: str, context: str, client, model: str = "gpt-4o-mini") -> str:
    """Generate a short answer using the configured LLM."""
    prompt = ANSWER_PROMPT.format(context=context, question=question)
    return _api_call_with_retry(
        client, [{"role": "user", "content": prompt}],
        max_tokens=100, model=model,
    )


def judge_answer(
    question: str, gold_answer: str, generated_answer: str,
    client, model: str = "gpt-4o-mini",
) -> int:
    """Judge whether the generated answer is correct. Returns 1 or 0."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=str(gold_answer),
        generated_answer=generated_answer,
    )
    text = _api_call_with_retry(
        client, [{"role": "user", "content": prompt}],
        max_tokens=100, model=model,
    )
    try:
        result = json.loads(text)
        label = result.get("label", "").upper()
    except json.JSONDecodeError:
        label = "CORRECT" if "CORRECT" in text.upper() else "WRONG"
    return 1 if label == "CORRECT" else 0


# ---------------------------------------------------------------------------
# Retrieval recall (evidence-based)
# ---------------------------------------------------------------------------

def _parse_retrieved_turns(retrieved_text: str) -> set[tuple[str, int]]:
    """Extract (session_prefix, turn_index) pairs from formatted retrieval output.

    The lookup() format includes headers like:
        --- [2026-03-10 12:00 session session_001] turn 42 ---

    Returns set of (session_id_prefix, turn_index) tuples.
    """
    turn_pattern = re.compile(
        r"session\s+(\S+)\]\s+turn\s+(\d+)\s+---"
    )
    retrieved_turns: set[tuple[str, int]] = set()
    for m in turn_pattern.finditer(retrieved_text):
        sid_prefix = m.group(1)
        turn_idx = int(m.group(2))
        retrieved_turns.add((sid_prefix, turn_idx))
    return retrieved_turns


def compute_retrieval_recall(
    retrieved_text: str,
    evidence: list[dict],
    session_texts: dict[str, list[str]],
    chunk_line_map: dict[str, dict[int, int]] | None = None,
) -> float:
    """Check how many evidence chunks were retrieved.

    Parses the retrieval output headers to find which (session, turn) pairs
    were returned, then checks if each evidence entry's chunk is among them.

    Args:
        retrieved_text: The formatted context returned by the SUT.
        evidence: List of evidence dicts with session_id, turn_index, etc.
        session_texts: Map from session_id to list of turn texts (indexed by
            turn_index — raw JSONL line numbers). Used as fallback.
        chunk_line_map: Optional map from session_id to {raw_line_num: chunk_turn_index}.
            When provided, maps evidence raw line numbers to chunk turn indices
            for precise header-based matching.

    Returns:
        Fraction of evidence turns found in the retrieved text (0.0 to 1.0).
    """
    if not evidence:
        return 0.0

    # Parse retrieved turn headers
    retrieved_turns = _parse_retrieved_turns(retrieved_text)

    found = 0
    for ev in evidence:
        sid = ev.get("session_id", "")
        tidx = ev.get("turn_index", -1)

        # Tier 1: header-based matching with chunk mapping
        if chunk_line_map and sid in chunk_line_map:
            chunk_tidx = chunk_line_map[sid].get(tidx)
            if chunk_tidx is not None:
                # Match against session prefix (session IDs may be truncated
                # to 8 chars in the header)
                for ret_prefix, ret_tidx in retrieved_turns:
                    if sid.startswith(ret_prefix) and ret_tidx == chunk_tidx:
                        found += 1
                        break
                continue

        # Tier 2: direct header matching (if turn_index is already a chunk index)
        for ret_prefix, ret_tidx in retrieved_turns:
            if sid.startswith(ret_prefix) and ret_tidx == tidx:
                found += 1
                break
        else:
            # Tier 3: raw-line text fallback
            turns = session_texts.get(sid, [])
            if 0 <= tidx < len(turns):
                text = turns[tidx].strip()
                if text and text[:80] in retrieved_text:
                    found += 1

    return found / len(evidence)


def _build_chunk_line_map(session_dir: Path) -> dict[str, dict[int, int]]:
    """Build a map from raw JSONL line numbers to chunk turn indices.

    For each session, parses the transcript into chunks and builds a
    line-number → byte-offset map. Then for each raw line, finds which
    chunk's byte range contains it.

    Returns:
        Map from session_id to {raw_line_number: chunk_turn_index}.
    """
    from synapt.recall.core import parse_transcript

    result: dict[str, dict[int, int]] = {}
    for sp in sorted(session_dir.glob("*.jsonl")):
        sid = sp.stem
        chunks = parse_transcript(sp)
        if not chunks:
            continue

        # Build line → byte offset map
        line_map: dict[int, int] = {}
        offset = 0
        line_num = 0
        with open(sp, "rb") as f:
            for raw_line in f:
                line_map[line_num] = offset
                offset += len(raw_line)
                line_num += 1

        # Single-pass: iterate sorted lines against sorted chunks
        raw_to_chunk: dict[int, int] = {}
        sorted_chunks = sorted(chunks, key=lambda c: c.byte_offset)
        ci = 0
        for ln in range(line_num):
            byte_off = line_map[ln]
            # Advance chunk pointer past chunks that end before this line
            while ci < len(sorted_chunks) - 1:
                c = sorted_chunks[ci]
                if byte_off < c.byte_offset + c.byte_length:
                    break
                ci += 1
            c = sorted_chunks[ci]
            if c.byte_offset <= byte_off < c.byte_offset + c.byte_length:
                raw_to_chunk[ln] = c.turn_index

        result[sid] = raw_to_chunk

    return result


def load_session_texts(session_dir: Path) -> dict[str, list[str]]:
    """Load all session transcripts and return a map of session_id -> turn texts."""
    texts: dict[str, list[str]] = {}
    for sp in sorted(session_dir.glob("*.jsonl")):
        session_id = sp.stem
        turn_texts = []
        for line in open(sp, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
                content = ""
                if isinstance(msg.get("message"), dict):
                    for block in msg["message"].get("content", []):
                        if isinstance(block, dict) and block.get("type") == "text":
                            content += block.get("text", "")
                        elif isinstance(block, dict) and block.get("type") == "tool_result":
                            tr = block.get("content", "")
                            if isinstance(tr, str):
                                content += tr
                            elif isinstance(tr, list):
                                for sub in tr:
                                    if isinstance(sub, dict) and sub.get("type") == "text":
                                        content += sub.get("text", "")
                                    elif isinstance(sub, str):
                                        content += sub
                        elif isinstance(block, str):
                            content += block
                turn_texts.append(content)
            except json.JSONDecodeError:
                turn_texts.append("")
        texts[session_id] = turn_texts
    return texts


# ---------------------------------------------------------------------------
# Project discovery
# ---------------------------------------------------------------------------

def discover_projects(project_filter: str | None = None) -> list[Path]:
    """Find all project directories under data/."""
    projects = []
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "manifest.json").exists():
            continue
        if not (d / "questions.json").exists():
            continue
        if project_filter and d.name != project_filter:
            continue
        projects.append(d)
    return projects


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    project_dirs: list[Path],
    sut: SystemUnderTest,
    retrieval_only: bool = False,
    max_chunks: int = 20,
    output_path: Path | None = None,
    model: str = "gpt-4o-mini",
) -> dict:
    """Run the CodeMemo evaluation across one or more projects.

    Args:
        project_dirs: List of project directories to evaluate.
        sut: SystemUnderTest implementation.
        retrieval_only: Skip answer generation and judging.
        max_chunks: Chunks to retrieve per question.
        output_path: Directory for results JSON.
        model: OpenAI model for answer generation and judging.

    Returns:
        Summary dict with per-category and overall scores.
    """
    client = None
    if not retrieval_only:
        try:
            if _is_anthropic_model(model):
                from anthropic import Anthropic
                client = Anthropic()
            else:
                from openai import OpenAI
                client = OpenAI()
        except Exception as e:
            print(f"API client failed: {e}")
            print("Falling back to retrieval-only mode.")
            retrieval_only = True

    all_results = []
    category_scores: dict[int, list] = defaultdict(list)
    category_f1: dict[int, list] = defaultdict(list)
    category_recall: dict[int, list] = defaultdict(list)
    total_retrieve_time = 0.0
    total_generate_time = 0.0
    total_judge_time = 0.0
    total_chunks = 0
    total_knowledge = 0
    total_questions = 0

    for project_dir in project_dirs:
        project_id = project_dir.name
        print(f"\n{'='*60}")
        print(f"Project: {project_id}")
        print(f"{'='*60}")

        # Load manifest
        with open(project_dir / "manifest.json") as f:
            manifest = json.load(f)
        print(f"  {manifest.get('description', '')}")
        print(f"  Tech: {', '.join(manifest.get('tech_stack', []))}")

        # Load questions
        with open(project_dir / "questions.json") as f:
            questions = json.load(f)
        print(f"  Questions: {len(questions)}")

        # Find session files
        session_dir = project_dir / "sessions"
        session_paths = sorted(session_dir.glob("*.jsonl"))
        if not session_paths:
            print(f"  WARNING: No session transcripts found in {session_dir}")
            print(f"  Skipping project (transcripts not yet generated)")
            continue

        # Load session texts for retrieval recall
        session_texts = load_session_texts(session_dir)

        # Build chunk-line map for precise retrieval recall
        chunk_line_map = _build_chunk_line_map(session_dir)

        # Ingest sessions into SUT
        print(f"  Ingesting {len(session_paths)} sessions...")
        t0 = time.time()
        sut.ingest(session_paths)
        ingest_time = time.time() - t0
        stats = sut.stats() if hasattr(sut, "stats") else {}
        chunk_count = stats.get("chunk_count", 0)
        kn_count = stats.get("knowledge_count", 0)
        total_chunks += chunk_count
        total_knowledge += kn_count
        kn_msg = f", {kn_count} knowledge nodes" if kn_count else ""
        print(f"  Indexed: {chunk_count} chunks{kn_msg} in {ingest_time:.1f}s")

        # Evaluate each question
        for i, qa in enumerate(questions):
            qid = qa["id"]
            question = qa["question"]
            gold = str(qa.get("answer_short", qa["answer"]))
            category = qa["category"]
            evidence = qa.get("evidence", [])

            if category not in ALL_CATEGORIES:
                continue

            # Retrieve
            t0 = time.time()
            context = sut.query(question, max_chunks=max_chunks)
            retrieve_time = time.time() - t0
            total_retrieve_time += retrieve_time

            # Retrieval recall
            recall_at_k = compute_retrieval_recall(
                context, evidence, session_texts,
                chunk_line_map=chunk_line_map,
            )
            category_recall[category].append(recall_at_k)

            result_entry = {
                "project": project_id,
                "id": qid,
                "question": question,
                "gold_answer": gold,
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, "?"),
                "context": context,
                "context_length": len(context),
                "retrieve_time_ms": round(retrieve_time * 1000, 1),
                "retrieval_recall": recall_at_k,
            }

            if not retrieval_only and client:
                # Generate
                t0 = time.time()
                generated = generate_answer(question, context, client, model=model)
                generate_time = time.time() - t0
                total_generate_time += generate_time

                # F1
                f1 = token_f1(generated, gold)
                category_f1[category].append(f1)

                # Judge
                t0 = time.time()
                j_score = judge_answer(question, gold, generated, client, model=model)
                judge_time = time.time() - t0
                total_judge_time += judge_time

                category_scores[category].append(j_score)

                result_entry.update({
                    "generated_answer": generated,
                    "j_score": j_score,
                    "f1": round(f1, 4),
                    "generate_time_ms": round(generate_time * 1000, 1),
                    "judge_time_ms": round(judge_time * 1000, 1),
                })

            all_results.append(result_entry)
            total_questions += 1

            # Progress
            if (i + 1) % 10 == 0 or i == len(questions) - 1:
                elapsed = total_retrieve_time + total_generate_time + total_judge_time
                if not retrieval_only and category_scores:
                    all_j = [s for scores in category_scores.values() for s in scores]
                    running_j = sum(all_j) / len(all_j) * 100 if all_j else 0
                    print(f"  [{i+1}/{len(questions)}] J={running_j:.1f}% "
                          f"({elapsed:.0f}s elapsed)")
                else:
                    print(f"  [{i+1}/{len(questions)}] retrieval done "
                          f"({elapsed:.0f}s elapsed)")

    # ---------------------------------------------------------------------------
    # Aggregate scores
    # ---------------------------------------------------------------------------
    # Resolve enrichment model for provenance
    enrich_model_name = ""
    if isinstance(sut, SynaptSUT):
        enrich_model_name = sut.enrich_model
        if not enrich_model_name and sut.mode == "full-pipeline":
            try:
                from synapt.recall._model_router import DEFAULT_DECODER_MODEL
                enrich_model_name = DEFAULT_DECODER_MODEL
            except ImportError:
                enrich_model_name = "(default — unknown)"

    summary: dict = {
        "benchmark": "codememo",
        "projects": [d.name for d in project_dirs],
        "questions_evaluated": total_questions,
        "max_chunks": max_chunks,
        "index_chunks": total_chunks,
        "model": model,
        "enrichment_model": enrich_model_name or "none",
        "pipeline_mode": sut.mode if isinstance(sut, SynaptSUT) else "external",
    }
    if total_knowledge:
        summary["knowledge_nodes"] = total_knowledge
    if total_questions:
        summary["avg_retrieve_ms"] = round(
            total_retrieve_time / total_questions * 1000, 1)

    # Retrieval recall by category
    for cat in sorted(ALL_CATEGORIES):
        if category_recall[cat]:
            avg = sum(category_recall[cat]) / len(category_recall[cat])
            summary[f"recall@{max_chunks}_{CATEGORY_NAMES[cat]}"] = round(avg * 100, 2)
            summary[f"n_{CATEGORY_NAMES[cat]}"] = len(category_recall[cat])

    if not retrieval_only and category_scores:
        summary["avg_generate_ms"] = round(
            total_generate_time / total_questions * 1000, 1)
        summary["avg_judge_ms"] = round(
            total_judge_time / total_questions * 1000, 1)

        # J scores by category
        for cat in sorted(ALL_CATEGORIES):
            if category_scores[cat]:
                j = sum(category_scores[cat]) / len(category_scores[cat]) * 100
                summary[f"j_score_{CATEGORY_NAMES[cat]}"] = round(j, 2)

        # Overall J score
        all_j = [s for scores in category_scores.values() for s in scores]
        summary["j_score_overall"] = round(
            sum(all_j) / len(all_j) * 100, 2) if all_j else 0

        # F1 by category
        for cat in sorted(ALL_CATEGORIES):
            if category_f1[cat]:
                avg_f1 = sum(category_f1[cat]) / len(category_f1[cat]) * 100
                summary[f"f1_{CATEGORY_NAMES[cat]}"] = round(avg_f1, 2)

        all_f1 = [f for scores in category_f1.values() for f in scores]
        summary["f1_overall"] = round(
            sum(all_f1) / len(all_f1) * 100, 2) if all_f1 else 0

    # Print summary
    print("\n" + "=" * 60)
    print("CODEMEMO BENCHMARK RESULTS")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Save results
    if output_path is None:
        output_path = Path("evaluation/codememo/results")
    output_path.mkdir(parents=True, exist_ok=True)

    with open(output_path / "codememo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_path / "codememo_detailed.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}/")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="CodeMemo benchmark for coding-assistant memory systems")
    parser.add_argument(
        "--project", type=str, default=None,
        help="Evaluate a single project (directory name under data/)",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Only measure retrieval quality (no API key needed)",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=20,
        help="Number of chunks to retrieve per question (default: 20)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--recalldb", action="store_true",
        help="Use RecallDB + FTS5 + embeddings (fast, no MLX enrichment)",
    )
    parser.add_argument(
        "--full-pipeline", action="store_true",
        help="Use full pipeline: RecallDB + enrich + consolidate + knowledge graph",
    )
    parser.add_argument(
        "--enrich-model", type=str, default="",
        help="MLX model for enrichment/consolidation",
    )
    parser.add_argument(
        "--max-knowledge", type=int, default=None,
        help="Maximum knowledge blocks per query",
    )
    parser.add_argument(
        "--knowledge-boost", type=float, default=None,
        help="Override knowledge node boost factor",
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4o-mini",
        help="OpenAI model for answer generation and judging (default: gpt-4o-mini)",
    )
    args = parser.parse_args()

    # Discover projects
    project_dirs = discover_projects(args.project)
    if not project_dirs:
        print(f"No projects found in {DATA_DIR}")
        if args.project:
            print(f"  (filtered for: {args.project})")
        sys.exit(1)

    print(f"Found {len(project_dirs)} project(s): "
          f"{', '.join(d.name for d in project_dirs)}")

    # Set up SUT
    if args.full_pipeline:
        mode = "full-pipeline"
    elif args.recalldb:
        mode = "recalldb"
    else:
        mode = "baseline"

    sut = SynaptSUT(
        mode=mode,
        enrich_model=args.enrich_model,
        max_knowledge=args.max_knowledge,
        knowledge_boost=args.knowledge_boost,
    )

    try:
        run_evaluation(
            project_dirs=project_dirs,
            sut=sut,
            retrieval_only=args.retrieval_only,
            max_chunks=args.max_chunks,
            output_path=args.output,
            model=args.model,
        )
    finally:
        sut.close()


if __name__ == "__main__":
    main()
