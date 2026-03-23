"""LOCOMO benchmark evaluation for synapt recall.

Evaluates synapt's hybrid RRF search against the LOCOMO dataset,
following Mem0's methodology (J score via gpt-4o-mini LLM-as-Judge).

Usage:
    # Full run (requires OPENAI_API_KEY):
    python evaluation/locomo_eval.py

    # Full pipeline (RecallDB + enrich + consolidate + knowledge graph):
    python evaluation/locomo_eval.py --full-pipeline

    # Retrieval-only (no API key needed):
    python evaluation/locomo_eval.py --retrieval-only

    # Quick test on 1 conversation:
    python evaluation/locomo_eval.py --max-conversations 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Category names (following LOCOMO paper)
# ---------------------------------------------------------------------------
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "single-hop",
    4: "open-domain",
    5: "adversarial",
}

# Exclude adversarial (cat 5) following Mem0's methodology
EVAL_CATEGORIES = {1, 2, 3, 4}

# Audit log path (relative to repo root)
_AUDIT_FILE = Path(__file__).parent.parent / "docs" / "audit.jsonl"


# ---------------------------------------------------------------------------
# Token usage tracker — accumulates across all API calls
# ---------------------------------------------------------------------------

class TokenTracker:
    """Accumulate token usage across real-time and batch API calls."""

    # gpt-4o-mini pricing (USD per 1M tokens)
    _INPUT_PRICE = 0.15
    _OUTPUT_PRICE = 0.60
    _CACHED_INPUT_PRICE = 0.075  # 50% discount for cached

    def __init__(self):
        self.reset()

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0
        self.calls = 0

    def add_openai(self, usage):
        """Add usage from an OpenAI response.usage object or dict."""
        u = usage if isinstance(usage, dict) else usage.model_dump()
        self.prompt_tokens += u.get("prompt_tokens", 0)
        self.completion_tokens += u.get("completion_tokens", 0)
        # OpenAI reports cached tokens inside prompt_tokens_details
        details = u.get("prompt_tokens_details") or {}
        self.cached_tokens += details.get("cached_tokens", 0)
        self.calls += 1

    def summary(self) -> dict:
        uncached = self.prompt_tokens - self.cached_tokens
        cost = (
            uncached * self._INPUT_PRICE / 1_000_000
            + self.cached_tokens * self._CACHED_INPUT_PRICE / 1_000_000
            + self.completion_tokens * self._OUTPUT_PRICE / 1_000_000
        )
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "uncached_tokens": uncached,
            "cache_hit_rate": round(
                self.cached_tokens / self.prompt_tokens * 100, 1
            ) if self.prompt_tokens else 0.0,
            "api_calls": self.calls,
            "estimated_cost_usd": round(cost, 4),
        }


_token_tracker = TokenTracker()


def _detect_enrich_model(enrich_model: str = "", backend: str = "") -> str:
    """Detect the enrichment model that will be used for this run."""
    if enrich_model:
        return enrich_model
    if backend == "modal":
        return "mistralai/Ministral-8B-Instruct-2410"
    if backend == "ollama":
        return "ollama (default)"
    # MLX default — read from model router
    try:
        from synapt.recall._model_router import DEFAULT_DECODER_MODEL
        return DEFAULT_DECODER_MODEL
    except ImportError:
        return "unknown"


def _audit_log(entry: dict) -> None:
    """Append a JSON entry to docs/audit.jsonl."""
    try:
        _AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_AUDIT_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        print(f"  [audit] Warning: could not write audit log: {e}")


def _audit_start(
    pipeline: str,
    max_conversations: int | None,
    max_chunks: int,
    batch: bool,
    enrich_model: str,
    backend: str,
    conv_offset: int = 0,
) -> str:
    """Log a RUNNING entry and return its timestamp (used as ID)."""
    ts = datetime.now(timezone.utc).isoformat()
    model = _detect_enrich_model(enrich_model, backend)
    entry = {
        "type": "eval",
        "eval_type": "locomo",
        "timestamp": ts,
        "outcome": "RUNNING",
        "pipeline": pipeline,
        "batch_mode": batch,
        "enrichment_model": model,
        "backend": backend or "auto",
        "max_conversations": max_conversations,
        "max_chunks": max_chunks,
        "conv_offset": conv_offset,
        "note": "Auto-logged by locomo_eval.py",
    }
    _audit_log(entry)
    print(f"  [audit] Logged RUNNING entry (enrichment: {model})")
    return ts


def _audit_finish(start_ts: str, summary: dict, outcome: str = "SUCCESS") -> None:
    """Log a completion entry with scores."""
    entry = {
        "type": "eval",
        "eval_type": "locomo",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "start_timestamp": start_ts,
        "outcome": outcome,
        "pipeline": summary.get("pipeline", "unknown"),
        "enrichment_model": summary.get("enrichment_model", "unknown"),
        "conversations": summary.get("conversations", 0),
        "questions_evaluated": summary.get("questions_evaluated", 0),
        "j_score_overall": summary.get("j_score_overall"),
        "f1_overall": summary.get("f1_overall"),
        "j_score_multi-hop": summary.get("j_score_multi-hop"),
        "j_score_temporal": summary.get("j_score_temporal"),
        "j_score_single-hop": summary.get("j_score_single-hop"),
        "j_score_open-domain": summary.get("j_score_open-domain"),
        "note": "Auto-logged by locomo_eval.py",
    }
    _audit_log(entry)
    j = summary.get("j_score_overall", "?")
    print(f"  [audit] Logged {outcome} — J-Score: {j}%")


def _resolve_work_dir(output_path) -> Path:
    """Get or create a work directory for enrichment, cached via work_dir.txt.

    Must be outside the git/gripspace tree so project_data_dir resolves
    to the temp dir itself, not the repo root.
    """
    work_dir = Path(tempfile.mkdtemp(prefix="synapt_locomo_"))
    if output_path:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        work_ref = Path(output_path) / "work_dir.txt"
        if work_ref.exists():
            cached_wd = Path(work_ref.read_text().strip())
            if cached_wd.exists():
                work_dir = cached_wd
            else:
                work_ref.write_text(str(work_dir))
        else:
            work_ref.write_text(str(work_dir))
    return work_dir


# ---------------------------------------------------------------------------
# Step 1: Convert LOCOMO conversations to synapt transcript format
# ---------------------------------------------------------------------------

_LOCOMO_DATE_FMTS = [
    "%I:%M %p on %d %B, %Y",   # "1:56 pm on 8 May, 2023"
    "%I:%M %p on %d %b, %Y",   # "1:56 pm on 8 May, 2023" (abbr)
    "%d %B, %Y",               # "8 May, 2023"
    "%d %B %Y",                # "8 May 2023"
    "%B %d, %Y",               # "May 8, 2023"
]


def parse_locomo_date(raw: str):
    """Parse a LOCOMO free-text date string into a timezone-aware datetime.

    Returns None if no format matches.
    """
    from datetime import datetime, timezone
    for fmt in _LOCOMO_DATE_FMTS:
        try:
            dt = datetime.strptime(raw.strip(), fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def extract_anchor_date(item: dict):
    """Extract the last session timestamp as a datetime for recency anchoring.

    Returns a datetime representing the conversation's final timestamp,
    enabling meaningful recency decay on historical data.
    """
    conv = item.get("conversation", {})
    dates = []
    for key in conv:
        if key.endswith("_date_time"):
            dates.append(conv[key])
    if not dates:
        return None
    return parse_locomo_date(dates[-1])


def extract_date_range(item: dict) -> str:
    """Extract the session date range from a LOCOMO conversation.

    Returns a string like "8 May, 2023 to 15 January, 2024" for use
    in answer prompts.
    """
    conv = item.get("conversation", {})
    dates = []
    for key in conv:
        if key.endswith("_date_time"):
            dates.append(conv[key])
    if not dates:
        return ""
    # LOCOMO dates are free-text like "1:56 pm on 8 May, 2023"
    # Return first and last for the range
    return f"{dates[0]} to {dates[-1]}"


def locomo_to_transcripts(data: list[dict], output_dir: Path) -> list[Path]:
    """Convert LOCOMO conversations into synapt-compatible JSONL transcripts.

    Each LOCOMO session becomes its own transcript file, matching synapt's
    one-session-per-file model. This gives the search engine proper session
    boundaries for progressive and time-ordered retrieval.
    """
    transcript_paths = []
    for conv_idx, item in enumerate(data):
        conv = item.get("conversation", {})
        speaker_a = conv.get("speaker_a", "Speaker_A")

        # Collect all sessions in order
        session_keys = sorted(
            [k for k in conv if k.startswith("session_") and not k.endswith("_date_time")],
            key=lambda k: int(k.split("_")[1]),
        )

        for session_key in session_keys:
            date_key = f"{session_key}_date_time"
            session_date = conv.get(date_key, "")
            turns = conv[session_key]
            session_num = session_key.split("_")[1]

            lines = []
            for turn in turns:
                speaker = turn.get("speaker", "")
                text = turn.get("text", "")
                dia_id = turn.get("dia_id", "")

                role = "user" if speaker == speaker_a else "assistant"
                # Session ID format: put session num first so session_id[:8]
                # is unique across sessions (chunk IDs use first 8 chars).
                sess_id = f"s{session_num:0>3}c{conv_idx:02d}"
                msg = {
                    "type": role,
                    "message": {
                        "role": role,
                        "content": [{"type": "text", "text": text}],
                    },
                    "timestamp": session_date,
                    "uuid": dia_id,
                    "sessionId": sess_id,
                    "parentUuid": "",
                }
                lines.append(json.dumps(msg))

            transcript_path = output_dir / f"s{session_num:0>3}c{conv_idx:02d}.jsonl"
            transcript_path.write_text("\n".join(lines))
            transcript_paths.append(transcript_path)

    return transcript_paths


# ---------------------------------------------------------------------------
# Step 2: Build synapt index over LOCOMO transcripts
# ---------------------------------------------------------------------------

def build_synapt_index(transcript_dir: Path, index_dir: Path, db=None):
    """Build a synapt recall index from LOCOMO transcripts."""
    from synapt.recall.core import build_index

    # build_index discovers .jsonl files in transcript_dir, parses them,
    # and returns a TranscriptIndex with in-memory BM25 + optional embeddings.
    # When db is provided and populated, uses FTS5 instead of in-memory BM25.
    index = build_index(
        source_dir=transcript_dir,
        use_embeddings=True,
        cache_dir=index_dir,
        db=db,
    )
    return index


# ---------------------------------------------------------------------------
# Full pipeline: RecallDB + enrich + consolidate + knowledge graph
# ---------------------------------------------------------------------------

def setup_recalldb_index(
    item: dict,
    conv_idx: int,
    conv_dir: Path,
) -> tuple:
    """Set up RecallDB-backed index for one LOCOMO conversation.

    Fast path: RecallDB + FTS5 + embeddings, no MLX enrichment.
    Returns (index, db) — caller must close db when done.
    """
    from synapt.recall.storage import RecallDB
    from synapt.recall.core import parse_transcript

    transcript_dir = conv_dir / "transcripts"
    index_dir = conv_dir / "index"
    transcript_dir.mkdir(parents=True)
    index_dir.mkdir(parents=True)

    transcript_paths = locomo_to_transcripts([item], transcript_dir)

    # Create RecallDB and populate chunks for FTS5
    db = RecallDB(index_dir / "recall.db")
    all_chunks = []
    for tp in transcript_paths:
        chunks = parse_transcript(tp)
        all_chunks.extend(chunks)
    db.save_chunks(all_chunks)

    # Build index using FTS5 backend + embeddings
    index = build_synapt_index(transcript_dir, index_dir, db=db)

    # Build cross-session links for threading
    if index._all_embeddings:
        try:
            n_links = index.build_cross_session_links()
            if n_links:
                print(f"    Cross-session links: {n_links}")
        except Exception:
            pass

    return index, db


def setup_full_pipeline_index(
    item: dict,
    conv_idx: int,
    conv_dir: Path,
    enrich_model: str = "",
) -> tuple:
    """Set up full synapt pipeline for one LOCOMO conversation.

    Creates the directory structure, writes transcripts, creates journal
    stubs, runs enrich + consolidate via MLX, builds RecallDB-backed
    index with knowledge graph.

    Returns (index, db) — caller must close db when done.
    """
    from synapt.recall.storage import RecallDB
    from synapt.recall.enrich import enrich_all
    from synapt.recall.consolidate import consolidate
    from synapt.recall.core import parse_transcript
    from synapt.recall.journal import JournalEntry, append_entry

    conv_dir = conv_dir.resolve()
    conv_name = conv_dir.name
    data_dir = conv_dir / ".synapt" / "recall"
    wt_dir = data_dir / "worktrees" / conv_name
    archive_dir = wt_dir / "transcripts"
    index_dir = data_dir / "index"

    # Check if enrichment data already exists (cache from previous run)
    kn_path = data_dir / "knowledge.jsonl"
    journal_path = wt_dir / "journal.jsonl"
    has_cache = (archive_dir.exists()
                 and journal_path.exists()
                 and kn_path.exists())

    if has_cache:
        print(f"    Using cached enrichment data from {conv_dir}")
        transcript_paths = sorted(archive_dir.glob("*.jsonl"))
    else:
        archive_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)

        # Write transcripts to archive directory (where enrich_all finds them)
        transcript_paths = locomo_to_transcripts([item], archive_dir)

        # Create journal auto-stubs for each session
        for tp in transcript_paths:
            session_id = tp.stem
            # Read first line to get session timestamp
            timestamp = ""
            try:
                first_line = tp.read_text().split("\n", 1)[0]
                timestamp = json.loads(first_line).get("timestamp", "")
            except (json.JSONDecodeError, IndexError):
                pass
            stub = JournalEntry(
                timestamp=timestamp or "2024-01-01T00:00:00+00:00",
                session_id=session_id,
                auto=True,
                enriched=False,
            )
            append_entry(stub, journal_path)

        # Run enrichment (LLM → structured session summaries)
        enrich_kwargs = {"project_dir": conv_dir, "max_entries": 0}
        if enrich_model:
            enrich_kwargs["model"] = enrich_model
        enriched_count = enrich_all(**enrich_kwargs)
        print(f"    Enriched {enriched_count} sessions")

        # Detect content profile for adaptive filtering
        from synapt.recall.content_profile import detect_content_profile
        profile_chunks = []
        for tp in transcript_paths:
            parsed = parse_transcript(tp)
            profile_chunks.extend(parsed)
        c_profile = detect_content_profile(profile_chunks)
        print(f"    Content profile: {c_profile.content_type}")

        # Run consolidation (MLX → knowledge nodes → syncs to DB)
        consolidate_kwargs = {
            "project_dir": conv_dir,
            "force": True,
            "min_entries": 2,
            "content_profile": c_profile,
        }
        if enrich_model:
            consolidate_kwargs["model"] = enrich_model
        con_result = consolidate(**consolidate_kwargs)
        print(f"    Knowledge: {con_result.nodes_created} created, "
              f"{con_result.nodes_corroborated} corroborated, "
              f"{con_result.clusters_found} clusters")

    # Create/recreate RecallDB and populate with chunks for FTS5
    # (Always rebuild — retrieval params may have changed)
    db_path = index_dir / "recall.db"
    for suffix in ("", "-wal", "-shm"):
        p = db_path.parent / (db_path.name + suffix)
        if p.exists():
            p.unlink()
    index_dir.mkdir(parents=True, exist_ok=True)
    db = RecallDB(db_path)
    all_chunks = []
    for tp in transcript_paths:
        chunks = parse_transcript(tp)
        all_chunks.extend(chunks)
    db.save_chunks(all_chunks)

    # Sync knowledge nodes into the SAME db object used for index building.
    # consolidate() writes to knowledge.jsonl and syncs via a separate DB
    # connection — WAL mode means the original connection may not see those
    # writes. Re-syncing here ensures knowledge is visible at index time.
    from synapt.recall.knowledge import read_nodes
    if kn_path.exists():
        kn_nodes = read_nodes(kn_path)
        if kn_nodes:
            db.save_knowledge_nodes([n.to_dict() for n in kn_nodes])

    # Release MLX resources before loading sentence-transformers embedding
    # model — Metal GPU context from enrichment can deadlock with MiniLM load.
    import gc
    try:
        import mlx.core as mx
        mx.metal.clear_cache()
    except Exception:
        pass
    gc.collect()

    # Build index with RecallDB (FTS5 + embeddings + knowledge)
    index = build_synapt_index(archive_dir, index_dir, db=db)

    # Build cross-session links for threading
    if index._all_embeddings:
        try:
            n_links = index.build_cross_session_links()
            if n_links:
                print(f"    Cross-session links: {n_links}")
        except Exception:
            pass

    return index, db


# ---------------------------------------------------------------------------
# Step 3: Retrieve context for each question
# ---------------------------------------------------------------------------

def retrieve_context(
    index, question: str, max_chunks: int = 5, max_tokens: int = 2000,
    anchor_date=None, knowledge_boost: float | None = None,
    max_knowledge: int | None = None,
) -> str:
    """Use synapt's hybrid search to retrieve context for a question.

    When anchor_date is set (datetime), recency decay uses that as "now"
    instead of the real current time. This enables meaningful recency bias
    on historical conversations (e.g., more recent sessions score higher).
    """
    # Use moderate recency decay when anchor is set — 90-day half-life
    # gives ~50% score for chunks 3 months before the anchor, which gently
    # prioritizes recent sessions without harshly penalizing older ones.
    half_life = 90.0 if anchor_date else 0
    result = index.lookup(
        query=question,
        max_chunks=max_chunks,
        max_tokens=max_tokens,
        half_life=half_life,
        threshold_ratio=0.0,  # Don't drop low-scoring results
        depth="full",
        now=anchor_date,
        knowledge_boost=knowledge_boost,
        max_knowledge=max_knowledge,
    )
    return result


# ---------------------------------------------------------------------------
# Step 4: Generate answer using LLM
# ---------------------------------------------------------------------------

ANSWER_PROMPT = """You are an intelligent memory assistant tasked with retrieving accurate \
information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These \
memories contain timestamped information that may be relevant to answering \
the question.{date_range}

# INSTRUCTIONS:
1. Carefully analyze ALL provided memories from both speakers
2. Pay special attention to timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct \
evidence in the memories — quote specific details (names, places, objects)
4. If the memories contain contradictory information, prioritize the most \
recent memory
5. If there is a question about time references (like "last year", "two \
months ago", etc.), calculate the actual date based on the memory timestamp
6. Always convert relative time references to specific dates, months, or years
7. Focus only on the content of the memories from both speakers
8. For questions asking "how many", "which ones", or listing multiple items, \
scan EVERY memory block and count/list ALL distinct instances — do not stop \
after the first mention
9. NEVER say "not mentioned" or "not specifically stated" if any relevant \
detail appears anywhere in the retrieved context. Extract and state the fact \
directly, even if it appears in casual conversation
10. Answer with the specific details from the memories. Be precise — include \
exact names, descriptions, and facts rather than general summaries. Be \
complete but concise — use 2-3 sentences if needed to cover all relevant details.

Retrieved context:
{context}

Question: {question}
Answer:"""

MULTIHOP_ANSWER_PROMPT = """You are an intelligent memory assistant. The question below \
requires combining information from MULTIPLE different conversations or memory blocks.

# CONTEXT:
You have access to memories from two speakers spanning multiple conversations. \
The answer requires synthesizing facts scattered across different memory blocks — \
no single block contains the complete answer.{date_range}

# CRITICAL INSTRUCTIONS:
1. Read EVERY memory block — the answer pieces are SCATTERED across different ones
2. Collect ALL relevant facts, details, and mentions from ALL memory blocks
3. If the question asks about preferences, hobbies, activities, or experiences, \
list EVERY distinct instance found across all memories
4. If the question asks about a person's career, goals, or plans, trace the \
progression across conversations — early mentions may set the foundation, later \
ones show the decision
5. DO NOT stop after finding one relevant memory — scan ALL remaining blocks \
for additional information
6. If different memories mention the same topic with different details, include \
ALL details (they often complement rather than contradict each other)
7. NEVER say "not mentioned" — if ANY relevant detail appears ANYWHERE in the \
memories, extract and state it
8. Answer with specific details from the memories. Be precise — include exact \
names, places, dates, and facts. Be complete — cover all relevant details found.

Retrieved context:
{context}

Question: {question}
Answer:"""

TEMPORAL_ANSWER_PROMPT = """You are a time traveler who has arrived at the exact moment \
each conversation happened. You must answer questions about WHEN things occurred by \
computing real dates from the timestamps you see.

# CONTEXT:
You have access to memories from two speakers in a conversation. Each memory \
block has a timestamp in its header showing when that conversation happened. \
Imagine you are THERE at that timestamp — "last week" means the week before \
the timestamp, "yesterday" means the day before the timestamp, etc.{date_range}

# CRITICAL INSTRUCTIONS FOR TEMPORAL QUESTIONS:
1. Find the memory block where the event is mentioned
2. Read the timestamp in that block's header (e.g., "1:56 pm on 8 May, 2023")
3. You are NOW at that timestamp. Resolve ALL relative time references from \
that anchor point:
   - "last Tuesday" with timestamp "8 May, 2023" (Monday) → 2 May, 2023
   - "two months ago" with timestamp June 2023 → April 2023
   - "next week" with timestamp "3 July, 2023" → ~10 July, 2023
   - "recently" or "the other day" → within the week before the timestamp
4. You MUST answer with a specific date, month, or year — NEVER use relative \
references like "last week" or "recently" in your answer
5. The conversation timestamp is ONLY an anchor for resolving relative phrases. \
It is NOT automatically the date of the event itself. If someone is recalling an \
earlier event, do not answer with the conversation date unless the text clearly \
says the event happened that same day.
6. If the event is described as happening relative to the conversation time, \
SHOW YOUR REASONING: state the anchor timestamp, the relative expression, \
and the computed date
7. If multiple memories mention the same event, prefer the memory that gives the \
clearest date evidence for WHEN it happened, not merely the latest conversation \
where it was discussed.
8. If no relative phrase is present, look for an explicit absolute date, month, \
year, season, holiday, birthday, or "during X" reference in the memory text. Do \
NOT default to the header timestamp just because the event is mentioned there.
9. Answer with specific details from the memories. Be precise — include \
exact dates, names, and facts. Be complete but concise — use 2-3 sentences \
if needed to show your date reasoning.

Retrieved context:
{context}

Question: {question}
Answer:"""


def _api_call_with_retry(client, messages, max_tokens=50, retries=3, model="gpt-4o-mini"):
    """Call OpenAI API with exponential backoff retry."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            if response.usage:
                _token_tracker.add_openai(response.usage)
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"API call failed after {retries} retries: {e}") from e


def build_answer_prompt(question: str, context: str, category: int = 0, date_range: str = "") -> str:
    """Build the category-appropriate answer prompt for LOCOMO."""
    dr_text = f"\nThe conversations span from {date_range}." if date_range else ""
    if category == 1:
        return MULTIHOP_ANSWER_PROMPT.format(
            context=context, question=question, date_range=dr_text,
        )
    if category == 2:
        return TEMPORAL_ANSWER_PROMPT.format(
            context=context, question=question, date_range=dr_text,
        )
    return ANSWER_PROMPT.format(
        context=context, question=question, date_range=dr_text,
    )


def generate_answer(
    question: str, context: str, client,
    category: int = 0, date_range: str = "",
) -> str:
    """Generate a short answer using gpt-4o-mini."""
    prompt = build_answer_prompt(
        question, context, category=category, date_range=date_range,
    )
    return _api_call_with_retry(
        client, [{"role": "user", "content": prompt}], max_tokens=100,
    )


# ---------------------------------------------------------------------------
# Step 5: LLM-as-Judge scoring
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """Your task is to label an answer to a question as "CORRECT" or "WRONG".
You will be given the following data: (1) a question (posed by one user \
to another user), (2) a 'gold' (ground truth) answer, (3) a generated \
answer which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know \
about the other user based on their prior conversations. The gold answer \
will usually be a concise and short answer that includes the referenced \
topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous \
with your grading - as long as it touches on the same topic as the gold \
answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, \
month, year, etc. The generated answer might be much longer or use \
relative time references (like 'last Tuesday' or 'next month'), but you \
should be generous with your grading - as long as it refers to the same \
date or time period as the gold answer, it should be counted as CORRECT. \
Even if the format differs (e.g., 'May 7th' vs '7 May'), consider it \
CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, \
then finish with CORRECT or WRONG. Do NOT include both CORRECT and WRONG \
in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key \
as "label"."""


def judge_answer(question: str, gold_answer: str, generated_answer: str, client) -> int:
    """Judge whether the generated answer is correct. Returns 1 or 0."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        gold_answer=str(gold_answer),
        generated_answer=generated_answer,
    )
    text = _api_call_with_retry(
        client, [{"role": "user", "content": prompt}], max_tokens=100,
    )

    # Parse JSON response
    try:
        result = json.loads(text)
        label = result.get("label", "").upper()
    except json.JSONDecodeError:
        # Fallback: look for CORRECT/WRONG in text
        label = "CORRECT" if "CORRECT" in text.upper() else "WRONG"

    return 1 if label == "CORRECT" else 0


# ---------------------------------------------------------------------------
# Step 6: F1 score (token overlap, following LOCOMO paper)
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """Normalize answer for F1 computation."""
    s = str(s).lower()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = re.sub(r"[^\w\s]", "", s)
    # Collapse whitespace
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


# ---------------------------------------------------------------------------
# Step 7: Retrieval recall (evidence-based)
# ---------------------------------------------------------------------------

def build_evidence_map(data: list[dict]) -> dict[tuple[int, str], str]:
    """Build a mapping from (conv_idx, dia_id) to text content.

    dia_ids like 'D1:3' repeat across conversations, so we scope by conv index.
    """
    emap: dict[tuple[int, str], str] = {}
    for conv_idx, item in enumerate(data):
        conv = item.get("conversation", {})
        for key in conv:
            if key.startswith("session_") and not key.endswith("_date_time"):
                for turn in conv[key]:
                    dia_id = turn.get("dia_id", "")
                    text = turn.get("text", "")
                    if dia_id and text:
                        emap[(conv_idx, dia_id)] = text
    return emap


def compute_retrieval_recall(
    retrieved_text: str,
    evidence_ids: list[str],
    conv_idx: int,
    evidence_map: dict[tuple[int, str], str],
) -> float:
    """Check how many evidence turns' text content appears in the retrieved text.

    Instead of matching dia_ids (which aren't in the formatted output), we check
    if a significant substring of each evidence turn's text is present.
    """
    if not evidence_ids:
        return 0.0
    found = 0
    for eid in evidence_ids:
        text = evidence_map.get((conv_idx, eid), "")
        if not text:
            continue
        # Check if a meaningful substring appears in retrieved text
        snippet = text[:80].strip()
        if snippet and snippet in retrieved_text:
            found += 1
    return found / len(evidence_ids)


def build_question_pairs(
    data_with_idx: list[tuple[int, dict]],
    *,
    question_order: str = "forward",
) -> list[tuple[int, dict]]:
    """Flatten evaluable QA pairs with a controllable per-conversation order.

    The eval reuses one mutable index per conversation, so order effects should
    be tested by reversing questions within each conversation instead of
    reversing the whole dataset globally.
    """
    all_qa: list[tuple[int, dict]] = []
    for conv_idx, item in data_with_idx:
        qa_items = [
            qa for qa in item.get("qa", [])
            if qa.get("category", 0) in EVAL_CATEGORIES
        ]
        if question_order == "reverse":
            qa_items.reverse()
        for qa in qa_items:
            all_qa.append((conv_idx, qa))
    return all_qa


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset_path: Path,
    max_conversations: int | None = None,
    retrieval_only: bool = False,
    max_chunks: int = 5,
    max_tokens: int = 2000,
    output_path: Path | None = None,
    full_pipeline: bool = False,
    recalldb: bool = False,
    enrich_model: str = "",
    conv_offset: int = 0,
    knowledge_boost: float | None = None,
    max_knowledge: int | None = None,
    question_order: str = "forward",
) -> dict:
    """Run the full LOCOMO evaluation.

    Args:
        dataset_path: Path to locomo10.json.
        max_conversations: Limit conversations for quick testing.
        retrieval_only: Skip answer generation and judging (no API key needed).
        max_chunks: Chunks to retrieve per question.
        max_tokens: Token budget for retrieval.
        output_path: Path to write detailed results JSON.
        full_pipeline: Use RecallDB + enrich + consolidate + knowledge graph.
        recalldb: Use RecallDB + FTS5 + embeddings (fast, no MLX).
        enrich_model: MLX model for enrichment/consolidation (default: Ministral-3B).

    Returns:
        Summary dict with scores per category and overall.
    """
    pipeline_name = "full" if full_pipeline else ("recalldb" if recalldb else "baseline")
    backend = os.environ.get("SYNAPT_SUMMARY_BACKEND", "")
    audit_ts = _audit_start(
        pipeline=pipeline_name,
        max_conversations=max_conversations,
        max_chunks=max_chunks,
        batch=False,
        enrich_model=enrich_model,
        backend=backend,
        conv_offset=conv_offset,
    )

    # Reset token tracker for this run
    _token_tracker.reset()

    print("Loading LOCOMO dataset...")
    with open(dataset_path) as f:
        full_data = json.load(f)

    # Build evidence map from ALL conversations (evidence may reference any turn)
    evidence_map = build_evidence_map(full_data)

    end = conv_offset + (max_conversations or len(full_data))
    data_with_idx = list(enumerate(full_data))[conv_offset:end]

    # Extract date ranges per conversation for temporal prompt injection
    conv_date_ranges: dict[int, str] = {}
    conv_anchor_dates: dict[int, object] = {}
    for conv_idx, item in data_with_idx:
        conv_date_ranges[conv_idx] = extract_date_range(item)
        conv_anchor_dates[conv_idx] = extract_anchor_date(item)

    # Set up OpenAI client if needed
    client = None
    if not retrieval_only:
        try:
            from openai import OpenAI
            client = OpenAI()
        except Exception as e:
            print(f"OpenAI client failed: {e}")
            print("Falling back to retrieval-only mode.")
            retrieval_only = True

    work_dir = _resolve_work_dir(output_path)

    # Track RecallDBs that need closing (full pipeline only)
    open_dbs: list = []

    try:
        # Step 1: Build one index per conversation
        if full_pipeline:
            mode_label = "full pipeline"
        elif recalldb:
            mode_label = "recalldb"
        else:
            mode_label = "baseline"
        print(f"Building indexes ({mode_label}) for {len(data_with_idx)} conversations...")
        indexes: dict[int, object] = {}
        total_chunks = 0
        total_knowledge = 0
        t0 = time.time()
        for conv_idx, item in data_with_idx:
            conv_dir = work_dir / f"conv_{conv_idx:02d}"

            if full_pipeline:
                print(f"  Conv {conv_idx}: setting up full pipeline...")
                index, db = setup_full_pipeline_index(
                    item, conv_idx, conv_dir,
                    enrich_model=enrich_model,
                )
                open_dbs.append(db)
            elif recalldb:
                index, db = setup_recalldb_index(item, conv_idx, conv_dir)
                open_dbs.append(db)
            else:
                transcript_dir = conv_dir / "transcripts"
                cache_dir = conv_dir / "cache"
                transcript_dir.mkdir(parents=True)
                cache_dir.mkdir()
                locomo_to_transcripts([item], transcript_dir)
                index = build_synapt_index(transcript_dir, cache_dir)

            indexes[conv_idx] = index
            stats = index.stats()
            total_chunks += stats["chunk_count"]
            kn_count = stats.get("knowledge_count", 0)
            total_knowledge += kn_count
            extra = f", {kn_count} knowledge nodes" if kn_count else ""
            print(f"  Conv {conv_idx}: {stats['chunk_count']} chunks, "
                  f"{stats['session_count']} sessions{extra}")
        build_time = time.time() - t0
        kn_msg = f", {total_knowledge} knowledge nodes" if total_knowledge else ""
        print(f"  Total: {total_chunks} chunks{kn_msg} in {build_time:.1f}s")

        # Step 2-5: Evaluate each question
        results = []
        category_scores: dict[int, list] = defaultdict(list)
        category_f1: dict[int, list] = defaultdict(list)
        category_recall: dict[int, list] = defaultdict(list)
        total_retrieve_time = 0.0
        total_generate_time = 0.0
        total_judge_time = 0.0

        all_qa = build_question_pairs(
            data_with_idx,
            question_order=question_order,
        )

        print(
            f"Evaluating {len(all_qa)} questions (categories 1-4, "
            f"question_order={question_order})..."
        )

        for i, (conv_idx, qa) in enumerate(all_qa):
            question = qa["question"]
            gold = str(qa["answer"])
            evidence = qa.get("evidence", [])
            category = qa["category"]

            # Retrieve from this conversation's index
            t0 = time.time()
            context = retrieve_context(
                indexes[conv_idx], question,
                max_chunks=max_chunks, max_tokens=max_tokens,
                anchor_date=conv_anchor_dates.get(conv_idx),
                knowledge_boost=knowledge_boost,
                max_knowledge=max_knowledge,
            )
            retrieve_time = time.time() - t0
            total_retrieve_time += retrieve_time

            # Retrieval recall
            recall_at_k = compute_retrieval_recall(
                context, evidence, conv_idx, evidence_map)
            category_recall[category].append(recall_at_k)

            result_entry = {
                "conv_idx": conv_idx,
                "question": question,
                "gold_answer": gold,
                "category": category,
                "category_name": CATEGORY_NAMES.get(category, "?"),
                "context_length": len(context),
                "retrieve_time_ms": round(retrieve_time * 1000, 1),
                "retrieval_recall": recall_at_k,
            }

            if not retrieval_only and client:
                # Generate answer
                t0 = time.time()
                generated = generate_answer(
                    question, context, client,
                    category=category,
                    date_range=conv_date_ranges.get(conv_idx, ""),
                )
                generate_time = time.time() - t0
                total_generate_time += generate_time

                # F1 score
                f1 = token_f1(generated, gold)
                category_f1[category].append(f1)

                # Judge
                t0 = time.time()
                j_score = judge_answer(question, gold, generated, client)
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

            results.append(result_entry)

            # Progress + incremental save
            if (i + 1) % 50 == 0 or i == len(all_qa) - 1:
                elapsed = total_retrieve_time + total_generate_time + total_judge_time
                if not retrieval_only and category_scores:
                    all_j = [s for scores in category_scores.values() for s in scores]
                    running_j = sum(all_j) / len(all_j) * 100
                    print(f"  [{i+1}/{len(all_qa)}] J={running_j:.1f}% "
                          f"({elapsed:.0f}s elapsed)")
                else:
                    print(f"  [{i+1}/{len(all_qa)}] retrieval done ({elapsed:.0f}s elapsed)")

                # Save checkpoint so progress isn't lost on crash
                save_dir = output_path if output_path else Path("evaluation/results")
                save_dir.mkdir(parents=True, exist_ok=True)
                with open(save_dir / "locomo_checkpoint.json", "w") as f:
                    json.dump(results, f)

        # Aggregate scores
        enrich_model_name = _detect_enrich_model(enrich_model, backend)
        summary: dict = {
            "dataset": str(dataset_path),
            "pipeline": pipeline_name,
            "enrichment_model": enrich_model_name,
            "conversations": len(data_with_idx),
            "questions_evaluated": len(all_qa),
            "question_order": question_order,
            "max_chunks": max_chunks,
            "max_tokens": max_tokens,
            "index_chunks": total_chunks,
            "index_conversations": len(indexes),
            "build_time_s": round(build_time, 1),
            "avg_retrieve_ms": round(total_retrieve_time / len(all_qa) * 1000, 1),
        }
        if full_pipeline:
            summary["knowledge_nodes"] = total_knowledge

        # Token usage tracking
        if not retrieval_only and _token_tracker.calls > 0:
            summary["token_usage"] = _token_tracker.summary()

        # Retrieval recall by category
        for cat in sorted(EVAL_CATEGORIES):
            if category_recall[cat]:
                avg = sum(category_recall[cat]) / len(category_recall[cat])
                summary[f"recall@{max_chunks}_{CATEGORY_NAMES[cat]}"] = round(avg * 100, 2)

        if not retrieval_only and category_scores:
            summary["avg_generate_ms"] = round(
                total_generate_time / len(all_qa) * 1000, 1)
            summary["avg_judge_ms"] = round(
                total_judge_time / len(all_qa) * 1000, 1)

            # J scores by category
            for cat in sorted(EVAL_CATEGORIES):
                if category_scores[cat]:
                    j = sum(category_scores[cat]) / len(category_scores[cat]) * 100
                    n = len(category_scores[cat])
                    summary[f"j_score_{CATEGORY_NAMES[cat]}"] = round(j, 2)
                    summary[f"n_{CATEGORY_NAMES[cat]}"] = n

            # Overall J score
            all_j = [s for scores in category_scores.values() for s in scores]
            overall_j = sum(all_j) / len(all_j) * 100
            summary["j_score_overall"] = round(overall_j, 2)

            # F1 by category
            for cat in sorted(EVAL_CATEGORIES):
                if category_f1[cat]:
                    avg_f1 = sum(category_f1[cat]) / len(category_f1[cat]) * 100
                    summary[f"f1_{CATEGORY_NAMES[cat]}"] = round(avg_f1, 2)

            all_f1 = [f for scores in category_f1.values() for f in scores]
            summary["f1_overall"] = round(sum(all_f1) / len(all_f1) * 100, 2)

        # Print summary
        print("\n" + "=" * 60)
        pipeline_tag = " (full pipeline)" if full_pipeline else (" (recalldb)" if recalldb else " (baseline)")
        print(f"LOCOMO BENCHMARK RESULTS — synapt recall v0.4.0{pipeline_tag}")
        print("=" * 60)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("=" * 60)

        # Save results
        if output_path is None:
            output_path = Path("evaluation/results")
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "locomo_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        with open(output_path / "locomo_detailed.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}/")

        _audit_finish(audit_ts, summary)
        return summary

    except Exception:
        _audit_finish(audit_ts, {"pipeline": pipeline_name, "enrichment_model": _detect_enrich_model(enrich_model, backend)}, outcome="FAILED")
        raise

    finally:
        # Close RecallDBs before cleanup
        for db in open_dbs:
            try:
                db.close()
            except Exception:
                pass
        # Clean up temp dir
        if not output_path:
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Batch API support
# ---------------------------------------------------------------------------

def _build_batch_request(custom_id: str, prompt: str, max_tokens: int = 50) -> dict:
    """Build one JSONL line for OpenAI Batch API."""
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        },
    }


def _submit_batch(client, batch_jsonl_path: Path, description: str) -> str:
    """Upload JSONL and submit as a batch. Returns batch ID."""
    with open(batch_jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": description},
    )
    print(f"  Batch submitted: {batch.id} ({description})")
    return batch.id


# Max batch file size in bytes to stay under OpenAI's 2M enqueued token limit.
# ~9KB avg per request * 750 requests ≈ 6.75MB is a safe margin.
_MAX_BATCH_BYTES = 7_000_000


def _submit_batch_sharded(
    client, batch_jsonl_path: Path, description: str,
) -> list[str]:
    """Split a large JSONL file into shards and submit each.

    Shards are submitted sequentially — each shard must complete before the
    next is submitted. This avoids hitting OpenAI's enqueued token limit
    (2M tokens) which fails when multiple large shards are queued at once.

    Returns list of batch IDs. If the file is small enough, submits as one.
    """
    file_size = batch_jsonl_path.stat().st_size
    if file_size <= _MAX_BATCH_BYTES:
        return [_submit_batch(client, batch_jsonl_path, description)]

    # Split into shards
    with open(batch_jsonl_path) as f:
        lines = f.readlines()

    n_shards = (file_size // _MAX_BATCH_BYTES) + 1
    shard_size = len(lines) // n_shards + 1
    print(f"  Splitting {len(lines)} requests into {n_shards} shards "
          f"({shard_size} requests each)")

    batch_ids = []
    for i in range(n_shards):
        shard_lines = lines[i * shard_size : (i + 1) * shard_size]
        if not shard_lines:
            continue
        shard_path = batch_jsonl_path.with_suffix(f".shard{i}.jsonl")
        with open(shard_path, "w") as f:
            f.writelines(shard_lines)
        bid = _submit_batch(client, shard_path, f"{description} (shard {i+1}/{n_shards})")
        batch_ids.append(bid)
        shard_path.unlink()  # Clean up shard file

        # Wait for this shard to complete before submitting the next.
        # This avoids exceeding the enqueued token limit.
        if i < n_shards - 1:
            print(f"  Waiting for shard {i+1}/{n_shards} to complete before submitting next...")
            batch = _poll_batch(client, bid)
            if batch.status != "completed":
                print(f"  WARNING: Shard {i+1} {batch.status} — continuing with remaining shards")

    return batch_ids


def _poll_batch_sharded(client, batch_ids: list[str], poll_interval: int = 30) -> list:
    """Poll multiple batch shards until all complete."""
    results = []
    for bid in batch_ids:
        batch = _poll_batch(client, bid, poll_interval)
        results.append(batch)
    return results


def _download_batch_results_sharded(client, batches: list) -> dict[str, str]:
    """Download and merge results from multiple batch shards."""
    merged = {}
    for batch in batches:
        merged.update(_download_batch_results(client, batch))
    return merged


def _poll_batch(client, batch_id: str, poll_interval: int = 30) -> dict:
    """Poll until batch completes. Returns the batch object."""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        completed = batch.request_counts.completed if batch.request_counts else 0
        total = batch.request_counts.total if batch.request_counts else 0
        failed = batch.request_counts.failed if batch.request_counts else 0

        if status in ("completed", "failed", "expired", "cancelled"):
            print(f"  Batch {batch_id}: {status} "
                  f"({completed}/{total} completed, {failed} failed)")
            return batch

        print(f"  Batch {batch_id}: {status} "
              f"({completed}/{total} completed) — waiting {poll_interval}s...")
        time.sleep(poll_interval)


def _download_batch_results(client, batch) -> dict[str, str]:
    """Download batch results. Returns {custom_id: response_text}."""
    if not batch.output_file_id:
        print(f"  WARNING: No output file for batch {batch.id}")
        return {}

    content = client.files.content(batch.output_file_id).text
    results = {}
    for line in content.strip().split("\n"):
        if not line:
            continue
        entry = json.loads(line)
        cid = entry["custom_id"]
        resp = entry.get("response", {})
        body = resp.get("body", {})
        choices = body.get("choices", [])
        if choices:
            results[cid] = choices[0]["message"]["content"].strip()
            # Track token usage from batch results
            usage = body.get("usage")
            if usage:
                _token_tracker.add_openai(usage)
        else:
            error = entry.get("error", {})
            print(f"  WARNING: No response for {cid}: {error}")
    return results


def run_batch_evaluation(
    dataset_path: Path,
    max_conversations: int | None = None,
    max_chunks: int = 20,
    max_tokens: int = 2000,
    output_path: Path | None = None,
    recalldb: bool = True,
    full_pipeline: bool = False,
    enrich_model: str = "",
    batch_id: str = "",
    conv_offset: int = 0,
    knowledge_boost: float | None = None,
    max_knowledge: int | None = None,
    question_order: str = "forward",
) -> dict:
    """Run LOCOMO eval using OpenAI Batch API (50% cheaper, ~1hr turnaround).

    Flow:
        1. Retrieval: run all queries locally → save retrieval results
        2. Generate batch: submit answer generation as batch
        3. Judge batch: submit judging as batch
        4. Score: compute Jaccard/F1 per category
    """
    from openai import OpenAI
    client = OpenAI()

    # Reset token tracker for this run
    _token_tracker.reset()

    pipeline_name = "full" if full_pipeline else ("recalldb" if recalldb else "baseline")
    backend = os.environ.get("SYNAPT_SUMMARY_BACKEND", "")
    audit_ts = _audit_start(
        pipeline=pipeline_name,
        max_conversations=max_conversations,
        max_chunks=max_chunks,
        batch=True,
        enrich_model=enrich_model,
        backend=backend,
        conv_offset=conv_offset,
    )

    if output_path is None:
        output_path = Path("evaluation/results")
    output_path.mkdir(parents=True, exist_ok=True)

    retrieval_path = output_path / "batch_retrieval.json"
    generate_batch_path = output_path / "batch_generate.jsonl"
    judge_batch_path = output_path / "batch_judge.jsonl"
    state_path = output_path / "batch_state.json"

    # Load or resume state
    state = {}
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)

    # ------- Phase 1: Retrieval (local) -------
    if not retrieval_path.exists():
        print("Phase 1: Running local retrieval...")
        print("Loading LOCOMO dataset...")
        with open(dataset_path) as f:
            full_data = json.load(f)

        evidence_map = build_evidence_map(full_data)
        end = conv_offset + (max_conversations or len(full_data))
        data_with_idx = list(enumerate(full_data))[conv_offset:end]

        conv_date_ranges: dict[int, str] = {}
        conv_anchor_dates: dict[int, object] = {}
        for conv_idx, item in data_with_idx:
            conv_date_ranges[conv_idx] = extract_date_range(item)
            conv_anchor_dates[conv_idx] = extract_anchor_date(item)

        work_dir = _resolve_work_dir(output_path)
        open_dbs: list = []

        try:
            mode_label = "full pipeline" if full_pipeline else ("recalldb" if recalldb else "baseline")
            print(f"Building indexes ({mode_label}) for {len(data_with_idx)} conversations...")
            indexes: dict[int, object] = {}
            t0 = time.time()
            for conv_idx, item in data_with_idx:
                conv_dir = work_dir / f"conv_{conv_idx:02d}"
                if full_pipeline:
                    print(f"  Conv {conv_idx}: setting up full pipeline...")
                    index, db = setup_full_pipeline_index(
                        item, conv_idx, conv_dir, enrich_model=enrich_model)
                    open_dbs.append(db)
                elif recalldb:
                    index, db = setup_recalldb_index(item, conv_idx, conv_dir)
                    open_dbs.append(db)
                else:
                    transcript_dir = conv_dir / "transcripts"
                    cache_dir = conv_dir / "cache"
                    transcript_dir.mkdir(parents=True)
                    cache_dir.mkdir()
                    locomo_to_transcripts([item], transcript_dir)
                    index = build_synapt_index(transcript_dir, cache_dir)
                indexes[conv_idx] = index
                stats = index.stats()
                kn_count = stats.get("knowledge_count", 0)
                extra = f", {kn_count} knowledge nodes" if kn_count else ""
                print(f"  Conv {conv_idx}: {stats['chunk_count']} chunks, "
                      f"{stats['session_count']} sessions{extra}")
            build_time = time.time() - t0
            print(f"  Indexes built in {build_time:.1f}s")

            all_qa = build_question_pairs(
                data_with_idx,
                question_order=question_order,
            )

            print(
                f"Retrieving context for {len(all_qa)} questions "
                f"(question_order={question_order})..."
            )
            retrieval_results = []
            t0 = time.time()
            for i, (conv_idx, qa) in enumerate(all_qa):
                question = qa["question"]
                gold = str(qa["answer"])
                evidence = qa.get("evidence", [])
                category = qa["category"]

                rt0 = time.time()
                context = retrieve_context(
                    indexes[conv_idx], question,
                    max_chunks=max_chunks, max_tokens=max_tokens,
                    anchor_date=conv_anchor_dates.get(conv_idx),
                    knowledge_boost=knowledge_boost,
                    max_knowledge=max_knowledge,
                )
                retrieve_time = time.time() - rt0
                recall_at_k = compute_retrieval_recall(
                    context, evidence, conv_idx, evidence_map)

                retrieval_results.append({
                    "idx": i,
                    "conv_idx": conv_idx,
                    "question": question,
                    "gold_answer": gold,
                    "category": category,
                    "category_name": CATEGORY_NAMES.get(category, "?"),
                    "evidence": evidence,
                    "context": context,
                    "context_length": len(context),
                    "date_range": conv_date_ranges.get(conv_idx, ""),
                    "retrieve_time_ms": round(retrieve_time * 1000, 1),
                    "retrieval_recall": recall_at_k,
                })

                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{len(all_qa)}] ({elapsed:.0f}s elapsed)")

            elapsed = time.time() - t0
            print(f"  Retrieval complete: {len(retrieval_results)} questions in {elapsed:.0f}s")

            with open(retrieval_path, "w") as f:
                json.dump(retrieval_results, f)
            print(f"  Saved to {retrieval_path}")

        finally:
            for db in open_dbs:
                try:
                    db.close()
                except Exception:
                    pass
            if not output_path:
                shutil.rmtree(work_dir, ignore_errors=True)
    else:
        print(f"Phase 1: Using cached retrieval from {retrieval_path}")
        with open(retrieval_path) as f:
            retrieval_results = json.load(f)
        print(f"  {len(retrieval_results)} questions loaded")

    # ------- Phase 2: Generate answers (batch) -------
    # Support both old (generate_batch_id: str) and new (generate_batch_ids: list) state
    gen_batch_ids = state.get("generate_batch_ids", [])
    if not gen_batch_ids and state.get("generate_batch_id"):
        gen_batch_ids = [state["generate_batch_id"]]
    if not gen_batch_ids:
        print("Phase 2: Submitting answer generation batch...")
        with open(generate_batch_path, "w") as f:
            for r in retrieval_results:
                dr_text = (f"\nThe conversations span from {r['date_range']}."
                           if r["date_range"] else "")
                if r["category"] == 1:
                    prompt = MULTIHOP_ANSWER_PROMPT.format(
                        context=r["context"], question=r["question"],
                        date_range=dr_text,
                    )
                elif r["category"] == 2:
                    prompt = TEMPORAL_ANSWER_PROMPT.format(
                        context=r["context"], question=r["question"],
                        date_range=dr_text,
                    )
                else:
                    prompt = ANSWER_PROMPT.format(
                        context=r["context"], question=r["question"],
                        date_range=dr_text,
                    )
                req = _build_batch_request(f"gen_{r['idx']}", prompt, max_tokens=100)
                f.write(json.dumps(req) + "\n")

        gen_batch_ids = _submit_batch_sharded(
            client, generate_batch_path, "LOCOMO answer generation")
        state["generate_batch_ids"] = gen_batch_ids
        with open(state_path, "w") as f:
            json.dump(state, f)

    # Poll for generation batch(es)
    gen_answers: dict[str, str] = state.get("generate_answers", {})
    if not gen_answers:
        print("  Waiting for generation batch...")
        gen_batches = _poll_batch_sharded(client, gen_batch_ids)
        failed = [b for b in gen_batches if b.status != "completed"]
        if failed:
            print(f"  ERROR: {len(failed)} generation shard(s) {failed[0].status}. Re-run to retry.")
            return {}
        gen_answers = _download_batch_results_sharded(client, gen_batches)
        state["generate_answers"] = gen_answers
        with open(state_path, "w") as f:
            json.dump(state, f)
        print(f"  Got {len(gen_answers)} generated answers")

    # Attach generated answers to retrieval results
    for r in retrieval_results:
        r["generated_answer"] = gen_answers.get(f"gen_{r['idx']}", "")

    # ------- Phase 3: Judge answers (batch) -------
    judge_batch_ids = state.get("judge_batch_ids", [])
    if not judge_batch_ids and state.get("judge_batch_id"):
        judge_batch_ids = [state["judge_batch_id"]]
    if not judge_batch_ids:
        print("Phase 3: Submitting judging batch...")
        with open(judge_batch_path, "w") as f:
            for r in retrieval_results:
                prompt = JUDGE_PROMPT.format(
                    question=r["question"],
                    gold_answer=r["gold_answer"],
                    generated_answer=r["generated_answer"],
                )
                req = _build_batch_request(f"judge_{r['idx']}", prompt, max_tokens=100)
                f.write(json.dumps(req) + "\n")

        judge_batch_ids = _submit_batch_sharded(
            client, judge_batch_path, "LOCOMO answer judging")
        state["judge_batch_ids"] = judge_batch_ids
        with open(state_path, "w") as f:
            json.dump(state, f)

    # Poll for judge batch(es)
    judge_results: dict[str, str] = state.get("judge_results", {})
    if not judge_results:
        print("  Waiting for judging batch...")
        judge_batches = _poll_batch_sharded(client, judge_batch_ids)
        failed = [b for b in judge_batches if b.status != "completed"]
        if failed:
            print(f"  ERROR: {len(failed)} judge shard(s) {failed[0].status}. Re-run to retry.")
            return {}
        judge_results = _download_batch_results_sharded(client, judge_batches)
        state["judge_results"] = judge_results
        with open(state_path, "w") as f:
            json.dump(state, f)
        print(f"  Got {len(judge_results)} judge responses")

    # ------- Phase 4: Score -------
    print("Phase 4: Scoring...")
    results = []
    category_scores: dict[int, list] = defaultdict(list)
    category_f1: dict[int, list] = defaultdict(list)
    category_recall: dict[int, list] = defaultdict(list)

    for r in retrieval_results:
        category = r["category"]
        category_recall[category].append(r["retrieval_recall"])

        # Parse judge response
        judge_text = judge_results.get(f"judge_{r['idx']}", "")
        try:
            j_result = json.loads(judge_text)
            label = j_result.get("label", "").upper()
        except (json.JSONDecodeError, ValueError):
            label = "CORRECT" if "CORRECT" in judge_text.upper() else "WRONG"
        j_score = 1 if label == "CORRECT" else 0

        f1 = token_f1(r["generated_answer"], r["gold_answer"])

        category_scores[category].append(j_score)
        category_f1[category].append(f1)

        results.append({
            "conv_idx": r["conv_idx"],
            "question": r["question"],
            "gold_answer": r["gold_answer"],
            "category": category,
            "category_name": r["category_name"],
            "context_length": r["context_length"],
            "retrieve_time_ms": r.get("retrieve_time_ms", 0),
            "retrieval_recall": r.get("retrieval_recall", 0),
            "generated_answer": r["generated_answer"],
            "j_score": j_score,
            "f1": round(f1, 4),
        })

    # Build summary
    enrich_model_name = _detect_enrich_model(enrich_model, backend)
    summary: dict = {
        "dataset": str(dataset_path),
        "pipeline": pipeline_name,
        "enrichment_model": enrich_model_name,
        "mode": "batch",
        "question_order": question_order,
        "conversations": max_conversations or len(retrieval_results),
        "questions_evaluated": len(results),
        "max_chunks": max_chunks,
        "max_tokens": max_tokens,
    }

    # Token usage tracking
    if _token_tracker.calls > 0:
        summary["token_usage"] = _token_tracker.summary()

    for cat in sorted(EVAL_CATEGORIES):
        if category_recall[cat]:
            avg = sum(category_recall[cat]) / len(category_recall[cat])
            summary[f"recall@{max_chunks}_{CATEGORY_NAMES[cat]}"] = round(avg * 100, 2)

    for cat in sorted(EVAL_CATEGORIES):
        if category_scores[cat]:
            j = sum(category_scores[cat]) / len(category_scores[cat]) * 100
            summary[f"j_score_{CATEGORY_NAMES[cat]}"] = round(j, 2)
            summary[f"n_{CATEGORY_NAMES[cat]}"] = len(category_scores[cat])

    all_j = [s for scores in category_scores.values() for s in scores]
    summary["j_score_overall"] = round(sum(all_j) / len(all_j) * 100, 2) if all_j else 0

    for cat in sorted(EVAL_CATEGORIES):
        if category_f1[cat]:
            avg_f1 = sum(category_f1[cat]) / len(category_f1[cat]) * 100
            summary[f"f1_{CATEGORY_NAMES[cat]}"] = round(avg_f1, 2)
    all_f1 = [f for scores in category_f1.values() for f in scores]
    summary["f1_overall"] = round(sum(all_f1) / len(all_f1) * 100, 2) if all_f1 else 0

    # Print summary
    print("\n" + "=" * 60)
    print(f"LOCOMO BENCHMARK RESULTS — synapt recall v0.4.0 ({pipeline_name}, batch)")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    # Save results
    with open(output_path / "locomo_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_path / "locomo_detailed.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}/")

    # Clean up state file on success
    state_path.unlink(missing_ok=True)

    _audit_finish(audit_ts, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="LOCOMO benchmark for synapt recall")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("evaluation/dataset/locomo10.json"),
        help="Path to locomo10.json",
    )
    parser.add_argument(
        "--max-conversations", type=int, default=None,
        help="Limit number of conversations (for quick testing)",
    )
    parser.add_argument(
        "--conv-offset", type=int, default=0,
        help="Start from this conversation index (e.g., --conv-offset 3 --max-conversations 1 tests conv 3 only)",
    )
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="Only measure retrieval quality (no LLM answer generation)",
    )
    parser.add_argument(
        "--max-chunks", type=int, default=5,
        help="Number of chunks to retrieve per question",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=2000,
        help="Token budget for retrieval",
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
        help="Use full pipeline: RecallDB + enrich + consolidate + knowledge graph (slow)",
    )
    parser.add_argument(
        "--enrich-model", type=str, default="",
        help="MLX model for enrichment/consolidation (default: Ministral-3B-4bit)",
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Use OpenAI Batch API (50%% cheaper, ~1hr turnaround instead of 4+ hrs)",
    )
    parser.add_argument(
        "--knowledge-boost", type=float, default=None,
        help="Override knowledge node boost (default: intent-classified, typically 2.0). "
             "Lower values let raw conversation chunks rank higher.",
    )
    parser.add_argument(
        "--max-knowledge", type=int, default=None,
        help="Maximum knowledge blocks per query (default: no cap). "
             "Set to e.g. 5 to prevent knowledge nodes from crowding out raw chunks.",
    )
    parser.add_argument(
        "--backend", type=str, default="",
        choices=["", "mlx", "modal", "ollama"],
        help="Model backend for enrichment/consolidation (default: auto). "
             "Use 'modal' for GPU-accelerated cloud inference.",
    )
    parser.add_argument(
        "--question-order",
        type=str,
        default="forward",
        choices=["forward", "reverse"],
        help="Evaluate questions in forward or reverse order within each conversation.",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found at {args.dataset}")
        print("Download from: https://github.com/snap-research/locomo/tree/main/data")
        sys.exit(1)

    # Set model backend if specified
    if args.backend:
        import os
        os.environ["SYNAPT_SUMMARY_BACKEND"] = args.backend
        print(f"Using backend: {args.backend}")

    # RecallDB/full-pipeline defaults: higher k to match Mem0's methodology
    max_chunks = args.max_chunks
    if (args.full_pipeline or args.recalldb) and max_chunks == 5:
        max_chunks = 20
        print(f"Using --max-chunks {max_chunks} (override with --max-chunks N)")

    if args.batch:
        run_batch_evaluation(
            dataset_path=args.dataset,
            max_conversations=args.max_conversations,
            max_chunks=max_chunks,
            max_tokens=args.max_tokens,
            output_path=args.output,
            recalldb=args.recalldb,
            full_pipeline=args.full_pipeline,
            enrich_model=args.enrich_model,
            conv_offset=args.conv_offset,
            knowledge_boost=args.knowledge_boost,
            max_knowledge=args.max_knowledge,
            question_order=args.question_order,
        )
    else:
        run_evaluation(
            dataset_path=args.dataset,
            max_conversations=args.max_conversations,
            retrieval_only=args.retrieval_only,
            max_chunks=max_chunks,
            max_tokens=args.max_tokens,
            output_path=args.output,
            full_pipeline=args.full_pipeline,
            recalldb=args.recalldb,
            enrich_model=args.enrich_model,
            conv_offset=args.conv_offset,
            knowledge_boost=args.knowledge_boost,
            max_knowledge=args.max_knowledge,
            question_order=args.question_order,
        )


if __name__ == "__main__":
    main()
