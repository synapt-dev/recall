from __future__ import annotations

import argparse
import ast
import json
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.locomo_eval import (
    ANSWER_PROMPT,
    MULTIHOP_ANSWER_PROMPT,
    TEMPORAL_ANSWER_PROMPT,
    _resolve_work_dir,
    _token_tracker,
    attach_search_summary,
    build_synapt_index,
    judge_answer,
    retrieve_context,
    token_f1,
)

AVAILABLE_SPLITS = ("100K", "500K", "1M")

BEAM_CATEGORY_NAMES = {
    "abstention": "abstention",
    "contradiction_resolution": "contradiction_resolution",
    "event_ordering": "event_ordering",
    "information_extraction": "information_extraction",
    "instruction_following": "instruction_following",
    "knowledge_update": "knowledge_update",
    "multi_session_reasoning": "multi_session_reasoning",
    "preference_following": "preference_following",
    "summarization": "summarization",
    "temporal_reasoning": "temporal_reasoning",
}


def parse_beam_time_anchor(raw: str) -> tuple[str, datetime | None]:
    """Convert BEAM's `Month-DD-YYYY` anchors into ISO timestamps."""
    raw = (raw or "").strip()
    if not raw:
        return "", None

    for fmt in (
        "%B-%d-%Y",
        "%b-%d-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d, %Y, %I:%M %p UTC",
        "%b %d, %Y, %I:%M %p UTC",
    ):
        try:
            dt = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
            return dt.isoformat(), dt
        except ValueError:
            continue
    return raw, None


def extract_date_range(item: dict[str, Any]) -> str:
    anchors = [
        q.get("time_anchor", "")
        for q in item.get("user_questions", [])
        if isinstance(q, dict) and q.get("time_anchor")
    ]
    if not anchors:
        return ""
    return f"{anchors[0]} to {anchors[-1]}"


def extract_anchor_date(item: dict[str, Any]) -> datetime | None:
    anchors = [
        q.get("time_anchor", "")
        for q in item.get("user_questions", [])
        if isinstance(q, dict) and q.get("time_anchor")
    ]
    if not anchors:
        return None
    _, dt = parse_beam_time_anchor(anchors[-1])
    return dt


def parse_probing_questions(raw: str) -> dict[str, list[dict[str, Any]]]:
    """BEAM stores probing questions as a serialized Python dict."""
    parsed = ast.literal_eval(raw)
    if not isinstance(parsed, dict):
        raise ValueError("BEAM probing questions must deserialize to a dict")
    return parsed


def build_questions(item: dict[str, Any]) -> list[dict[str, Any]]:
    probes = parse_probing_questions(item.get("probing_questions", "{}"))
    rows: list[dict[str, Any]] = []
    for category_name, questions in probes.items():
        if category_name not in BEAM_CATEGORY_NAMES:
            continue
        for idx, probe in enumerate(questions):
            answer = (
                probe.get("ideal_response")
                or probe.get("ideal_answer")
                or probe.get("ideal_summary")
                or probe.get("answer")
                or probe.get("expected_compliance")
                or " ".join(str(x) for x in probe.get("rubric", []))
            )
            rows.append(
                {
                    "id": f"{item.get('conversation_id', 'conv')}-{category_name}-{idx}",
                    "question": probe["question"],
                    "answer": answer,
                    "category_name": category_name,
                    "difficulty": probe.get("difficulty", ""),
                    "rubric": probe.get("rubric", []),
                }
            )
    return rows


def beam_to_transcripts(item: dict[str, Any], output_dir: Path) -> list[Path]:
    """Convert BEAM chat batches into synapt-compatible transcript files."""
    transcript_paths: list[Path] = []
    conversation_id = str(item.get("conversation_id", "beam"))

    for batch_idx, turns in enumerate(item.get("chat", []), start=1):
        if not turns:
            continue
        batch_anchor = ""
        for turn in turns:
            batch_anchor, _ = parse_beam_time_anchor(turn.get("time_anchor", ""))
            if batch_anchor:
                break
        lines = []
        for turn_idx, turn in enumerate(turns):
            role = turn.get("role", "user")
            text = turn.get("content", "")
            timestamp, _ = parse_beam_time_anchor(turn.get("time_anchor", ""))
            if not timestamp:
                timestamp = batch_anchor
            msg = {
                "type": role,
                "message": {
                    "role": role,
                    "content": [{"type": "text", "text": text}],
                },
                "timestamp": timestamp or "2024-01-01T00:00:00+00:00",
                "uuid": f"{conversation_id}-{turn.get('index', turn_idx)}",
                "sessionId": f"b{batch_idx:03d}c{conversation_id}",
                "parentUuid": "",
            }
            lines.append(json.dumps(msg))

        transcript_path = output_dir / f"b{batch_idx:03d}c{conversation_id}.jsonl"
        transcript_path.write_text("\n".join(lines), encoding="utf-8")
        transcript_paths.append(transcript_path)

    return transcript_paths


def build_answer_prompt(
    question: str,
    context: str,
    category_name: str,
    date_range: str = "",
) -> str:
    dr_text = f"\nThe conversations span from {date_range}." if date_range else ""
    if category_name == "temporal_reasoning":
        return TEMPORAL_ANSWER_PROMPT.format(
            context=context,
            question=question,
            date_range=dr_text,
        )
    if category_name in {
        "multi_session_reasoning",
        "event_ordering",
        "contradiction_resolution",
        "knowledge_update",
        "summarization",
    }:
        return MULTIHOP_ANSWER_PROMPT.format(
            context=context,
            question=question,
            date_range=dr_text,
        )
    return ANSWER_PROMPT.format(
        context=context,
        question=question,
        date_range=dr_text,
    )


def generate_beam_answer(
    question: str,
    context: str,
    client,
    *,
    category_name: str,
    date_range: str = "",
) -> str:
    prompt = build_answer_prompt(
        question=question,
        context=context,
        category_name=category_name,
        date_range=date_range,
    )
    return generate_answer(
        question=question,
        context=context,
        client=client,
        category=0,
        date_range="",
    ) if False else _generate_from_prompt(prompt, client)


def _generate_from_prompt(prompt: str, client) -> str:
    from evaluation.locomo_eval import _api_call_with_retry

    return _api_call_with_retry(
        client,
        [{"role": "user", "content": prompt}],
        max_tokens=100,
    )


def load_beam_split(split: str):
    if split not in AVAILABLE_SPLITS:
        raise ValueError(f"Unknown BEAM split {split!r}; expected one of {AVAILABLE_SPLITS}")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install `datasets` to run the BEAM evaluator") from exc
    return load_dataset("Mohammadta/BEAM", split=split)


def run_evaluation(
    *,
    split: str,
    max_conversations: int | None = None,
    retrieval_only: bool = False,
    max_chunks: int = 5,
    max_tokens: int = 2000,
    output_path: Path | None = None,
) -> dict[str, Any]:
    start_wall = time.time()
    start_ts = datetime.now(timezone.utc).isoformat()
    data = load_beam_split(split)
    if max_conversations is not None:
        data = data.select(range(min(max_conversations, len(data))))

    client = None
    if not retrieval_only:
        try:
            from openai import OpenAI

            client = OpenAI()
        except Exception:
            retrieval_only = True
    _token_tracker.reset()

    work_dir = _resolve_work_dir(output_path)
    results: list[dict[str, Any]] = []
    category_scores: dict[str, list[int]] = defaultdict(list)
    category_f1: dict[str, list[float]] = defaultdict(list)
    total_retrieve_time = 0.0
    total_generate_time = 0.0
    total_judge_time = 0.0

    try:
        for conv_idx, item in enumerate(data):
            conv_dir = work_dir / f"beam_{split}_{conv_idx:03d}"
            transcript_dir = conv_dir / "transcripts"
            cache_dir = conv_dir / "cache"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            cache_dir.mkdir(parents=True, exist_ok=True)

            beam_to_transcripts(item, transcript_dir)
            index = build_synapt_index(transcript_dir, cache_dir)
            date_range = extract_date_range(item)
            anchor_date = extract_anchor_date(item)

            for qa in build_questions(item):
                t0 = time.time()
                context = retrieve_context(
                    index,
                    qa["question"],
                    max_chunks=max_chunks,
                    max_tokens=max_tokens,
                    anchor_date=anchor_date,
                )
                retrieve_time = time.time() - t0
                total_retrieve_time += retrieve_time

                row = {
                    "split": split,
                    "conv_idx": conv_idx,
                    "question": qa["question"],
                    "gold_answer": qa["answer"],
                    "category_name": qa["category_name"],
                    "difficulty": qa["difficulty"],
                    "context_length": len(context),
                    "retrieve_time_ms": round(retrieve_time * 1000, 1),
                }
                attach_search_summary(row, index)

                if not retrieval_only and client is not None:
                    t0 = time.time()
                    generated = _generate_from_prompt(
                        build_answer_prompt(
                            qa["question"],
                            context,
                            qa["category_name"],
                            date_range=date_range,
                        ),
                        client,
                    )
                    generate_time = time.time() - t0
                    total_generate_time += generate_time

                    f1 = token_f1(generated, qa["answer"])
                    category_f1[qa["category_name"]].append(f1)

                    t0 = time.time()
                    j_score = judge_answer(
                        qa["question"],
                        qa["answer"],
                        generated,
                        client,
                    )
                    judge_time = time.time() - t0
                    total_judge_time += judge_time
                    category_scores[qa["category_name"]].append(j_score)

                    row.update(
                        {
                            "generated_answer": generated,
                            "j_score": j_score,
                            "f1": f1,
                            "generate_time_ms": round(generate_time * 1000, 1),
                            "judge_time_ms": round(judge_time * 1000, 1),
                        }
                    )

                results.append(row)
    finally:
        pass

    elapsed_s = time.time() - start_wall
    summary: dict[str, Any] = {
        "benchmark": "BEAM",
        "split": split,
        "conversations": len(data),
        "questions_evaluated": len(results),
        "retrieval_only": retrieval_only,
        "started_at": start_ts,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_s, 1),
        "methodology": {
            "dataset": "Mohammadta/BEAM",
            "public_split_availability": list(AVAILABLE_SPLITS),
            "missing_public_split": "10M",
            "retrieval": "synapt hybrid recall search via transcript adapter",
            "answer_model": None if retrieval_only else "gpt-4o-mini",
            "judge_model": None if retrieval_only else "gpt-4o-mini",
            "judge_method": None if retrieval_only else "LOCOMO-style LLM-as-judge reuse",
            "max_chunks": max_chunks,
            "max_tokens": max_tokens,
            "notes": [
                "BEAM probe rows do not expose LOCOMO-style evidence turn IDs.",
                "This evaluator reports answer quality metrics but not retrieval recall.",
                "The public dataset currently exposes 100K, 500K, and 1M only.",
            ],
        },
        "avg_retrieve_ms": round(
            total_retrieve_time / len(results) * 1000, 1
        ) if results else 0.0,
    }
    if not retrieval_only and results:
        summary["j_score_overall"] = round(
            sum(sum(scores) for scores in category_scores.values()) / len(results) * 100,
            2,
        )
        summary["f1_overall"] = round(
            sum(sum(scores) for scores in category_f1.values()) / len(results) * 100,
            2,
        )
        summary["avg_generate_ms"] = round(total_generate_time / len(results) * 1000, 1)
        summary["avg_judge_ms"] = round(total_judge_time / len(results) * 1000, 1)
        for category_name in sorted(category_scores):
            summary[f"j_score_{category_name}"] = round(
                sum(category_scores[category_name]) / len(category_scores[category_name]) * 100,
                2,
            )
            summary[f"f1_{category_name}"] = round(
                sum(category_f1[category_name]) / len(category_f1[category_name]) * 100,
                2,
            )
        summary["token_usage"] = _token_tracker.summary()

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "beam_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        (output_path / "beam_detailed.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        (output_path / "beam_methodology.md").write_text(
            "\n".join(
                [
                    "# BEAM Eval Methodology",
                    "",
                    f"- Dataset: `Mohammadta/BEAM`",
                    f"- Split: `{split}`",
                    f"- Conversations evaluated: `{len(data)}`",
                    f"- Retrieval: `synapt` transcript-adapter + hybrid recall search",
                    f"- Answer model: `{'none (retrieval-only)' if retrieval_only else 'gpt-4o-mini'}`",
                    f"- Judge model: `{'none (retrieval-only)' if retrieval_only else 'gpt-4o-mini'}`",
                    f"- Max chunks: `{max_chunks}`",
                    f"- Max tokens: `{max_tokens}`",
                    "- Public BEAM splits currently available: `100K`, `500K`, `1M`",
                    "- Public `10M` split is not present in the current Hugging Face release",
                    "- Retrieval recall is not reported because BEAM does not expose LOCOMO-style evidence turn IDs",
                    "",
                    f"- Started at: `{summary['started_at']}`",
                    f"- Finished at: `{summary['finished_at']}`",
                    f"- Elapsed seconds: `{summary['elapsed_seconds']}`",
                    "",
                    f"- Average retrieve ms: `{summary['avg_retrieve_ms']}`",
                    f"- Questions evaluated: `{len(results)}`",
                ]
                + (
                    [
                        f"- J-score overall: `{summary['j_score_overall']}`",
                        f"- F1 overall: `{summary['f1_overall']}`",
                        f"- Average generate ms: `{summary['avg_generate_ms']}`",
                        f"- Average judge ms: `{summary['avg_judge_ms']}`",
                        f"- Estimated cost USD: `{summary['token_usage']['estimated_cost_usd']}`",
                        f"- Prompt tokens: `{summary['token_usage']['prompt_tokens']}`",
                        f"- Completion tokens: `{summary['token_usage']['completion_tokens']}`",
                    ]
                    if not retrieval_only and results
                    else []
                )
            ),
            encoding="utf-8",
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run synapt against the BEAM benchmark.")
    parser.add_argument("--split", default="100K", choices=AVAILABLE_SPLITS)
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    summary = run_evaluation(
        split=args.split,
        max_conversations=args.max_conversations,
        retrieval_only=args.retrieval_only,
        max_chunks=args.max_chunks,
        max_tokens=args.max_tokens,
        output_path=args.output_dir,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
