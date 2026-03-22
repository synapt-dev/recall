"""Audit utilities for CodeMemo question quality.

This script is intentionally conservative. It does not try to automatically
rewrite the benchmark taxonomy. It surfaces:

- near-duplicate questions
- temporal-labeled questions whose phrasing looks more like debug/root-cause
  analysis than time/order reasoning

Use it to support benchmark hardening work, not as a source of truth.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from evaluation.codememo.schema import CATEGORY_NAMES, Category
from synapt.recall.hybrid import classify_query_intent

CODEMEMO_DIR = Path(__file__).parent
_LOCAL_DATA = CODEMEMO_DIR / "data"
HF_DATASET_ID = "laynepro/codememo-benchmark"

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "did",
    "do",
    "does",
    "for",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "to",
    "was",
    "we",
    "what",
    "when",
    "which",
    "why",
    "with",
}

_TEMPORAL_TERMS = re.compile(
    r"\b("
    r"after|before|during|earlier|final|first|follow-up|how many|in what session|"
    r"last|later|next step|order|previous|session|then|time|timeline|version"
    r")\b",
    re.IGNORECASE,
)

_DEBUG_TERMS = re.compile(
    r"\b("
    r"bug|captur|crash|debug|dedup|duplicate|error|exact mechanism|explain|"
    r"failed|fix|issue|lock|performance|plan output|query plan|root cause|stderr"
    r")\b",
    re.IGNORECASE,
)


def _resolve_data_dir() -> Path:
    """Return the dataset directory, downloading from HuggingFace if needed."""
    if _LOCAL_DATA.is_dir() and any(_LOCAL_DATA.iterdir()):
        return _LOCAL_DATA
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(HF_DATASET_ID, repo_type="dataset")
        return Path(path)
    except ImportError:
        return _LOCAL_DATA


def _token_set(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if token not in _STOPWORDS
    }


def near_duplicate_pairs(
    questions: list[dict],
    *,
    threshold: float = 0.85,
) -> list[dict]:
    """Find likely duplicate question pairs using token-set Jaccard similarity."""
    pairs: list[dict] = []
    for left, right in combinations(questions, 2):
        left_tokens = _token_set(left["question"])
        right_tokens = _token_set(right["question"])
        if not left_tokens or not right_tokens:
            continue
        score = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        if score >= threshold:
            pairs.append(
                {
                    "left_id": left["id"],
                    "right_id": right["id"],
                    "similarity": round(score, 3),
                    "left_category": CATEGORY_NAMES.get(left["category"], str(left["category"])),
                    "right_category": CATEGORY_NAMES.get(right["category"], str(right["category"])),
                }
            )
    return sorted(pairs, key=lambda item: (-item["similarity"], item["left_id"], item["right_id"]))


def temporal_mismatch_candidates(questions: list[dict]) -> list[dict]:
    """Flag temporal-labeled questions that read more like debug/root-cause prompts."""
    candidates: list[dict] = []
    for question in questions:
        if question["category"] != Category.TEMPORAL:
            continue
        text = question["question"]
        has_temporal = bool(_TEMPORAL_TERMS.search(text))
        has_debug = bool(_DEBUG_TERMS.search(text))
        predicted_intent = classify_query_intent(text)
        if has_debug and not has_temporal:
            candidates.append(
                {
                    "id": question["id"],
                    "question": text,
                    "reason": "temporal label but debug/root-cause phrasing with no explicit temporal cues",
                }
            )
            continue
        if predicted_intent != "temporal":
            candidates.append(
                {
                    "id": question["id"],
                    "question": text,
                    "reason": f"temporal label but query intent classifier reads it as '{predicted_intent}'",
                }
            )
    return candidates


def build_audit_report(project_dir: Path, *, threshold: float = 0.85) -> dict:
    questions = json.loads((project_dir / "questions.json").read_text())
    counts = Counter(CATEGORY_NAMES.get(q["category"], str(q["category"])) for q in questions)
    return {
        "project": project_dir.name,
        "question_count": len(questions),
        "category_counts": dict(sorted(counts.items())),
        "near_duplicates": near_duplicate_pairs(questions, threshold=threshold),
        "temporal_mismatch_candidates": temporal_mismatch_candidates(questions),
    }


def _format_markdown(report: dict) -> str:
    lines = [
        f"# CodeMemo Audit: {report['project']}",
        "",
        f"- Questions: {report['question_count']}",
        "- Category counts:",
    ]
    for name, count in report["category_counts"].items():
        lines.append(f"  - {name}: {count}")

    lines.append("")
    lines.append("## Near Duplicates")
    if not report["near_duplicates"]:
        lines.append("- None")
    else:
        for pair in report["near_duplicates"]:
            lines.append(
                f"- {pair['left_id']} vs {pair['right_id']} "
                f"(similarity {pair['similarity']:.3f}, "
                f"{pair['left_category']} / {pair['right_category']})"
            )

    lines.append("")
    lines.append("## Temporal Mismatch Candidates")
    if not report["temporal_mismatch_candidates"]:
        lines.append("- None")
    else:
        for item in report["temporal_mismatch_candidates"]:
            lines.append(f"- {item['id']}: {item['reason']}")
            lines.append(f"  - {item['question']}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CodeMemo benchmark question quality")
    parser.add_argument(
        "--project",
        type=str,
        required=True,
        help="Project directory name under the CodeMemo dataset",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=0.85,
        help="Jaccard threshold for near-duplicate detection (default: 0.85)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of markdown",
    )
    args = parser.parse_args()

    project_dir = _resolve_data_dir() / args.project
    report = build_audit_report(project_dir, threshold=args.duplicate_threshold)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(_format_markdown(report))


if __name__ == "__main__":
    main()
