"""Capture user corrections as benchmark data and knowledge updates.

When a user corrects a wrong answer, this module:
1. Logs the correction to `.synapt/recall/corrections.jsonl` for benchmark use
2. Triggers a knowledge contradiction to supersede the wrong information

See issue #347 for design.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from synapt.recall.core import project_data_dir

logger = logging.getLogger(__name__)


def _corrections_path(project: Path | None = None) -> Path:
    """Return the path to the corrections JSONL file."""
    data_dir = project_data_dir(project)
    return data_dir / "recall" / "corrections.jsonl"


def log_correction(
    question: str,
    wrong_answer: str,
    correct_answer: str,
    category: str = "",
    project: Path | None = None,
) -> Path:
    """Append a correction entry to the corrections log.

    Args:
        question: The question that was answered incorrectly.
        wrong_answer: The incorrect answer that was given.
        correct_answer: The correct answer from the user.
        category: Optional category (e.g., "convention", "factual", "temporal").
        project: Project root. Defaults to cwd.

    Returns:
        Path to the corrections file.
    """
    path = _corrections_path(project)
    path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "question": question,
        "wrong_answer": wrong_answer,
        "correct_answer": correct_answer,
        "category": category,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source": "user_correction",
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info("Logged correction: %s", question[:80])
    return path


def read_corrections(project: Path | None = None) -> list[dict]:
    """Read all corrections from the log file.

    Returns:
        List of correction entries, newest last.
    """
    path = _corrections_path(project)
    if not path.exists():
        return []

    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries
