from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_codememo_audit():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "evaluation" / "codememo" / "audit.py"
    spec = importlib.util.spec_from_file_location("tests_codememo_audit_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


codememo_audit = _load_codememo_audit()


def test_near_duplicate_pairs_flags_question_rewording():
    questions = [
        {
            "id": "q1",
            "category": 5,
            "question": (
                "What naming convention do we use for database migration files, "
                "and what mechanism tracks which migrations have been applied?"
            ),
        },
        {
            "id": "q2",
            "category": 5,
            "question": (
                "What is the naming convention for database migration files, "
                "and what mechanism tracks which migrations have been applied?"
            ),
        },
        {
            "id": "q3",
            "category": 2,
            "question": "What was the root cause of the timezone bug?",
        },
    ]

    pairs = codememo_audit.near_duplicate_pairs(questions, threshold=0.85)

    assert len(pairs) == 1
    assert pairs[0]["left_id"] == "q1"
    assert pairs[0]["right_id"] == "q2"
    assert pairs[0]["similarity"] >= 0.85


def test_temporal_mismatch_candidates_flags_debug_shaped_temporal_questions():
    questions = [
        {
            "id": "q1",
            "category": 4,
            "question": "What was the root cause of the database lock bug and how did we fix it?",
        },
        {
            "id": "q2",
            "category": 4,
            "question": "In what session did we switch the SQLite journal mode to WAL?",
        },
        {
            "id": "q3",
            "category": 2,
            "question": "What was the exact error and which function was affected?",
        },
    ]

    candidates = codememo_audit.temporal_mismatch_candidates(questions)

    assert [item["id"] for item in candidates] == ["q1", "q2"]
    assert "debug/root-cause" in candidates[0]["reason"]
    assert "query intent classifier reads it as" in candidates[1]["reason"]


def test_temporal_mismatch_candidates_uses_intent_classifier_for_non_temporal_reads():
    questions = [
        {
            "id": "q1",
            "category": 4,
            "question": "What was the final test count and coverage percentage after the CI/CD setup session?",
        }
    ]

    candidates = codememo_audit.temporal_mismatch_candidates(questions)

    assert [item["id"] for item in candidates] == ["q1"]
    assert "query intent classifier reads it as" in candidates[0]["reason"]
