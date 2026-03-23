from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_locomo_eval():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "evaluation" / "locomo_eval.py"
    spec = importlib.util.spec_from_file_location("tests_locomo_eval_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


locomo_eval = _load_locomo_eval()


def test_build_question_pairs_reverses_within_each_conversation():
    data_with_idx = [
        (
            0,
            {
                "qa": [
                    {"question": "q0a", "category": 1},
                    {"question": "q0b", "category": 2},
                    {"question": "skip", "category": 5},
                ]
            },
        ),
        (
            1,
            {
                "qa": [
                    {"question": "q1a", "category": 3},
                    {"question": "q1b", "category": 4},
                ]
            },
        ),
    ]

    forward = locomo_eval.build_question_pairs(data_with_idx, question_order="forward")
    reverse = locomo_eval.build_question_pairs(data_with_idx, question_order="reverse")

    assert [(conv_idx, qa["question"]) for conv_idx, qa in forward] == [
        (0, "q0a"),
        (0, "q0b"),
        (1, "q1a"),
        (1, "q1b"),
    ]
    assert [(conv_idx, qa["question"]) for conv_idx, qa in reverse] == [
        (0, "q0b"),
        (0, "q0a"),
        (1, "q1b"),
        (1, "q1a"),
    ]
