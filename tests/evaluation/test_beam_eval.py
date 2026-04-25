from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_beam_eval():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "evaluation" / "beam_eval.py"
    spec = importlib.util.spec_from_file_location("tests_beam_eval_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


beam_eval = _load_beam_eval()


def _sample_item() -> dict:
    return {
        "conversation_id": "7",
        "user_questions": [
            {"time_anchor": "January-05-2024", "messages": [["u1", "a1"]]},
            {"time_anchor": "March-15-2024", "messages": [["u2", "a2"]]},
        ],
        "chat": [
            [
                {
                    "role": "user",
                    "content": "I want to ship the MVP before March 31.",
                    "index": "1,1",
                    "time_anchor": "January-05-2024",
                },
                {
                    "role": "assistant",
                    "content": "Start with login, then budgeting, then analytics.",
                    "index": "1,2",
                    "time_anchor": "January-05-2024",
                },
            ],
            [
                {
                    "role": "user",
                    "content": "We moved the MVP date to April 15.",
                    "index": "2,1",
                    "time_anchor": "March-15-2024",
                }
            ],
        ],
        "probing_questions": repr(
            {
                "temporal_reasoning": [
                    {
                        "question": "When was the MVP date moved?",
                        "ideal_response": "March 15, 2024.",
                        "difficulty": "easy",
                        "rubric": ["March 15, 2024"],
                    }
                ],
                "information_extraction": [
                    {
                        "question": "What features should ship first?",
                        "ideal_response": "Login, budgeting, and analytics.",
                        "difficulty": "easy",
                        "rubric": ["login", "budgeting", "analytics"],
                    }
                ],
            }
        ),
    }


def test_parse_probing_questions_returns_category_dict():
    item = _sample_item()

    probes = beam_eval.parse_probing_questions(item["probing_questions"])

    assert sorted(probes) == ["information_extraction", "temporal_reasoning"]
    assert probes["temporal_reasoning"][0]["ideal_response"] == "March 15, 2024."


def test_build_questions_flattens_all_probe_categories():
    item = _sample_item()

    questions = beam_eval.build_questions(item)

    assert [q["category_name"] for q in questions] == [
        "temporal_reasoning",
        "information_extraction",
    ]
    assert questions[0]["question"] == "When was the MVP date moved?"
    assert questions[1]["answer"] == "Login, budgeting, and analytics."


def test_beam_to_transcripts_writes_one_file_per_chat_batch(tmp_path):
    item = _sample_item()

    paths = beam_eval.beam_to_transcripts(item, tmp_path)

    assert [path.name for path in paths] == ["b001c7.jsonl", "b002c7.jsonl"]
    first_lines = paths[0].read_text(encoding="utf-8").splitlines()
    second_lines = paths[1].read_text(encoding="utf-8").splitlines()
    first_message = json.loads(first_lines[0])
    second_message = json.loads(second_lines[0])

    assert first_message["sessionId"] == "b001c7"
    assert first_message["timestamp"] == "2024-01-05T00:00:00+00:00"
    assert first_message["message"]["content"][0]["text"].startswith("I want to ship")
    assert second_message["sessionId"] == "b002c7"
    assert second_message["timestamp"] == "2024-03-15T00:00:00+00:00"


def test_build_answer_prompt_uses_temporal_guardrails_for_temporal_reasoning():
    prompt = beam_eval.build_answer_prompt(
        question="When was the MVP date moved?",
        context="memory block",
        category_name="temporal_reasoning",
        date_range="January-05-2024 to March-15-2024",
    )

    assert "ONLY an anchor for resolving relative phrases" in prompt
    assert "Do NOT default to the header timestamp" in prompt
    assert "The conversations span from January-05-2024 to March-15-2024." in prompt
