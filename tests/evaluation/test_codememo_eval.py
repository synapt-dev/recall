from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_codememo_eval():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "evaluation" / "codememo" / "eval.py"
    spec = importlib.util.spec_from_file_location("tests_codememo_eval_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


codememo_eval = _load_codememo_eval()


class FakeSUT:
    def __init__(self) -> None:
        self.ingest_calls = 0
        self.query_calls = 0
        self.reset_calls = 0
        self.mode = "recalldb"
        self.enrich_model = ""

    def ingest(self, session_paths: list[Path]) -> None:
        self.ingest_calls += 1

    def query(self, question: str, max_chunks: int = 20) -> str:
        self.query_calls += 1
        return "Session 1\nUser: use pytest\nAssistant: use pytest -x"

    def stats(self) -> dict:
        return {"chunk_count": 4, "knowledge_count": 0}

    def reset_working_memory(self) -> None:
        self.reset_calls += 1


def _write_project(root: Path) -> Path:
    project = root / "project_demo"
    sessions = project / "sessions"
    sessions.mkdir(parents=True)

    (project / "manifest.json").write_text(
        json.dumps({"description": "demo project", "tech_stack": ["python"]}),
        encoding="utf-8",
    )
    (project / "questions.json").write_text(
        json.dumps(
            [
                {
                    "id": "q1",
                    "question": "What test runner is used?",
                    "answer": "pytest",
                    "answer_short": "pytest",
                    "category": 1,
                    "evidence": [],
                },
                {
                    "id": "q2",
                    "question": "What flag is used for fail-fast?",
                    "answer": "-x",
                    "answer_short": "-x",
                    "category": 2,
                    "evidence": [],
                },
            ]
        ),
        encoding="utf-8",
    )
    (sessions / "session_001.jsonl").write_text(
        json.dumps(
            {
                "timestamp": "2026-03-20T10:00:00Z",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "use pytest -x"}],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return project


def test_run_evaluation_repeat_runs_records_per_run(tmp_path):
    project = _write_project(tmp_path)
    sut = FakeSUT()

    summary = codememo_eval.run_evaluation(
        project_dirs=[project],
        sut=sut,
        retrieval_only=True,
        output_path=tmp_path / "results",
        repeat_runs=2,
    )

    assert sut.ingest_calls == 1
    assert sut.query_calls == 4
    assert summary["repeat_runs"] == 2
    assert summary["questions_per_run"] == 2
    assert summary["questions_evaluated_total"] == 4
    assert [row["run"] for row in summary["run_summaries"]] == [1, 2]

    detailed = json.loads((tmp_path / "results" / "codememo_detailed.json").read_text())
    assert {(row["run"], row["id"]) for row in detailed} == {
        (1, "q1"), (1, "q2"), (2, "q1"), (2, "q2")
    }


def test_run_evaluation_can_reset_working_memory_between_runs(tmp_path):
    project = _write_project(tmp_path)
    sut = FakeSUT()

    summary = codememo_eval.run_evaluation(
        project_dirs=[project],
        sut=sut,
        retrieval_only=True,
        output_path=tmp_path / "results",
        repeat_runs=3,
        reset_working_memory_between_runs=True,
    )

    assert summary["repeat_runs"] == 3
    assert sut.reset_calls == 2
