from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


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


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )


class _FakeClient:
    def __init__(self) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions())


def _write_project(root: Path, name: str = "project_demo") -> Path:
    project = root / name
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


def test_api_call_uses_legacy_chat_args_for_gpt4o():
    client = _FakeClient()

    result = codememo_eval._api_call_with_retry(
        client,
        [{"role": "user", "content": "hi"}],
        max_tokens=77,
        retries=1,
        model="gpt-4o-mini",
    )

    assert result == "ok"
    call = client.chat.completions.calls[0]
    assert call["model"] == "gpt-4o-mini"
    assert call["max_tokens"] == 77
    assert call["temperature"] == 0.0
    assert call["seed"] == 42
    assert "max_completion_tokens" not in call


def test_api_call_uses_gpt5_completion_arg_shape():
    client = _FakeClient()

    result = codememo_eval._api_call_with_retry(
        client,
        [{"role": "user", "content": "hi"}],
        max_tokens=88,
        retries=1,
        model="gpt-5-mini",
    )

    assert result == "ok"
    call = client.chat.completions.calls[0]
    assert call["model"] == "gpt-5-mini"
    assert call["max_completion_tokens"] == 88
    assert call["reasoning_effort"] == "minimal"
    assert "max_tokens" not in call
    assert "temperature" not in call
    assert "seed" not in call


def test_generate_answer_uses_higher_budget_for_gpt5():
    client = _FakeClient()

    result = codememo_eval.generate_answer(
        "What version of croniter did we pin?",
        "We pinned croniter 1.3.8 for recurring task support.",
        client,
        model="gpt-5-mini",
    )

    assert result == "ok"
    call = client.chat.completions.calls[0]
    assert call["model"] == "gpt-5-mini"
    assert call["max_completion_tokens"] == 150
    assert call["reasoning_effort"] == "minimal"


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


def test_synapt_sut_namespaces_persistent_work_dir_per_project(tmp_path, monkeypatch):
    project_a = _write_project(tmp_path / "a", name="project_alpha")
    project_b = _write_project(tmp_path / "b", name="project_beta")

    import synapt.recall.core as recall_core
    import synapt.recall.storage as recall_storage

    monkeypatch.setattr(recall_core, "parse_transcript", lambda path: [])
    monkeypatch.setattr(recall_core, "build_index", lambda **kwargs: SimpleNamespace(chunks=[]))

    class DummyRecallDB:
        def __init__(self, path: Path) -> None:
            self.path = path

        def save_chunks(self, chunks) -> None:
            return None

        def close(self) -> None:
            return None

    monkeypatch.setattr(recall_storage, "RecallDB", DummyRecallDB)

    sut = codememo_eval.SynaptSUT(mode="recalldb", work_dir=str(tmp_path / "work"))
    sut.ingest(sorted((project_a / "sessions").glob("*.jsonl")))
    first_dir = sut._work_dir
    sut.ingest(sorted((project_b / "sessions").glob("*.jsonl")))
    second_dir = sut._work_dir

    assert first_dir is not None
    assert second_dir is not None
    assert first_dir != second_dir
    assert first_dir.name == project_a.name
    assert second_dir.name == project_b.name


def test_compute_retrieval_recall_prefers_better_aligned_chunk():
    retrieved = (
        "Past session context:\n"
        "--- [2026-03-10 12:00 session session_001] turn 211 ---\n"
        "Assistant: Embeddings are built with sentence-transformers "
        "all-MiniLM-L6-v2 for semantic search.\n"
    )
    evidence = [
        {
            "session_id": "session_001",
            "turn_index": 1543,
            "description": "Assistant implements embedding search using all-MiniLM-L6-v2",
        }
    ]
    session_texts = {
        "session_001": [""] * 1544,
    }
    session_texts["session_001"][1543] = (
        "This session is being continued from a previous conversation that "
        "ran out of context."
    )
    chunk_line_map = {"session_001": {1543: 133}}
    chunk_text_map = {
        "session_001": {
            133: "Continuation summary of transcript-rag extraction work.",
            211: (
                "Embeddings are built with sentence-transformers "
                "all-MiniLM-L6-v2 for semantic search."
            ),
        }
    }

    recall = codememo_eval.compute_retrieval_recall(
        retrieved,
        evidence,
        session_texts,
        chunk_line_map=chunk_line_map,
        chunk_text_map=chunk_text_map,
    )

    assert recall == 1.0


def test_compute_retrieval_recall_keeps_direct_chunk_when_it_matches():
    retrieved = (
        "Past session context:\n"
        "--- [2026-03-10 12:00 session session_001] turn 12 ---\n"
        "Assistant: We pinned croniter 1.3.8 for recurring tasks.\n"
    )
    evidence = [
        {
            "session_id": "session_001",
            "turn_index": 42,
            "description": "Assistant pins croniter 1.3.8 for recurring task support",
        }
    ]
    session_texts = {"session_001": [""] * 43}
    session_texts["session_001"][42] = "Pinned croniter 1.3.8 in pyproject.toml."
    chunk_line_map = {"session_001": {42: 12}}
    chunk_text_map = {
        "session_001": {
            12: "We pinned croniter 1.3.8 for recurring tasks.",
            27: "Discussed packaging and changelog cleanup.",
        }
    }

    recall = codememo_eval.compute_retrieval_recall(
        retrieved,
        evidence,
        session_texts,
        chunk_line_map=chunk_line_map,
        chunk_text_map=chunk_text_map,
    )

    assert recall == 1.0


def test_parse_retrieved_turns_accepts_age_suffix():
    retrieved = (
        "Past session context:\n"
        "--- [2026-03-01 16:00 session session_001] turn 192, 2w ago ---\n"
        "Assistant: Embeddings are built and cached.\n"
        "--- [2026-03-10 12:00 session session_007] turn 8, 1w ago ---\n"
        "Assistant: Embeddings are back.\n"
    )

    turns = codememo_eval._parse_retrieved_turns(retrieved)

    assert turns == {("session_001", 192), ("session_007", 8)}
