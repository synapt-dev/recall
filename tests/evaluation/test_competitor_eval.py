from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_competitor_eval():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "evaluation" / "codememo" / "competitor_eval.py"
    spec = importlib.util.spec_from_file_location("tests_competitor_eval_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


competitor_eval = _load_competitor_eval()


def test_mem0_ingest_raises_when_infer_stores_zero_memories(tmp_path, monkeypatch):
    class _FakeMemory:
        @classmethod
        def from_config(cls, config):
            return cls()

        def add(self, chunk, user_id=None, infer=True, metadata=None):
            return {"results": []}

    monkeypatch.setitem(sys.modules, "mem0", SimpleNamespace(Memory=_FakeMemory))

    session = tmp_path / "session_001.jsonl"
    session.write_text(
        json.dumps(
            {
                "type": "user",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "remember this detail"}],
                },
            }
        ) + "\n",
        encoding="utf-8",
    )

    sut = competitor_eval.Mem0SUT(llm_model="gpt-5-mini", infer=True)

    try:
        try:
            sut.ingest([session])
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected zero-memory ingest to raise RuntimeError")
    finally:
        sut.close()

    assert "stored 0 memories" in message
    assert "gpt-5-mini" in message
