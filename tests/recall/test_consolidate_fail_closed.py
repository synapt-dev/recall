"""Consolidation refuses when no model backend could run, instead of reporting "nothing extracted".

When every inference attempt fails (no backend, or the backend raises), the CLI exits 1 with the
reason on stderr and the MCP tool returns a refusal. A run where the model did run keeps its old
output and exit code: nothing found, an unparseable reply, a mixed run, a dry run, a cache hit.

The client getter is forged, so no real model runs and the rows behave the same on every platform.
"""
import argparse
import functools
import json
import os

import pytest

import synapt.recall.consolidate as cons
from synapt.recall import cli, server

NODE_REPLY = json.dumps({"nodes": [{
    "action": "create", "content": "Pin web to core tag v0.3.7-zeta", "category": "convention",
    "confidence": 0.7, "tags": [], "contradiction_note": "", "source_turns": [],
}]})
EMPTY_REPLY = json.dumps({"nodes": []})

TOPICS = {
    "auth": ("auth token refresh bug in login service", "added lock around token refresh in login service"),
    "cache": ("disk cache eviction in image loader", "bounded disk cache eviction in image loader"),
}


def _write_journal(path, topics, per_topic=4):
    lines = []
    for t in topics:
        focus, done = TOPICS[t]
        for i in range(per_topic):
            lines.append(json.dumps({
                "timestamp": "2026-03-01T00:%02d:00+00:00" % (len(lines)),
                "session_id": "%s-%d" % (t, i), "branch": "main",
                "focus": "%s, attempt %d" % (focus, i), "done": [done],
                "decisions": ["use a lock for %s" % focus], "next_steps": ["test %s" % focus],
                "files_modified": [], "git_log": [], "auto": False, "enriched": True,
            }))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class FakeClient:
    """chat() plays a script: an Exception instance is raised, a string is returned."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def chat(self, **kwargs):
        step = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(step, BaseException):
            raise step
        return step


@pytest.fixture
def project(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    if "SYSTEMROOT" in os.environ:
        monkeypatch.setenv("SYSTEMROOT", os.environ["SYSTEMROOT"])
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    journal = tmp_path / "journal.jsonl"
    knowledge = tmp_path / "knowledge.jsonl"
    real = cons.consolidate
    monkeypatch.setattr(cons, "consolidate", functools.partial(
        real, explicit_journal_path=journal, explicit_knowledge_path=knowledge))
    return journal


def _client(monkeypatch, client):
    monkeypatch.setattr(cons, "_get_consolidation_client", lambda *a, **k: client)


def _cli(capsys, dry_run=False):
    args = argparse.Namespace(show=False, model="m-test", dry_run=dry_run, force=True,
                              min_entries=3, adapter_path="")
    code = None
    try:
        cli.cmd_consolidate(args)
    except SystemExit as exc:
        code = exc.code
    out, err = capsys.readouterr()
    return code, out, err


def test_cli_refuses_when_backend_import_fails(project, monkeypatch, capsys):
    _write_journal(project, ["auth"])
    _client(monkeypatch, FakeClient([ModuleNotFoundError("No module named 'mlx_lm'")]))
    code, out, err = _cli(capsys)
    assert code == 1
    assert "[consolidate] refused: no model backend could run" in err
    assert "FakeClient: No module named 'mlx_lm'" in err
    assert "No knowledge extracted" not in out
    assert "Traceback" not in err


def test_cli_refuses_when_no_client(project, monkeypatch, capsys):
    _write_journal(project, ["auth"])
    _client(monkeypatch, None)
    code, out, err = _cli(capsys)
    assert code == 1
    assert "no backend: router found none" in err
    assert "No knowledge extracted" not in out


def test_cli_refuses_when_backend_raises_every_cluster(project, monkeypatch, capsys):
    _write_journal(project, ["auth", "cache"])
    fake = FakeClient([RuntimeError("boom")])
    _client(monkeypatch, fake)
    code, out, err = _cli(capsys)
    assert code == 1
    assert fake.calls >= 2
    assert "%d of %d inference attempts failed" % (fake.calls, fake.calls) in err


def test_cli_mixed_run_stays_zero(project, monkeypatch, capsys):
    _write_journal(project, ["auth"])
    _client(monkeypatch, FakeClient([RuntimeError("boom"), NODE_REPLY]))
    code, out, err = _cli(capsys)
    assert code is None
    assert "Done!" in out
    assert "refused" not in err


def test_cli_model_ran_returned_nothing_stays_zero(project, monkeypatch, capsys):
    _write_journal(project, ["auth"])
    _client(monkeypatch, FakeClient([EMPTY_REPLY]))
    code, out, err = _cli(capsys)
    assert code is None
    assert "No knowledge extracted" in out


def test_cli_unparseable_stays_zero(project, monkeypatch, capsys):
    _write_journal(project, ["auth"])
    _client(monkeypatch, FakeClient(["this is not json"]))
    code, out, err = _cli(capsys)
    assert code is None
    assert "No knowledge extracted" in out
    assert "refused" not in err


def test_cli_dry_run_stays_zero(project, monkeypatch, capsys):
    _write_journal(project, ["auth"])
    _client(monkeypatch, FakeClient([RuntimeError("boom")]))
    code, out, err = _cli(capsys, dry_run=True)
    assert code is None
    assert "Dry run" in out


def test_cli_cache_hit_stays_zero(project, monkeypatch, capsys):
    _write_journal(project, ["auth"])
    _client(monkeypatch, FakeClient([NODE_REPLY]))
    assert _cli(capsys)[0] is None
    fake = FakeClient([RuntimeError("boom")])
    _client(monkeypatch, fake)
    code, out, err = _cli(capsys)
    assert code is None
    assert fake.calls == 0
    assert "refused" not in err


def test_cli_no_clusters_stays_zero(project, monkeypatch, capsys):
    _write_journal(project, ["auth"], per_topic=1)
    fake = FakeClient([RuntimeError("boom")])
    _client(monkeypatch, fake)
    code, out, err = _cli(capsys)
    assert code is None
    assert fake.calls == 0


def test_mcp_refusal_text(project, monkeypatch):
    _write_journal(project, ["auth"])
    _client(monkeypatch, FakeClient([ModuleNotFoundError("No module named 'mlx_lm'")]))
    text = server.recall_consolidate(force=True, model="m-test")
    assert text.startswith("Consolidation refused: no model backend could run")
    assert "mlx_lm" in text and "Model: m-test." in text
    assert "No knowledge extracted" not in text


def test_mcp_mixed_run_reports_consolidated(project, monkeypatch):
    _write_journal(project, ["auth"])
    _client(monkeypatch, FakeClient([RuntimeError("boom"), NODE_REPLY]))
    text = server.recall_consolidate(force=True, model="m-test")
    assert text.startswith("Consolidated:")
