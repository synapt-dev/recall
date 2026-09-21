"""`synapt recall stats` must not list uninstalled models as active.

Cold-pass finding 2: the "Active Models" block printed five models
(embedding/summarization/enrichment/consolidation/reranker) as active on a
base install whose own stderr says "No embedding provider found — search will
use BM25 only". The one output a first-time user checks contradicted runtime
reality.

Contract after the fix:
- a model row is ACTIVE only when its provider stack actually resolves in
  this process (embedding: the resolved embedding provider, the same
  resolution the runtime search uses; the others: their backend stacks);
- when no row resolves, the section is titled "Configured models" and every
  row says it is not installed — agreeing with the BM25-only line;
- the header says "Active Models" only when at least one row resolved.
"""

import importlib.util
from unittest.mock import patch


class _StubProvider:
    """Minimal provider stand-in for the resolution check."""

    @staticmethod
    def embed(_texts):
        return []


class TestResolveModelStates:
    def test_providerless_env_marks_every_row_not_installed(self):
        # The stranger's environment: base install, no provider resolves.
        from synapt.recall import cli

        with patch("synapt.recall.cli.get_embedding_provider", return_value=None), \
             patch("synapt.recall.cli._stack_importable", return_value=False), \
             patch("synapt.recall.cli._ollama_reachable", return_value=False):
            rows = cli_resolve()

        assert rows, "expected one row per configured model"
        for key, model, state in rows:
            assert state == "not installed", f"{key}/{model} claimed {state} with no stack"
        # Every row is a configured model name, not an empty dict.
        assert {r[0] for r in rows} == {
            "embedding", "summarization", "enrichment", "consolidation", "reranker",
        }

    def test_resolved_embedding_provider_marks_embedding_active(self):
        from synapt.recall import cli as cli_mod

        with patch("synapt.recall.cli.get_embedding_provider", return_value=_StubProvider()), \
             patch("synapt.recall.cli._stack_importable", return_value=False), \
             patch("synapt.recall.cli._ollama_reachable", return_value=False):
            rows = cli_resolve()

        by_key = {k: (m, s) for k, m, s in rows}
        assert by_key["embedding"][1] == "active"
        # The others stay honest: no stack, no activity.
        assert by_key["summarization"][1] == "not installed"
        assert by_key["reranker"][1] == "not installed"

    def test_reranker_active_when_sentence_transformers_present(self):
        from synapt.recall import cli as cli_mod

        def stack(name):
            return name == "sentence_transformers"

        with patch("synapt.recall.cli.get_embedding_provider", return_value=None), \
             patch("synapt.recall.cli._stack_importable", side_effect=stack), \
             patch("synapt.recall.cli._ollama_reachable", return_value=False):
            rows = cli_resolve()

        by_key = {k: s for k, _, s in rows}
        assert by_key["reranker"] == "active"
        assert by_key["embedding"] == "not installed"

    def test_ollama_timeout_says_unknown_never_active(self):
        # A reachable-but-slow Ollama server must not hang stats nor guess:
        # the Ollama-dependent rows say unknown with the bound named.
        from synapt.recall import cli as cli_mod

        with patch("synapt.recall.cli.get_embedding_provider", return_value=None), \
             patch("synapt.recall.cli._stack_importable", return_value=False), \
             patch("synapt.recall.cli._ollama_reachable", return_value=None):
            rows = cli_resolve()

        by_key = {k: s for k, _, s in rows}
        assert by_key["enrichment"].startswith("unknown (Ollama did not answer in 2s)") or \
               "unknown (Ollama did not answer in" in by_key["enrichment"], by_key["enrichment"]
        assert "active" not in by_key["enrichment"]
        assert "active" not in by_key["consolidation"]


def cli_resolve():
    """Call the seam under its real name (kept short in assertions above)."""
    from synapt.recall.cli import resolve_model_states
    return resolve_model_states()


class TestStatsHeader:
    def test_all_unresolved_header_is_configured_not_active(self, capsys):
        from synapt.recall.cli import print_model_status

        rows = [(k, m, "not installed") for k, m in (
            ("embedding", "all-MiniLM-L6-v2"),
            ("reranker", "cross-encoder/x"),
        )]
        print_model_status_wrapper(rows)
        out = capsys.readouterr().out
        assert "Active Models" not in out
        assert "Configured models" in out
        assert "not installed" in out

    def test_some_active_header_is_active_models(self, capsys):
        from synapt.recall.cli import print_model_status

        rows = [("embedding", "all-MiniLM-L6-v2", "active"), ("reranker", "cross-encoder/x", "not installed")]
        print_model_status_wrapper(rows)
        out = capsys.readouterr().out
        assert "Active Models" in out
        # The inactive row still says so inside an active table.
        assert "not installed" in out

    def test_unknown_rows_never_flip_the_header_or_guess(self, capsys):
        from synapt.recall.cli import print_model_status

        rows = [("embedding", "all-MiniLM-L6-v2", "not installed"),
                ("enrichment", "laynepro/x", "unknown (Ollama did not answer in 2s)")]
        print_model_status_wrapper(rows)
        out = capsys.readouterr().out
        assert "Active Models" not in out  # unknown is not active
        assert "unknown (Ollama did not answer in 2s)" in out


def print_model_status_wrapper(rows):
    from synapt.recall.cli import print_model_status
    print_model_status(rows)

class TestReadmeCategories:
    """Ref (7) of the cold-pass scope: the README's category table must name
    all eleven valid categories and the silent default, not three."""

    def test_readme_names_all_categories_and_default(self):
        from pathlib import Path
        from synapt.recall.knowledge import VALID_CATEGORIES

        readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
        for cat in sorted(VALID_CATEGORIES):
            assert f"`{cat}`" in readme, f"README does not name category `{cat}`"
        assert "`workflow`" in readme  # the silent default is named


class TestOllamaProbeLadder:
    """R2 A2/A3: exactly ONE _ollama_reachable definition (the bounded one),
    and the REAL function classifies a silent server as unknown and a
    refused connection as not-installed."""

    def test_exactly_one_ollama_probe_definition(self):
        # A2: the unbounded shadow def once won the module binding and the
        # 2s bound was dead code. The mutation that re-adds a shadow def
        # reds this witness.
        import ast
        from pathlib import Path
        src = (Path(__file__).resolve().parents[2] / "src" / "synapt" / "recall" / "cli.py").read_text()
        defs = [n.lineno for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.FunctionDef) and n.name == "_ollama_reachable"]
        assert len(defs) == 1, f"_ollama_reachable defined {len(defs)}x at {defs}"

    def test_silent_server_is_unknown_not_not_installed(self):
        # A3: a server that ACCEPTS and never answers raises a bare
        # TimeoutError from getresponse (urllib does not wrap it in
        # URLError); the ladder must return None ("unknown"), not False.
        import socket
        import threading
        import time
        from unittest.mock import patch as _patch
        from synapt.recall import cli as cli_mod

        with socket.socket() as srv:
            srv.bind(("127.0.0.1", 0))
            srv.listen(1)
            port = srv.getsockname()[1]

            def accept_and_hold():
                try:
                    conn, _ = srv.accept()
                    time.sleep(5)  # accept, never answer, outlive the probe
                    conn.close()
                except OSError:
                    pass

            t = threading.Thread(target=accept_and_hold, daemon=True)
            t.start()
            time.sleep(0.1)  # let the listener stand

            class _Stub:
                api_url = f"http://127.0.0.1:{port}/api/embed"
                model = "probe-model"
            with _patch.object(cli_mod, "_OLLAMA_PROBE_TIMEOUT", 0.3), \
                 _patch("synapt.recall.embeddings.OllamaEmbeddings", _Stub):
                t0 = time.time()
                result = cli_mod._ollama_reachable()
                elapsed = time.time() - t0
        assert result is None, result
        assert elapsed < 1.5, f"probe took {elapsed:.2f}s — the bound did not hold"

    def test_dead_port_is_not_serving(self):
        # Control: a closed port answers by refusing → URLError → False
        # ("not installed"), the measured (False, 0.0s) shape.  On CI runners
        # whose firewall DROPS packets to closed ports instead of refusing
        # (measured: Windows runners, macOS 3.10/3.12 runners), the same
        # closed port behaves as a silent server and the probe correctly
        # returns None (unknown) — that is the platform's honest answer, not
        # a probe defect.  The discriminating witness for unknown-vs-
        # not-installed is test_silent_server_is_unknown (a socket that
        # ACCEPTS and never answers); this control proves the probe neither
        # raises nor hangs on a dead port.
        import socket
        from unittest.mock import patch as _patch
        from synapt.recall import cli as cli_mod
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            dead_port = s.getsockname()[1]  # bound here, closed with the ctx
        class _Stub:
            api_url = f"http://127.0.0.1:{dead_port}/api/embed"
            model = "probe-model"
        with _patch.object(cli_mod, "_OLLAMA_PROBE_TIMEOUT", 0.3), \
             _patch("synapt.recall.embeddings.OllamaEmbeddings", _Stub):
            result = cli_mod._ollama_reachable()
        assert result in (False, None), f"dead port: expected not-serving, got {result!r}"


class TestBackendRow:
    def test_backend_row_printed_when_not_auto(self, capsys):
        # A5: the backend override row is kept when the backend is not auto.
        from synapt.recall.cli import print_model_status
        rows = [("embedding", "all-MiniLM-L6-v2", "active")]
        print_model_status(rows, backend="onnx")
        out = capsys.readouterr().out
        assert "backend" in out and "onnx" in out

    def test_backend_row_absent_for_auto_and_default(self, capsys):
        from synapt.recall.cli import print_model_status
        rows = [("embedding", "all-MiniLM-L6-v2", "active")]
        print_model_status(rows, backend="auto")
        print_model_status(rows)  # default None: row absent
        out = capsys.readouterr().out
        assert "  backend" not in out
