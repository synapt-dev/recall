"""Regression suite for a reported ranking defect: the nine questions from
the stranger-run-2 report (a stranger asking `synapt code "where is X"`
against this gripspace), replayed against a hermetic fixture that
reproduces the two measured defect shapes rather than the real gripspace's
own code:

  1. ONE FLAT INDEX SPANS SIBLINGS. server.py's real recall_code() wrapper
     sets db/repo/repo_root all from the SAME root, and when that root has
     no git identity of its own (the gripspace root, sitting above several
     sibling checkouts), index_repo indexes every subdirectory under one
     repo tag with no home/foreign distinction -- ``reference/`` here.

  2. THE RANKING SIGNAL ITSELF. Even with zero foreign noise, a clean-index
     baseline (Atlas, 2026-09-09) showed 7 of the original 9 rows still
     wrong: token_coverage rewards incidental overlap (the repo's own name
     matching every ``recall_``-prefixed MCP wrapper, a query's content
     word living only in a file's path and never in any candidate's own
     name) and ties were broken alphabetically.

Two commits fixed part of this (path affinity + tests-after-production;
definition preference via name-match ratio + kind rank). The measured count
on the REAL gripspace clone was 2/9 RIGHT, not 9/9 -- an earlier "8 of 9 is
a path-only defect" reading was withdrawn once the clean-index baseline
showed most of the defect was the ranking signal, not path alone, and a
path-only fix was judged not to be the fruit the stranger needs.

On THIS hermetic fixture, that breaks down as: 2 deterministically RIGHT
(Q7, Q9, plain assertions below); 5 deterministically WRONG, pinned as
xfail(strict=True) with the specific residual defect each demonstrates --
not left silently passing on the wrong answer, since an accidental future
fix flips a strict xfail to a failure, forcing whoever changed the ranking
to notice and re-register the count; and 2 genuine, undiscriminated ties
(Q2, Q10) whose specific winner depends on find_symbols' row order rather
than on anything this lane ranks by -- asserted below as tie-existence, not
a specific winner, since the winner was measured to differ between two runs
on the same machine and pinning one would make this suite itself flaky.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from synapt.recall.code_index import index_repo
from synapt.recall.code_search import recall_code


REPO_NAME = "fixture-gripspace"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n")


@pytest.fixture
def gripspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    """A synthetic gripspace: a `home/` production package shaped like this
    repo's own recall naming conventions, plus a `reference/competitor/`
    sibling with generic, plausible-sounding names that compete on raw
    word-overlap -- indexed as ONE flat tree under ONE repo tag, exactly as
    server.py's real wrapper does when repo_root has no git identity of its
    own (the confounded shape; see module docstring). Returns (grip, db)."""
    monkeypatch.setattr(
        "synapt.recall.code_search.recall_server.recall_search",
        lambda query, max_chunks=3: "No results found.",
    )
    grip = tmp_path / "fixture-gripspace"
    home = grip / "home" / "src" / "synapt" / "recall"
    integrations = grip / "home" / "src" / "synapt" / "integrations"
    plugins = grip / "home" / "src" / "synapt"
    tests = grip / "home" / "tests" / "recall"
    foreign = grip / "reference" / "competitor"

    _write(
        home / "channel.py",
        """
        def channel_post(message, channel="dev"):
            \"\"\"Post a message to a channel.\"\"\"


        class ChannelMessage:
            \"\"\"A single posted message.\"\"\"
        """,
    )
    _write(
        home / "enrich.py",
        """
        ENRICHMENT_PROMPT = "summarize this conversation"


        def enrich_all(entries):
            \"\"\"Run enrichment over every entry.\"\"\"
        """,
    )
    _write(
        home / "server.py",
        """
        def recall_channel(project_dir=None):
            \"\"\"MCP wrapper: read a channel.\"\"\"


        def recall_build(project_dir=None):
            \"\"\"MCP wrapper: rebuild the index.\"\"\"
        """,
    )
    _write(
        home / "hybrid.py",
        """
        class HybridSearch:
            \"\"\"BM25 + embedding hybrid search over the recall store.\"\"\"

            def search(self, query):
                \"\"\"Run the hybrid ranking pass.\"\"\"
        """,
    )
    _write(
        home / "storage.py",
        """
        class RecallDB:
            \"\"\"The sqlite-backed storage layer.\"\"\"
        """,
    )
    _write(
        home / "config.py",
        """
        class RecallConfig:
            \"\"\"Recall's own configuration surface.\"\"\"
        """,
    )
    _write(
        integrations / "anthropic.py",
        """
        class AnthropicAdapter:
            \"\"\"Adapter exposing recall as an Anthropic tool.\"\"\"

            def search(self, query):
                \"\"\"Forward a search call to the adapter's own client.\"\"\"
        """,
    )
    _write(
        plugins / "plugins.py",
        """
        ENTRY_POINT_GROUP = "synapt.plugins"
        \"\"\"The importlib.metadata entry-point group name plugins register under.\"\"\"
        """,
    )
    _write(
        tests / "test_channel.py",
        """
        def test_recall_channel_uses_registry_dispatch():
            \"\"\"A test whose NAME carries several query words at once.\"\"\"
        """,
    )

    # Foreign sibling: plausible generic names, deliberately not copied from
    # any real project, shaped to win pure word-overlap the way a stranger's
    # `reference/`-nested competitor content did in the real report.
    _write(
        foreign / "eval_harness.py",
        """
        def run_search_eval_for_channel_class_config():
            \"\"\"Long generic name overlapping several query words at once.\"\"\"


        class MemoryEngineCoreStorage:
            \"\"\"Generic competing name for a 'core storage' question.\"\"\"

            def enrich_and_search_hybrid_module(self):
                \"\"\"Generic competing name for a 'hybrid'/'enrich' question.\"\"\"
        """,
    )

    db = grip / ".synapt" / "recall" / "code_index.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    stats = index_repo(grip, db, repo=REPO_NAME)
    assert not stats.errors, stats.errors
    return grip, db


def _top(grip: Path, db: Path, query: str) -> tuple[str, str, str] | None:
    """Ask recall_code exactly as server.py's real wrapper does: db/repo/
    repo_root all derived from the SAME flat root, which is the confounded
    shape itself, not an artificially narrower repo_root."""
    result = recall_code(
        query, db_path=str(db), repo=REPO_NAME, repo_root=str(grip),
        max_symbols=1, max_chunks=0,
    )
    if not result["symbols"]:
        return None
    top = result["symbols"][0]
    return top["path"], top["name"], top["kind"]


# --- The nine stranger-run-2 questions, RIGHT (2/9, measured 2026-09-09) ---

def test_q7_enrich_module_resolves_inside_enrich_py(gripspace: tuple[Path, Path]) -> None:
    grip, db = gripspace
    path, name, kind = _top(grip, db, "How does the enrich module work")
    assert "enrich.py" in path
    # Definition preference (kind rank): the function, not the coincidentally
    # equal-coverage ENRICHMENT_PROMPT constant.
    assert kind in ("function", "class", "method")


def test_q9_channel_post_logic_resolves_to_channel_post(gripspace: tuple[Path, Path]) -> None:
    grip, db = gripspace
    path, name, kind = _top(grip, db, "Where is the channel post logic")
    assert "channel.py" in path
    assert name == "channel_post"


# --- The remaining seven, WRONG as measured -- pinned, not silently green ---

@pytest.mark.xfail(
    strict=True,
    reason=(
        "recall_channel (server.py, an MCP wrapper) substring-matches BOTH "
        "'recall' and 'channel', so raw token_coverage (2) beats a real "
        "channel.py symbol matching only 'channel' (1) -- the repo's own "
        "product name, appearing in the query as scope-context ('...in "
        "recall'), gets credited as a second distinguishing content word. "
        "Out of this lane's two named signals (path affinity, definition "
        "preference); token_coverage's own primacy is a separate, "
        "incident-justified invariant (2026-09-02 crowding fix) this lane "
        "was not asked to touch."
    ),
)
def test_q1_channel_class_resolves_to_channel_py(gripspace: tuple[Path, Path]) -> None:
    grip, db = gripspace
    path, _name, _kind = _top(grip, db, "Where is the Channel class defined in recall")
    assert "channel.py" in path


def _assert_undiscriminated_tie(grip: Path, db: Path, query: str) -> None:
    """Q2/Q10's defect is NOT "the wrong answer wins" -- which candidate
    lands on top is decided by find_symbols' own row order, which follows
    an unsorted os.walk (code_index.py's _iter_source_files carries no
    sort), so it is filesystem/CI-dependent and was measured to differ
    between two runs on the SAME machine in the same process. Pinning a
    specific winner via xfail would make this test itself flaky. What IS
    deterministic, and what this asserts: HybridSearch.search and
    AnthropicAdapter.search are genuine ties on every signal this lane
    added (token_coverage, name_match_ratio, match_kind, kind) -- nothing
    in the two named commits (path affinity, definition preference)
    distinguishes a package's OWN search implementation from an adapter's
    identically-named method, and Stromus's ruling explicitly named
    alphabetical-as-decider-between-cov=1-peers as the thing to avoid, with
    no replacement signal named for this case."""
    from synapt.recall.code_search import (
        _KIND_RANK,
        _MATCH_KIND_RANK,
        _name_match_ratio,
    )

    result = recall_code(
        query, db_path=str(db), repo=REPO_NAME, repo_root=str(grip),
        max_symbols=10, max_chunks=0,
    )
    searches = [s for s in result["symbols"] if s["name"] == "search"]
    assert len(searches) >= 2, (
        f"expected >=2 tied 'search' candidates, got {searches}"
    )
    query_words = {
        w for w in query.lower().split() if w not in ("the", "does", "is", "a")
    }
    keys = {
        (
            s["token_coverage"],
            round(_name_match_ratio(s["name"], query_words), 6),
            _MATCH_KIND_RANK[s["match_kind"]],
            _KIND_RANK.get(s.get("kind"), 1),
        )
        for s in searches[:2]
    }
    assert len(keys) == 1, (
        f"expected the top two 'search' candidates to be a genuine tie "
        f"on every signal this lane ranks by, got distinct keys {keys}"
    )


def test_q2_hybrid_search_module_is_an_undiscriminated_tie(gripspace: tuple[Path, Path]) -> None:
    grip, db = gripspace
    _assert_undiscriminated_tie(grip, db, "What does the hybrid search module do")


def test_q10_search_ranking_is_an_undiscriminated_tie(gripspace: tuple[Path, Path]) -> None:
    grip, db = gripspace
    _assert_undiscriminated_tie(grip, db, "How does search ranking work")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ENTRY_POINT_GROUP (a constant) covers both 'entry' and 'point', "
        "the same raw token_coverage a real MCP entry-point symbol would "
        "need -- token_coverage's primacy over kind_rank means a "
        "high-coverage constant is never actually reached by the "
        "definition-preference tie-break; kind_rank only fires between "
        "candidates whose coverage already ties. A generic-sounding "
        "constant winning purely on coverage is a real residual gap in "
        "this lexical ranker, named rather than patched in this lane."
    ),
)
def test_q3_mcp_entry_point_resolves_to_server_py(gripspace: tuple[Path, Path]) -> None:
    grip, db = gripspace
    path, _name, _kind = _top(grip, db, "Where is the MCP server entry point")
    assert "server.py" in path


@pytest.mark.xfail(
    strict=True,
    reason=(
        "RecallConfig, RecallDB (in storage.py), and RecallTask-shaped "
        "peers all cover exactly 'recall' (cov=1) with an identical "
        "name_match_ratio (0.5) and kind (class) -- the query's actual "
        "distinguishing word ('storage') lives only in the FILE's own "
        "module name, never in any candidate's own symbol name, so "
        "name_match_ratio has nothing to discriminate on and alphabetical "
        "order decides. A path/module-name matching signal would fix this "
        "but is a third mechanism beyond the two this lane was scoped to."
    ),
)
def test_q4_recall_core_storage_resolves_to_storage_py(gripspace: tuple[Path, Path]) -> None:
    grip, db = gripspace
    path, _name, _kind = _top(grip, db, "How does recall core storage work")
    assert "storage.py" in path


@pytest.mark.xfail(
    strict=True,
    reason="Same defect as Q4: 'storage' is a path/module word, not a name word, "
    "on every tied cov=1 production class candidate.",
)
def test_q8_storage_backend_resolves_to_storage_py(gripspace: tuple[Path, Path]) -> None:
    grip, db = gripspace
    path, _name, _kind = _top(grip, db, "What storage backend does recall use")
    assert "storage.py" in path


@pytest.mark.xfail(
    strict=True,
    reason=(
        "This fixture's cli.py is deliberately absent: the real gripspace "
        "measurement found no candidate at all under 'cli.py' for this "
        "question (the CLI's real symbols don't literally contain 'cli') "
        "-- a missing-candidate gap in find_symbols' own token matching, "
        "not a ranking-order defect this lane's sort key can fix."
    ),
)
def test_q6_recall_cli_resolves_to_cli_py(gripspace: tuple[Path, Path]) -> None:
    grip, db = gripspace
    path, _name, _kind = _top(grip, db, "Where is the recall CLI defined")
    assert "cli.py" in path


# The bar, stated as one number: on this hermetic fixture, 2 of the nine
# resolve correctly and deterministically (Q7, Q9, asserted above as plain
# passes); 5 resolve wrong deterministically (Q1, Q3, Q4, Q6, Q8, pinned as
# strict xfail above -- an accidental future fix flips one to an unexpected
# pass, which fails the suite and forces a deliberate update); and 2 are
# genuine, undiscriminated ties whose specific winner depends on
# find_symbols' row order (itself dependent on an unsorted os.walk in
# code_index.py, measured to differ between two runs on the same machine),
# asserted above as tie-existence rather than a specific winner so this
# suite does not become the thing that is flaky. No single integer speaks
# for all nine without erasing that last distinction; that is why there is
# no test asserting a single overall count here.
