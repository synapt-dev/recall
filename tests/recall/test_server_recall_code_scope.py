"""server.recall_code()'s repo_root resolution.

Two measured defects, one root cause. ``repo_root`` defaults to
``Path.cwd()`` with no check that the effective root corresponds to ONE
repo. When it is instead a directory containing several sibling repos (a
gripspace root, or any ancestor spanning more than one member repo),
``index_repo`` walks and indexes the whole subtree as one undifferentiated
"repo" tag -- symbols from every sibling repo become candidates for every
query (the wrong-workspace symptom), and because the repo tag is a raw
basename derived from whatever root happened to be in play, the SAME
physical files re-index from scratch every time that basename varies
between calls (the "13,347 files re-parsed, 0 unchanged" symptom).

The fix scopes ``recall_code`` to refuse an ambiguous multi-repo root
outright, rather than silently merging repos into one answer.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _make_git_repo(path: Path) -> None:
    (path / ".git").mkdir(parents=True)


@pytest.fixture(autouse=True)
def _isolated_project_data_dir(tmp_path, monkeypatch):
    """Every case gets its own SYNAPT_RECALL_ROOT so this test never
    touches the real gripspace's shared code_index.db."""
    isolated_root = tmp_path / "isolated-recall-root"
    isolated_root.mkdir()
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(isolated_root))


class TestAmbiguousMultiRepoRootIsRefused:
    def test_root_with_two_sibling_repos_is_refused_not_merged(self, tmp_path):
        from synapt.recall.server import recall_code

        gripspace = tmp_path / "gripspace"
        repo_a = gripspace / "repo-a"
        repo_b = gripspace / "repo-b"
        _make_git_repo(repo_a)
        _make_git_repo(repo_b)
        (repo_a / "a.py").write_text("def repo_a_func():\n    return 1\n")
        (repo_b / "b.py").write_text("def repo_b_func():\n    return 2\n")

        result = recall_code("repo_a_func", repo_root=str(gripspace))

        assert "repo-a" in result and "repo-b" in result, (
            "refusal must name the ambiguous repos so the caller knows what to pick"
        )
        assert "repo_a_func" not in result, (
            "an ambiguous root must never fall through to indexing and answering"
        )

    def test_single_repo_root_is_unaffected(self, tmp_path):
        """Control: the common, correct case (repo_root IS a git repo top)
        must proceed exactly as before."""
        from synapt.recall.server import recall_code

        repo = tmp_path / "one-repo"
        _make_git_repo(repo)
        (repo / "widget.py").write_text("def find_me():\n    return 1\n")

        result = recall_code("find_me", repo_root=str(repo))
        assert "find_me" in result

    def test_subdirectory_of_a_single_repo_is_unaffected(self, tmp_path):
        """Control: pointing repo_root at a subdirectory that has no child
        repos of its own (the common narrower-scope case) must not trip
        the ambiguous-root refusal -- only >1 sibling repo does."""
        from synapt.recall.server import recall_code

        repo = tmp_path / "one-repo"
        _make_git_repo(repo)
        sub = repo / "src" / "pkg"
        sub.mkdir(parents=True)
        (sub / "widget.py").write_text("def find_me():\n    return 1\n")

        result = recall_code("find_me", repo_root=str(sub))
        assert "find_me" in result

    def test_member_repos_nested_two_levels_deep_are_still_refused(self, tmp_path):
        """Follow-on to the v1 ambiguity guard: it checked only DIRECT
        children of an un-rooted repo_root, so container/sub/{repo-a,
        repo-b} with repo_root=container read zero sibling repos (sub
        itself carries no .git) and walked straight through both,
        answering from whichever repo's content matched -- the merged-tag
        bug the guard exists to prevent, one level below where it looked.
        Every live MCP-server caller on this host passes a gripspace root
        (repos as direct children, the v1 shape); a stranger's ~/Development
        with repos nested arbitrarily deep is the shape this closes."""
        from synapt.recall.server import recall_code

        container = tmp_path / "container"
        repo_a = container / "sub" / "repo-a"
        repo_b = container / "sub" / "repo-b"
        _make_git_repo(repo_a)
        _make_git_repo(repo_b)
        (repo_a / "deep_alpha.py").write_text("def deep_alpha():\n    return 1\n")
        (repo_b / "deep_beta.py").write_text("def deep_beta():\n    return 2\n")

        result = recall_code("deep_beta", repo_root=str(container))

        assert "repo-a" in result and "repo-b" in result, (
            "refusal must name the ambiguous repos, found at whatever depth"
        )
        assert "deep_beta" not in result, (
            "a root containing member repos nested below its direct "
            "children must never fall through to indexing and answering"
        )


class TestReparseStabilityFollowsFromScope:
    def test_second_identical_call_reparses_zero_files(self, tmp_path):
        """Once repo_root names a stable single repo,
        the repo tag stops varying between calls, and the content-hash
        cache holds across calls exactly as it already does within a
        single index_repo() call (test_code_index.py's own contract)."""
        from synapt.recall.server import recall_code

        repo = tmp_path / "one-repo"
        _make_git_repo(repo)
        (repo / "widget.py").write_text("def find_me():\n    return 1\n")

        first = recall_code("find_me", repo_root=str(repo))
        assert "1 files re-parsed" in first, (
            f"control: the first call must index exactly the one file; got: {first.splitlines()[0]!r}"
        )

        second = recall_code("find_me", repo_root=str(repo))
        assert "0 files re-parsed" in second, (
            f"expected a fully-cached second call; got: {second.splitlines()[0]!r}"
        )
