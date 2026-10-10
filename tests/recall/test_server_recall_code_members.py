"""server.recall_code() at a gripspace root whose manifest declares its members.

The refusal of an ambiguous multi-repo root (see test_server_recall_code_scope.py)
exists because walking such a root indexed every member under ONE repo tag. The
remedy is not to drop the refusal: a root that DECLARES its members in
``.gitgrip/spaces/main/gripspace.yml`` is searched member by member, each under
its own stable tag, and the hits are merged with the member named on each. A
root with no manifest keeps the refusal; the members are declared, never
inferred from whatever git repos happen to sit below the root.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def _repo(path: Path, files: dict[str, str]) -> None:
    (path / ".git").mkdir(parents=True)
    for rel, text in files.items():
        f = path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)


def _manifest(root: Path, members: dict[str, dict]) -> None:
    """members: key -> {path, reference?}"""
    lines = ["version: 2", "repos:"]
    for key, spec in members.items():
        lines.append(f"  {key}:")
        lines.append(f"    url: https://example.invalid/{key}.git")
        lines.append(f"    path: {spec['path']}")
        lines.append("    default_branch: main")
        if spec.get("reference"):
            lines.append("    reference: true")
    lines.append("settings:")
    lines.append("  default_branch: main")
    manifest = root / ".gitgrip" / "spaces" / "main" / "gripspace.yml"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("\n".join(lines) + "\n")


@pytest.fixture(autouse=True)
def _isolated_project_data_dir(tmp_path, monkeypatch):
    isolated_root = tmp_path / "isolated-recall-root"
    isolated_root.mkdir()
    monkeypatch.setenv("SYNAPT_RECALL_ROOT", str(isolated_root))


@pytest.fixture
def gripspace(tmp_path) -> Path:
    root = tmp_path / "gripspace"
    _repo(root / "runner", {"src/runner/cost.py": "def generation_cost_surface():\n    return 1\n"})
    _repo(root / "eval", {"src/eval/score.py": "def score_the_answers():\n    return 2\n"})
    _manifest(root, {"runner": {"path": "./runner"}, "eval": {"path": "./eval"}})
    return root


class TestDeclaredMembersAreSearchedEach:
    def test_a_member_answers_from_the_root_and_is_named_on_the_hit(self, gripspace):
        from synapt.recall.server import recall_code

        result = recall_code("generation_cost_surface", repo_root=str(gripspace))

        assert "is not itself a git repository" not in result, f"refused: {result[:200]}"
        assert "runner/src/runner/cost.py" in result, (
            "the hit path must be reachable from the root, member directory first"
        )
        assert "generation_cost_surface" in result

    def test_every_member_is_searched_not_only_the_first(self, gripspace):
        from synapt.recall.server import recall_code

        result = recall_code("score_the_answers", repo_root=str(gripspace))

        assert "eval/src/eval/score.py" in result

    def test_a_git_repo_the_manifest_does_not_declare_is_not_searched(self, gripspace):
        """The walk-based refusal listed a scratch clone as a member; a declared
        root must not search it."""
        from synapt.recall.server import recall_code

        _repo(gripspace / "scratch-clone", {"x.py": "def only_in_the_scratch_clone():\n    return 3\n"})

        result = recall_code("only_in_the_scratch_clone", repo_root=str(gripspace))

        # the answer header echoes the query, so judge by the file the hit would name
        assert "scratch-clone" not in result
        assert "x.py" not in result

    def test_a_reference_member_is_not_searched(self, gripspace):
        from synapt.recall.server import recall_code

        _repo(gripspace / "reference" / "rival", {"r.py": "def rival_secret_sauce():\n    return 4\n"})
        _manifest(
            gripspace,
            {
                "runner": {"path": "./runner"},
                "eval": {"path": "./eval"},
                "rival": {"path": "reference/rival", "reference": True},
            },
        )

        result = recall_code("rival_secret_sauce", repo_root=str(gripspace))

        assert "reference/rival" not in result and "r.py" not in result
        assert "No code symbol matched" in result

    def test_a_declared_member_that_is_not_cloned_is_named_not_hidden(self, gripspace):
        from synapt.recall.server import recall_code

        _manifest(
            gripspace,
            {
                "runner": {"path": "./runner"},
                "eval": {"path": "./eval"},
                "extract": {"path": "./extract"},
            },
        )

        result = recall_code("generation_cost_surface", repo_root=str(gripspace))

        assert "runner/src/runner/cost.py" in result
        assert "extract" in result and "not cloned" in result, (
            "an answer that silently omits a declared member reads as 'checked, nothing there'"
        )

    def test_a_query_no_member_matches_says_so(self, gripspace):
        from synapt.recall.server import recall_code

        result = recall_code("zzqxv_nothing_like_this", repo_root=str(gripspace))

        assert "No code symbol matched" in result


class TestEachMemberKeepsAStableTag:
    def test_second_identical_call_reparses_zero_files_in_every_member(self, gripspace):
        """The cache property the refusal protected: a stable tag per member
        means the second call re-parses nothing."""
        from synapt.recall.server import recall_code

        first = recall_code("generation_cost_surface", repo_root=str(gripspace))
        assert "1 files re-parsed" in first, f"control, first call must index: {first[:300]}"

        second = recall_code("generation_cost_surface", repo_root=str(gripspace))

        assert "1 files re-parsed" not in second
        assert second.count("0 files re-parsed") == 2, (
            f"both members must report fully cached; got: {second[:400]}"
        )


class TestNoManifestKeepsTheRefusal:
    def test_two_repos_and_no_manifest_is_still_refused(self, tmp_path):
        from synapt.recall.server import recall_code

        root = tmp_path / "no-manifest"
        _repo(root / "a", {"a.py": "def only_a():\n    return 1\n"})
        _repo(root / "b", {"b.py": "def only_b():\n    return 2\n"})

        result = recall_code("only_a", repo_root=str(root))

        assert "is not itself a git repository" in result
        assert "only_a" not in result.split("Pass repo_root")[0].split("including")[0]
