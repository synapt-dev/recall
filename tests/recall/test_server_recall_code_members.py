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


class TestMergedAnswerRanksCoverageBeforePath:
    """Across many repos a one-word path hit is noise: every member has files
    whose path shares a word with a plain-English question. A symbol covering
    two of the question's words must outrank one that covers one, or the answer
    is buried (measured on a real ten-member root: the symbol the question
    named ranked 60th)."""

    @staticmethod
    def _hit(name, coverage, path_ratio):
        return {
            "name": name, "kind": "class", "is_foreign": False, "is_test": False,
            "path_match_ratio": path_ratio, "token_coverage": coverage,
            "name_match_ratio": 0.5, "match_kind": "prefix",
        }

    def test_two_words_in_the_name_beat_one_word_in_the_path(self):
        from synapt.recall.code_search import merged_hit_sort_key

        two_words = self._hit("GenerationCostSurface", coverage=2, path_ratio=0.0)
        one_word_path = self._hit("run", coverage=1, path_ratio=0.17)

        assert sorted([one_word_path, two_words], key=merged_hit_sort_key)[0] is two_words

    def test_a_single_repo_answer_keeps_its_own_order(self):
        """Control: the single-repo key is unchanged by the merged one."""
        from synapt.recall.code_search import hit_sort_key

        two_words = self._hit("GenerationCostSurface", coverage=2, path_ratio=0.0)
        one_word_path = self._hit("run", coverage=1, path_ratio=0.17)

        assert sorted([two_words, one_word_path], key=hit_sort_key)[0] is one_word_path


class TestRankingSurvivesTheCuts:
    """A member keeps only its own top max_symbols before the merge. If that cut uses the
    single-repo order, the best-coverage symbol can be dropped before the merged order
    ever sees it; and if the final sort uses the single-repo order, it is buried again."""

    def _members(self, root: Path):
        # member one: six symbols whose PATH shares the query word (the single-repo order
        # prefers these) and one symbol covering both query words in a path with no match
        files = {f"src/alpha_{i}.py": f"def alpha_item_{i}():\n    return {i}\n" for i in range(6)}
        files["src/util.py"] = "def alpha_beta_thing():\n    return 1\n"
        _repo(root / "one", files)
        files2 = {f"src/alpha_{i}.py": f"def alpha_other_{i}():\n    return {i}\n" for i in range(2)}
        files2["src/util2.py"] = "def alpha_beta_other():\n    return 2\n"
        _repo(root / "two", files2)
        _manifest(root, {"one": {"path": "./one"}, "two": {"path": "./two"}})

    def test_the_best_coverage_symbol_survives_each_members_own_cut(self, tmp_path):
        from synapt.recall.server import recall_code

        root = tmp_path / "gripspace"
        self._members(root)

        result = recall_code("alpha beta", repo_root=str(root))

        assert "alpha_beta_thing" in result, "cut by the member's own single-repo top-5 before the merge"

    def test_the_final_merge_uses_the_merged_order(self, tmp_path):
        from synapt.recall.server import recall_code

        root = tmp_path / "gripspace"
        self._members(root)

        result = recall_code("alpha beta", repo_root=str(root))

        assert "alpha_beta_thing" in result and "alpha_beta_other" in result, (
            "both two-word symbols must outrank the one-word path hits of either member"
        )


class TestManifestPathsThatAreNotMembers:
    def test_a_nul_byte_in_a_path_skips_that_entry_not_the_answer(self, gripspace):
        from synapt.recall.server import recall_code

        manifest = gripspace / ".gitgrip" / "spaces" / "main" / "gripspace.yml"
        manifest.write_text(manifest.read_text().replace("settings:", "  bad:\n    url: https://example.invalid/b.git\n    path: ./ba\x00d\nsettings:"))

        result = recall_code("generation_cost_surface", repo_root=str(gripspace))

        assert "runner/src/runner/cost.py" in result, f"one bad entry must not take the answer down: {result[:200]}"

    def test_a_path_outside_the_root_is_named_and_not_searched(self, tmp_path):
        from synapt.recall.server import recall_code

        root = tmp_path / "gripspace"
        _repo(root / "runner", {"src/cost.py": "def inside_the_root():\n    return 1\n"})
        _repo(tmp_path / "outside", {"o.py": "def outside_the_root():\n    return 2\n"})
        _manifest(root, {"runner": {"path": "./runner"}, "outside": {"path": "../outside"}})

        result = recall_code("outside_the_root", repo_root=str(root))

        assert "o.py" not in result, "a declared path outside the root must never be searched"
        assert "outside the gripspace root" in result, "and the answer must say it was skipped"

    def test_two_keys_for_one_path_are_one_member(self, gripspace):
        from synapt.recall.server import recall_code

        _manifest(gripspace, {"runner": {"path": "./runner"}, "runner-again": {"path": "./runner"}, "eval": {"path": "./eval"}})

        result = recall_code("generation_cost_surface", repo_root=str(gripspace))

        assert result.count("runner/src/runner/cost.py") == 1
        assert "across the 2 declared members" in result

    def test_members_with_the_same_directory_name_do_not_re_parse_each_other(self, tmp_path):
        from synapt.recall.server import recall_code

        root = tmp_path / "gripspace"
        _repo(root / "a" / "x", {"one.py": "def only_in_a():\n    return 1\n"})
        _repo(root / "b" / "x", {"two.py": "def only_in_b():\n    return 2\n"})
        _manifest(root, {"ax": {"path": "./a/x"}, "bx": {"path": "./b/x"}})

        first = recall_code("only_in_a", repo_root=str(root))
        assert first.count("1 files re-parsed") == 2, f"control, both index once: {first[:300]}"

        second = recall_code("only_in_a", repo_root=str(root))

        assert second.count("0 files re-parsed") == 2, f"one tag shared by two members: {second[:400]}"
