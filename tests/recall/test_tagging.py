"""Tests for auto-tag extraction (Phase 10)."""

from __future__ import annotations

from synapt.recall.journal import JournalEntry
from synapt.recall.tagging import (
    extract_branch_tag,
    extract_issue_refs,
    extract_tags,
)


class TestExtractIssueRefs:
    def test_single_ref(self):
        assert extract_issue_refs("fix #287") == ["issue:287"]

    def test_multiple_refs(self):
        assert extract_issue_refs("fix #287 and #290") == ["issue:287", "issue:290"]

    def test_dedup(self):
        assert extract_issue_refs("#123 then #123 again") == ["issue:123"]

    def test_no_refs(self):
        assert extract_issue_refs("no issues here") == []

    def test_empty(self):
        assert extract_issue_refs("") == []

    def test_none(self):
        assert extract_issue_refs(None) == []

    def test_in_commit_message(self):
        assert extract_issue_refs("feat: add clustering (#290)") == ["issue:290"]

    def test_skips_hex_hashes(self):
        # a1b2c3#456 — the #456 should NOT be preceded by hex char
        # but standalone #456 should still match
        result = extract_issue_refs("commit a1b2c3 and #456")
        assert "issue:456" in result


class TestExtractBranchTag:
    def test_feature_branch(self):
        assert extract_branch_tag("feat/clustering") == "branch:feat/clustering"

    def test_main_skipped(self):
        assert extract_branch_tag("main") is None

    def test_master_skipped(self):
        assert extract_branch_tag("master") is None

    def test_develop_skipped(self):
        assert extract_branch_tag("develop") is None

    def test_empty(self):
        assert extract_branch_tag("") is None

    def test_custom_branch(self):
        assert extract_branch_tag("fix/recall-309") == "branch:fix/recall-309"


class TestExtractTags:
    def _entry(self, **kwargs) -> JournalEntry:
        defaults = {
            "timestamp": "2026-03-06T00:00:00Z",
            "session_id": "sess-1",
            "branch": "",
            "focus": "",
            "done": [],
            "git_log": [],
        }
        defaults.update(kwargs)
        return JournalEntry(**defaults)

    def test_issue_from_git_log(self):
        entry = self._entry(
            session_id="s1",
            git_log=["fix: Jaccard clustering stability (#306)"],
        )
        cluster = {"session_ids": ["s1"]}
        tags = extract_tags(cluster, [entry])
        assert "issue:306" in tags

    def test_branch_tag(self):
        entry = self._entry(session_id="s1", branch="feat/clustering")
        cluster = {"session_ids": ["s1"]}
        tags = extract_tags(cluster, [entry])
        assert "branch:feat/clustering" in tags

    def test_keywords_from_focus(self):
        entry = self._entry(
            session_id="s1",
            focus="Jaccard clustering with inverted index optimization",
        )
        cluster = {"session_ids": ["s1"]}
        tags = extract_tags(cluster, [entry])
        keyword_tags = [t for t in tags if t.startswith("keyword:")]
        assert len(keyword_tags) <= 3
        assert len(keyword_tags) > 0

    def test_no_match_returns_empty(self):
        entry = self._entry(session_id="s1")
        cluster = {"session_ids": ["s2"]}  # No overlap
        tags = extract_tags(cluster, [entry])
        assert tags == []

    def test_empty_inputs(self):
        assert extract_tags({}, []) == []
        assert extract_tags({"session_ids": []}, []) == []

    def test_combined_tags(self):
        entry = self._entry(
            session_id="s1",
            branch="feat/recall-benchmarks-305",
            focus="recall benchmarks search latency",
            git_log=["feat: recall benchmarks (#305)"],
        )
        cluster = {"session_ids": ["s1"]}
        tags = extract_tags(cluster, [entry])
        assert "issue:305" in tags
        assert "branch:feat/recall-benchmarks-305" in tags
        # Should also have keyword tags
        assert any(t.startswith("keyword:") for t in tags)
