"""Consolidation says what its content filters rejected, and keeps hyphenated ALLCAPS identifiers.

Two behaviours, one story:

1. ``_apply_consolidation_result`` reports how many nodes the model returned and how many the content
   filters rejected, with the first words of each rejected node, so a run that printed only "3 created"
   no longer hides that more came back. What is KEPT does not change.
2. ``_lacks_specificity`` reads an ALLCAPS-hyphen identifier (``STAMP-PARITY-91``) as a project-specific
   signal, as it already does for a path, a version or a CamelCase name.

Synthetic text only.
"""
import json
import tempfile
import unittest
from pathlib import Path

from synapt.recall.consolidate import (
    ConsolidationResult,
    _apply_consolidation_result,
    _lacks_specificity,
)
from synapt.recall.journal import JournalEntry
from synapt.recall.knowledge import read_nodes

GENERIC = "Use Gradle for building Android apps"
IDENTIFIED = "The merge rule STAMP-PARITY-91 prevents merging when a stamp differs"


def _entry(session_id):
    return JournalEntry(
        timestamp="2026-03-01T00:00:00", session_id=session_id, focus="synthetic",
        done=[], decisions=[], next_steps=[], files_modified=[], enriched=True,
    )


def _node(content):
    return {"action": "create", "content": content, "category": "convention", "confidence": 0.7,
            "tags": [], "contradiction_note": "", "source_turns": ["s1:1"]}


def _apply_one(content):
    kn = Path(tempfile.mkdtemp()) / "knowledge.jsonl"
    return _apply_consolidation_result({"nodes": [_node(content)]}, [], [_entry("s1")], kn)


class TestRejectionsAreCounted(unittest.TestCase):
    def setUp(self):
        self.kn = Path(tempfile.mkdtemp()) / "knowledge.jsonl"
        self.cluster = [_entry("s1"), _entry("s2")]

    def _apply(self, contents):
        parsed = {"nodes": [_node(c) for c in contents]}
        return _apply_consolidation_result(parsed, [], self.cluster, self.kn)

    def test_a_rejected_node_is_counted_and_shown(self):
        r = self._apply([GENERIC, "Pin web to core tag v0.3.7-zeta"])
        self.assertEqual(r.nodes_emitted, 2)
        self.assertEqual(r.nodes_rejected, 1)
        self.assertEqual(r.nodes_created, 1)
        self.assertEqual(len(r.rejected_preview), 1)
        self.assertTrue(r.rejected_preview[0].startswith("Use Gradle"))

    def test_what_is_kept_does_not_change(self):
        self._apply([GENERIC, "Pin web to core tag v0.3.7-zeta"])
        kept = [n.content for n in read_nodes(self.kn)]
        self.assertEqual(kept, ["Pin web to core tag v0.3.7-zeta"])

    def test_the_preview_is_short_and_bounded(self):
        many = ["Use Gradle for building Android apps number %d with a long tail of words" % i for i in range(12)]
        r = self._apply(many)
        self.assertEqual(r.nodes_rejected, 12)
        self.assertLessEqual(len(r.rejected_preview), 5)
        self.assertTrue(all(len(p) <= 48 for p in r.rejected_preview))

    def test_a_clean_run_reports_zero_rejected(self):
        r = self._apply(["Pin web to core tag v0.3.7-zeta"])
        self.assertEqual((r.nodes_emitted, r.nodes_rejected, r.rejected_preview), (1, 0, []))

    def test_the_preview_is_one_printable_line(self):
        r = self._apply(["Use Gradle for \x1b[31mbuilding\x1b[0m Android apps\nsecond line",
                         "Use Gradle \x1b]0;title\x07 for Android builds\r\nand more"])
        self.assertEqual(r.nodes_rejected, 2)
        for p in r.rejected_preview:
            self.assertTrue(p.isprintable(), repr(p))
            self.assertNotIn("\x1b", p)
            self.assertNotIn("\n", p)

    def test_the_defaults_are_empty(self):
        r = ConsolidationResult()
        self.assertEqual((r.nodes_emitted, r.nodes_rejected, r.rejected_preview), (0, 0, []))


class TestAllCapsHyphenIdentifier(unittest.TestCase):
    def test_an_identifier_is_specific(self):
        self.assertFalse(_lacks_specificity(IDENTIFIED))
        self.assertFalse(_lacks_specificity("The codelab rule for cold trials is called CLEAN-TRIAL-58"))
        self.assertFalse(_lacks_specificity("Merging is refused by LANE-GATE-PARITY-7 when a stamp differs"))

    def test_generic_advice_is_still_dropped(self):
        for line in (
            "Use Gradle for building Android apps",
            "Store secrets in environment variables",
            "Keep dependencies up-to-date",           # kebab-case words are NOT a signal
            "Set request timeouts to 30 seconds",     # a number with a unit is NOT a signal
            "Review pull requests within 24 hours",
            "Prefer well-known, state-of-the-art tools",
        ):
            self.assertTrue(_lacks_specificity(line), line)

    def test_standard_identifiers_are_not_a_signal(self):
        """Hyphenated ALLCAPS names that every project shares say nothing about THIS project."""
        for line in (
            "Use SHA-256 for hashing passwords.",
            "Always use UTF-8 encoding for text files.",
            "Prefer HTTP-2 for API traffic.",
            "Set up CI-CD for every repository.",
            "Use AES-256 for data at rest.",
            "Validate JSON-LD before publishing.",
            "Follow ISO-8601 for dates.",
            "Prefer RFC-3339 timestamps.",
            "Keep PEP-8 style in Python files.",
        ):
            self.assertTrue(_lacks_specificity(line), line)

    def test_a_bare_ticket_number_is_not_a_signal(self):
        """A known loss, stated: one word and a number (ABC-123) is indistinguishable from UTF-8."""
        self.assertTrue(_lacks_specificity("Use ABC-123 as the label for tracked work"))

    def test_a_single_capital_word_is_not_an_identifier(self):
        self.assertTrue(_lacks_specificity("Run the TESTS before you deploy it"))

    def test_the_identifier_form_requires_a_hyphenated_tail(self):
        self.assertTrue(_lacks_specificity("Use a CI system for every build you make"))


class TestCliSaysWhatWasRejected(unittest.TestCase):
    def _run(self, result):
        import argparse, contextlib, io
        from unittest.mock import patch
        import synapt.recall.cli as cli
        args = argparse.Namespace(show=False, model=None, dry_run=False, force=False, min_entries=1, adapter_path="")
        out, err = io.StringIO(), io.StringIO()
        with patch("synapt.recall.consolidate._resolve_consolidation_model", return_value="m"), \
             patch("synapt.recall.consolidate.consolidate", return_value=result), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            cli.cmd_consolidate(args)
        return out.getvalue(), err.getvalue()

    def test_a_hostile_node_prints_as_exactly_one_stderr_line(self):
        import argparse, contextlib, io
        from unittest.mock import patch
        import synapt.recall.cli as cli
        r = _apply_one("Use Gradle \x1b]0;t\x07 for builds\nline two\nline three")
        out, err = self._run(r)
        self.assertEqual(len(err.splitlines()), 1)
        self.assertNotIn("\x1b", err)

    def test_a_rejection_is_named_on_stderr_and_stdout_is_unchanged(self):
        r = ConsolidationResult(nodes_created=3, nodes_emitted=7, nodes_rejected=4,
                                rejected_preview=["The rule is for merging", "Use Gradle for building"])
        out, err = self._run(r)
        self.assertIn("3 created", out)
        self.assertNotIn("rejected", out)
        self.assertIn("returned 7 node(s); 4 rejected by the content filters", err)
        self.assertIn('"The rule is for merging"', err)

    def test_a_run_with_no_rejection_prints_nothing_extra(self):
        out, err = self._run(ConsolidationResult(nodes_created=1, nodes_emitted=1))
        self.assertIn("1 created", out)
        self.assertEqual(err, "")


class _Client:
    """A model seam that returns one canned completion per call."""

    def __init__(self, completion, action_completion="{}"):
        self.completion, self.action_completion, self.calls = completion, action_completion, []

    def chat(self, *, model, messages, **kwargs):
        self.calls.append(model)
        content = messages[0].content if messages else ""
        return self.action_completion if "New Facts (indexed)" in content else self.completion


def _journal(tmp_path):
    from synapt.recall.journal import _journal_path, append_entry
    path = _journal_path(tmp_path)
    for sid, ts, done in [
        ("s1", "2026-07-13T10:00:00Z", ["wired extract_batch into consolidate step three"]),
        ("s2", "2026-07-13T11:00:00Z", ["tested extract_batch count invariance in consolidate"]),
        ("s3", "2026-07-13T12:00:00Z", ["extract_batch consolidate wiring behind the flag"]),
    ]:
        append_entry(JournalEntry(timestamp=ts, session_id=sid, done=done), path)


class TestTheRunCarriesTheCounts(unittest.TestCase):
    """The per-cluster counts reach the run's result on BOTH paths, not only inside the apply function."""

    def _run(self, completion, *, extract, action_completion="{}"):
        import os
        from unittest.mock import patch
        from synapt.recall.consolidate import consolidate
        tmp = Path(tempfile.mkdtemp())
        _journal(tmp)
        client = _Client(completion, action_completion)
        env = {"SYNAPT_USE_EXTRACT": "1"} if extract else {}
        with patch.dict(os.environ, env, clear=False), \
             patch("synapt.recall.consolidate._get_consolidation_client", lambda *a, **k: client):
            if not extract:
                os.environ.pop("SYNAPT_USE_EXTRACT", None)
            return consolidate(project_dir=tmp, force=True, min_entries=3)

    def test_the_monolith_path_reports_emitted_and_rejected(self):
        nodes = [_node(GENERIC), _node("recall#875 wired extract_batch into consolidate")]
        r = self._run(json.dumps({"nodes": nodes}), extract=False)
        self.assertEqual(r.nodes_emitted, 2)
        self.assertEqual(r.nodes_rejected, 1)
        self.assertEqual(r.nodes_created, 1)
        self.assertTrue(r.rejected_preview[0].startswith("Use Gradle"))

    def test_the_extract_path_reports_emitted_and_rejected(self):
        envelope = json.dumps({"extracted_at": "2026-07-13T12:00:00Z", "decisions": [], "temporal_refs": [],
                               "facts": [{"text": GENERIC}]})
        r = self._run(envelope, extract=True, action_completion='{"actions": [{"index": 0, "action": "create"}]}')
        # one fact per journal entry reaches reconcile, every one generic: all emitted, all rejected
        self.assertGreaterEqual(r.nodes_emitted, 1)
        self.assertEqual(r.nodes_rejected, r.nodes_emitted)
        self.assertEqual(r.nodes_created, 0)
        self.assertTrue(r.rejected_preview)


if __name__ == "__main__":
    unittest.main()
