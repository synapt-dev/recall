"""Tests for journal enrichment pipeline."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from synapt.recall.enrich import (
    ENRICHMENT_PROMPT,
    TranscriptSegment,
    _FACT_LIMITS,
    _backfill_stubs,
    _build_summary_from_chunks,
    _build_transcript_summary,
    _detect_content_type,
    _get_fact_limits,
    _has_conversation,
    _parse_enrichment_text,
    _parse_llm_response,
    _segment_transcript,
    enrich_entry,
    enrich_transcript_segments,
    iter_enrichable_entries,
)
from synapt.recall.core import project_archive_dir, project_worktree_dir
from synapt.recall.journal import JournalEntry, append_entry


def _make_entry(
    session_id: str = "sess-A",
    timestamp: str = "2026-03-01T00:00:00",
    focus: str = "",
    auto: bool = True,
    enriched: bool = False,
    done: list | None = None,
) -> JournalEntry:
    return JournalEntry(
        timestamp=timestamp,
        session_id=session_id,
        branch="main",
        focus=focus,
        done=done or [],
        auto=auto,
        enriched=enriched,
    )


class TestParseLlmResponse(unittest.TestCase):
    """Test JSON parsing for enrichment responses."""

    def test_clean_json(self):
        response = '{"focus": "Fix the bug", "done": ["Fixed it"]}'
        result = _parse_llm_response(response)
        self.assertIsNotNone(result)
        self.assertEqual(result["focus"], "Fix the bug")

    def test_json_with_markdown_fences(self):
        response = '```json\n{"focus": "Fix the bug", "done": ["Fixed it"]}\n```'
        result = _parse_llm_response(response)
        self.assertIsNotNone(result)
        self.assertEqual(result["focus"], "Fix the bug")

    def test_json_with_surrounding_text(self):
        response = 'Here is the result:\n{"focus": "Fix the bug", "done": ["Fixed it"]}\nDone!'
        result = _parse_llm_response(response)
        self.assertIsNotNone(result)

    def test_garbage_returns_none(self):
        result = _parse_llm_response("not valid json at all")
        self.assertIsNone(result)


class TestParseEnrichmentText(unittest.TestCase):
    """Test fallback structured-text parser for enrichment responses."""

    def test_focus_only(self):
        text = "Focus: Discussed authentication setup\n"
        result = _parse_enrichment_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["focus"], "Discussed authentication setup")

    def test_focus_and_done(self):
        text = (
            "Focus: Set up the auth pipeline\n"
            "Done:\n"
            "- Configured OAuth provider\n"
            "- Added login endpoint\n"
        )
        result = _parse_enrichment_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["focus"], "Set up the auth pipeline")
        self.assertEqual(result["done"], ["Configured OAuth provider", "Added login endpoint"])

    def test_all_fields(self):
        text = (
            "Focus: Refactored database layer\n"
            "Done:\n"
            "- Migrated to SQLAlchemy 2.0\n"
            "- Updated all queries\n"
            "Decisions:\n"
            "- Use async sessions everywhere\n"
            "Next steps:\n"
            "- Add connection pooling\n"
            "- Write migration tests\n"
        )
        result = _parse_enrichment_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["focus"], "Refactored database layer")
        self.assertEqual(len(result["done"]), 2)
        self.assertEqual(result["decisions"], ["Use async sessions everywhere"])
        self.assertEqual(len(result["next_steps"]), 2)

    def test_bullet_star_format(self):
        text = (
            "Focus: Bug fixes\n"
            "Done:\n"
            "* Fixed login crash\n"
            "* Patched XSS vulnerability\n"
        )
        result = _parse_enrichment_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["done"], ["Fixed login crash", "Patched XSS vulnerability"])

    def test_strips_quotes_from_values(self):
        text = 'Focus: "Worked on the API"\n'
        result = _parse_enrichment_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["focus"], "Worked on the API")

    def test_no_matching_fields_returns_none(self):
        text = "This is just random text with no structure."
        result = _parse_enrichment_text(text)
        self.assertIsNone(result)

    def test_case_insensitive_focus(self):
        text = "focus: lowercase focus line\n"
        result = _parse_enrichment_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["focus"], "lowercase focus line")

    def test_done_only_no_focus(self):
        text = (
            "Done:\n"
            "- Completed the task\n"
        )
        result = _parse_enrichment_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["done"], ["Completed the task"])

    def test_llm_response_fallback_to_text(self):
        """_parse_llm_response falls back to text parser when JSON fails."""
        text = (
            "Focus: Discussed project architecture\n"
            "Done:\n"
            "- Designed the API schema\n"
        )
        result = _parse_llm_response(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["focus"], "Discussed project architecture")
        self.assertEqual(result["done"], ["Designed the API schema"])

    def test_all_caps_headers(self):
        """FOCUS:, DONE: etc should work."""
        text = (
            "FOCUS: Set up authentication\n"
            "DONE:\n"
            "- Configured OAuth\n"
        )
        result = _parse_enrichment_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["focus"], "Set up authentication")
        self.assertEqual(result["done"], ["Configured OAuth"])

    def test_numbered_list_items(self):
        """Numbered lists (1. 2.) should be parsed like bullets."""
        text = (
            "Focus: Database migration\n"
            "Done:\n"
            "1. Migrated schema\n"
            "2. Updated queries\n"
            "3. Ran integration tests\n"
        )
        result = _parse_enrichment_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["done"]), 3)
        self.assertEqual(result["done"][0], "Migrated schema")

    def test_inline_items(self):
        """Items on same line as header (no bullets)."""
        text = "Focus: Quick fix\nDone: Fixed the redirect, Updated config\n"
        result = _parse_enrichment_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(result["done"], ["Fixed the redirect", "Updated config"])

    def test_decisions_only_sufficient(self):
        """Decisions alone (no focus or done) should return a result."""
        text = (
            "Decisions:\n"
            "- Use async everywhere\n"
            "- Switch to PostgreSQL\n"
        )
        result = _parse_enrichment_text(text)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["decisions"]), 2)


class TestEnrichmentPrompt(unittest.TestCase):
    """Test prompt construction."""

    def test_has_transcript_placeholder(self):
        self.assertIn("{transcript}", ENRICHMENT_PROMPT)

    def test_format_with_transcript(self):
        prompt = ENRICHMENT_PROMPT.format(
            transcript="[Turn 1] User: hello",
            session_date="Monday, March 22, 2026",
            done_limit="1-5",
            decisions_limit="0-3",
            next_steps_limit="0-3",
        )
        self.assertIn("[Turn 1] User: hello", prompt)
        self.assertIn("focus", prompt)
        self.assertIn("done", prompt)
        self.assertIn("decisions", prompt)
        self.assertIn("next_steps", prompt)
        self.assertIn("March 22, 2026", prompt)

    def test_has_session_date_placeholder(self):
        self.assertIn("{session_date}", ENRICHMENT_PROMPT)

    def test_has_fact_limit_placeholders(self):
        self.assertIn("{done_limit}", ENRICHMENT_PROMPT)
        self.assertIn("{decisions_limit}", ENRICHMENT_PROMPT)
        self.assertIn("{next_steps_limit}", ENRICHMENT_PROMPT)

    def test_personal_extraction_rules_in_prompt(self):
        self.assertIn("emotional reactions", ENRICHMENT_PROMPT)
        self.assertIn("social relationships", ENRICHMENT_PROMPT)
        self.assertIn("possessions, pets, addresses", ENRICHMENT_PROMPT)


class TestContentAwareFactLimits(unittest.TestCase):
    """Test content-profile-aware fact limits (#307 P1+P2)."""

    def test_personal_limits_higher(self):
        limits = _get_fact_limits("personal")
        self.assertEqual(limits["done"], 15)
        self.assertEqual(limits["decisions"], 5)
        self.assertEqual(limits["next_steps"], 5)

    def test_code_limits_default(self):
        limits = _get_fact_limits("code")
        self.assertEqual(limits["done"], 5)
        self.assertEqual(limits["decisions"], 3)
        self.assertEqual(limits["next_steps"], 3)

    def test_mixed_limits_default(self):
        limits = _get_fact_limits("mixed")
        self.assertEqual(limits["done"], 5)

    def test_unknown_falls_back_to_mixed(self):
        limits = _get_fact_limits("unknown")
        self.assertEqual(limits, _FACT_LIMITS["mixed"])

    def test_detect_personal_content(self):
        text = "User: I had dinner with my friend Sarah last weekend. We talked about vacation plans."
        self.assertEqual(_detect_content_type(text), "personal")

    def test_detect_code_content(self):
        text = "[Turn 1] User: fix the bug in main.py\n[Turn 1] Tools: Read, Edit\nOutput: ```python\ndef foo(): pass\n```"
        self.assertEqual(_detect_content_type(text), "code")

    def test_detect_mixed_content(self):
        # No strong signals either way
        text = "[Turn 1] User: hello\n[Turn 1] Assistant: hi there"
        self.assertEqual(_detect_content_type(text), "mixed")


class TestIterEnrichableEntries(unittest.TestCase):
    """Test entry eligibility filtering."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.journal_path = Path(self.tmp) / "journal.jsonl"

    def test_yields_auto_entries(self):
        entry = _make_entry(auto=True, enriched=False)
        append_entry(entry, self.journal_path)
        result = list(iter_enrichable_entries(self.journal_path))
        self.assertEqual(len(result), 1)

    def test_skips_non_auto(self):
        entry = _make_entry(auto=False, enriched=False)
        append_entry(entry, self.journal_path)
        result = list(iter_enrichable_entries(self.journal_path))
        self.assertEqual(len(result), 0)

    def test_skips_already_enriched(self):
        entry = _make_entry(auto=True, enriched=True)
        append_entry(entry, self.journal_path)
        result = list(iter_enrichable_entries(self.journal_path))
        self.assertEqual(len(result), 0)

    def test_skips_entries_with_done(self):
        """Entries with done items are already rich — not eligible."""
        entry = _make_entry(auto=True, enriched=False, done=["Finished task"])
        append_entry(entry, self.journal_path)
        result = list(iter_enrichable_entries(self.journal_path))
        self.assertEqual(len(result), 0)


class TestEnrichEntry(unittest.TestCase):
    """Test the single-entry enrichment function."""

    def test_adapter_path_passed_to_client(self):
        """adapter_path should be forwarded to MLXClient.chat()."""
        entry = _make_entry(session_id="test-sess")

        mock_client = MagicMock()
        mock_client.chat.return_value = json.dumps({
            "focus": "Test session",
            "done": ["Did something"],
        })

        with patch(
            "synapt.recall.enrich._build_transcript_summary",
            return_value="[Turn 1] User: hello",
        ):
            result = enrich_entry(
                entry,
                Path("/tmp/fake"),
                client=mock_client,
                adapter_path="adapters/enrichment-v1",
            )

        # Verify adapter_path was passed
        call_kwargs = mock_client.chat.call_args[1]
        self.assertEqual(call_kwargs["adapter_path"], "adapters/enrichment-v1")
        self.assertIsNotNone(result)
        self.assertTrue(result.enriched)
        self.assertFalse(result.auto)

    def test_no_adapter_path_passes_none(self):
        """Without adapter_path, should pass None to client.chat()."""
        entry = _make_entry(session_id="test-sess")

        mock_client = MagicMock()
        mock_client.chat.return_value = json.dumps({
            "focus": "Test session",
            "done": ["Did something"],
        })

        with patch(
            "synapt.recall.enrich._build_transcript_summary",
            return_value="[Turn 1] User: hello",
        ):
            enrich_entry(entry, Path("/tmp/fake"), client=mock_client)

        call_kwargs = mock_client.chat.call_args[1]
        self.assertIsNone(call_kwargs["adapter_path"])

    def test_returns_none_on_missing_transcript(self):
        entry = _make_entry(session_id="nonexistent")
        with patch(
            "synapt.recall.enrich._build_transcript_summary",
            return_value="",
        ):
            result = enrich_entry(entry, Path("/tmp/fake"))
        self.assertIsNone(result)

    def test_returns_none_on_bad_llm_response(self):
        entry = _make_entry(session_id="test-sess")

        mock_client = MagicMock()
        mock_client.chat.return_value = "not json"

        with patch(
            "synapt.recall.enrich._build_transcript_summary",
            return_value="[Turn 1] User: hello",
        ):
            result = enrich_entry(
                entry, Path("/tmp/fake"), client=mock_client,
            )
        self.assertIsNone(result)


class TestAutoStubPersistence(unittest.TestCase):
    """Test that auto-stubs are persisted for later enrichment."""

    def test_auto_stub_eligible_for_enrichment(self):
        """Auto-stub with files but no rich content should be enrichable."""
        tmp = tempfile.mkdtemp()
        journal_path = Path(tmp) / "journal.jsonl"
        # Simulate what SessionEnd writes: auto=True, files but no focus/done
        entry = JournalEntry(
            timestamp="2026-03-04T00:00:00",
            session_id="sess-auto",
            branch="main",
            files_modified=["src/foo.py"],
            auto=True,
            enriched=False,
        )
        append_entry(entry, journal_path)
        result = list(iter_enrichable_entries(journal_path))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].session_id, "sess-auto")


class TestBackfillStubs(unittest.TestCase):
    """Test backfill of auto-stubs from archived transcripts."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.project = Path(self.tmp)
        self.journal_path = project_worktree_dir(self.project) / "journal.jsonl"
        self.archive = project_archive_dir(self.project)
        self.archive.mkdir(parents=True)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_conversation_transcript(self, sid: str):
        """Write a transcript with actual conversation content."""
        import json
        path = self.archive / f"{sid}.jsonl"
        path.write_text(
            json.dumps({"type": "user", "message": {"role": "user", "content": "hello"}}) + "\n"
            + json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "hi"}}) + "\n"
        )

    def _write_snapshot_transcript(self, sid: str):
        """Write a transcript with only file-history-snapshot entries."""
        import json
        path = self.archive / f"{sid}.jsonl"
        path.write_text(
            json.dumps({"type": "file-history-snapshot", "snapshot": {}}) + "\n"
        )

    def test_creates_stubs_for_unjournaled_transcripts(self):
        """Transcripts with conversation content get auto-stubs."""
        self._write_conversation_transcript("sess-aaa")
        self._write_conversation_transcript("sess-bbb")
        self._write_conversation_transcript("sess-ccc")

        # Journal has 1 existing entry
        entry = _make_entry(session_id="sess-aaa", auto=False)
        append_entry(entry, self.journal_path)

        journaled = {"sess-aaa"}
        count = _backfill_stubs(self.project, self.journal_path, journaled)
        self.assertEqual(count, 2)  # bbb and ccc

        # Verify stubs are enrichable
        enrichable = list(iter_enrichable_entries(self.journal_path))
        enrichable_sids = {e.session_id for e in enrichable}
        self.assertIn("sess-bbb", enrichable_sids)
        self.assertIn("sess-ccc", enrichable_sids)
        self.assertNotIn("sess-aaa", enrichable_sids)

    def test_skips_snapshot_only_transcripts(self):
        """Snapshot-only transcripts should not get stubs."""
        self._write_snapshot_transcript("sess-snap")
        self._write_conversation_transcript("sess-conv")

        count = _backfill_stubs(self.project, self.journal_path, set())
        self.assertEqual(count, 1)  # only the conversation transcript

        enrichable = list(iter_enrichable_entries(self.journal_path))
        self.assertEqual(len(enrichable), 1)
        self.assertEqual(enrichable[0].session_id, "sess-conv")

    def test_no_archive_returns_zero(self):
        """No archive directory → no stubs created."""
        import shutil
        shutil.rmtree(self.archive)
        count = _backfill_stubs(self.project, self.journal_path, set())
        self.assertEqual(count, 0)

    def test_idempotent(self):
        """Running backfill twice doesn't create duplicate stubs."""
        self._write_conversation_transcript("sess-xyz")
        journaled: set[str] = set()
        _backfill_stubs(self.project, self.journal_path, journaled)
        _backfill_stubs(self.project, self.journal_path, journaled)
        enrichable = list(iter_enrichable_entries(self.journal_path))
        self.assertEqual(len(enrichable), 1)

    def test_has_conversation_true(self):
        """Transcript with user/assistant turns has conversation."""
        self._write_conversation_transcript("sess-conv")
        self.assertTrue(_has_conversation(self.archive / "sess-conv.jsonl"))

    def test_has_conversation_false(self):
        """Snapshot-only transcript has no conversation."""
        self._write_snapshot_transcript("sess-snap")
        self.assertFalse(_has_conversation(self.archive / "sess-snap.jsonl"))


def _write_multi_segment_transcript(path: Path, session_id: str = "sess-big"):
    """Write a transcript with multiple time-separated segments."""
    lines = []
    # Segment 0: Feb 24, 13:00-14:00 (3 user/assistant pairs)
    for i, (minute, text) in enumerate([
        (0, "Set up auth UI"),
        (15, "Add login form"),
        (45, "Fix styling"),
    ]):
        ts = f"2026-02-24T13:{minute:02d}:00.000Z"
        lines.append(json.dumps({
            "type": "user", "timestamp": ts,
            "sessionId": session_id, "uuid": f"u-s0-{i}",
            "message": {"role": "user", "content": text},
        }))
        lines.append(json.dumps({
            "type": "assistant", "timestamp": ts,
            "sessionId": session_id, "uuid": f"a-s0-{i}", "parentUuid": f"u-s0-{i}",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": f"Working on: {text}"},
            ]},
        }))

    # 2-hour gap → Segment 1: Feb 24, 16:00-17:00 (2 pairs)
    for i, (minute, text) in enumerate([
        (0, "Implement Apple Sign-In"),
        (30, "Test sandbox auth flow"),
    ]):
        ts = f"2026-02-24T16:{minute:02d}:00.000Z"
        lines.append(json.dumps({
            "type": "user", "timestamp": ts,
            "sessionId": session_id, "uuid": f"u-s1-{i}",
            "message": {"role": "user", "content": text},
        }))
        lines.append(json.dumps({
            "type": "assistant", "timestamp": ts,
            "sessionId": session_id, "uuid": f"a-s1-{i}", "parentUuid": f"u-s1-{i}",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": f"Done: {text}"},
            ]},
        }))

    # 3-hour gap → Segment 2: Feb 24, 20:00 (1 pair — short segment)
    ts = "2026-02-24T20:00:00.000Z"
    lines.append(json.dumps({
        "type": "user", "timestamp": ts,
        "sessionId": session_id, "uuid": "u-s2-0",
        "message": {"role": "user", "content": "Quick fix for login redirect"},
    }))
    lines.append(json.dumps({
        "type": "assistant", "timestamp": ts,
        "sessionId": session_id, "uuid": "a-s2-0", "parentUuid": "u-s2-0",
        "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Fixed the redirect URL"},
        ]},
    }))

    path.write_text("\n".join(lines) + "\n")


class TestBuildSummaryFromChunks(unittest.TestCase):
    """Test extracted summary builder."""

    def test_builds_from_chunks(self):
        from synapt.recall.core import TranscriptChunk
        chunks = [
            TranscriptChunk(
                id="s:t0", session_id="s", timestamp="2026-01-01T00:00:00",
                turn_index=0, user_text="Hello world",
                assistant_text="Hi there",
                tools_used=["Read"], files_touched=["src/main.py"],
            ),
            TranscriptChunk(
                id="s:t1", session_id="s", timestamp="2026-01-01T00:01:00",
                turn_index=1, user_text="Fix the bug",
                assistant_text="Done",
            ),
        ]
        result = _build_summary_from_chunks(chunks)
        self.assertIn("[Turn 0] User: Hello world", result)
        self.assertIn("[Turn 0] Assistant: Hi there", result)
        self.assertIn("[Turn 0] Tools: Read", result)
        self.assertIn("[Turn 1] User: Fix the bug", result)

    def test_truncates_long_output(self):
        from synapt.recall.core import TranscriptChunk
        chunks = [
            TranscriptChunk(
                id="s:t0", session_id="s", timestamp="2026-01-01T00:00:00",
                turn_index=0, user_text="x" * 5000,
                assistant_text="y" * 5000,
            ),
        ]
        result = _build_summary_from_chunks(chunks, max_chars=100)
        self.assertLessEqual(len(result), 130)  # 100 + truncation message
        self.assertIn("(truncated)", result)

    def test_empty_chunks(self):
        result = _build_summary_from_chunks([])
        self.assertEqual(result, "")


class TestSegmentTranscript(unittest.TestCase):
    """Test transcript segmentation by time gaps."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.transcript_path = Path(self.tmp) / "sess-big.jsonl"

    def test_segments_by_time_gap(self):
        _write_multi_segment_transcript(self.transcript_path)
        segments = _segment_transcript(self.transcript_path, gap_minutes=60)
        self.assertEqual(len(segments), 3)

    def test_segment_ids_are_unique(self):
        _write_multi_segment_transcript(self.transcript_path)
        segments = _segment_transcript(self.transcript_path, gap_minutes=60)
        ids = [s.segment_id for s in segments]
        self.assertEqual(len(ids), len(set(ids)))

    def test_segment_has_correct_chunks(self):
        _write_multi_segment_transcript(self.transcript_path)
        segments = _segment_transcript(self.transcript_path, gap_minutes=60)
        # Segment 0: 3 turns, Segment 1: 2 turns, Segment 2: 1 turn
        self.assertEqual(len(segments[0].chunks), 3)
        self.assertEqual(len(segments[1].chunks), 2)
        self.assertEqual(len(segments[2].chunks), 1)

    def test_segment_timestamps(self):
        _write_multi_segment_transcript(self.transcript_path)
        segments = _segment_transcript(self.transcript_path, gap_minutes=60)
        self.assertIn("13:00", segments[0].start_timestamp)
        self.assertIn("16:00", segments[1].start_timestamp)
        self.assertIn("20:00", segments[2].start_timestamp)

    def test_larger_gap_fewer_segments(self):
        _write_multi_segment_transcript(self.transcript_path)
        segments = _segment_transcript(self.transcript_path, gap_minutes=180)
        # 3-hour gap needed: only the last gap (16:30 → 20:00 = 3.5h) qualifies
        self.assertEqual(len(segments), 2)

    def test_empty_transcript(self):
        self.transcript_path.write_text("")
        segments = _segment_transcript(self.transcript_path)
        self.assertEqual(segments, [])

    def test_snapshot_only_dropped(self):
        """Segments with no user text are dropped."""
        self.transcript_path.write_text(
            json.dumps({"type": "file-history-snapshot", "snapshot": {}}) + "\n"
        )
        segments = _segment_transcript(self.transcript_path)
        self.assertEqual(segments, [])


class TestEnrichTranscriptSegments(unittest.TestCase):
    """Test end-to-end segment enrichment."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.project = Path(self.tmp)
        self.transcript_path = Path(self.tmp) / "transcript.jsonl"
        self.journal_path = project_worktree_dir(self.project) / "journal.jsonl"
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    def test_dry_run_counts_segments(self):
        _write_multi_segment_transcript(self.transcript_path)
        count = enrich_transcript_segments(
            self.transcript_path, self.project, dry_run=True,
        )
        self.assertEqual(count, 3)

    def test_enriches_segments_with_mock_client(self):
        _write_multi_segment_transcript(self.transcript_path)

        mock_client = MagicMock()
        mock_client.chat.return_value = json.dumps({
            "focus": "Auth UI implementation",
            "done": ["Set up login form"],
        })

        with patch("synapt.recall.enrich._MLX_AVAILABLE", True), \
             patch("synapt.recall.enrich.MLXClient", return_value=mock_client), \
             patch("synapt.recall.enrich.MLXOptions"):
            count = enrich_transcript_segments(
                self.transcript_path, self.project,
            )

        self.assertEqual(count, 3)
        # Verify journal entries were written
        entries = []
        with open(self.journal_path) as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        self.assertEqual(len(entries), 3)
        # Each should be enriched
        for e in entries:
            self.assertTrue(e["enriched"])
            self.assertFalse(e["auto"])

    def test_skips_already_journaled_segments(self):
        _write_multi_segment_transcript(self.transcript_path)

        # Pre-journal one segment — session_id comes from filename stem
        # transcript_path is "transcript.jsonl" → session_id = "transcript"
        existing = JournalEntry(
            timestamp="2026-02-24T13:00:00",
            session_id="transcript:s0",
            branch="", enriched=True, auto=False,
            focus="Already done",
        )
        append_entry(existing, self.journal_path)

        count = enrich_transcript_segments(
            self.transcript_path, self.project, dry_run=True,
        )
        self.assertEqual(count, 2)  # skipped s0

    def test_idempotent(self):
        _write_multi_segment_transcript(self.transcript_path)
        # First run (dry)
        c1 = enrich_transcript_segments(
            self.transcript_path, self.project, dry_run=True,
        )
        self.assertEqual(c1, 3)


if __name__ == "__main__":
    unittest.main()
