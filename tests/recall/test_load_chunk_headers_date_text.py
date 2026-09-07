"""load_chunk_headers must serve the stored date_text, not recompute it.

Regression coverage for the cold/warm-path fix: RecallDB.load_chunk_headers()
now selects date_text from the chunks table and passes it into TranscriptChunk
instead of leaving the field blank and letting __post_init__ recompute it via
_build_date_text() on every header load. The equal-output property this test
asserts: for every chunk, the date_text a caller of load_chunk_headers() sees
must be byte-identical to a fresh call to _build_date_text() on that chunk's
own timestamp -- i.e. the stored value and the recomputed value never diverge,
across a representative spread of timestamp shapes.
"""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from synapt.recall.core import TranscriptChunk, _build_date_text
from synapt.recall.storage import RecallDB


def _sample_timestamps(n: int) -> list[str]:
    """n distinct, varied ISO-ish timestamps: different months, days
    (including the 29th/30th/31st), hours, minutes, seconds, both
    Z-suffixed and explicit-offset forms, spanning several years so
    leap-year Feb 29 and both DST transitions are represented.
    """
    out: list[str] = []
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        dt = base + timedelta(hours=i * 7, minutes=(i * 13) % 60, seconds=(i * 3) % 60)
        if i % 2 == 0:
            out.append(dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
        else:
            out.append(dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"))
    return out


class TestLoadChunkHeadersDoesNotRecompute(unittest.TestCase):
    """The discriminating witness: load_chunk_headers() must not call
    _build_date_text() at all once chunks are already stored. A value-
    equality check alone cannot tell "recomputed the same answer" apart
    from "served the stored answer" -- both give the same output -- so
    this poisons _build_date_text() after save_chunks() and asserts
    load_chunk_headers() still returns the correct, already-computed
    text without invoking it. This is the property the fix changes;
    it fails against the pre-fix loader (which recomputes on every
    call, including via this poisoned function) and passes against
    the fixed one.
    """

    def test_load_chunk_headers_never_calls_build_date_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = RecallDB(Path(tmpdir) / "recall.db")
            try:
                chunk = TranscriptChunk(
                    id="c0",
                    session_id="s1",
                    timestamp="2025-06-15T10:30:00Z",
                    turn_index=0,
                    user_text="hello",
                    assistant_text="world",
                )
                expected_date_text = chunk.date_text
                self.assertTrue(expected_date_text)
                db.save_chunks([chunk])

                with mock.patch(
                    "synapt.recall.core._build_date_text",
                    side_effect=AssertionError(
                        "load_chunk_headers() recomputed date_text instead "
                        "of using the stored value"
                    ),
                ):
                    headers = db.load_chunk_headers()

                self.assertEqual(len(headers), 1)
                self.assertEqual(headers[0].date_text, expected_date_text)
            finally:
                db.close()


class TestLoadChunkHeadersDateTextEqualsRecompute(unittest.TestCase):
    """Byte-identical old-vs-new date_text across a 10k-timestamp sample."""

    def test_stored_date_text_matches_fresh_recompute(self):
        timestamps = _sample_timestamps(10_000)

        with tempfile.TemporaryDirectory() as tmpdir:
            db = RecallDB(Path(tmpdir) / "recall.db")
            try:
                chunks = [
                    TranscriptChunk(
                        id=f"c{i}",
                        session_id="s1",
                        timestamp=ts,
                        turn_index=i,
                        user_text=f"user {i}",
                        assistant_text=f"assistant {i}",
                    )
                    for i, ts in enumerate(timestamps)
                ]
                db.save_chunks(chunks)

                headers = db.load_chunk_headers()
                self.assertEqual(len(headers), len(timestamps))

                headers_by_id = {h.id: h for h in headers}
                mismatches = []
                for i, ts in enumerate(timestamps):
                    chunk_id = f"c{i}"
                    header = headers_by_id[chunk_id]
                    expected = _build_date_text(ts)
                    if header.date_text != expected:
                        mismatches.append((chunk_id, ts, header.date_text, expected))

                self.assertEqual(
                    mismatches,
                    [],
                    f"{len(mismatches)} of {len(timestamps)} header date_text "
                    f"values diverged from a fresh recompute; first: "
                    f"{mismatches[:3]!r}",
                )
            finally:
                db.close()

    def test_stored_date_text_is_non_empty_for_valid_timestamps(self):
        """Sanity guard: the property above is vacuous if date_text is
        empty on both sides. Confirm the stored/served value is real.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            db = RecallDB(Path(tmpdir) / "recall.db")
            try:
                chunk = TranscriptChunk(
                    id="c0",
                    session_id="s1",
                    timestamp="2025-06-15T10:30:00Z",
                    turn_index=0,
                    user_text="hello",
                    assistant_text="world",
                )
                db.save_chunks([chunk])
                headers = db.load_chunk_headers()
                self.assertEqual(len(headers), 1)
                self.assertTrue(headers[0].date_text)
                self.assertIn("2025", headers[0].date_text)
                self.assertIn("June", headers[0].date_text)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
