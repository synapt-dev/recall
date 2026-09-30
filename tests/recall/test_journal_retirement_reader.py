"""The retirement READER: what `pending_next_steps` serves after a retirement.

A step is retired by naming it under ``done``. `merge_carried_forward_with_report`
honours that on the WRITE side, and `session_done_items` reads the RAW entries so a
retirement made earlier in the same session is still visible -- its docstring says
exactly why:

    The RAW entries are read, not ``read_entries``: that view dedupes to one entry
    per session, so the session's own earlier writes would be missing from it and
    the union would silently equal the newest entry -- the defect wearing the fix's
    clothes.

`pending_next_steps` is the READER behind ``recall_journal action=pending``, and it
built its ``all_done`` set from ``read_entries`` -- the deduping view. So a
retirement was invisible whenever the retiring write was not the session's newest,
which is the ordinary case: retire a step, then write again before the session ends.
The writer said "16 retired by done" and the reader handed all 16 back.

Two sources, and they are deliberately NOT the same one:

* ``all_done`` takes the union of ``done`` over the RAW entries for the sessions in
  the window -- the read-side twin of ``same_session_done``. A retirement is a
  fact about the session, and every write that recorded one must count.
* ``next_steps`` stay on the DEDUPED newest-per-session entries. The newest write's
  list is the handoff; reading them raw would bring back a step a newer same-session
  write deliberately dropped.
"""

import unittest
from pathlib import Path
from tempfile import mkdtemp

from synapt.recall.journal import (
    JournalEntry,
    append_entry,
    merge_carried_forward_with_report,
    pending_next_steps,
    session_done_items,
)


class TestARetirementIsVisibleToTheReader(unittest.TestCase):
    """The three rows, and each names the arm it pins."""

    def setUp(self):
        self.tmpdir = mkdtemp()
        self.path = Path(self.tmpdir) / "journal.jsonl"
        self.steps = [f"carried step {i:02d}" for i in range(16)]
        self._clock = 0

    def _stamp(self) -> str:
        self._clock += 1
        return f"2026-09-29T00:{self._clock:02d}:00+00:00"

    def _write(self, session_id: str, done=(), next_steps=()) -> JournalEntry:
        """One journal write, in the sequence cli.py performs it.

        Mirrors the real call shape so the fixture holds what a real write leaves:
        the merge is what actually decides the stored next_steps.
        """
        entry = JournalEntry(timestamp=self._stamp(), session_id=session_id,
                             done=list(done), next_steps=list(next_steps))
        previous = None
        same_session_done = session_done_items(session_id, self.path)
        entry.next_steps, _ = merge_carried_forward_with_report(
            entry.next_steps, entry.done, previous, same_session_done=same_session_done)
        append_entry(entry, self.path)
        return entry

    def test_arm_a_a_retirement_on_a_non_final_write_is_not_served_again(self):
        """ARM A: the session retires the steps on its FIRST write, then writes again.

        The retiring write is not the session's newest, so the deduping view drops
        the done list that records it. The reader must still not serve those steps.
        """
        append_entry(JournalEntry(timestamp=self._stamp(), session_id="S0",
                                  next_steps=list(self.steps)), self.path)
        self._write("S1", done=self.steps, next_steps=["new work from write 1"])
        self._write("S1", done=[], next_steps=["new work from write 2"])

        pending = pending_next_steps(self.path)

        # This row's whole claim is the RETIREMENT, so it asserts nothing about
        # which steps a newer write drops -- that is ARM C's claim, and asserting it
        # here would make a mutation of the other half redden this row too.
        served = [s for s in pending if s.split(" [carried since")[0] in self.steps]
        self.assertEqual(
            served, [],
            f"a retired step was served again by pending_next_steps: {served[:3]} ...")
        # Deliberately membership, not exact equality: which steps a session DROPS
        # is ARM C's claim, and asserting it here too would make a mutation of that
        # half redden this row as well, so neither row would isolate its own arm.
        self.assertIn("new work from write 2", pending,
                      "the session's own newest work is still served")

    def test_arm_b_control_a_retirement_on_the_final_write_is_not_served(self):
        """ARM B (control): the retiring write IS the session's newest.

        This passes before any fix, which is what makes Arm A's failure a statement
        about WHICH write retired the step rather than about retirement in general.
        """
        append_entry(JournalEntry(timestamp=self._stamp(), session_id="S0",
                                  next_steps=list(self.steps)), self.path)
        self._write("S1", done=[], next_steps=["new work from write 1"])
        self._write("S1", done=self.steps, next_steps=["new work from write 2"])

        pending = pending_next_steps(self.path)
        served = [s for s in pending if s.split(" [carried since")[0] in self.steps]
        self.assertEqual(served, [])
        self.assertIn("new work from write 2", pending)

    def test_arm_c_a_step_the_newest_write_dropped_is_not_served(self):
        """ARM C: next_steps come from the DEDUPED newest-per-session entry.

        Pins the half that must NOT change. A session writes {a, b}, then writes {b}
        alone. The newest list is the handoff, so only b is pending. Serving the raw
        union of next_steps across the session would bring ``a`` back, and this row
        is what refuses that fix.
        """
        self._write("S1", done=[], next_steps=["step a", "step b"])
        self._write("S1", done=[], next_steps=["step b"])

        pending = pending_next_steps(self.path)
        self.assertEqual(pending, ["step b"])
        self.assertNotIn("step a", pending,
                         "a step the newest same-session write dropped was served again")


if __name__ == "__main__":
    unittest.main()
