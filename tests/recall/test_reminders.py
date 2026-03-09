"""Tests for synapt.recall.reminders — session reminder CRUD and lifecycle."""

import json
import tempfile
import unittest
from pathlib import Path

from synapt.recall.reminders import (
    Reminder,
    add_reminder,
    clear_reminder,
    format_for_session_start,
    get_pending,
    load_reminders,
    mark_shown,
    pop_pending,
    save_reminders,
)


class TestReminderDataclass(unittest.TestCase):
    def test_round_trip(self):
        r = Reminder(id="abc12345", text="test", created_at="2026-03-02T12:00:00")
        d = r.to_dict()
        restored = Reminder.from_dict(d)
        self.assertEqual(restored.id, r.id)
        self.assertEqual(restored.text, r.text)
        self.assertFalse(restored.sticky)
        self.assertEqual(restored.shown_count, 0)

    def test_from_dict_ignores_extra_keys(self):
        d = {"id": "x", "text": "t", "created_at": "now", "extra": "ignored"}
        r = Reminder.from_dict(d)
        self.assertEqual(r.text, "t")


class TestReminderStorage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "reminders.json"

    def test_load_nonexistent(self):
        self.assertEqual(load_reminders(self.path), [])

    def test_save_and_load(self):
        reminders = [
            Reminder(id="a", text="first", created_at="2026-03-02T12:00:00"),
            Reminder(id="b", text="second", created_at="2026-03-02T12:01:00", sticky=True),
        ]
        save_reminders(reminders, self.path)
        loaded = load_reminders(self.path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].text, "first")
        self.assertTrue(loaded[1].sticky)

    def test_load_corrupt_file(self):
        self.path.write_text("not json")
        self.assertEqual(load_reminders(self.path), [])

    def test_save_creates_parent_dirs(self):
        nested = Path(self.tmpdir) / "a" / "b" / "reminders.json"
        save_reminders([], nested)
        self.assertTrue(nested.exists())


class TestReminderCRUD(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "reminders.json"

    def test_add_reminder(self):
        r = add_reminder("test note", path=self.path)
        self.assertEqual(r.text, "test note")
        self.assertFalse(r.sticky)
        self.assertEqual(len(r.id), 8)

        loaded = load_reminders(self.path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].id, r.id)

    def test_add_sticky_reminder(self):
        r = add_reminder("sticky note", sticky=True, path=self.path)
        self.assertTrue(r.sticky)

    def test_clear_specific(self):
        r1 = add_reminder("first", path=self.path)
        r2 = add_reminder("second", path=self.path)
        count = clear_reminder(r1.id, self.path)
        self.assertEqual(count, 1)
        remaining = load_reminders(self.path)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].id, r2.id)

    def test_clear_all(self):
        add_reminder("a", path=self.path)
        add_reminder("b", path=self.path)
        count = clear_reminder(None, self.path)
        self.assertEqual(count, 2)
        self.assertEqual(load_reminders(self.path), [])

    def test_clear_nonexistent_id(self):
        add_reminder("a", path=self.path)
        count = clear_reminder("nonexistent", self.path)
        self.assertEqual(count, 0)
        self.assertEqual(len(load_reminders(self.path)), 1)

    def test_clear_empty(self):
        count = clear_reminder(None, self.path)
        self.assertEqual(count, 0)


class TestReminderPendingLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "reminders.json"

    def test_get_pending_non_sticky(self):
        add_reminder("todo", path=self.path)
        pending = get_pending(self.path)
        self.assertEqual(len(pending), 1)

    def test_mark_shown_removes_non_sticky(self):
        r = add_reminder("one-time", path=self.path)
        mark_shown([r.id], self.path)
        # Non-sticky with shown_count >= 1 should be removed
        self.assertEqual(load_reminders(self.path), [])

    def test_mark_shown_keeps_sticky(self):
        r = add_reminder("persistent", sticky=True, path=self.path)
        mark_shown([r.id], self.path)
        remaining = load_reminders(self.path)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].shown_count, 1)

    def test_sticky_always_pending(self):
        r = add_reminder("persistent", sticky=True, path=self.path)
        # Show it once
        mark_shown([r.id], self.path)
        # Still pending
        pending = get_pending(self.path)
        self.assertEqual(len(pending), 1)

    def test_mark_shown_all(self):
        r1 = add_reminder("a", path=self.path)
        r2 = add_reminder("b", path=self.path)
        mark_shown(None, self.path)  # Mark all
        self.assertEqual(load_reminders(self.path), [])

    def test_mixed_pending(self):
        r1 = add_reminder("one-time", path=self.path)
        r2 = add_reminder("persistent", sticky=True, path=self.path)
        pending = get_pending(self.path)
        self.assertEqual(len(pending), 2)
        mark_shown([r.id for r in pending], self.path)
        # Only sticky survives
        remaining = load_reminders(self.path)
        self.assertEqual(len(remaining), 1)
        self.assertTrue(remaining[0].sticky)


class TestPopPending(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "reminders.json"

    def test_pop_returns_and_clears_non_sticky(self):
        add_reminder("one-time", path=self.path)
        pending = pop_pending(self.path)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].text, "one-time")
        # Should be gone from disk
        self.assertEqual(load_reminders(self.path), [])

    def test_pop_keeps_sticky(self):
        add_reminder("persistent", sticky=True, path=self.path)
        pending = pop_pending(self.path)
        self.assertEqual(len(pending), 1)
        remaining = load_reminders(self.path)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].shown_count, 1)

    def test_pop_empty(self):
        self.assertEqual(pop_pending(self.path), [])


class TestLoadRobustness(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "reminders.json"

    def test_single_bad_entry_skipped(self):
        """One malformed entry shouldn't discard the entire list."""
        data = [
            {"id": "good", "text": "valid", "created_at": "2026-03-02T12:00:00"},
            {"bad": "entry"},  # Missing required fields
            {"id": "also-good", "text": "valid2", "created_at": "2026-03-02T12:01:00"},
        ]
        self.path.write_text(json.dumps(data))
        loaded = load_reminders(self.path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].id, "good")
        self.assertEqual(loaded[1].id, "also-good")

    def test_non_list_json(self):
        self.path.write_text('{"not": "a list"}')
        self.assertEqual(load_reminders(self.path), [])


class TestReminderFormatting(unittest.TestCase):
    def test_format_empty(self):
        self.assertEqual(format_for_session_start([]), "")

    def test_format_non_sticky(self):
        r = Reminder(id="abc", text="do thing", created_at="now")
        text = format_for_session_start([r])
        self.assertIn("do thing", text)
        self.assertIn("abc", text)
        self.assertNotIn("[sticky]", text)

    def test_format_sticky(self):
        r = Reminder(id="def", text="always", created_at="now", sticky=True)
        text = format_for_session_start([r])
        self.assertIn("[sticky]", text)
        self.assertIn("always", text)


if __name__ == "__main__":
    unittest.main()
