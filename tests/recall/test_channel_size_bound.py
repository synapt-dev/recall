"""A channel read is bounded by size, not only by count.

One 300,000-char post read with limit=2 returned the whole body. These rows pin
the per-message cap, the per-read budget, and chunked read_message.
"""
import shutil
import tempfile
import unittest

from synapt.recall.channel import (
    MAX_BODY_CHARS,
    MAX_READ_CHARS,
    _channel_path,
    _read_messages,
    channel_directive,
    channel_join,
    channel_pin,
    channel_post,
    channel_read,
    channel_read_message,
    channel_unread_read,
)
from tests.recall.test_channel import _patch_data_dir

BIG = 300_000
OVERHEAD = 1_000


def _body(n: int, tag: str = "a") -> str:
    return (tag + "lorem ipsum dolor sit amet " * (n // 27 + 1))[:n - 4] + "TAIL"


def _chunk_body(result: str) -> str:
    """The body slice of a read_message reply: after the header, before the footer."""
    return result.split("\n\n", 1)[1].rsplit("\n[", 1)[0]


class TestChannelSizeBound(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._patcher = _patch_data_dir(self.tmpdir)
        self._patcher.start()
        channel_join("dev", agent_name="reader", display_name="Reader")

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _post(self, body: str) -> str:
        channel_post("dev", body, agent_name="writer")
        return _read_messages(_channel_path("dev"))[-1].id

    def test_constants_are_the_ruled_guesses(self):
        self.assertEqual(MAX_BODY_CHARS, 8_000)
        self.assertEqual(MAX_READ_CHARS, 32_000)

    # R1
    def test_r1_big_post_is_capped_with_resume_marker(self):
        body = _body(BIG)
        mid = self._post(body)
        out = channel_read("dev", agent_name="reader", limit=2)
        self.assertLess(len(out), MAX_BODY_CHARS + OVERHEAD)
        self.assertIn(body[:MAX_BODY_CHARS], out)
        self.assertNotIn("TAIL", out)
        self.assertIn(f"message={mid} offset={MAX_BODY_CHARS}", out)
        self.assertIn(f"[truncated: {MAX_BODY_CHARS} of {BIG} chars shown", out)

    # R2
    def test_r2_boundary_exact_cap_is_whole(self):
        body = _body(MAX_BODY_CHARS)
        self._post(body)
        out = channel_read("dev", agent_name="reader")
        self.assertIn(body, out)
        self.assertNotIn("truncated", out)

    def test_r2_boundary_cap_plus_one_is_marked(self):
        body = _body(MAX_BODY_CHARS + 1)
        self._post(body)
        out = channel_read("dev", agent_name="reader")
        self.assertNotIn("TAIL", out)
        self.assertIn(f"[truncated: {MAX_BODY_CHARS} of {MAX_BODY_CHARS + 1} chars shown", out)

    # R3
    def test_r3_every_full_detail_level_is_bounded(self):
        self._post(_body(BIG))
        for detail in ("max", "high", "medium"):
            with self.subTest(detail=detail):
                out = channel_read("dev", agent_name="reader", detail=detail)
                self.assertLess(len(out), MAX_BODY_CHARS + OVERHEAD)
                self.assertIn("[truncated:", out)

    def test_r3_low_and_min_ordinary_rendering_unchanged(self):
        body = "y" * 1_000
        mid = self._post(body)
        low = channel_read("dev", agent_name="reader", detail="low")
        self.assertIn(f"[{mid}]  writer: {'y' * 200}... [truncated ~200 tok omitted]", low)
        self.assertIn(
            f"truncated 1 message(s), ~200 tok omitted total: {mid} (~200 tok). "
            "Use action='read_message' with message=<id> to inspect the full body.",
            low,
        )
        mn = channel_read("dev", agent_name="reader", detail="min")
        self.assertIn(f"[{mid}]  writer: {'y' * 80}... [truncated ~230 tok omitted]", mn)

    # R4
    def test_r4_big_directive_to_reader_is_bounded_at_low_and_medium(self):
        body = _body(BIG)
        channel_directive("dev", body, to="reader", agent_name="writer")
        mid = _read_messages(_channel_path("dev"))[-1].id
        for detail in ("low", "medium"):
            with self.subTest(detail=detail):
                out = channel_read("dev", agent_name="reader", detail=detail)
                self.assertLess(len(out), MAX_BODY_CHARS + OVERHEAD)
                self.assertIn("[DIRECTIVE]", out)
                self.assertIn(mid, out)
                self.assertIn(f"offset={MAX_BODY_CHARS}", out)

    # R5
    def test_r5_read_budget_keeps_newest_whole_and_refs_older(self):
        bodies = [_body(5_000, tag=f"<{i:02d}>") for i in range(20)]
        ids = [self._post(b) for b in bodies]
        out = channel_read("dev", agent_name="reader", limit=20)
        self.assertLess(len(out), MAX_READ_CHARS + 3 * OVERHEAD)
        self.assertIn(bodies[-1], out)
        self.assertIn(bodies[-2], out)
        self.assertNotIn(bodies[0], out)
        for mid in ids:
            self.assertIn(mid, out)
        self.assertIn(f"[{ids[0]}]  writer  [5000 chars, not shown]", out)

    # R6
    def test_r6_read_message_chunks_by_offset(self):
        body = _body(BIG)
        mid = self._post(body)
        res = channel_read_message(mid, channel="dev", offset=8_000, length=8_000)
        self.assertEqual(_chunk_body(res), body[8_000:16_000])
        self.assertIn(f"[chars 8000-16000 of {BIG}; next offset=16000]", res)
        last = channel_read_message(mid, channel="dev", offset=BIG - 100, length=8_000)
        self.assertIn("[end of message]", last)
        parts, off = [], 0
        while off < BIG:
            parts.append(_chunk_body(channel_read_message(mid, channel="dev", offset=off)))
            off += MAX_BODY_CHARS
        self.assertEqual("".join(parts), body)

    # R7
    def test_r7_read_message_default_is_first_chunk_only(self):
        body = _body(BIG)
        mid = self._post(body)
        res = channel_read_message(mid, channel="dev")
        self.assertLess(len(res), MAX_BODY_CHARS + OVERHEAD)
        self.assertEqual(_chunk_body(res), body[:MAX_BODY_CHARS])

    def test_r7_read_message_small_body_unchanged(self):
        mid = self._post("short body")
        res = channel_read_message(mid, channel="dev")
        self.assertTrue(res.endswith("\n\nshort body"))

    # R8
    def test_r8_big_pin_is_bounded(self):
        mid = self._post(_body(BIG))
        channel_pin("dev", mid, agent_name="reader")
        out = channel_read("dev", agent_name="reader", detail="high", limit=1)
        self.assertLess(len(out), 2 * MAX_BODY_CHARS + OVERHEAD)
        self.assertNotIn("TAIL", out)
        self.assertIn("## Pinned in #dev", out)

    # R9
    def test_r9_multibyte_counts_characters(self):
        body = "\U0001F600" * 100_000
        mid = self._post(body)
        out = channel_read("dev", agent_name="reader")
        self.assertIn("\U0001F600" * MAX_BODY_CHARS, out)
        self.assertNotIn("\U0001F600" * (MAX_BODY_CHARS + 1), out)
        out.encode("utf-8")
        parts, off = [], 0
        while off < len(body):
            parts.append(_chunk_body(channel_read_message(mid, channel="dev", offset=off)))
            off += MAX_BODY_CHARS
        self.assertEqual("".join(parts), body)

    # R10
    def test_r10_unread_read_is_bounded(self):
        self._post(_body(BIG))
        out = channel_unread_read(agent_name="reader")
        self.assertLess(len(out), MAX_BODY_CHARS + 2 * OVERHEAD)
        self.assertNotIn("TAIL", out)

    def test_r10_mcp_read_and_read_message_are_bounded(self):
        from synapt.recall import server

        body = _body(BIG)
        mid = self._post(body)
        out = server.recall_channel(action="read", channel="dev", limit=2, name="reader")
        self.assertLess(len(out), MAX_BODY_CHARS + 2 * OVERHEAD)
        self.assertNotIn("TAIL", out)
        res = server.recall_channel(
            action="read_message", channel="dev", message=mid, offset=8_000, length=100,
        )
        self.assertIn(body[8_000:8_100], res)
        self.assertNotIn(body[8_000:8_101], res)


if __name__ == "__main__":
    unittest.main()
