"""A channel post is stored exactly as it was given.

The store write used to rewrite every literal backslash-n / backslash-t pair
into a real newline / tab, so a body that carried those characters came back
shorter than it went in (one char lost per pair) with no error. These rows
pin that the stored body, and the body a read returns, equal the posted body.
"""

import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.channel import _channel_path, channel_post, channel_read_message


class VerbatimBodyTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        data_dir = Path(self.tmpdir) / "project" / ".synapt" / "recall"
        self.patchers = [
            patch("synapt.recall.channel.project_data_dir", return_value=data_dir),
            patch("synapt.recall.channel._read_manifest_url", return_value=None),
        ]
        for p in self.patchers:
            p.start()

    def tearDown(self):
        for p in reversed(self.patchers):
            p.stop()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _post(self, body: str) -> str:
        result = channel_post(channel="dev", message=body, display_name="poster")
        return re.search(r"\[id=([^\]]+)\]$", result).group(1)

    def _stored_body(self) -> str:
        line = _channel_path("dev").read_text(encoding="utf-8").splitlines()[-1]
        return json.loads(line)["body"]

    def test_r1_large_body_with_literal_backslash_n_is_stored_whole(self):
        body = "\\n".join(f"{i:010d}" for i in range(30_000))[:300_000]
        self.assertEqual(len(body), 300_000)
        self.assertEqual(body.count("\\n"), 25_000)
        self._post(body)
        stored = self._stored_body()
        self.assertEqual(len(stored), 300_000)
        self.assertEqual(stored, body)

    def test_r2_literal_backslash_t_is_stored_whole(self):
        body = "col_a\\tcol_b\\tcol_c"
        self._post(body)
        self.assertEqual(self._stored_body(), body)

    def test_r3_read_returns_the_posted_body(self):
        body = 'print("one\\ntwo")'
        mid = self._post(body)
        out = channel_read_message(mid, channel="dev")
        self.assertEqual(out.splitlines()[-1], body)


if __name__ == "__main__":
    unittest.main()
