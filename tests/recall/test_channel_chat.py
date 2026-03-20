import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synapt.recall.channel_chat import ChatUI, _extract_attachment_paths


class TestChannelChatAttachments(unittest.TestCase):
    def test_extract_attachment_paths_from_drag_drop_line(self):
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "image one.png"
        path.write_text("img")

        found = _extract_attachment_paths(f'"{path}"')
        self.assertEqual(found, [str(path)])

    def test_attach_command_posts_attachments(self):
        tmpdir = tempfile.mkdtemp()
        path = Path(tmpdir) / "clip.txt"
        path.write_text("hello")

        ui = ChatUI(channel="dev", agent_name="layne")
        ui.agent_id = "human"

        with patch("synapt.recall.channel_chat.channel_post") as post_mock:
            ok = ui._handle_command(f'/attach "{path}"')

        self.assertTrue(ok)
        post_mock.assert_called_once_with(
            "dev",
            "",
            agent_name="human",
            attachment_paths=[str(path)],
        )

    def test_chat_ui_init_without_readline(self):
        with patch("synapt.recall.channel_chat._readline", None):
            ui = ChatUI(channel="dev", agent_name="layne")

        self.assertEqual(ui.channel, "dev")
