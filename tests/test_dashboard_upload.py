from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


def test_render_message_includes_inline_image():
    from synapt.dashboard.app import _render_message

    html = _render_message(
        {
            "timestamp": "2026-03-30T04:00:00.000000Z",
            "from_display": "Layne",
            "body": "see screenshot",
            "attachments": ["attachments/m_abc/screenshot.png"],
        }
    )

    assert "/api/attachments/attachments/m_abc/screenshot.png" in html
    assert "attachment-image" in html


def test_render_message_includes_file_link_for_non_images():
    from synapt.dashboard.app import _render_message

    html = _render_message(
        {
            "timestamp": "2026-03-30T04:00:00.000000Z",
            "from_display": "Layne",
            "body": "log attached",
            "attachments": ["attachments/m_abc/debug.log"],
        }
    )

    assert "attachment-link" in html
    assert "debug.log" in html
    assert "attachment-image" not in html


def test_resolve_attachment_path_rejects_traversal(tmp_path):
    from fastapi import HTTPException
    from synapt.dashboard.app import _resolve_attachment_path

    channels_dir = tmp_path / ".synapt" / "recall" / "channels"
    channels_dir.mkdir(parents=True)
    with patch("synapt.dashboard.app._channels_dir", return_value=channels_dir):
        try:
            _resolve_attachment_path("../outside.txt")
        except HTTPException as exc:
            assert exc.status_code == 404
        else:
            raise AssertionError("expected HTTPException for traversal path")


def test_attachment_route_serves_file(tmp_path):
    from synapt.dashboard.app import create_app

    channels_dir = tmp_path / ".synapt" / "recall" / "channels"
    attachment = channels_dir / "attachments" / "m_abc" / "shot.png"
    attachment.parent.mkdir(parents=True)
    attachment.write_bytes(b"pngbytes")

    with patch("synapt.dashboard.app._channels_dir", return_value=channels_dir):
        client = TestClient(create_app())
        resp = client.get("/api/attachments/attachments/m_abc/shot.png")

    assert resp.status_code == 200
    assert resp.content == b"pngbytes"


def test_post_message_accepts_attachment_upload(tmp_path):
    from synapt.dashboard.app import create_app

    captured: dict = {}

    def fake_join(**kwargs):
        return "joined"

    def fake_post(**kwargs):
        captured.update(kwargs)
        attachment_paths = kwargs.get("attachment_paths") or []
        assert len(attachment_paths) == 1
        assert Path(attachment_paths[0]).is_file()
        return "posted"

    with patch("synapt.dashboard.app.channel_post", side_effect=fake_post), \
         patch("synapt.recall.channel.channel_join", side_effect=fake_join):
        client = TestClient(create_app())
        resp = client.post(
            "/api/post/dev",
            data={"message": "", "name": "Layne"},
            files={"attachment": ("shot.png", b"pngbytes", "image/png")},
        )

    assert resp.status_code == 200
    assert captured["channel"] == "dev"
    assert captured["agent_name"] == "dashboard"


def test_join_route_registers_dashboard_presence():
    from synapt.dashboard.app import create_app

    joined: dict = {}

    def fake_join(**kwargs):
        joined.update(kwargs)
        return "joined"

    with patch("synapt.recall.channel.channel_join", side_effect=fake_join):
        client = TestClient(create_app())
        resp = client.post("/api/join/dev", data={"name": "Layne"})

    assert resp.status_code == 200
    assert joined == {
        "channel": "dev",
        "agent_name": "dashboard",
        "display_name": "Layne",
        "role": "human",
    }


def test_template_initializes_dashboard_join():
    template = Path("src/synapt/dashboard/template.html").read_text()

    assert "ensureDashboardJoin('dev');" in template
    assert "ensureDashboardJoin(ch);" in template
