"""synapt dashboard — web-based mission control for multi-agent sessions.

Launch with: synapt dashboard [--port 8420] [--no-open]
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from html import escape
from pathlib import Path
from urllib.parse import quote

import markdown as _md
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse

from synapt.recall.channel import _channels_dir
from synapt.recall.core import project_data_dir
from synapt.recall.channel import (
    ChannelMessage,
    channel_agents_json,
    channel_list_channels,
    channel_messages_json,
    channel_post,
)

_MD = _md.Markdown(extensions=["fenced_code", "tables", "nl2br"])

# ---------------------------------------------------------------------------
# HTML fragment renderers
# ---------------------------------------------------------------------------

_STATUS_COLORS = {
    "online": "#4ade80",
    "idle": "#facc15",
    "away": "#fb923c",
    "offline": "#6b7280",
}

# Distinct colors per agent for the message feed
_AGENT_COLORS = [
    "#8b5cf6",  # purple (Opus)
    "#06b6d4",  # cyan (Apollo)
    "#4ade80",  # green (Sentinel)
    "#fb923c",  # orange (Atlas)
    "#f472b6",  # pink
    "#facc15",  # yellow
    "#a78bfa",  # light purple
    "#34d399",  # emerald
]
_agent_color_cache: dict[str, str] = {}


def _agent_color(name: str) -> str:
    """Assign a stable color to an agent name."""
    if name not in _agent_color_cache:
        idx = len(_agent_color_cache) % len(_AGENT_COLORS)
        _agent_color_cache[name] = _AGENT_COLORS[idx]
    return _agent_color_cache[name]


def _render_agent_tile(agent: dict) -> str:
    status = agent["status"]
    color = _STATUS_COLORS.get(status, "#6b7280")
    name = agent["display_name"] or agent["griptree"] or agent["agent_id"]
    role = agent["role"] if agent["role"] != "agent" else ""
    channels = ", ".join(f"#{c}" for c in agent["channels"]) or "no channels"
    seen = agent["last_seen"][11:16] if len(agent["last_seen"]) > 16 else ""
    return (
        f'<div class="tile" style="border-left:4px solid {color}">'
        f'<div class="tile-name">{escape(name)}</div>'
        f'<div class="tile-role">{escape(role)}</div>'
        f'<div class="tile-meta">'
        f'<span style="color:{color}">{status}</span>'
        f' &middot; {escape(channels)}'
        f' &middot; {seen}'
        f'</div></div>'
    )


def _attachment_url(rel_path: str) -> str:
    """Return the dashboard URL for a stored attachment."""
    return f"/api/attachments/{quote(rel_path, safe='/')}"


def _is_image_attachment(rel_path: str) -> bool:
    """Return True when the attachment should render inline as an image."""
    suffix = Path(rel_path).suffix.lower()
    return suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def _render_attachments(msg: dict) -> str:
    """Render stored attachments as inline images or links."""
    attachments = msg.get("attachments") or []
    if not attachments:
        return ""

    parts: list[str] = ['<div class="attachments">']
    for rel_path in attachments:
        url = _attachment_url(rel_path)
        label = escape(Path(rel_path).name)
        if _is_image_attachment(rel_path):
            parts.append(
                '<a class="attachment-link image-link" href="{}" target="_blank" rel="noopener">'
                '<img class="attachment-image" src="{}" alt="{}">'
                "</a>".format(url, url, label)
            )
        else:
            parts.append(
                '<a class="attachment-link" href="{}" target="_blank" rel="noopener">{}</a>'.format(
                    url, label
                )
            )
    parts.append("</div>")
    return "".join(parts)


def _resolve_attachment_path(rel_path: str) -> Path:
    """Resolve a stored attachment path safely inside the channel store."""
    base = _channels_dir().resolve()
    candidate = (base / rel_path).resolve()
    if not candidate.is_relative_to(base):
        raise HTTPException(status_code=404, detail="Attachment not found")
    return candidate


def _render_message(msg: dict) -> str:
    ts = msg.get("timestamp", "")
    ts_short = ts[11:16] if len(ts) > 16 else ts
    name = msg.get("from_display") or msg.get("from", "")
    body = msg.get("body", "")
    msg_type = msg.get("type", "message")
    to = msg.get("to", "")
    attachments_html = _render_attachments(msg)

    if msg_type in ("join", "leave"):
        return (
            f'<div class="msg sys">'
            f'<span class="ts">{ts_short}</span> '
            f'<span class="sys-text">-- {escape(name)} {"joined" if msg_type == "join" else "left"}</span>'
            f'</div>'
        )
    color = _agent_color(name)
    _MD.reset()
    body_html = _MD.convert(body)
    if msg_type == "directive":
        return (
            f'<div class="msg directive">'
            f'<span class="ts">{ts_short}</span> '
            f'<b style="color:{color}">{escape(name)}</b> &rarr; @{escape(to)}: {body_html}'
            f'{attachments_html}'
            f'</div>'
        )
    return (
        f'<div class="msg">'
        f'<span class="ts">{ts_short}</span> '
        f'<b style="color:{color}">{escape(name)}</b>: {body_html}'
        f'{attachments_html}'
        f'</div>'
    )


def _render_agents_html(agents: list[dict]) -> str:
    if not agents:
        return '<div class="tile empty">No agents online</div>'
    return "".join(_render_agent_tile(a) for a in agents)


def _render_messages_html(messages: list[dict]) -> str:
    return "".join(_render_message(m) for m in messages)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

_TEMPLATE: str | None = None


def _load_template() -> str:
    global _TEMPLATE
    if _TEMPLATE is None:
        path = Path(__file__).parent / "template.html"
        _TEMPLATE = path.read_text()
    return _TEMPLATE


def _ensure_dashboard_join(
    joined: set[tuple[str, str]],
    channel: str,
    name: str,
) -> None:
    """Ensure the dashboard user has an explicit human presence entry."""
    from synapt.recall.channel import channel_join

    clean_name = (name or "dashboard").strip() or "dashboard"
    join_key = (channel, clean_name)
    if join_key in joined:
        return
    channel_join(channel=channel, agent_name="dashboard", display_name=clean_name, role="human")
    joined.add(join_key)


def _dashboard_pid_path(project_dir: Path | None = None) -> Path:
    """Return the dashboard PID file under the project .synapt root."""
    return project_data_dir(project_dir).parent / "dashboard.pid"


def _dashboard_log_path(project_dir: Path | None = None) -> Path:
    """Return the dashboard log file under the project .synapt root."""
    return project_data_dir(project_dir).parent / "dashboard.log"


def _read_pid(pid_path: Path) -> int | None:
    """Read a PID file, returning None for missing or invalid content."""
    try:
        raw = pid_path.read_text().strip()
    except FileNotFoundError:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _pid_is_running(pid: int) -> bool:
    """Return True when the process exists and is signalable."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _cleanup_stale_pidfile(pid_path: Path) -> None:
    """Remove the PID file if it points at no live process."""
    pid = _read_pid(pid_path)
    if pid is None or not _pid_is_running(pid):
        pid_path.unlink(missing_ok=True)


def _stop_dashboard(project_dir: Path | None = None) -> bool:
    """Stop a background dashboard server if one is running."""
    pid_path = _dashboard_pid_path(project_dir)
    pid = _read_pid(pid_path)
    if pid is None:
        pid_path.unlink(missing_ok=True)
        return False
    if not _pid_is_running(pid):
        pid_path.unlink(missing_ok=True)
        return False

    os.kill(pid, signal.SIGTERM)
    deadline = time.time() + 3.0
    while time.time() < deadline:
        if not _pid_is_running(pid):
            pid_path.unlink(missing_ok=True)
            return True
        time.sleep(0.1)
    os.kill(pid, signal.SIGKILL)
    pid_path.unlink(missing_ok=True)
    return True


def _background_command(host: str, port: int) -> list[str]:
    """Build the detached child command for the dashboard server."""
    return [
        sys.executable,
        "-m",
        "synapt.cli",
        "dashboard",
        "--foreground",
        "--host",
        host,
        "--port",
        str(port),
        "--no-open",
    ]


def _start_dashboard_background(
    host: str,
    port: int,
    no_open: bool,
    project_dir: Path | None = None,
) -> int:
    """Spawn the dashboard server in the background and persist its PID."""
    synapt_dir = project_data_dir(project_dir).parent
    synapt_dir.mkdir(parents=True, exist_ok=True)
    pid_path = _dashboard_pid_path(project_dir)
    log_path = _dashboard_log_path(project_dir)
    _cleanup_stale_pidfile(pid_path)

    existing_pid = _read_pid(pid_path)
    if existing_pid is not None and _pid_is_running(existing_pid):
        if not no_open:
            import webbrowser

            webbrowser.open(f"http://{host}:{port}")
        return existing_pid

    cmd = _background_command(host, port)
    with log_path.open("ab") as log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            cwd=str(Path.cwd()),
            start_new_session=True,
        )

    time.sleep(0.2)
    if proc.poll() is not None:
        raise RuntimeError(
            f"Dashboard exited immediately with status {proc.returncode}. "
            f"See {log_path}."
        )

    pid_path.write_text(f"{proc.pid}\n")
    if not no_open:
        import webbrowser

        webbrowser.open(f"http://{host}:{port}")
    return proc.pid


def create_app() -> FastAPI:
    app = FastAPI(title="synapt dashboard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _load_template()

    @app.get("/api/agents")
    async def api_agents():
        return channel_agents_json()

    @app.get("/api/channels")
    async def api_channels():
        return channel_list_channels()

    @app.get("/api/messages/{channel}", response_class=HTMLResponse)
    async def api_messages(channel: str, limit: int = 50, since: str | None = None):
        msgs = channel_messages_json(channel=channel, limit=limit, since=since)
        return _render_messages_html(msgs)

    @app.get("/api/attachments/{attachment_path:path}")
    async def api_attachment(attachment_path: str):
        path = _resolve_attachment_path(attachment_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Attachment not found")
        return FileResponse(path)

    _dashboard_joined: set[tuple[str, str]] = set()

    @app.post("/api/join/{channel}")
    async def api_join(channel: str, name: str = Form("dashboard")):
        _ensure_dashboard_join(_dashboard_joined, channel=channel, name=name)
        return {"ok": True}

    @app.post("/api/post/{channel}")
    async def api_post(
        channel: str,
        message: str = Form(""),
        name: str = Form("dashboard"),
        attachment: UploadFile | None = File(None),
    ):
        agent_name = "dashboard"
        _ensure_dashboard_join(_dashboard_joined, channel=channel, name=name)

        attachment_paths: list[str] | None = None
        tmp_path: Path | None = None
        if attachment is not None and attachment.filename:
            suffix = Path(attachment.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(attachment.file, tmp)
                tmp_path = Path(tmp.name)
            attachment_paths = [str(tmp_path)]

        try:
            if not message.strip() and not attachment_paths:
                raise HTTPException(status_code=400, detail="Message or attachment required")
            channel_post(
                channel=channel,
                message=message,
                agent_name=agent_name,
                attachment_paths=attachment_paths,
            )
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        return {"ok": True}

    @app.get("/api/stream")
    async def stream(request: Request, channel: str = "dev"):
        async def generate():
            last_agents_hash = ""
            last_msg_ts = ""

            try:
                while True:
                    if await request.is_disconnected():
                        return

                    # Agent status
                    try:
                        agents = channel_agents_json()
                    except Exception:
                        agents = []
                    agents_hash = json.dumps(agents, sort_keys=True)
                    if agents_hash != last_agents_hash:
                        last_agents_hash = agents_hash
                        html = _render_agents_html(agents)
                        yield {"event": "agents", "data": html}

                    # New messages
                    try:
                        msgs = channel_messages_json(
                            channel=channel,
                            limit=20,
                            since=last_msg_ts or None,
                        )
                    except Exception:
                        msgs = []
                    if msgs:
                        last_msg_ts = msgs[-1].get("timestamp", last_msg_ts)
                        html = _render_messages_html(msgs)
                        yield {"event": "messages", "data": html}

                    await asyncio.sleep(2)
            except asyncio.CancelledError:
                return

        return EventSourceResponse(generate())

    return app


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="synapt dashboard")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--foreground", action="store_true", help="Run server in the terminal")
    group.add_argument("--stop", action="store_true", help="Stop the background dashboard server")
    group.add_argument("--launch", action="store_true", help="Explicitly launch in background mode")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    if args.stop:
        if _stop_dashboard():
            print("synapt dashboard: stopped")
        else:
            print("synapt dashboard: not running")
        return

    if args.foreground:
        print(f"synapt dashboard: {url}")
        if not args.no_open:
            import webbrowser

            webbrowser.open(url)

        import uvicorn

        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
        return

    pid = _start_dashboard_background(
        host=args.host,
        port=args.port,
        no_open=args.no_open,
    )
    print(f"synapt dashboard: {url} (pid {pid})")


if __name__ == "__main__":
    main()
