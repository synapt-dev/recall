"""synapt dashboard — web-based mission control for multi-agent sessions.

Launch with: synapt dashboard [--port 8420] [--no-open]
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
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
    _resolve_org_id,
    _resolve_project_id,
)
from synapt.recall.registry import list_agents as _registry_list_agents

_MD = _md.Markdown(extensions=["fenced_code", "tables", "nl2br"])

# ---------------------------------------------------------------------------
# Agent tool detection (codex vs claude)
# ---------------------------------------------------------------------------

_CODEX_AGENTS: set[str] = set()


def _load_codex_agents() -> None:
    """Load agents.toml and populate _CODEX_AGENTS with names using the codex tool."""
    from synapt.recall.core import _find_gripspace_root

    grip_root = _find_gripspace_root(Path.cwd())
    if grip_root is None:
        return
    toml_path = grip_root / ".gitgrip" / "agents.toml"
    if not toml_path.is_file():
        toml_path = grip_root / "config" / "agents.toml"
    if not toml_path.is_file():
        return
    try:
        with open(toml_path, "rb") as f:
            cfg = tomllib.load(f)
        for name, agent_cfg in cfg.get("agents", {}).items():
            if agent_cfg.get("tool") == "codex":
                _CODEX_AGENTS.add(name)
    except (OSError, tomllib.TOMLDecodeError):
        pass


_load_codex_agents()

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


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, OSError):
        return False


def _combined_agents_json() -> list[dict]:
    """Return agents from both team.db (process tracking) and channel presence.

    Fixes recall#552: spawned agents weren't visible because presence is
    per-gripspace (local channels.db) but the dashboard may be in a
    different gripspace. team.db is global and updated by gr spawn.

    Merge strategy: team.db agents are the authority for process status.
    Channel presence supplements with channel memberships and heartbeat.
    """
    # Start with channel presence (existing behavior)
    agents_by_name: dict[str, dict] = {}
    try:
        for agent in channel_agents_json():
            name = agent.get("display_name") or agent.get("agent_id", "")
            agents_by_name[name] = agent
    except Exception:
        pass

    # Overlay with team.db agents (global process tracking)
    try:
        for org_dir in (Path.home() / ".synapt" / "orgs").iterdir():
            if not org_dir.is_dir():
                continue
            db_path = org_dir / "team.db"
            if not db_path.exists():
                continue
            org_id = org_dir.name
            try:
                registered = _registry_list_agents(org_id, db_path=db_path)
            except Exception:
                continue
            for agent in registered:
                name = agent.get("display_name", "")
                status = agent.get("status") or "offline"
                # Verify PID is alive — team.db status can be stale if the
                # process exited without updating the DB (crash, kill, etc.)
                pid = agent.get("pid")
                if status in ("running", "online") and pid:
                    if not _is_pid_alive(pid):
                        status = "offline"
                if status in ("offline", "stopped") and name not in agents_by_name:
                    continue  # Don't show offline agents that aren't in presence
                if name not in agents_by_name:
                    # Agent visible in team.db but not in local presence
                    agents_by_name[name] = {
                        "agent_id": agent.get("agent_id", ""),
                        "display_name": name,
                        "griptree": "",
                        "role": agent.get("role", "agent"),
                        "status": status,
                        "last_seen": agent.get("last_seen_at", ""),
                        "channels": [],
                    }
                else:
                    # Merge: use team.db status if agent is running/online
                    existing = agents_by_name[name]
                    if status in ("running", "online"):
                        existing["status"] = status
    except FileNotFoundError:
        pass  # No orgs directory

    return sorted(agents_by_name.values(), key=lambda a: a.get("display_name", ""))


def _render_agent_tile(agent: dict) -> str:
    status = agent["status"]
    color = _STATUS_COLORS.get(status, "#6b7280")
    name = agent["display_name"] or agent["griptree"] or agent["agent_id"]
    role = agent["role"] if agent["role"] != "agent" else ""
    griptree = agent.get("griptree", "")
    channels = ", ".join(f"#{c}" for c in agent["channels"]) or "no channels"
    seen = agent["last_seen"][11:16] if len(agent["last_seen"]) > 16 else ""
    project_badge = (
        f'<div class="tile-project">{escape(griptree)}</div>' if griptree else ""
    )
    return (
        f'<div class="tile clickable" data-agent="{escape(name)}" style="border-left:4px solid {color}">'
        f'<div class="tile-name">{escape(name)}</div>'
        f'{project_badge}'
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
            f'<span class="ts" data-utc="{escape(ts)}">{ts_short}</span> '
            f'<span class="sys-text">-- {escape(name)} {"joined" if msg_type == "join" else "left"}</span>'
            f'</div>'
        )
    color = _agent_color(name)
    _MD.reset()
    # Escape leading '#' not followed by space — prevents markdown
    # from turning "#celebrate" into an <h1> heading. (recall#630)
    body_escaped = re.sub(r'^(#{1,6})(?=[^ #])', r'\\\1', body, flags=re.MULTILINE)
    body_html = _MD.convert(body_escaped)
    # Color @mentions — skip content inside <code> and <pre> tags
    def _color_mentions(html: str) -> str:
        parts = re.split(r'(<code.*?>.*?</code>|<pre.*?>.*?</pre>)', html, flags=re.DOTALL)
        for i, part in enumerate(parts):
            if not part.startswith(('<code', '<pre')):
                parts[i] = re.sub(
                    r'@(\w+)',
                    lambda m: f'<span style="color:{_agent_color(m.group(1))};font-weight:600">@{m.group(1)}</span>',
                    part,
                )
        return ''.join(parts)
    body_html = _color_mentions(body_html)
    if msg_type == "directive":
        return (
            f'<div class="msg directive">'
            f'<span class="ts" data-utc="{escape(ts)}">{ts_short}</span> '
            f'<b style="color:{color}">{escape(name)}</b> &rarr; @{escape(to)}: '
            f'<span class="msg-body">{body_html}</span>'
            f'{attachments_html}'
            f'</div>'
        )
    return (
        f'<div class="msg">'
        f'<span class="ts" data-utc="{escape(ts)}">{ts_short}</span> '
        f'<b style="color:{color}">{escape(name)}</b>: '
        f'<span class="msg-body">{body_html}</span>'
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


def _all_channels_nav() -> dict:
    """Return hierarchical org → project → channel navigation data.

    Scans ``~/.synapt/channels/<org>/<project>/`` globally so the dashboard
    can show all visible channels, not just the current project's.
    Falls back to local channel list for non-gripspace repos.
    """
    global_dir = Path.home() / ".synapt" / "channels"
    current_org = _resolve_org_id(None) or ""
    current_project = _resolve_project_id(None) or ""

    projects = []

    if global_dir.exists():
        for org_dir in sorted(global_dir.iterdir()):
            if not org_dir.is_dir() or org_dir.name.startswith("_"):
                continue
            for proj_dir in sorted(org_dir.iterdir()):
                if not proj_dir.is_dir():
                    continue
                channels = sorted(p.stem for p in proj_dir.glob("*.jsonl"))
                if not channels:
                    continue
                is_active = (
                    org_dir.name == current_org and proj_dir.name == current_project
                )
                projects.append(
                    {
                        "org": org_dir.name,
                        "project": proj_dir.name,
                        "channels": channels,
                        "active": is_active,
                    }
                )

    # Fallback: no global store found — use local channels
    if not projects:
        local_channels = channel_list_channels()
        if local_channels:
            projects.append(
                {
                    "org": current_org or "local",
                    "project": current_project or "local",
                    "channels": local_channels,
                    "active": True,
                }
            )

    return {
        "org": current_org,
        "project": current_project,
        "projects": projects,
    }


def _channels_dir_for(org: str | None, project: str | None) -> "Path | None":
    """Return the channels directory for a specific org/project, or None for current."""
    if not org or not project:
        return None
    return Path.home() / ".synapt" / "channels" / org / project


def create_app() -> FastAPI:
    app = FastAPI(title="synapt dashboard", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _load_template()

    @app.get("/api/agents")
    async def api_agents():
        return _combined_agents_json()

    @app.get("/api/org")
    async def api_org():
        org = _resolve_org_id(None) or "unknown"
        project = _resolve_project_id(None) or "unknown"
        return {"org": org, "project": project}

    @app.get("/api/channels")
    async def api_channels():
        return channel_list_channels()

    @app.get("/api/nav")
    async def api_nav():
        return _all_channels_nav()

    @app.get("/api/messages/{channel}", response_class=HTMLResponse)
    async def api_messages(
        channel: str,
        limit: int = 50,
        since: str | None = None,
        org: str | None = None,
        project: str | None = None,
    ):
        ch_dir = _channels_dir_for(org, project)
        msgs = channel_messages_json(
            channel=channel, limit=limit, since=since, channels_dir=ch_dir
        )
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
        org: str = Form(""),
        project: str = Form(""),
        attachment: UploadFile | None = File(None),
    ):
        agent_name = "dashboard"
        ch_dir = _channels_dir_for(org or None, project or None)
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
                channels_dir=ch_dir,
            )
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        return {"ok": True}

    @app.get("/api/stream")
    async def stream(
        request: Request,
        channel: str = "dev",
        org: str | None = None,
        project: str | None = None,
    ):
        ch_dir = _channels_dir_for(org, project)

        async def generate():
            last_msg_ts = ""

            try:
                while True:
                    if await request.is_disconnected():
                        return

                    # New messages
                    try:
                        msgs = channel_messages_json(
                            channel=channel,
                            limit=20,
                            since=last_msg_ts or None,
                            channels_dir=ch_dir,
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

    # -----------------------------------------------------------------
    # Mission Control: per-agent tmux integration (Sprint 9)
    # -----------------------------------------------------------------

    @app.post("/api/agent/{name}/input")
    async def api_agent_input(name: str, text: str = Form("")):
        """Send input to an agent's tmux pane via send-keys.

        Codex agents require a second Enter to confirm the prompt (two-step
        input protocol).  The agent tool type is detected from agents.toml at
        startup; codex agents get an extra Enter after a short delay.
        """
        if not text.strip():
            raise HTTPException(status_code=400, detail="Input text required")
        # Resolve tmux target from team.db or convention
        target = f"{name}"  # Will be refined with session:window format
        is_codex = name in _CODEX_AGENTS
        try:
            result = subprocess.run(
                ["tmux", "send-keys", "-t", target, text, "Enter"],
                capture_output=True,
                timeout=5,
            )
            if result.returncode != 0:
                raise HTTPException(
                    status_code=502,
                    detail=f"tmux send-keys failed: {result.stderr.decode().strip()}",
                )
            # Codex needs a second Enter to confirm the prompt
            if is_codex:
                await asyncio.sleep(0.3)
                subprocess.run(
                    ["tmux", "send-keys", "-t", target, "Enter"],
                    capture_output=True,
                    timeout=5,
                )
        except FileNotFoundError:
            raise HTTPException(status_code=503, detail="tmux not available")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="tmux send-keys timed out")
        return {"ok": True, "agent": name, "codex": is_codex}

    @app.get("/api/agent/{name}/output")
    async def api_agent_output(request: Request, name: str, lines: int = 50):
        """Stream agent output from pipe-pane log file via SSE."""
        # Resolve log path from team.db or convention
        log_dir = project_data_dir() / ".." / "logs" / name
        log_path = log_dir / "output.log"

        async def tail_log():
            last_pos = 0
            try:
                while True:
                    if await request.is_disconnected():
                        return
                    if log_path.exists():
                        with open(log_path, "r") as f:
                            f.seek(last_pos)
                            new_content = f.read()
                            if new_content:
                                last_pos = f.tell()
                                yield {
                                    "event": "output",
                                    "data": escape(new_content),
                                }
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                return

        return EventSourceResponse(tail_log())

    @app.get("/api/agent/{name}/snapshot")
    async def api_agent_snapshot(name: str, lines: int = 50):
        """One-shot capture of agent's tmux pane content."""
        target = f"{name}"
        try:
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", target, "-p", "-S", f"-{lines}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                return {"agent": name, "content": "", "error": "pane not found"}
            return {"agent": name, "content": result.stdout}
        except FileNotFoundError:
            return {"agent": name, "content": "", "error": "tmux not available"}

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
