"""synapt dashboard — web-based mission control for multi-agent sessions.

Launch with: synapt dashboard [--port 8420] [--no-open]
"""

from __future__ import annotations

import asyncio
import json
import re
from html import escape
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

try:
    import markdown as _md

    def _render_markdown(text: str) -> str:
        """Convert markdown to HTML with fenced code blocks and tables."""
        html = _md.markdown(
            text,
            extensions=["fenced_code", "tables", "nl2br"],
        )
        # Strip wrapping <p> for single-line messages
        if html.startswith("<p>") and html.count("<p>") == 1:
            html = html[3:]
            if html.endswith("</p>"):
                html = html[:-4]
        return html

except ImportError:

    def _render_markdown(text: str) -> str:
        """Fallback: escape HTML and convert newlines to <br>."""
        return escape(text).replace("\n", "<br>")

from synapt.recall.channel import (
    ChannelMessage,
    channel_agents_json,
    channel_list_channels,
    channel_messages_json,
    channel_post,
)

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


def _render_message(msg: dict) -> str:
    ts = msg.get("timestamp", "")
    ts_short = ts[11:16] if len(ts) > 16 else ts
    name = msg.get("from_display") or msg.get("from", "")
    body = msg.get("body", "")
    msg_type = msg.get("type", "message")
    to = msg.get("to", "")

    if msg_type in ("join", "leave"):
        return (
            f'<div class="msg sys">'
            f'<span class="ts">{ts_short}</span> '
            f'<span class="sys-text">-- {escape(name)} {"joined" if msg_type == "join" else "left"}</span>'
            f'</div>'
        )
    color = _agent_color(name)
    body_html = _render_markdown(body)
    if msg_type == "directive":
        return (
            f'<div class="msg directive">'
            f'<span class="ts">{ts_short}</span> '
            f'<b style="color:{color}">{escape(name)}</b> &rarr; @{escape(to)}: {body_html}'
            f'</div>'
        )
    return (
        f'<div class="msg">'
        f'<span class="ts">{ts_short}</span> '
        f'<b style="color:{color}">{escape(name)}</b>: {body_html}'
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

    _dashboard_joined: set[tuple[str, str]] = set()

    @app.post("/api/post/{channel}")
    async def api_post(channel: str, message: str = Form(...), name: str = Form("dashboard")):
        from synapt.recall.channel import channel_join
        agent_name = "dashboard"
        join_key = (channel, name)
        if join_key not in _dashboard_joined:
            channel_join(channel=channel, agent_name=agent_name, display_name=name, role="human")
            _dashboard_joined.add(join_key)
        channel_post(channel=channel, message=message, agent_name=agent_name)
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
    import webbrowser

    parser = argparse.ArgumentParser(description="synapt dashboard")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open browser")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"
    print(f"synapt dashboard: {url}")

    if not args.no_open:
        webbrowser.open(url)

    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
