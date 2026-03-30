"""synapt dashboard — web-based mission control for multi-agent sessions.

Launch with: synapt dashboard [--port 8420] [--no-open]
"""

from __future__ import annotations

import asyncio
import json
from html import escape
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from sse_starlette.sse import EventSourceResponse

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
            f'<span class="sys-text">-- {escape(name)} {msg_type}ed</span>'
            f'</div>'
        )
    if msg_type == "directive":
        return (
            f'<div class="msg directive">'
            f'<span class="ts">{ts_short}</span> '
            f'<b>{escape(name)}</b> &rarr; @{escape(to)}: {escape(body)}'
            f'</div>'
        )
    return (
        f'<div class="msg">'
        f'<span class="ts">{ts_short}</span> '
        f'<b>{escape(name)}</b>: {escape(body)}'
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

    @app.post("/api/post/{channel}")
    async def api_post(channel: str, message: str = Form(...)):
        channel_post(channel=channel, message=message, agent_name="dashboard")
        return {"ok": True}

    @app.get("/api/stream")
    async def stream(request: Request, channel: str = "dev"):
        async def generate():
            last_agents_hash = ""
            last_msg_ts = ""

            while True:
                if await request.is_disconnected():
                    break

                # Agent status
                agents = channel_agents_json()
                agents_hash = json.dumps(agents, sort_keys=True)
                if agents_hash != last_agents_hash:
                    last_agents_hash = agents_hash
                    html = _render_agents_html(agents)
                    yield {"event": "agents", "data": html}

                # New messages
                msgs = channel_messages_json(
                    channel=channel,
                    limit=20,
                    since=last_msg_ts or None,
                )
                if msgs:
                    last_msg_ts = msgs[-1].get("timestamp", last_msg_ts)
                    html = _render_messages_html(msgs)
                    yield {"event": "messages", "data": html}

                await asyncio.sleep(2)

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
