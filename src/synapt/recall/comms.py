"""The first thin ``comms send`` path.

This is deliberately one transport: a declared tmux pane.  ``direct`` remains
the durable inbox and message ledger; this module adds the receipt that says
what the pane transport could actually observe.
"""

from __future__ import annotations

import re
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from synapt.recall import direct

_GUESSED_TIME = re.compile(r"(?<!\d)(?:[01]?\d|2[0-3]):(?:[0-5]x|xx)(?![\w])")


@dataclass(frozen=True)
class Receipt:
    message_id: str | None
    state: str
    detail: str


def _record_receipt(receipt: Receipt, *, to: str) -> Receipt:
    """Persist the transport measurement beside direct's durable envelope."""
    conn = direct._get_db()
    try:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS comms_receipts (
                receipt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT,
                to_agent TEXT NOT NULL,
                state TEXT NOT NULL,
                detail TEXT NOT NULL,
                created_at REAL NOT NULL
            )"""
        )
        conn.execute(
            "INSERT INTO comms_receipts(message_id, to_agent, state, detail, created_at) VALUES (?, ?, ?, ?, ?)",
            (receipt.message_id, to, receipt.state, receipt.detail, time.time()),
        )
        conn.commit()
    finally:
        conn.close()
    return receipt


def ledger(to: str | None = None) -> list[Receipt]:
    conn = direct._get_db()
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS comms_receipts (
            receipt_id INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT,
            to_agent TEXT NOT NULL, state TEXT NOT NULL, detail TEXT NOT NULL,
            created_at REAL NOT NULL)""")
        sql = "SELECT message_id, state, detail FROM comms_receipts"
        values: tuple[str, ...] = ()
        if to:
            sql += " WHERE to_agent = ?"
            values = (to,)
        rows = conn.execute(sql + " ORDER BY receipt_id", values).fetchall()
        return [Receipt(row["message_id"], row["state"], row["detail"]) for row in rows]
    finally:
        conn.close()


def _verify_submitted(target: str, needle: str) -> bool:
    """Return only the positive fact this proof needs: the sent needle rendered.

    A negative capture is intentionally ``unknown``.  The hardening path will
    port the verifier's bounded-region and capture-adequacy distinctions.
    """
    result = subprocess.run(
        ["tmux", "capture-pane", "-J", "-p", "-S", "-", "-t", target],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.returncode == 0 and needle in result.stdout


def send(to: str, body: str, *, from_agent: str) -> Receipt:
    """Write durable-first, paste once into a declared pane, then measure it."""
    guessed = _GUESSED_TIME.search(body)
    if guessed:
        return _record_receipt(Receipt(None, "refused", f"guessed time label: {guessed.group(0)}"), to=to)

    recipient = direct.resolve_registered_recipient(to)
    pane = direct.resolve_pane(recipient.agent_id, direct.load_pane_map())
    if pane is None:
        return _record_receipt(Receipt(None, "undeliverable", f"no declared pane for {recipient.agent_id}"), to=recipient.agent_id)

    message = direct.send_message(
        from_agent=from_agent,
        to_agent=recipient.agent_id,
        body=body,
        recipient_store_coordinate=recipient.store_coordinate,
    )
    delivery = direct.deliver_via_tmux(pane.target, pane.runtime, body)
    if not delivery.delivered:
        state = "undeliverable" if "can't find" in delivery.detail else "unknown"
        return _record_receipt(Receipt(message.message_id, state, delivery.detail), to=recipient.agent_id)
    time.sleep(1)
    if _verify_submitted(pane.target, body.splitlines()[-1]):
        return _record_receipt(Receipt(message.message_id, "submitted", f"{pane.target}; {delivery.detail}"), to=recipient.agent_id)
    return _record_receipt(Receipt(message.message_id, "unknown", f"{pane.target}; paste ran but rendered receipt was absent"), to=recipient.agent_id)


def read_body(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")
