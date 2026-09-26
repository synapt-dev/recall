from __future__ import annotations

import subprocess

from synapt.recall import comms


def test_guessed_time_refuses_before_recipient_resolution(monkeypatch):
    monkeypatch.setattr(comms, "_record_receipt", lambda receipt, **_: receipt)
    monkeypatch.setattr(comms.direct, "resolve_registered_recipient", lambda _: (_ for _ in ()).throw(AssertionError()))
    receipt = comms.send("atlas", "COMMS PROOF TEST, ignore\nclock 09:xx", from_agent="fathom-001")
    assert receipt.message_id is None
    assert receipt.state == "refused"


def test_missing_pane_is_undeliverable(monkeypatch):
    monkeypatch.setattr(comms, "_record_receipt", lambda receipt, **_: receipt)
    recipient = comms.direct.RegisteredRecipient("atlas-001", "synapt", "atlas")
    monkeypatch.setattr(comms.direct, "resolve_registered_recipient", lambda _: recipient)
    monkeypatch.setattr(comms.direct, "load_pane_map", lambda: {})
    receipt = comms.send("atlas", "COMMS PROOF TEST, ignore\nnonce", from_agent="fathom-001")
    assert receipt.message_id is None
    assert receipt.state == "undeliverable"


def test_declared_pane_produces_submitted_receipt(monkeypatch):
    monkeypatch.setattr(comms, "_record_receipt", lambda receipt, **_: receipt)
    recipient = comms.direct.RegisteredRecipient("atlas-001", "synapt", "atlas")
    message = comms.direct.DirectMessage("dm_test", "fathom-001", "atlas-001", "now", "line one\nunique needle")
    pane = comms.direct.PaneTarget("synapt:atlas", "claude")
    delivery = comms.direct.TmuxDelivery(True, "synapt:atlas", 2, "pasted")
    monkeypatch.setattr(comms.direct, "resolve_registered_recipient", lambda _: recipient)
    monkeypatch.setattr(comms.direct, "load_pane_map", lambda: {"atlas": {"target": "synapt:atlas", "runtime": "claude"}})
    monkeypatch.setattr(comms.direct, "send_message", lambda **_: message)
    monkeypatch.setattr(comms.direct, "deliver_via_tmux", lambda *_: delivery)
    monkeypatch.setattr(comms, "_verify_submitted", lambda *_: True)
    monkeypatch.setattr(comms.time, "sleep", lambda _: None)
    receipt = comms.send("atlas", message.body, from_agent="fathom-001")
    assert receipt.message_id == "dm_test"
    assert receipt.state == "submitted"


def test_empty_needle_cannot_verify_a_capture(monkeypatch):
    """An empty string matches every capture unless the verifier refuses it first."""
    monkeypatch.setattr(
        comms.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("capture must not run")),
    )
    assert comms._verify_submitted("synapt:atlas", "") is False


def test_trailing_blank_body_uses_last_non_blank_line(monkeypatch):
    monkeypatch.setattr(comms, "_record_receipt", lambda receipt, **_: receipt)
    recipient = comms.direct.RegisteredRecipient("atlas-001", "synapt", "atlas")
    message = comms.direct.DirectMessage("dm_test", "fathom-001", "atlas-001", "now", "line one\nnonce\n")
    pane = comms.direct.PaneTarget("synapt:atlas", "claude")
    delivery = comms.direct.TmuxDelivery(True, "synapt:atlas", 2, "pasted")
    monkeypatch.setattr(comms.direct, "resolve_registered_recipient", lambda _: recipient)
    monkeypatch.setattr(comms.direct, "load_pane_map", lambda: {"atlas": {"target": "synapt:atlas", "runtime": "claude"}})
    monkeypatch.setattr(comms.direct, "send_message", lambda **_: message)
    monkeypatch.setattr(comms.direct, "deliver_via_tmux", lambda *_: delivery)
    observed_needles: list[str] = []
    monkeypatch.setattr(comms, "_verify_submitted", lambda _target, needle: observed_needles.append(needle) or False)
    monkeypatch.setattr(comms.time, "sleep", lambda _: None)
    receipt = comms.send("atlas", message.body, from_agent="fathom-001")
    assert observed_needles == ["nonce"]
    assert receipt.state == "unknown"


def test_empty_body_returns_refused_receipt(monkeypatch):
    monkeypatch.setattr(comms, "_record_receipt", lambda receipt, **_: receipt)
    recipient = comms.direct.RegisteredRecipient("atlas-001", "synapt", "atlas")
    message = comms.direct.DirectMessage("dm_test", "fathom-001", "atlas-001", "now", "")
    pane = comms.direct.PaneTarget("synapt:atlas", "claude")
    delivery = comms.direct.TmuxDelivery(True, "synapt:atlas", 2, "pasted")
    monkeypatch.setattr(comms.direct, "resolve_registered_recipient", lambda _: recipient)
    monkeypatch.setattr(comms.direct, "load_pane_map", lambda: {"atlas": {"target": "synapt:atlas", "runtime": "claude"}})
    monkeypatch.setattr(comms.direct, "send_message", lambda **_: message)
    monkeypatch.setattr(comms.direct, "deliver_via_tmux", lambda *_: delivery)
    receipt = comms.send("atlas", "", from_agent="fathom-001")
    assert receipt == comms.Receipt("dm_test", "refused", "empty body: no non-blank needle to witness")


def test_capture_timeout_returns_unknown_receipt(monkeypatch):
    monkeypatch.setattr(comms, "_record_receipt", lambda receipt, **_: receipt)
    recipient = comms.direct.RegisteredRecipient("atlas-001", "synapt", "atlas")
    message = comms.direct.DirectMessage("dm_test", "fathom-001", "atlas-001", "now", "nonce")
    pane = comms.direct.PaneTarget("synapt:atlas", "claude")
    delivery = comms.direct.TmuxDelivery(True, "synapt:atlas", 2, "pasted")
    monkeypatch.setattr(comms.direct, "resolve_registered_recipient", lambda _: recipient)
    monkeypatch.setattr(comms.direct, "load_pane_map", lambda: {"atlas": {"target": "synapt:atlas", "runtime": "claude"}})
    monkeypatch.setattr(comms.direct, "send_message", lambda **_: message)
    monkeypatch.setattr(comms.direct, "deliver_via_tmux", lambda *_: delivery)
    monkeypatch.setattr(comms.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired("tmux", 10)))
    monkeypatch.setattr(comms.time, "sleep", lambda _: None)
    assert comms.send("atlas", message.body, from_agent="fathom-001").state == "unknown"


def test_missing_declared_tmux_pane_is_undeliverable(monkeypatch):
    recipient = comms.direct.RegisteredRecipient("atlas-001", "synapt", "atlas")
    message = comms.direct.DirectMessage("dm_test", "fathom-001", "atlas-001", "now", "COMMS PROOF TEST, ignore\nnonce")
    pane = comms.direct.PaneTarget("synapt:missing", "claude")
    delivery = comms.direct.TmuxDelivery(False, "synapt:missing", 2, "paste-buffer failed for synapt:missing: can't find window")
    monkeypatch.setattr(comms, "_record_receipt", lambda receipt, **_: receipt)
    monkeypatch.setattr(comms.direct, "resolve_registered_recipient", lambda _: recipient)
    monkeypatch.setattr(comms.direct, "load_pane_map", lambda: {"atlas": {"target": "synapt:missing", "runtime": "claude"}})
    monkeypatch.setattr(comms.direct, "send_message", lambda **_: message)
    monkeypatch.setattr(comms.direct, "deliver_via_tmux", lambda *_: delivery)
    assert comms.send("atlas", message.body, from_agent="fathom-001").state == "undeliverable"


def test_receipt_ledger_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNAPT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SYNAPT_SHARED_CHANNELS_DIR", str(tmp_path / "channels"))
    receipt = comms._record_receipt(comms.Receipt(None, "refused", "guessed time label: 09:xx"), to="atlas")
    assert comms.ledger("atlas") == [receipt]
