from __future__ import annotations

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
