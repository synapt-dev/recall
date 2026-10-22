"""_watch_room_path must resolve every tier to the same log as _channel_path."""

from __future__ import annotations

from pathlib import Path

import pytest

from synapt.recall import channel as ch

TIERS = ("shared", "mapped", "global", "local")


@pytest.fixture
def tier_env(tmp_path, monkeypatch):
    def set_tier(tier: str) -> Path:
        want = tmp_path / tier
        want.mkdir()
        monkeypatch.setattr(ch, "_guard_store_path", lambda _op, p: p)
        if tier == "shared":
            monkeypatch.setenv("SYNAPT_SHARED_CHANNELS_DIR", str(want))
        else:
            monkeypatch.delenv("SYNAPT_SHARED_CHANNELS_DIR", raising=False)
        monkeypatch.setattr(
            ch, "_mapped_room_dir", lambda _c: want if tier == "mapped" else None)
        monkeypatch.setattr(
            ch, "_global_channels_dir", lambda _p=None: want if tier == "global" else None)
        monkeypatch.setattr(
            ch, "_local_channels_dir", lambda _p=None: want if tier == "local" else tmp_path / "other")
        return want
    return set_tier


@pytest.mark.parametrize("tier", TIERS)
def test_watch_room_path_matches_channel_path(tier_env, tier):
    want = tier_env(tier)
    got = ch._watch_room_path("dev")
    assert got == want / "dev.jsonl"
    assert got == ch._channel_path("dev")
