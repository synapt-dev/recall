"""With a tier and every lower tier present, both resolvers pick the top one."""

from __future__ import annotations

from pathlib import Path

import pytest

from synapt.recall import channel as ch

TIERS = ("shared", "mapped", "global", "local")


@pytest.fixture
def stack_env(tmp_path, monkeypatch):
    def set_from(top: str) -> dict[str, Path]:
        present = TIERS[TIERS.index(top):]
        dirs = {t: tmp_path / t for t in TIERS}
        for t in present:
            dirs[t].mkdir()
        monkeypatch.setattr(ch, "_guard_store_path", lambda _op, p: p)
        if "shared" in present:
            monkeypatch.setenv("SYNAPT_SHARED_CHANNELS_DIR", str(dirs["shared"]))
        else:
            monkeypatch.delenv("SYNAPT_SHARED_CHANNELS_DIR", raising=False)
        monkeypatch.setattr(
            ch, "_mapped_room_dir",
            lambda _c: dirs["mapped"] if "mapped" in present else None)
        monkeypatch.setattr(
            ch, "_global_channels_dir",
            lambda _p=None: dirs["global"] if "global" in present else None)
        monkeypatch.setattr(ch, "_local_channels_dir", lambda _p=None: dirs["local"])
        return dirs
    return set_from


@pytest.mark.parametrize("top", TIERS[:3])
@pytest.mark.parametrize("resolver", ("_channel_path", "_watch_room_path"))
def test_top_tier_wins(stack_env, top, resolver):
    dirs = stack_env(top)
    assert getattr(ch, resolver)("dev") == dirs[top] / "dev.jsonl"
