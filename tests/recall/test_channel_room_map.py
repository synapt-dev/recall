"""A room shared across gripspaces lives in one store.

``#head`` is one room for several teams, but each caller resolves its own
store from its own manifest, so a post from a second gripspace landed in that
gripspace's copy of ``head.jsonl`` and the shared room never saw it. The
shared-room map names the store a room lives in; only the room's log moves,
and a mapping that cannot be honoured is refused by name rather than falling
back to the caller's store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from synapt.recall import channel as channel_mod
from synapt.recall.channel import (
    SharedRoomMapError,
    _db_path,
    channel_list_channels,
    channel_post,
    channel_search,
)
from synapt.recall.core import _gripspace_cache


SHARED = ("synapt-dev", "synapt-gripspace")
OTHER = ("synapt-dev", "other-gripspace")


def _gripspace(root: Path, repo: str) -> Path:
    grip = root / repo
    spaces_main = grip / ".gitgrip" / "spaces" / "main"
    spaces_main.mkdir(parents=True)
    (grip / ".gitgrip" / "griptrees.json").write_text('{"griptrees": {}}')
    (spaces_main / "gripspace.yml").write_text(
        f"version: 2\nmanifest:\n  url: git@github.com:synapt-dev/{repo}.git\nrepos: {{}}\n"
    )
    return grip


def _store(home: Path, org_project: tuple[str, str]) -> Path:
    return home / ".synapt" / "channels" / org_project[0] / org_project[1]


def _ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def world(tmp_path, monkeypatch, owned_recall_root):
    """A home with the shared store, and a caller inside the second gripspace.

    Recall data (the local state db) lives in the test's owned root; the room
    stores live under the test's home, where the manifest resolves them.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    # The suite points the map at nothing; these rows use the default location.
    monkeypatch.delenv("SYNAPT_ROOM_MAP", raising=False)
    monkeypatch.setenv("SYNAPT_AGENT_ID", "room-map-witness")
    _store(home, SHARED).mkdir(parents=True)
    grip = _gripspace(tmp_path, OTHER[1])
    monkeypatch.chdir(grip)
    _gripspace_cache.clear()
    yield {"home": home, "grip": grip, "data": owned_recall_root}
    _gripspace_cache.clear()


def _write_map(home: Path, body: str) -> Path:
    path = home / ".synapt" / "channels" / "rooms.toml"
    path.write_text(body)
    return path


HEAD_MAP = '[rooms]\nhead = "synapt-dev/synapt-gripspace"\n'


def test_a_mapped_room_lands_in_its_mapped_store(world):
    home = world["home"]
    _write_map(home, HEAD_MAP)

    channel_post("head", "from the second gripspace")

    shared = _ids(_store(home, SHARED) / "head.jsonl")
    own = _ids(channel_mod._channels_dir() / "head.jsonl")
    assert len(shared) == 1, f"shared head.jsonl holds {shared}"
    assert own == [], f"the caller's own head.jsonl was written: {own}"


def test_the_hash_spelling_of_a_mapped_room_is_mapped_too(world):
    home = world["home"]
    _write_map(home, HEAD_MAP)

    channel_post("#head", "written the way prose names it")

    assert len(_ids(_store(home, SHARED) / "head.jsonl")) == 1


def test_an_unmapped_room_stays_in_the_callers_store(world):
    home = world["home"]
    _write_map(home, HEAD_MAP)

    channel_post("dev", "team room")

    assert len(_ids(channel_mod._channels_dir() / "dev.jsonl")) == 1
    assert not (_store(home, SHARED) / "dev.jsonl").exists()


def test_the_callers_state_db_stays_local(world):
    home = world["home"]
    _write_map(home, HEAD_MAP)

    channel_post("head", "state stays home")

    state_db = _db_path()
    assert state_db.is_relative_to(world["data"]), state_db
    assert not (_store(home, SHARED) / "channels.db").exists()


def test_a_mapped_room_whose_store_is_missing_is_refused_by_name(world):
    home = world["home"]
    _write_map(home, '[rooms]\nhead = "synapt-dev/no-such-gripspace"\n')

    with pytest.raises(SharedRoomMapError, match="room 'head'.*no-such-gripspace"):
        channel_post("head", "must not fall back")

    assert not (channel_mod._channels_dir() / "head.jsonl").exists()


@pytest.mark.parametrize("target", ["synapt-gripspace", "synapt-dev/../x", "a/b/c", "/abs", 7])
def test_a_malformed_mapping_is_refused_by_name(world, target):
    home = world["home"]
    _write_map(home, f"[rooms]\nhead = {json.dumps(target)}\n")

    with pytest.raises(SharedRoomMapError, match="room 'head'"):
        channel_post("head", "must not fall back")

    assert not (channel_mod._channels_dir() / "head.jsonl").exists()


def test_an_unreadable_map_is_refused(world):
    _write_map(world["home"], "[rooms\nhead = ")

    with pytest.raises(SharedRoomMapError, match="rooms.toml"):
        channel_post("head", "must not fall back")


def test_without_a_map_every_room_stays_in_the_callers_store(world):
    home = world["home"]

    channel_post("head", "no map")

    assert len(_ids(channel_mod._channels_dir() / "head.jsonl")) == 1
    assert not (_store(home, SHARED) / "head.jsonl").exists()


def test_listing_and_search_reach_a_mapped_room(world):
    home = world["home"]
    _write_map(home, HEAD_MAP)
    channel_post("head", "findable needle in the shared room")
    channel_post("dev", "team room")

    assert {"head", "dev"} <= set(channel_list_channels())
    hits = channel_search("needle")
    assert [h["channel"] for h in hits] == ["head"], hits


def test_a_stale_local_copy_of_a_mapped_room_is_read_once(world):
    home = world["home"]
    stale = channel_mod._channels_dir() / "head.jsonl"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text(json.dumps({
        "timestamp": "2026-10-15T11:29:00Z", "id": "m_stale", "from": "x",
        "channel": "head", "type": "message", "body": "old needle",
    }) + "\n")
    _write_map(home, HEAD_MAP)
    channel_post("head", "new needle")

    hits = channel_search("needle")
    assert len(hits) == 1, hits
    assert "new needle" in hits[0]["body"]


def test_the_shared_channels_override_still_pins_every_room(world, tmp_path, monkeypatch):
    home = world["home"]
    _write_map(home, HEAD_MAP)
    pinned = tmp_path / "pinned"
    pinned.mkdir()
    monkeypatch.setenv("SYNAPT_SHARED_CHANNELS_DIR", str(pinned))

    channel_post("head", "pinned")

    assert len(_ids(pinned / "head.jsonl")) == 1
    assert not (_store(home, SHARED) / "head.jsonl").exists()


def test_the_store_line_names_the_mapped_store(world):
    from synapt.recall import server
    from synapt.recall.actions import reset_action_registry

    home = world["home"]
    map_path = _write_map(home, HEAD_MAP)
    reset_action_registry()
    server._last_store_preamble = None
    try:
        result = server.recall_channel(action="post", channel="head", message="line check")
    finally:
        reset_action_registry()
        server._last_store_preamble = None

    shared = _store(home, SHARED).resolve()
    assert f"Channel log store: {shared} (source: room map {map_path})" in result, result
    assert len(_ids(shared / "head.jsonl")) == 1


def test_the_suite_points_the_map_at_nothing(monkeypatch):
    """The default map lives under the real home; the harness must not read it."""
    import os

    assert os.environ.get("SYNAPT_ROOM_MAP") == os.devnull
    assert channel_mod._load_room_map() == {}


def test_the_resolver_maps_the_hash_spelling_itself(world):
    """Callers that skip the post path's normalisation still reach the mapped store."""
    home = world["home"]
    _write_map(home, HEAD_MAP)

    assert channel_mod._room_channels_dir("#head") == _store(home, SHARED)


def test_a_map_key_written_with_a_hash_maps_the_room(world):
    home = world["home"]
    _write_map(home, '[rooms]\n"#head" = "synapt-dev/synapt-gripspace"\n')

    channel_post("head", "keyed the way prose names it")

    assert len(_ids(_store(home, SHARED) / "head.jsonl")) == 1
    assert not (channel_mod._channels_dir() / "head.jsonl").exists()


def test_a_room_mapped_twice_to_different_stores_is_refused_by_name(world):
    home = world["home"]
    _write_map(
        home,
        '[rooms]\nhead = "synapt-dev/synapt-gripspace"\n"#head" = "synapt-dev/other"\n',
    )

    with pytest.raises(SharedRoomMapError, match="room 'head' is mapped twice"):
        channel_post("head", "ambiguous")

    assert not (channel_mod._channels_dir() / "head.jsonl").exists()
