"""A watched room name is held to the same containment as post and read."""

import pytest

from synapt.recall import channel as ch


@pytest.mark.parametrize("name", ["../escape", "sub/room", "nul\0room", "back\\slash"])
def test_watch_room_path_refuses_what_post_refuses(name, tmp_path, monkeypatch):
    monkeypatch.setattr(ch, "_shared_channels_dir", lambda: tmp_path)
    monkeypatch.setattr(ch, "_guard_store_path", lambda _op, p: p)
    with pytest.raises(ch.ChannelNameError):
        ch._contained_channel_file(tmp_path, ch._channel_to_filename(name))
    with pytest.raises(ch.ChannelNameError):
        ch._watch_room_path(name)
