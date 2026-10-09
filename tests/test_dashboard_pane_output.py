from synapt.dashboard.pane_output import PaneOutputReader


def test_rotation_drains_unread_old_tail_once(tmp_path):
    path = tmp_path / "output.log"
    reader = PaneOutputReader()
    try:
        path.write_bytes(b"AAAA")
        assert reader.read_new(path) == "AAAA"
        with path.open("ab") as stream:
            stream.write(b"-LAST-WORDS-OF-THE-OLD-FILE")
        path.replace(tmp_path / "output.log.1")
        path.write_bytes(b"NEW")
        assert reader.read_new(path) == "-LAST-WORDS-OF-THE-OLD-FILE" + "NEW"
        assert reader.read_new(path) == ""
    finally:
        reader.close()


def test_unreadable_log_retries_without_losing_position(tmp_path, monkeypatch):
    from pathlib import Path

    path = tmp_path / "output.log"
    path.write_bytes(b"first")
    reader = PaneOutputReader()
    try:
        assert reader.read_new(path) == "first"
        with path.open("ab") as stream:
            stream.write(b" next")
        with monkeypatch.context() as patch:
            def refused(*args, **kwargs):
                raise PermissionError("unreadable fixture")
            patch.setattr(Path, "open", refused)
            assert reader.read_new(path) == ""
        assert reader.read_new(path) == " next"
    finally:
        reader.close()


def test_first_connect_limits_legacy_log_to_latest_mib(tmp_path):
    path = tmp_path / "output.log"
    path.write_bytes(b"old" + b"x" * (1024 * 1024 - 6) + b"LATEST")
    reader = PaneOutputReader()
    try:
        content = reader.read_new(path)
        assert len(content) == 1024 * 1024
        assert content.endswith("LATEST")
        assert not content.startswith("old")
    finally:
        reader.close()


def test_actual_sse_route_keeps_delivering_across_rotation(tmp_path, monkeypatch):
    import asyncio
    from synapt.dashboard import app as dashboard

    data = tmp_path / ".synapt" / "recall"
    data.mkdir(parents=True)
    path = tmp_path / ".synapt" / "logs" / "fixture" / "output.log"
    path.parent.mkdir(parents=True)
    monkeypatch.setattr(dashboard, "project_data_dir", lambda: data)
    application = dashboard.create_app()
    endpoint = next(route.endpoint for route in application.routes
                    if getattr(route, "path", None) == "/api/agent/{name}/output")

    class Request:
        step = 0

        async def is_disconnected(self):
            self.step += 1
            if self.step == 1:
                path.write_bytes(b"first\xe2")
            elif self.step == 2:
                path.replace(path.with_name("output.log.1"))
                path.write_bytes(b"\x82\xac current after cap")
            elif self.step == 3:
                path.replace(path.with_name("output.log.1"))
                path.write_bytes(b"latest live sentinel longer than prior position")
            else:
                return True
            return False

    async def collect():
        response = await endpoint(Request(), "fixture")
        return [event async for event in response.body_iterator]

    events = asyncio.run(collect())
    assert events[0] == {"event": "output", "data": "first"}
    assert "current after cap" in events[1]["data"]
    assert events[2] == {"event": "output", "data": "latest live sentinel longer than prior position"}


def test_append_and_missing_rotation_gap(tmp_path):
    path = tmp_path / "output.log"
    reader = PaneOutputReader()
    try:
        assert reader.read_new(path) == ""
        path.write_bytes(b"first")
        assert reader.read_new(path) == "first"
        assert reader.read_new(path) == ""
        with path.open("ab") as stream:
            stream.write(b" next")
        assert reader.read_new(path) == " next"
        path.rename(tmp_path / "output.log.1")
        assert reader.read_new(path) == ""
        path.write_bytes(b"new")
        assert reader.read_new(path) == "new"
    finally:
        reader.close()


def test_replacement_larger_than_old_offset_and_repeated_crossings(tmp_path):
    path = tmp_path / "output.log"
    reader = PaneOutputReader()
    try:
        path.write_bytes(b"old")
        assert reader.read_new(path) == "old"
        for index in range(4):
            path.replace(tmp_path / "output.log.1")
            new = f"current sentinel {index}".encode()
            path.write_bytes(new)
            assert reader.read_new(path) == new.decode()
    finally:
        reader.close()


def test_same_inode_shrink_resets_offset(tmp_path):
    path = tmp_path / "output.log"
    reader = PaneOutputReader()
    try:
        path.write_bytes(b"long previous segment")
        assert reader.read_new(path) == "long previous segment"
        inode = path.stat().st_ino
        path.write_bytes(b"new")
        assert path.stat().st_ino == inode
        assert reader.read_new(path) == "new"
    finally:
        reader.close()


def test_utf8_split_during_append_and_at_rotation_keeps_stream_alive(tmp_path):
    path = tmp_path / "output.log"
    reader = PaneOutputReader()
    try:
        path.write_bytes(b"before\xe2")
        assert reader.read_new(path) == "before"
        with path.open("ab") as stream:
            stream.write(b"\x82\xac")
        assert reader.read_new(path) == "\u20ac"
        with path.open("ab") as stream:
            stream.write(b"\xe2")
        assert reader.read_new(path) == ""
        path.replace(tmp_path / "output.log.1")
        path.write_bytes(b"\x82\xac after rotation")
        assert "after rotation" in reader.read_new(path)
        with path.open("ab") as stream:
            stream.write(b" live sentinel")
        assert reader.read_new(path) == " live sentinel"
    finally:
        reader.close()
