"""access.db creation survives Windows file-handle rules, checked on any OS."""

import os
import stat

from synapt.recall import storage


def _windows_fsync(monkeypatch):
    """os.open/os.fsync as Windows behaves: fsync needs a writable handle."""
    real_open, real_fsync = os.open, os.fsync
    writable = set()

    def fake_open(path, flags, *a, **kw):
        fd = real_open(path, flags, *a, **kw)
        writable.discard(fd)
        if flags & (os.O_WRONLY | os.O_RDWR):
            writable.add(fd)
        return fd

    def fake_fsync(fd):
        if fd not in writable:
            raise OSError(9, "Bad file descriptor")
        real_fsync(fd)

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "fsync", fake_fsync)


def test_create_access_db_fsyncs_tmp_through_a_writable_handle(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_FSYNC_DIRS", False, raising=False)
    _windows_fsync(monkeypatch)
    storage._create_access_db(tmp_path / "access.db")
    assert (tmp_path / "access.db").is_file()


def test_create_access_db_never_opens_a_directory_the_windows_way(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "_FSYNC_DIRS", False, raising=False)
    real_open = os.open
    dirs = []

    def fake_open(path, flags, *a, **kw):
        if stat.S_ISDIR(os.stat(path).st_mode):
            dirs.append(path)
        return real_open(path, flags, *a, **kw)

    monkeypatch.setattr(os, "open", fake_open)
    storage._create_access_db(tmp_path / "access.db")
    assert dirs == []
    assert (tmp_path / "access.db").is_file()
