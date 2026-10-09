"""Read new terminal bytes while following a rotated pane capture."""

import codecs
import os
from pathlib import Path


def _open_log(path: Path):
    if os.name != "nt":
        return path.open("rb")
    # The old descriptor stays open while rotation renames its file. The CRT's
    # normal open denies that on Windows; share deletion as well as reads/writes.
    import ctypes
    from ctypes import wintypes
    import msvcrt

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                           wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                           wintypes.HANDLE)
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    # GENERIC_READ; FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
    # OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL. No creation or write access.
    handle = create_file(str(path), 0x80000000, 0x7, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except OSError:
        close_handle(handle)
        raise
    try:
        return os.fdopen(fd, "rb")
    except BaseException:
        os.close(fd)
        raise


class PaneOutputReader:
    """Follow the active file, resetting on replacement or observed shrink.

    Hold the previous descriptor until the next file is open so its inode cannot
    be recycled between polls. This is a live view, not a lossless archive: a
    reader that falls behind several rotations receives the current segment.
    Same-inode truncation that regrows past the old offset between polls cannot
    be detected; the capture writer uses rename instead. First connect starts
    with at most the latest MiB, including for legacy oversized logs.
    """

    def __init__(self):
        self._file = None
        self._identity = None
        self._position = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def read_new(self, path: Path) -> str:
        try:
            current = _open_log(path)
        except OSError:
            return ""
        previous = self._file
        try:
            stat = os.fstat(current.fileno())
            identity = (stat.st_dev, stat.st_ino)
            reset = identity != self._identity or stat.st_size < self._position
            rest = b""
            if identity != self._identity and previous is not None:
                previous.seek(self._position)
                rest = previous.read()
            position = 0 if reset else self._position
            if self._identity is None:
                position = max(0, stat.st_size - 1024 * 1024)
            current.seek(position)
            content = current.read()
            position = current.tell()
        except OSError:
            current.close()
            return ""
        prefix = ""
        if reset:
            prefix = self._decoder.decode(rest, final=True)
            self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._file = current
        self._identity = identity
        if previous is not None:
            previous.close()
        self._position = position
        return prefix + self._decoder.decode(content)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
