"""Read new terminal bytes while following a rotated pane capture."""

import codecs
import os
from pathlib import Path


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
            current = path.open("rb")
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
