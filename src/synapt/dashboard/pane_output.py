"""Read new terminal bytes while following a rotated pane capture."""

import codecs
import os
from pathlib import Path


class PaneOutputReader:
    """Follow the active file, resetting on replacement or observed shrink.

    Hold the previous descriptor until the next file is open so its inode cannot
    be recycled between polls. This is a live view, not a lossless archive: a
    reader that falls behind several rotations receives the current segment.
    """

    def __init__(self):
        self._file = None
        self._identity = None
        self._position = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def read_new(self, path: Path) -> str:
        try:
            current = path.open("rb")
        except FileNotFoundError:
            return ""
        stat = os.fstat(current.fileno())
        identity = (stat.st_dev, stat.st_ino)
        prefix = ""
        if identity != self._identity or stat.st_size < self._position:
            prefix = self._decoder.decode(b"", final=True)
            self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            self._position = 0
        previous = self._file
        self._file = current
        self._identity = identity
        if previous is not None:
            previous.close()
        current.seek(self._position)
        content = current.read()
        self._position = current.tell()
        return prefix + self._decoder.decode(content)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
