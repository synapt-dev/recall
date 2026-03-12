"""Cross-platform file locking.

Replaces direct fcntl.flock() calls with wrappers that work on
macOS, Linux, AND Windows. On Unix, uses fcntl.flock(). On Windows,
uses msvcrt.locking().

Three operations:
- lock_exclusive(f): Block until exclusive lock acquired on file object
- lock_exclusive_nb(fd): Non-blocking exclusive lock on file descriptor (int)
- unlock(f_or_fd): Release lock on file object or file descriptor
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    import msvcrt
    import os

    def lock_exclusive(f) -> None:
        """Acquire an exclusive lock on an open file (blocking)."""
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

    def lock_exclusive_nb(fd: int) -> None:
        """Acquire an exclusive lock on a file descriptor (non-blocking).

        Raises OSError (errno.EACCES or EDEADLOCK) if the lock cannot
        be acquired immediately.
        """
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def unlock(f_or_fd) -> None:
        """Release a lock on a file object or file descriptor."""
        fd = f_or_fd if isinstance(f_or_fd, int) else f_or_fd.fileno()
        try:
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass  # Already unlocked or fd closing

else:
    import fcntl

    def lock_exclusive(f) -> None:  # type: ignore[misc]
        """Acquire an exclusive lock on an open file (blocking)."""
        fcntl.flock(f, fcntl.LOCK_EX)

    def lock_exclusive_nb(fd: int) -> None:  # type: ignore[misc]
        """Acquire an exclusive lock on a file descriptor (non-blocking).

        Raises OSError (errno.EAGAIN/EWOULDBLOCK) if the lock cannot
        be acquired immediately.
        """
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unlock(f_or_fd) -> None:  # type: ignore[misc]
        """Release a lock on a file object or file descriptor."""
        fd = f_or_fd if isinstance(f_or_fd, int) else f_or_fd.fileno()
        fcntl.flock(fd, fcntl.LOCK_UN)
