"""The CURRENT-pointer swap (generations.py) must survive a Windows
open-handle race on os.replace (WinError 5 / PermissionError) with a bounded retry,
and must still fail loudly when the switch is genuinely stuck. POSIX PermissionError
is a real permission problem and is raised at once (no retry)."""
import pytest

from synapt.recall import generations


def _flaky_replace(fail_times, exc=PermissionError):
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= fail_times:
            # WinError 5 shape: PermissionError(errno, "Access is denied")
            raise exc(13, "Access is denied")

    return flaky, calls


def test_windows_retry_succeeds_after_transient_permission_error(monkeypatch):
    monkeypatch.setattr(generations.os, "name", "nt")
    flaky, calls = _flaky_replace(fail_times=3)
    monkeypatch.setattr(generations.os, "replace", flaky)
    generations._atomic_replace_with_retry("src", "dst", base_delay=0.0)
    assert calls["n"] == 4  # three transient failures, then the switch lands


def test_windows_retry_reraises_after_exhausting_attempts(monkeypatch):
    # never a silent skip: a genuinely stuck switch still raises, bounded.
    monkeypatch.setattr(generations.os, "name", "nt")
    flaky, calls = _flaky_replace(fail_times=999)
    monkeypatch.setattr(generations.os, "replace", flaky)
    with pytest.raises(PermissionError):
        generations._atomic_replace_with_retry("src", "dst", attempts=5, base_delay=0.0)
    assert calls["n"] == 5


def test_posix_does_not_retry_permission_error(monkeypatch):
    monkeypatch.setattr(generations.os, "name", "posix")
    flaky, calls = _flaky_replace(fail_times=1)
    monkeypatch.setattr(generations.os, "replace", flaky)
    with pytest.raises(PermissionError):
        generations._atomic_replace_with_retry("src", "dst", base_delay=0.0)
    assert calls["n"] == 1  # raised on the first attempt, no Windows-style retry
