"""Channel timestamps render as local wall time with the zone name, never as a bare UTC slice."""
import os
import time

import pytest

from synapt.recall.channel import _render_ts

STORED = "2026-09-09T20:03:12.345678Z"


def _under_tz(tz: str, fn):
    old = os.environ.get("TZ")
    os.environ["TZ"] = tz
    time.tzset()
    try:
        return fn()
    finally:
        if old is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = old
        time.tzset()


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="tzset is POSIX only")
def test_render_ts_is_local_wall_time_with_zone():
    assert _under_tz("America/Chicago", lambda: _render_ts(STORED)) == "2026-09-09 15:03 CDT"


@pytest.mark.skipif(not hasattr(time, "tzset"), reason="tzset is POSIX only")
def test_render_ts_on_a_utc_host_says_utc():
    assert _under_tz("UTC", lambda: _render_ts(STORED)) == "2026-09-09 20:03 UTC"


def test_render_ts_never_drops_the_zone():
    out = _render_ts(STORED)
    assert out.startswith("2026-09-09 ") or out.startswith("2026-09-10 ")
    assert out.split(" ")[-1].strip() != "" and "T" not in out.split(" ")[0]


def test_render_ts_malformed_falls_back_to_slice():
    assert _render_ts("not a timestamp at all") == "not a timestamp "
