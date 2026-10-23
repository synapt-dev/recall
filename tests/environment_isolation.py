"""Shared process-environment isolation for the test harness.

Tests inherit the shell that launched pytest.  Values exported by a real
agent session are therefore ambient inputs unless the harness removes them
before each test.  Keep that policy in one importable seam so a new suite does
not have to rediscover which variables change recall semantics.

A test that deliberately exercises one of these variables sets it after the
autouse fixture runs.
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import pytest


AMBIENT_PROCESS_ENV_VARS = (
    "SYNAPT_SHARED_CHANNELS_DIR",
    "SYNAPT_RECALL_ROOT",
    "SYNAPT_RECALL_WORKTREE",
    "GRIPSPACE_ROOT",
    "SYNAPT_AGENT_ID",
    "CLAUDE_CONFIG_DIR",
)


def scrub_ambient_process_env(
    monkeypatch: pytest.MonkeyPatch,
    variables: Iterable[str] = AMBIENT_PROCESS_ENV_VARS,
) -> None:
    """Remove agent-session inputs before a test establishes its own state."""
    for variable in variables:
        monkeypatch.delenv(variable, raising=False)
    # The shared-room map defaults to a file under the real home, which no
    # test redirects. Point it at nothing so a map the operator wrote never
    # moves a test's room; a test of the map sets its own after this runs.
    monkeypatch.setenv("SYNAPT_ROOM_MAP", os.devnull)
