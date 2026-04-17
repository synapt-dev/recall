"""TDD specs for recall#665: agent posts appearing as human identity.

Problem: The session-start hook unconditionally joins #dev with role="human",
even when SYNAPT_AGENT_ID is set (indicating an agent session). The agent's
registered ID gets the "human" role in presence, and the escalation guard
(recall#546) prevents correction. All subsequent posts from that agent render
with the [human] tag.

Two fixes:
1. Session-start hook: detect agent sessions via SYNAPT_AGENT_ID and join
   with role="agent" instead of role="human".
2. Defense-in-depth: when channel_join detects an agent colliding with a
   human's s_* hash, derive a distinct a_* identity for the agent.
"""

import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from synapt.recall.channel import (
    _agent_id,
    _open_db,
    _AGENT_ID_CACHE,
    channel_join,
    channel_post,
    channel_read,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_agent(agent_id: str, agent_name: str | None = None):
    """Patch env vars to simulate a gr-spawn agent session."""
    env = {"SYNAPT_AGENT_ID": agent_id}
    if agent_name:
        env["SYNAPT_AGENT_NAME"] = agent_name
    return patch.dict(os.environ, env)


def _patch_no_agent():
    """Ensure no SYNAPT_AGENT_ID is set (human session)."""
    env = {k: v for k, v in os.environ.items() if k not in ("SYNAPT_AGENT_ID", "SYNAPT_AGENT_NAME")}
    return patch.dict(os.environ, env, clear=True)


def _get_presence_role(project_dir: Path, agent_id: str) -> str | None:
    """Read role from presence table for a given agent_id."""
    conn = _open_db(project_dir)
    try:
        row = conn.execute(
            "SELECT role FROM presence WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        return row["role"] if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Fix 1: Session-start hook should detect agent sessions
# ---------------------------------------------------------------------------

class TestSessionStartRoleDetection:
    """The session-start hook must not register agent sessions as human."""

    def test_agent_session_joins_as_agent(self, tmp_path):
        """When SYNAPT_AGENT_ID is set, channel_join from session-start
        should use role='agent', not role='human'."""
        with _patch_agent("apollo-001", "Apollo"):
            # Simulate what the session-start hook should do after the fix:
            # detect SYNAPT_AGENT_ID and join with role="agent"
            role = "agent" if os.environ.get("SYNAPT_AGENT_ID") else "human"
            channel_join("dev", project_dir=tmp_path, role=role)

            assert _get_presence_role(tmp_path, "apollo-001") == "agent"

    def test_human_session_joins_as_human(self, tmp_path):
        """Without SYNAPT_AGENT_ID, session-start should join as human."""
        with _patch_no_agent():
            role = "agent" if os.environ.get("SYNAPT_AGENT_ID") else "human"
            channel_join("dev", project_dir=tmp_path, role=role)

            aid = _agent_id(tmp_path)
            assert _get_presence_role(tmp_path, aid) == "human"

    def test_agent_post_not_tagged_human(self, tmp_path):
        """After joining as agent, posts must not render with [human] tag."""
        with _patch_agent("apollo-001", "Apollo"):
            channel_join("dev", project_dir=tmp_path, role="agent",
                         display_name="Apollo")
            channel_post("dev", "test message from agent",
                         project_dir=tmp_path, display_name="Apollo")

            output = channel_read("dev", project_dir=tmp_path, limit=5)
            assert "[human]" not in output
            assert "Apollo" in output


# ---------------------------------------------------------------------------
# Fix 2: Collision guard should split agent identity
# ---------------------------------------------------------------------------

class TestCollisionGuardSplitsIdentity:
    """When an agent joins with the same s_* hash as a human, the agent
    should get a new distinct identity instead of being absorbed."""

    def test_agent_gets_split_identity_on_collision(self, tmp_path):
        """If a human owns s_xxxx with role='human', an agent joining with
        the same s_xxxx should get a new a_* identity."""
        _AGENT_ID_CACHE.clear()
        with _patch_no_agent():
            # Human joins first
            channel_join("dev", project_dir=tmp_path, role="human")
            human_aid = _agent_id(tmp_path)
            assert _get_presence_role(tmp_path, human_aid) == "human"

            # Clear cache to simulate a new process (MCP server)
            _AGENT_ID_CACHE.clear()

            # Agent joins without display_name — same s_* hash
            result = channel_join("dev", project_dir=tmp_path, role="agent")

            # Human's role must be preserved
            assert _get_presence_role(tmp_path, human_aid) == "human"

            # Agent should have gotten a distinct identity
            new_aid = _agent_id(tmp_path)
            assert new_aid != human_aid, (
                f"Agent should have a split identity, not share {human_aid}"
            )
            assert new_aid.startswith("a_"), (
                f"Split identity should be a_* hash, got {new_aid}"
            )
            assert _get_presence_role(tmp_path, new_aid) == "agent"

    def test_split_identity_cached_for_subsequent_posts(self, tmp_path):
        """After collision split, channel_post should use the split identity,
        not the original s_* hash."""
        _AGENT_ID_CACHE.clear()
        with _patch_no_agent():
            # Human joins
            channel_join("dev", project_dir=tmp_path, role="human")
            human_aid = _agent_id(tmp_path)
            _AGENT_ID_CACHE.clear()

            # Agent joins (gets split identity)
            channel_join("dev", project_dir=tmp_path, role="agent")

            # Post without display_name — should use cached split identity
            channel_post("dev", "message from split agent",
                         project_dir=tmp_path)

            output = channel_read("dev", project_dir=tmp_path, limit=5)
            assert "[human]" not in output

    def test_no_split_when_no_collision(self, tmp_path):
        """When no human owns the s_* hash, agent keeps its original identity."""
        _AGENT_ID_CACHE.clear()
        with _patch_no_agent():
            channel_join("dev", project_dir=tmp_path, role="agent")
            aid = _agent_id(tmp_path)
            assert aid.startswith("s_")
            assert _get_presence_role(tmp_path, aid) == "agent"

    def test_registered_agent_not_absorbed_by_human_hook(self, tmp_path):
        """A registered agent (SYNAPT_AGENT_ID) whose session-start hook
        mistakenly ran with role='human' can still correct its role on
        the next explicit join with role='agent'."""
        with _patch_agent("opus-001", "Opus"):
            # Bug scenario: hook ran channel_join with role="human"
            channel_join("dev", project_dir=tmp_path, role="human")
            assert _get_presence_role(tmp_path, "opus-001") == "human"

            # Agent's MCP join with role="agent" should correct this
            channel_join("dev", project_dir=tmp_path, role="agent",
                         display_name="Opus")
            assert _get_presence_role(tmp_path, "opus-001") == "agent"
