"""TDD specs for recall#639: membership/presence separation.

Problem: _reap_stale_agents() deletes memberships rows for reaped agents.
After reaping, the agent has no channel memberships, so the next poll
returns "No channel memberships -- join a channel first." This breaks
monitoring loops that run across session boundaries.

Fix: Reaping should only affect ephemeral presence state (status, last_seen).
Durable membership (which channels an agent belongs to) must survive reaping.

All tests in this file are failing until the fix is implemented.
"""

import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import os

import pytest

from synapt.recall.channel import (
    _open_db,
    _reap_stale_agents,
    channel_join,
    channel_heartbeat,
    channel_unread,
    channel_who,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_agent(agent_id: str):
    """Patch SYNAPT_AGENT_ID env var to set the current agent identity."""
    return patch.dict(os.environ, {"SYNAPT_AGENT_ID": agent_id, "SYNAPT_AGENT_NAME": agent_id})


def _make_stale_agent(conn: sqlite3.Connection, agent_id: str, channel: str,
                      project_dir: Path, minutes_stale: int = 130) -> None:
    """Insert a presence row + membership row that is stale enough to be reaped."""
    stale_ts = (
        datetime.now(timezone.utc) - timedelta(minutes=minutes_stale)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    joined_ts = (
        datetime.now(timezone.utc) - timedelta(hours=3)
    ).strftime("%Y-%m-%dT%H:%M:%SZ")

    conn.execute(
        "INSERT OR REPLACE INTO presence "
        "(agent_id, griptree, display_name, role, status, last_seen, joined_at) "
        "VALUES (?, 'test', ?, 'agent', 'away', ?, ?)",
        (agent_id, agent_id, stale_ts, joined_ts),
    )
    conn.execute(
        "INSERT OR REPLACE INTO memberships (agent_id, channel, joined_at) "
        "VALUES (?, ?, ?)",
        (agent_id, channel, joined_ts),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Test: Membership rows survive reaping
# ---------------------------------------------------------------------------

class TestMembershipDurabilityAfterReap:
    """After reaping, membership rows must still exist."""

    def test_membership_row_exists_after_reap(self, tmp_path):
        """Reaping an agent must not delete their memberships row."""
        with _patch_agent("sentinel-test"):
            conn = _open_db(tmp_path)
            _make_stale_agent(conn, "sentinel-test", "dev", tmp_path)

            # Sanity: membership exists before reap
            before = conn.execute(
                "SELECT COUNT(*) FROM memberships WHERE agent_id = 'sentinel-test'",
            ).fetchone()[0]
            assert before == 1, "membership should exist before reap"

            _reap_stale_agents(conn, tmp_path)

            # After reap: membership must still be there
            after = conn.execute(
                "SELECT COUNT(*) FROM memberships WHERE agent_id = 'sentinel-test'",
            ).fetchone()[0]
            assert after == 1, "membership must survive reaping"

    def test_membership_joined_at_preserved_after_reap(self, tmp_path):
        """The original joined_at timestamp must not change after reaping."""
        with _patch_agent("sentinel-test"):
            conn = _open_db(tmp_path)
            _make_stale_agent(conn, "sentinel-test", "dev", tmp_path)

            original_joined_at = conn.execute(
                "SELECT joined_at FROM memberships WHERE agent_id = 'sentinel-test'",
            ).fetchone()[0]

            _reap_stale_agents(conn, tmp_path)

            after_joined_at = conn.execute(
                "SELECT joined_at FROM memberships WHERE agent_id = 'sentinel-test'",
            ).fetchone()[0]
            assert after_joined_at == original_joined_at, \
                "joined_at must be unchanged by reaping"

    def test_membership_across_multiple_channels_survives_reap(self, tmp_path):
        """All of an agent's memberships survive when they are reaped."""
        with _patch_agent("sentinel-test"):
            conn = _open_db(tmp_path)
            _make_stale_agent(conn, "sentinel-test", "dev", tmp_path)
            # Add a second membership manually
            joined_ts = conn.execute(
                "SELECT joined_at FROM memberships WHERE agent_id = 'sentinel-test'",
            ).fetchone()[0]
            conn.execute(
                "INSERT OR REPLACE INTO memberships (agent_id, channel, joined_at) "
                "VALUES ('sentinel-test', 'general', ?)",
                (joined_ts,),
            )
            conn.commit()

            _reap_stale_agents(conn, tmp_path)

            count = conn.execute(
                "SELECT COUNT(*) FROM memberships WHERE agent_id = 'sentinel-test'",
            ).fetchone()[0]
            assert count == 2, "both memberships must survive reaping"


# ---------------------------------------------------------------------------
# Test: Presence is still cleared by reaping (existing behavior preserved)
# ---------------------------------------------------------------------------

class TestPresenceUpdatedByReap:
    """Reaping should update presence to offline — existing behaviour."""

    def test_presence_status_offline_after_reap(self, tmp_path):
        """Presence row status becomes 'offline' after reaping."""
        with _patch_agent("sentinel-test"):
            conn = _open_db(tmp_path)
            _make_stale_agent(conn, "sentinel-test", "dev", tmp_path)

            _reap_stale_agents(conn, tmp_path)

            status = conn.execute(
                "SELECT status FROM presence WHERE agent_id = 'sentinel-test'",
            ).fetchone()
            # Presence row may persist with offline status, or be cleared.
            # Either is acceptable — what matters is the agent appears offline.
            # If the row exists it must be 'offline'.
            if status is not None:
                assert status[0] == "offline", "reaped agent must appear offline"


# ---------------------------------------------------------------------------
# Test: Monitoring loop use case — unread after reap does not fail
# ---------------------------------------------------------------------------

class TestUnreadSurvivesReap:
    """The monitoring loop: channel_unread must not error after agent is reaped."""

    def test_unread_returns_result_after_reap(self, tmp_path):
        """channel_unread must return a result (not 'No channel memberships') after reap."""
        with _patch_agent("sentinel-loop"):
            # Join and post something so there's a channel to read
            channel_join("dev", project_dir=tmp_path)
            channel_heartbeat(project_dir=tmp_path)

            # Simulate reaping by directly running the reaper
            conn = _open_db(tmp_path)
            _make_stale_agent(conn, "sentinel-loop", "dev", tmp_path)
            _reap_stale_agents(conn, tmp_path)

            # Now simulate the next poll tick — unread must not fail
            result = channel_unread(project_dir=tmp_path)
            assert "No channel memberships" not in result, \
                "channel_unread must work after being reaped (membership is durable)"

    def test_heartbeat_restores_presence_without_rejoin(self, tmp_path):
        """After reaping, a heartbeat should restore presence without losing memberships."""
        with _patch_agent("sentinel-loop"):
            channel_join("dev", project_dir=tmp_path)

            conn = _open_db(tmp_path)
            _make_stale_agent(conn, "sentinel-loop", "dev", tmp_path)
            _reap_stale_agents(conn, tmp_path)

            # Heartbeat should work and restore presence
            channel_heartbeat(project_dir=tmp_path)

            # Membership still intact
            conn2 = _open_db(tmp_path)
            count = conn2.execute(
                "SELECT COUNT(*) FROM memberships WHERE agent_id = 'sentinel-loop'",
            ).fetchone()[0]
            assert count >= 1, "membership must persist after heartbeat post-reap"


# ---------------------------------------------------------------------------
# Test: who() still excludes offline/reaped agents
# ---------------------------------------------------------------------------

class TestWhoExcludesOfflineAgents:
    """channel_who must not list reaped (offline) agents as active."""

    def test_reaped_agent_not_in_who(self, tmp_path):
        """After reaping, the agent must not appear in channel_who active list."""
        with _patch_agent("active-agent"):
            channel_join("dev", project_dir=tmp_path)
            channel_heartbeat(project_dir=tmp_path)

        # Insert and reap a stale agent
        conn = _open_db(tmp_path)
        _make_stale_agent(conn, "stale-agent", "dev", tmp_path)
        _reap_stale_agents(conn, tmp_path)

        with _patch_agent("active-agent"):
            result = channel_who(project_dir=tmp_path)

        assert "stale-agent" not in result, \
            "reaped agent must not appear in who() output after reaping"
