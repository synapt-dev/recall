"""TDD specs for recall#637: bind recall agent identity to gr2 workspaces.

All tests should FAIL before implementation. They define the expected
behavior for gr2-aware identity resolution in the recall channel system.
"""

import hashlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gr2_workspace(tmp_path):
    """Create a minimal gr2 workspace structure."""
    ws = tmp_path / "my-workspace"
    ws.mkdir()
    grip = ws / ".grip"
    grip.mkdir()
    (grip / "workspace.toml").write_text(
        'version = 2\nname = "my-workspace"\nlayout = "team-workspace"\n'
    )
    agents = ws / "agents"
    agents.mkdir()
    repos = ws / "repos"
    repos.mkdir()
    # Create recall data dir
    recall_dir = ws / ".synapt" / "recall"
    recall_dir.mkdir(parents=True)
    return ws


@pytest.fixture
def gr2_agent(gr2_workspace):
    """Create a gr2 agent workspace within the gr2 workspace."""
    agent_dir = gr2_workspace / "agents" / "opus"
    agent_dir.mkdir()
    (agent_dir / "agent.toml").write_text(
        'name = "opus"\nkind = "agent-workspace"\n'
    )
    return agent_dir


@pytest.fixture
def gr2_repo(gr2_workspace):
    """Create a gr2 repo within the workspace."""
    repo_dir = gr2_workspace / "repos" / "synapt"
    repo_dir.mkdir()
    (repo_dir / "repo.toml").write_text(
        'name = "synapt"\nurl = "https://github.com/synapt-dev/recall.git"\n'
    )
    return repo_dir


@pytest.fixture
def gr1_gripspace(tmp_path):
    """Create a minimal gr1 gripspace structure for comparison."""
    gs = tmp_path / "old-gripspace"
    gs.mkdir()
    gitgrip = gs / ".gitgrip"
    gitgrip.mkdir()
    (gitgrip / "griptrees.json").write_text("{}")
    recall_dir = gs / ".synapt" / "recall"
    recall_dir.mkdir(parents=True)
    return gs


# ---------------------------------------------------------------------------
# 1. Workspace detection
# ---------------------------------------------------------------------------


def test_detect_gr2_workspace(gr2_workspace):
    """_detect_workspace_context finds .grip/workspace.toml and returns Gr2Context."""
    from synapt.recall.channel import _detect_workspace_context

    with patch("synapt.recall.channel.Path.cwd", return_value=gr2_workspace):
        ctx = _detect_workspace_context()

    assert ctx.kind == "gr2"
    assert ctx.name == "my-workspace"
    assert ctx.root == gr2_workspace


def test_detect_gr1_gripspace(gr1_gripspace):
    """_detect_workspace_context falls back to Gr1Context for gr1 gripspaces."""
    from synapt.recall.channel import _detect_workspace_context

    with patch("synapt.recall.channel.Path.cwd", return_value=gr1_gripspace):
        ctx = _detect_workspace_context()

    assert ctx.kind == "gr1"
    assert ctx.root == gr1_gripspace


def test_detect_standalone(tmp_path):
    """_detect_workspace_context returns StandaloneContext outside any workspace."""
    from synapt.recall.channel import _detect_workspace_context

    with patch("synapt.recall.channel.Path.cwd", return_value=tmp_path):
        ctx = _detect_workspace_context()

    assert ctx.kind == "standalone"


def test_detect_from_subdirectory(gr2_workspace, gr2_repo):
    """Detection works when CWD is inside a repo subdirectory."""
    from synapt.recall.channel import _detect_workspace_context

    subdir = gr2_repo / "src" / "lib"
    subdir.mkdir(parents=True)

    with patch("synapt.recall.channel.Path.cwd", return_value=subdir):
        ctx = _detect_workspace_context()

    assert ctx.kind == "gr2"
    assert ctx.name == "my-workspace"


# ---------------------------------------------------------------------------
# 2. Agent ID derivation in gr2
# ---------------------------------------------------------------------------


def test_agent_id_from_agent_toml(gr2_workspace, gr2_agent):
    """When CWD is inside agents/{name}/, derive stable g2_ ID from metadata."""
    from synapt.recall.channel import _agent_id

    env = {"SYNAPT_AGENT_ID": ""}
    with patch.dict(os.environ, env, clear=False), \
         patch("synapt.recall.channel.Path.cwd", return_value=gr2_agent):
        aid = _agent_id()

    assert aid.startswith("g2_")
    assert "my-workspace" in aid
    assert "opus" in aid


def test_agent_id_stable_across_clones(tmp_path):
    """Same workspace+agent metadata at different paths produces same ID."""
    from synapt.recall.channel import _agent_id

    def make_workspace(base, name="my-workspace", agent="opus"):
        ws = base / name
        ws.mkdir(parents=True)
        (ws / ".grip").mkdir()
        (ws / ".grip" / "workspace.toml").write_text(
            f'version = 2\nname = "{name}"\nlayout = "team-workspace"\n'
        )
        (ws / "agents").mkdir()
        agent_dir = ws / "agents" / agent
        agent_dir.mkdir()
        (agent_dir / "agent.toml").write_text(
            f'name = "{agent}"\nkind = "agent-workspace"\n'
        )
        (ws / ".synapt" / "recall").mkdir(parents=True)
        return agent_dir

    clone_a = make_workspace(tmp_path / "location-a")
    clone_b = make_workspace(tmp_path / "location-b")

    env = {"SYNAPT_AGENT_ID": ""}
    with patch.dict(os.environ, env, clear=False):
        with patch("synapt.recall.channel.Path.cwd", return_value=clone_a):
            id_a = _agent_id()
        with patch("synapt.recall.channel.Path.cwd", return_value=clone_b):
            id_b = _agent_id()

    assert id_a == id_b, "Same workspace+agent metadata must produce identical IDs"


def test_agent_id_differs_per_agent_name(gr2_workspace):
    """Different agent names in the same workspace produce different IDs."""
    from synapt.recall.channel import _agent_id

    # Create a second agent
    atlas_dir = gr2_workspace / "agents" / "atlas"
    atlas_dir.mkdir()
    (atlas_dir / "agent.toml").write_text(
        'name = "atlas"\nkind = "agent-workspace"\n'
    )
    opus_dir = gr2_workspace / "agents" / "opus"
    opus_dir.mkdir(exist_ok=True)
    (opus_dir / "agent.toml").write_text(
        'name = "opus"\nkind = "agent-workspace"\n'
    )

    env = {"SYNAPT_AGENT_ID": ""}
    with patch.dict(os.environ, env, clear=False):
        with patch("synapt.recall.channel.Path.cwd", return_value=opus_dir):
            id_opus = _agent_id()
        with patch("synapt.recall.channel.Path.cwd", return_value=atlas_dir):
            id_atlas = _agent_id()

    assert id_opus != id_atlas


def test_synapt_agent_id_env_still_takes_priority(gr2_workspace, gr2_agent):
    """SYNAPT_AGENT_ID env var overrides gr2 metadata (backward compat)."""
    from synapt.recall.channel import _agent_id

    with patch.dict(os.environ, {"SYNAPT_AGENT_ID": "org-registered-42"}), \
         patch("synapt.recall.channel.Path.cwd", return_value=gr2_agent):
        aid = _agent_id()

    assert aid == "org-registered-42"


# ---------------------------------------------------------------------------
# 3. Griptree resolution in gr2
# ---------------------------------------------------------------------------


def test_griptree_from_gr2_repo(gr2_workspace, gr2_repo):
    """Griptree in gr2 = workspace_name/repo_name from metadata."""
    from synapt.recall.channel import _resolve_griptree

    with patch("synapt.recall.channel.Path.cwd", return_value=gr2_repo):
        gt = _resolve_griptree()

    assert gt == "my-workspace/synapt"


def test_griptree_from_gr2_agent_dir(gr2_workspace, gr2_agent):
    """Griptree from agent directory = workspace_name/agents/agent_name."""
    from synapt.recall.channel import _resolve_griptree

    with patch("synapt.recall.channel.Path.cwd", return_value=gr2_agent):
        gt = _resolve_griptree()

    # Agent dirs are not repos, so griptree should still include workspace name
    assert gt.startswith("my-workspace")


def test_griptree_gr1_unchanged(gr1_gripspace):
    """gr1 gripspace griptree resolution is unchanged."""
    from synapt.recall.channel import _resolve_griptree

    with patch("synapt.recall.channel.Path.cwd", return_value=gr1_gripspace):
        gt = _resolve_griptree()

    assert gt == "old-gripspace"


# ---------------------------------------------------------------------------
# 4. Presence table workspace column
# ---------------------------------------------------------------------------


def test_join_stores_workspace_in_presence(gr2_workspace, gr2_agent):
    """channel_join stores workspace name in presence table."""
    from synapt.recall.channel import channel_join, _open_db

    with patch("synapt.recall.channel.Path.cwd", return_value=gr2_agent), \
         patch("synapt.recall.channel.project_data_dir",
               return_value=gr2_workspace / ".synapt" / "recall"):
        channel_join(channel="dev", display_name="opus")

    conn = _open_db(gr2_workspace / ".synapt" / "recall")
    try:
        row = conn.execute(
            "SELECT workspace FROM presence WHERE display_name = 'opus'"
        ).fetchone()
    finally:
        conn.close()

    assert row is not None
    assert row[0] == "my-workspace"


# ---------------------------------------------------------------------------
# 5. Clone survival (integration)
# ---------------------------------------------------------------------------


def test_cursor_survives_workspace_recreation(tmp_path):
    """Agent joins, reads, workspace is recreated: cursor is preserved."""
    from synapt.recall.channel import (
        channel_join, channel_post, channel_read, _open_db,
    )

    def make_ws(base):
        ws = base
        ws.mkdir(exist_ok=True)
        (ws / ".grip").mkdir(exist_ok=True)
        (ws / ".grip" / "workspace.toml").write_text(
            'version = 2\nname = "team-x"\nlayout = "team-workspace"\n'
        )
        (ws / "agents" / "opus").mkdir(parents=True, exist_ok=True)
        (ws / "agents" / "opus" / "agent.toml").write_text(
            'name = "opus"\nkind = "agent-workspace"\n'
        )
        recall = ws / ".synapt" / "recall"
        recall.mkdir(parents=True, exist_ok=True)
        return ws

    # Clone A: join, post, read
    ws_a = make_ws(tmp_path / "clone-a")
    agent_dir_a = ws_a / "agents" / "opus"

    with patch("synapt.recall.channel.Path.cwd", return_value=agent_dir_a), \
         patch("synapt.recall.channel.project_data_dir",
               return_value=ws_a / ".synapt" / "recall"):
        channel_join(channel="test", display_name="opus")
        channel_post(channel="test", message="msg1")
        channel_post(channel="test", message="msg2")
        channel_read(channel="test", limit=10)

    # Clone B: same workspace metadata, different path
    ws_b = make_ws(tmp_path / "clone-b")
    agent_dir_b = ws_b / "agents" / "opus"

    # Copy the channels DB so we have shared state
    import shutil
    ch_dir_a = ws_a / ".synapt" / "recall" / "channels"
    ch_dir_b = ws_b / ".synapt" / "recall" / "channels"
    if ch_dir_a.exists():
        shutil.copytree(ch_dir_a, ch_dir_b)

    # Capture cursor from clone A before moving to clone B
    conn_a = _open_db(ws_a / ".synapt" / "recall")
    try:
        from synapt.recall.channel import _agent_id as _aid_fn
        with patch("synapt.recall.channel.Path.cwd", return_value=agent_dir_a):
            aid_a = _aid_fn()
        cursor_a = conn_a.execute(
            "SELECT last_read_at FROM cursors WHERE agent_id = ? AND channel = 'test'",
            (aid_a,),
        ).fetchone()
    finally:
        conn_a.close()

    assert cursor_a is not None, "Clone A must have a cursor after channel_read"

    with patch("synapt.recall.channel.Path.cwd", return_value=agent_dir_b), \
         patch("synapt.recall.channel.project_data_dir",
               return_value=ws_b / ".synapt" / "recall"):
        from synapt.recall.channel import _agent_id, channel_unread
        aid = _agent_id()
        # Join should find the existing cursor (INSERT OR IGNORE)
        channel_join(channel="test", display_name="opus")
        # Post a new message AFTER the cursor position
        channel_post(channel="test", message="msg3")
        # Unread count should reflect only messages after the cursor
        unread = channel_unread()

    # The key property: agent ID is identical across clones (g2_ prefix)
    assert aid.startswith("g2_")
    assert aid == aid_a, "Clone A and Clone B must produce the same agent ID"

    # Verify cursor VALUE survived: the cursor from clone A should be
    # present under the same agent_id in clone B's DB.
    conn = _open_db(ws_b / ".synapt" / "recall")
    try:
        cursor_b = conn.execute(
            "SELECT last_read_at FROM cursors WHERE agent_id = ? AND channel = 'test'",
            (aid,),
        ).fetchone()
    finally:
        conn.close()

    assert cursor_b is not None, "Cursor should exist in clone B's DB"

    # Behavioral contract: unread count should be small (just msg3 and
    # possibly the join event), NOT the full history (msg1 + msg2 + msg3).
    # If the cursor was reset, unread would include all messages.
    assert unread.get("test", 0) <= 2, (
        f"Expected at most 2 unread (msg3 + join event), got {unread.get('test', 0)}. "
        "Cursor from clone A did not survive -- identity or cursor was reset."
    )


# ---------------------------------------------------------------------------
# 6. Backward compatibility
# ---------------------------------------------------------------------------


def test_gr1_identity_unchanged_when_no_grip_dir(gr1_gripspace):
    """Without .grip/workspace.toml, all identity behavior is unchanged."""
    from synapt.recall.channel import _agent_id, _resolve_griptree

    env = {"SYNAPT_AGENT_ID": ""}
    with patch.dict(os.environ, env, clear=False), \
         patch("synapt.recall.channel.Path.cwd", return_value=gr1_gripspace), \
         patch("synapt.recall.channel.project_data_dir",
               return_value=gr1_gripspace / ".synapt" / "recall"):
        aid = _agent_id(name="opus")
        gt = _resolve_griptree()

    # Should produce a_ format (existing behavior), not g2_
    assert aid.startswith("a_")
    assert gt == "old-gripspace"


def test_standalone_identity_unchanged(tmp_path):
    """Without any workspace manager, identity falls back to session hash."""
    from synapt.recall.channel import _agent_id

    recall_dir = tmp_path / ".synapt" / "recall"
    recall_dir.mkdir(parents=True)

    env = {"SYNAPT_AGENT_ID": ""}
    with patch.dict(os.environ, env, clear=False), \
         patch("synapt.recall.channel.Path.cwd", return_value=tmp_path), \
         patch("synapt.recall.channel.project_data_dir", return_value=recall_dir):
        aid = _agent_id()

    # Should produce s_ format (session-scoped fallback)
    assert aid.startswith("s_")
