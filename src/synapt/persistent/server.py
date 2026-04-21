"""OSS stubs for persistent-agent career memory tools.

Premium overrides these via the synapt.plugins entry point. When premium
is installed, its register_tools replaces these stubs with the real
implementation backed by career.db.
"""

from __future__ import annotations

PLUGIN_NAME = "persistent"
PLUGIN_VERSION = "0.1.0"

_PREMIUM_MSG = (
    "recall_career requires premium. "
    "Install synapt-private to unlock persistent agent memory."
)


def recall_career(
    action: str = "search",
    query: str = "",
    lesson: str = "",
    scope: str = "agent",
    source_project: str = "",
    lesson_id: str = "",
    agent_name: str | None = None,
    project_dir: str = "",
) -> str:
    """Manage career lessons learned across projects (premium stub).

    Career memory stores durable lessons that persist across projects and
    sessions. Lessons have three scopes: project (local), team (org-shared),
    and agent (personal career knowledge).

    Args:
        action: "search", "list", "save", "retract".
        query: Search query for "search" action.
        lesson: Lesson text for "save" action.
        scope: "project", "team", or "agent" (default "agent").
        source_project: Project that generated the lesson.
        lesson_id: Lesson ID for "retract" action.
        agent_name: Agent identity (resolved from env if omitted).
        project_dir: Project directory for scope resolution.
    """
    return _PREMIUM_MSG


def register_tools(mcp) -> None:
    """Register career memory stubs on the MCP server."""
    mcp.tool()(recall_career)
