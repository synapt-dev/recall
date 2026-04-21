from __future__ import annotations

import inspect

from synapt.plugins import LoadedPlugin, register_plugins


def test_career_plugin_stub_is_discoverable_via_entry_points():
    """OSS should advertise recall_career even before premium is installed."""
    import synapt.persistent.server as career_plugin

    assert hasattr(career_plugin, "register_tools")
    assert callable(career_plugin.register_tools)


def test_career_tool_registers_locked_stub_with_action_api():
    """The OSS stub must expose the future action-based MCP contract."""
    import synapt.persistent.server as career_plugin

    captured: list[object] = []

    class FakeMCP:
        def tool(self):
            def decorator(fn):
                captured.append(fn)
                return fn

            return decorator

    career_plugin.register_tools(FakeMCP())

    recall_career = next(fn for fn in captured if getattr(fn, "__name__", "") == "recall_career")
    sig = inspect.signature(recall_career)

    assert "action" in sig.parameters
    assert "query" in sig.parameters
    assert "agent_name" in sig.parameters
    assert "project_dir" in sig.parameters
    assert "scope" in sig.parameters

    message = recall_career(action="search", query="lessons", agent_name="sentinel")
    assert "premium" in message.lower()
    assert "recall_career" in message


def test_career_plugin_registers_through_standard_plugin_loader():
    """The career stub should fit the same plugin registration seam as premium plugins."""
    import synapt.persistent.server as career_plugin

    calls: list[str] = []

    class FakeMCP:
        def tool(self):
            def decorator(fn):
                calls.append(fn.__name__)
                return fn

            return decorator

    plugin = LoadedPlugin("persistent", "", career_plugin, "persistent")
    registered = register_plugins(FakeMCP(), [plugin])

    assert [p.name for p in registered] == ["persistent"]
    assert "recall_career" in calls
