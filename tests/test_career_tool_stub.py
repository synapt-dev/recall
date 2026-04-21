"""Tests for recall_career OSS stub — Story 11.

The OSS recall_career tool is a no-op stub that returns a premium-required
message for all actions. Premium replaces it via the plugin entry point seam.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from synapt.plugins import LoadedPlugin, register_plugins
from synapt.recall.server import recall_career, register_tools


class TestRecallCareerStub:
    """recall_career OSS stub returns premium-required for all actions."""

    def test_search_action_returns_premium_message(self):
        result = recall_career(action="search", query="testing lessons")
        assert "premium" in result.lower()

    def test_list_action_returns_premium_message(self):
        result = recall_career(action="list")
        assert "premium" in result.lower()

    def test_save_action_returns_premium_message(self):
        result = recall_career(
            action="save",
            lesson="Always validate inputs",
            scope="agent",
            source_project="recall",
        )
        assert "premium" in result.lower()

    def test_retract_action_returns_premium_message(self):
        result = recall_career(action="retract", lesson_id="abc123")
        assert "premium" in result.lower()

    def test_unknown_action_returns_premium_message(self):
        result = recall_career(action="unknown_action")
        assert "premium" in result.lower()

    def test_returns_string(self):
        result = recall_career(action="list")
        assert isinstance(result, str)

    def test_default_action_is_search(self):
        result = recall_career()
        assert "premium" in result.lower()

    def test_message_includes_tool_name(self):
        result = recall_career(action="search")
        assert "recall_career" in result


class TestRecallCareerRegistration:
    """recall_career is registered as an MCP tool via register_tools."""

    def test_career_registered_in_mcp(self):
        mcp = MagicMock()
        register_tools(mcp)
        registered_names = {
            call.args[0].__name__
            for call in mcp.tool.return_value.call_args_list
            if call.args
        }
        assert "recall_career" in registered_names


class TestRecallCareerSignature:
    """recall_career accepts the documented parameters."""

    def test_accepts_action_param(self):
        recall_career(action="search")

    def test_accepts_query_param(self):
        recall_career(action="search", query="test")

    def test_accepts_lesson_param(self):
        recall_career(action="save", lesson="test lesson")

    def test_accepts_scope_param(self):
        recall_career(action="save", scope="team")

    def test_accepts_source_project_param(self):
        recall_career(action="save", source_project="recall")

    def test_accepts_lesson_id_param(self):
        recall_career(action="retract", lesson_id="abc123")

    def test_accepts_agent_name_param(self):
        recall_career(action="search", agent_name="apollo")

    def test_accepts_project_dir_param(self):
        recall_career(action="search", project_dir="/tmp/project")

    def test_all_params_together(self):
        result = recall_career(
            action="save",
            query="test",
            lesson="learned something",
            scope="agent",
            source_project="grip",
            lesson_id="",
            agent_name="apollo",
            project_dir="/tmp",
        )
        assert isinstance(result, str)


class TestPersistentPluginSeam:
    """The synapt.persistent.server module provides the plugin seam."""

    def test_persistent_module_is_importable(self):
        import synapt.persistent.server as career_plugin
        assert hasattr(career_plugin, "register_tools")
        assert callable(career_plugin.register_tools)

    def test_persistent_module_has_recall_career(self):
        import synapt.persistent.server as career_plugin
        assert hasattr(career_plugin, "recall_career")
        assert callable(career_plugin.recall_career)

    def test_persistent_stub_returns_premium_message(self):
        import synapt.persistent.server as career_plugin
        result = career_plugin.recall_career(action="search", query="test")
        assert "premium" in result.lower()
        assert "recall_career" in result

    def test_persistent_register_tools_registers_career(self):
        import synapt.persistent.server as career_plugin
        captured = []

        class FakeMCP:
            def tool(self):
                def decorator(fn):
                    captured.append(fn)
                    return fn
                return decorator

        career_plugin.register_tools(FakeMCP())
        names = {fn.__name__ for fn in captured}
        assert "recall_career" in names

    def test_persistent_stub_has_action_api(self):
        import synapt.persistent.server as career_plugin
        sig = inspect.signature(career_plugin.recall_career)
        assert "action" in sig.parameters
        assert "query" in sig.parameters
        assert "agent_name" in sig.parameters
        assert "project_dir" in sig.parameters
        assert "scope" in sig.parameters

    def test_persistent_plugin_registers_through_loader(self):
        import synapt.persistent.server as career_plugin
        calls = []

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

    def test_persistent_module_has_plugin_metadata(self):
        import synapt.persistent.server as career_plugin
        assert career_plugin.PLUGIN_NAME == "persistent"
        assert career_plugin.PLUGIN_VERSION == "0.1.0"
