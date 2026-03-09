"""Tests for synapt.plugins — plugin discovery and registration."""

from __future__ import annotations

import types
from unittest.mock import MagicMock, patch

from synapt.plugins import (
    ENTRY_POINT_GROUP,
    LoadedPlugin,
    discover_plugins,
    register_plugins,
)


def _make_entry_point(
    name: str,
    module: types.ModuleType | None = None,
    load_error: Exception | None = None,
):
    """Create a mock EntryPoint that loads the given module."""
    ep = MagicMock()
    ep.name = name
    ep.group = ENTRY_POINT_GROUP
    if load_error:
        ep.load.side_effect = load_error
    else:
        ep.load.return_value = module
    return ep


def _make_plugin_module(
    has_register: bool = True,
    plugin_name: str | None = None,
    plugin_version: str | None = None,
    register_error: Exception | None = None,
):
    """Create a fake plugin module."""
    mod = types.ModuleType("fake_plugin")
    if has_register:
        if register_error:
            def register_tools(mcp):
                raise register_error
        else:
            def register_tools(mcp):
                mcp.tool()(lambda: "test")
        mod.register_tools = register_tools
    if plugin_name is not None:
        mod.PLUGIN_NAME = plugin_name
    if plugin_version is not None:
        mod.PLUGIN_VERSION = plugin_version
    return mod


_PATCH_TARGET = "synapt.plugins.importlib.metadata.entry_points"


class TestDiscoverPlugins:
    def test_discovers_valid_plugin(self):
        mod = _make_plugin_module(plugin_name="test-repair", plugin_version="1.0")
        ep = _make_entry_point("repair", mod)
        with patch(_PATCH_TARGET, return_value=[ep]):
            plugins = discover_plugins()
        assert len(plugins) == 1
        assert plugins[0].name == "test-repair"
        assert plugins[0].version == "1.0"
        assert plugins[0].entry_point_name == "repair"

    def test_falls_back_to_entry_point_name(self):
        mod = _make_plugin_module()  # no PLUGIN_NAME
        ep = _make_entry_point("watch", mod)
        with patch(_PATCH_TARGET, return_value=[ep]):
            plugins = discover_plugins()
        assert plugins[0].name == "watch"
        assert plugins[0].version == ""

    def test_skips_import_failure(self):
        ep = _make_entry_point("broken", load_error=ImportError("no such module"))
        with patch(_PATCH_TARGET, return_value=[ep]):
            plugins = discover_plugins()
        assert len(plugins) == 0

    def test_skips_missing_register_tools(self):
        mod = _make_plugin_module(has_register=False)
        ep = _make_entry_point("incomplete", mod)
        with patch(_PATCH_TARGET, return_value=[ep]):
            plugins = discover_plugins()
        assert len(plugins) == 0

    def test_discovers_multiple_plugins(self):
        mod1 = _make_plugin_module(plugin_name="repair")
        mod2 = _make_plugin_module(plugin_name="watch")
        eps = [_make_entry_point("repair", mod1), _make_entry_point("watch", mod2)]
        with patch(_PATCH_TARGET, return_value=eps):
            plugins = discover_plugins()
        assert len(plugins) == 2
        names = {p.name for p in plugins}
        assert names == {"repair", "watch"}

    def test_empty_when_no_plugins_installed(self):
        with patch(_PATCH_TARGET, return_value=[]):
            plugins = discover_plugins()
        assert plugins == []


class TestRegisterPlugins:
    def test_registers_tools_on_mcp(self):
        mod = _make_plugin_module()
        mcp = MagicMock()
        plugin = LoadedPlugin("test", "1.0", mod, "test")
        registered = register_plugins(mcp, [plugin])
        assert len(registered) == 1
        assert mcp.tool.called

    def test_skips_registration_failure(self):
        mod = _make_plugin_module(register_error=RuntimeError("boom"))
        mcp = MagicMock()
        plugin = LoadedPlugin("broken", "1.0", mod, "broken")
        registered = register_plugins(mcp, [plugin])
        assert len(registered) == 0

    def test_auto_discovers_when_no_plugins_passed(self):
        mod = _make_plugin_module()
        ep = _make_entry_point("auto", mod)
        mcp = MagicMock()
        with patch(_PATCH_TARGET, return_value=[ep]):
            registered = register_plugins(mcp)
        assert len(registered) == 1

    def test_partial_failure_registers_remaining(self):
        good_mod = _make_plugin_module(plugin_name="good")
        bad_mod = _make_plugin_module(plugin_name="bad", register_error=RuntimeError("fail"))
        mcp = MagicMock()
        plugins = [
            LoadedPlugin("good", "", good_mod, "good"),
            LoadedPlugin("bad", "", bad_mod, "bad"),
        ]
        registered = register_plugins(mcp, plugins)
        assert len(registered) == 1
        assert registered[0].name == "good"


class TestLoadedPlugin:
    def test_repr(self):
        p = LoadedPlugin("repair", "1.0", None, "repair")
        assert repr(p) == "LoadedPlugin('repair', '1.0')"

    def test_slots(self):
        p = LoadedPlugin("repair", "1.0", None, "repair")
        assert not hasattr(p, "__dict__")
