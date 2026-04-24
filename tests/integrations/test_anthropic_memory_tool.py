"""Tests for the Anthropic Memory Tool backend (SynaptMemoryTool)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("anthropic")

from anthropic.types.beta import (
    BetaMemoryTool20250818ViewCommand,
    BetaMemoryTool20250818CreateCommand,
    BetaMemoryTool20250818DeleteCommand,
    BetaMemoryTool20250818InsertCommand,
    BetaMemoryTool20250818RenameCommand,
    BetaMemoryTool20250818StrReplaceCommand,
)


@pytest.fixture
def tool():
    with patch("synapt.integrations.anthropic._RecallBridge") as MockBridge:
        bridge_instance = MockBridge.return_value
        bridge_instance.save.return_value = "saved"
        bridge_instance.retract.return_value = "retracted"
        bridge_instance.search.return_value = "No results found."
        bridge_instance.load_knowledge_files.return_value = None

        from synapt.integrations.anthropic import SynaptMemoryTool
        t = SynaptMemoryTool(search_augment=False)
        t._core._bridge = bridge_instance
        yield t


@pytest.fixture
def tool_with_search():
    with patch("synapt.integrations.anthropic._RecallBridge") as MockBridge:
        bridge_instance = MockBridge.return_value
        bridge_instance.save.return_value = "saved"
        bridge_instance.retract.return_value = "retracted"
        bridge_instance.search.return_value = "Relevant context: deploy uses blue-green"
        bridge_instance.load_knowledge_files.return_value = None

        from synapt.integrations.anthropic import SynaptMemoryTool
        t = SynaptMemoryTool(search_augment=True)
        t._core._bridge = bridge_instance
        yield t


def _view(path, view_range=None):
    return BetaMemoryTool20250818ViewCommand(command="view", path=path, view_range=view_range)


def _create(path, file_text):
    return BetaMemoryTool20250818CreateCommand(command="create", path=path, file_text=file_text)


def _str_replace(path, old_str, new_str):
    return BetaMemoryTool20250818StrReplaceCommand(
        command="str_replace", path=path, old_str=old_str, new_str=new_str
    )


def _insert(path, insert_line, insert_text):
    return BetaMemoryTool20250818InsertCommand(
        command="insert", path=path, insert_line=insert_line, insert_text=insert_text
    )


def _delete(path):
    return BetaMemoryTool20250818DeleteCommand(command="delete", path=path)


def _rename(old_path, new_path):
    return BetaMemoryTool20250818RenameCommand(command="rename", old_path=old_path, new_path=new_path)


class TestViewCommand:

    def test_view_root_lists_directories(self, tool):
        result = tool.view(_view("/memory"))
        assert "knowledge/" in result
        assert "sessions/" in result
        assert "notes/" in result

    def test_view_empty_directory(self, tool):
        result = tool.view(_view("/memory/knowledge"))
        assert "empty directory" in result

    def test_view_nonexistent_file(self, tool):
        result = tool.view(_view("/memory/knowledge/nope.md"))
        assert "Error" in result
        assert "does not exist" in result

    def test_view_file_with_line_numbers(self, tool):
        tool.create(_create("/memory/notes/deploy.md", "line one\nline two\nline three"))
        result = tool.view(_view("/memory/notes/deploy.md"))
        assert "1\tline one" in result
        assert "2\tline two" in result
        assert "3\tline three" in result

    def test_view_file_with_range(self, tool):
        tool.create(_create("/memory/notes/multi.md", "a\nb\nc\nd\ne"))
        result = tool.view(_view("/memory/notes/multi.md", view_range=[2, 4]))
        assert "2\tb" in result
        assert "3\tc" in result
        assert "4\td" in result
        assert "1\ta" not in result
        assert "5\te" not in result

    def test_view_directory_listing(self, tool):
        tool.create(_create("/memory/notes/a.md", "aaa"))
        tool.create(_create("/memory/notes/b.md", "bbb"))
        result = tool.view(_view("/memory/notes"))
        assert "a.md" in result
        assert "b.md" in result


class TestCreateCommand:

    def test_create_new_file(self, tool):
        result = tool.create(_create("/memory/notes/test.md", "hello world"))
        assert "created successfully" in result

    def test_create_duplicate_fails(self, tool):
        tool.create(_create("/memory/notes/dup.md", "first"))
        result = tool.create(_create("/memory/notes/dup.md", "second"))
        assert "Error" in result
        assert "already exists" in result

    def test_create_triggers_recall_save(self, tool):
        tool.create(_create("/memory/knowledge/fact.md", "Python 3.13 is latest"))
        tool._core._bridge.save.assert_called()

    def test_create_relative_path(self, tool):
        result = tool.create(_create("notes/rel.md", "relative path"))
        assert "created successfully" in result
        view_result = tool.view(_view("/memory/notes/rel.md"))
        assert "relative path" in view_result


class TestStrReplaceCommand:

    def test_replace_succeeds(self, tool):
        tool.create(_create("/memory/notes/edit.md", "old text here"))
        result = tool.str_replace(_str_replace("/memory/notes/edit.md", "old text", "new text"))
        assert "edited successfully" in result
        view = tool.view(_view("/memory/notes/edit.md"))
        assert "new text" in view
        assert "old text" not in view

    def test_replace_not_found(self, tool):
        tool.create(_create("/memory/notes/nf.md", "some content"))
        result = tool.str_replace(_str_replace("/memory/notes/nf.md", "missing", "new"))
        assert "Error" in result
        assert "not found" in result

    def test_replace_ambiguous(self, tool):
        tool.create(_create("/memory/notes/amb.md", "aaa aaa"))
        result = tool.str_replace(_str_replace("/memory/notes/amb.md", "aaa", "bbb"))
        assert "Error" in result
        assert "2 times" in result

    def test_replace_nonexistent_file(self, tool):
        result = tool.str_replace(_str_replace("/memory/notes/nope.md", "a", "b"))
        assert "Error" in result
        assert "does not exist" in result


class TestInsertCommand:

    def test_insert_at_beginning(self, tool):
        tool.create(_create("/memory/notes/ins.md", "line 1\nline 2"))
        result = tool.insert(_insert("/memory/notes/ins.md", 0, "new first"))
        assert "edited successfully" in result
        view = tool.view(_view("/memory/notes/ins.md"))
        assert "1\tnew first" in view

    def test_insert_at_end(self, tool):
        tool.create(_create("/memory/notes/ins2.md", "line 1\nline 2"))
        tool.insert(_insert("/memory/notes/ins2.md", 2, "line 3"))
        view = tool.view(_view("/memory/notes/ins2.md"))
        assert "3\tline 3" in view

    def test_insert_out_of_range(self, tool):
        tool.create(_create("/memory/notes/ins3.md", "one"))
        result = tool.insert(_insert("/memory/notes/ins3.md", 100, "nope"))
        assert "Error" in result
        assert "out of range" in result

    def test_insert_nonexistent_file(self, tool):
        result = tool.insert(_insert("/memory/notes/nope.md", 0, "text"))
        assert "Error" in result


class TestDeleteCommand:

    def test_delete_file(self, tool):
        tool.create(_create("/memory/notes/del.md", "to delete"))
        result = tool.delete(_delete("/memory/notes/del.md"))
        assert "deleted" in result
        view = tool.view(_view("/memory/notes/del.md"))
        assert "does not exist" in view

    def test_delete_nonexistent(self, tool):
        result = tool.delete(_delete("/memory/notes/nope.md"))
        assert "Error" in result
        assert "does not exist" in result

    def test_delete_directory(self, tool):
        tool.create(_create("/memory/notes/sub/a.md", "aaa"))
        tool.create(_create("/memory/notes/sub/b.md", "bbb"))
        result = tool.delete(_delete("/memory/notes/sub"))
        assert "deleted" in result
        assert "2 file(s)" in result

    def test_delete_triggers_recall_retract(self, tool):
        tool.create(_create("/memory/notes/ret.md", "retract me"))
        tool.delete(_delete("/memory/notes/ret.md"))
        tool._core._bridge.retract.assert_called()


class TestRenameCommand:

    def test_rename_succeeds(self, tool):
        tool.create(_create("/memory/notes/old.md", "content"))
        result = tool.rename(_rename("/memory/notes/old.md", "/memory/notes/new.md"))
        assert "Renamed" in result
        view_old = tool.view(_view("/memory/notes/old.md"))
        assert "does not exist" in view_old
        view_new = tool.view(_view("/memory/notes/new.md"))
        assert "content" in view_new

    def test_rename_source_missing(self, tool):
        result = tool.rename(_rename("/memory/notes/nope.md", "/memory/notes/new.md"))
        assert "Error" in result
        assert "does not exist" in result

    def test_rename_dest_exists(self, tool):
        tool.create(_create("/memory/notes/a.md", "aaa"))
        tool.create(_create("/memory/notes/b.md", "bbb"))
        result = tool.rename(_rename("/memory/notes/a.md", "/memory/notes/b.md"))
        assert "Error" in result
        assert "already exists" in result


class TestSearchAugmentation:

    def test_view_includes_search_context(self, tool_with_search):
        tool_with_search.create(_create("/memory/notes/deploy.md", "blue-green deploys"))
        result = tool_with_search.view(_view("/memory/notes/deploy.md"))
        assert "Related recall context" in result
        assert "blue-green" in result

    def test_view_range_skips_search(self, tool_with_search):
        tool_with_search.create(_create("/memory/notes/deploy2.md", "line1\nline2\nline3"))
        result = tool_with_search.view(_view("/memory/notes/deploy2.md", view_range=[1, 2]))
        assert "Related recall context" not in result


class TestExecuteDispatch:

    def test_execute_routes_view(self, tool):
        result = tool.execute(_view("/memory"))
        assert "knowledge/" in result

    def test_execute_routes_create(self, tool):
        result = tool.execute(_create("/memory/notes/exec.md", "via execute"))
        assert "created successfully" in result


class TestClearAll:

    def test_clear_all_memory(self, tool):
        tool.create(_create("/memory/notes/a.md", "aaa"))
        tool.create(_create("/memory/notes/b.md", "bbb"))
        result = tool.clear_all_memory()
        assert "cleared" in result
        view = tool.view(_view("/memory/notes/a.md"))
        assert "does not exist" in view


class TestPathNormalization:

    def test_relative_path_normalized(self, tool):
        tool.create(_create("knowledge/fact.md", "fact"))
        result = tool.view(_view("/memory/knowledge/fact.md"))
        assert "fact" in result

    def test_bare_filename(self, tool):
        tool.create(_create("test.md", "bare"))
        result = tool.view(_view("/memory/test.md"))
        assert "bare" in result


class TestImportability:

    def test_sync_importable(self):
        from synapt.integrations.anthropic import SynaptMemoryTool
        assert SynaptMemoryTool is not None

    def test_async_importable(self):
        from synapt.integrations.anthropic import SynaptAsyncMemoryTool
        assert SynaptAsyncMemoryTool is not None

    def test_to_dict(self, tool):
        d = tool.to_dict()
        assert d["type"] == "memory_20250818"
        assert d["name"] == "memory"
