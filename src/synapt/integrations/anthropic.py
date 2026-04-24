"""Anthropic Memory Tool backend powered by synapt recall.

Provides SynaptMemoryTool, a drop-in replacement for Anthropic's
BetaAbstractMemoryTool that routes all memory operations through recall's
hybrid search and knowledge persistence.

Usage:
    from anthropic import Anthropic
    from synapt.integrations.anthropic import SynaptMemoryTool

    client = Anthropic()
    memory = SynaptMemoryTool()

    response = client.beta.messages.run_tools(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Remember that deployments use blue-green strategy"}],
        tools=[memory],
    ).until_done()

    # Async variant
    from synapt.integrations.anthropic import SynaptAsyncMemoryTool

    async_memory = SynaptAsyncMemoryTool()
    response = await client.beta.messages.run_tools(
        model="claude-sonnet-4-6",
        messages=[...],
        tools=[async_memory],
    ).until_done()
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from pathlib import PurePosixPath
from typing import Any

try:
    from anthropic.lib.tools import (
        BetaAbstractMemoryTool,
        BetaAsyncAbstractMemoryTool,
        BetaFunctionToolResultType,
    )
    from anthropic.types.beta import (
        BetaCacheControlEphemeralParam,
        BetaMemoryTool20250818CreateCommand,
        BetaMemoryTool20250818DeleteCommand,
        BetaMemoryTool20250818InsertCommand,
        BetaMemoryTool20250818RenameCommand,
        BetaMemoryTool20250818StrReplaceCommand,
        BetaMemoryTool20250818ViewCommand,
    )
except ImportError as exc:
    BetaAbstractMemoryTool = None  # type: ignore[assignment,misc]
    BetaAsyncAbstractMemoryTool = None  # type: ignore[assignment,misc]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

log = logging.getLogger("synapt.integrations.anthropic")

_VIRTUAL_ROOT = "/memories"
_VIRTUAL_DIRS = ("knowledge", "sessions", "notes")


def _require_anthropic() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "Install anthropic>=0.77.0 to use the Anthropic memory backend"
        ) from _IMPORT_ERROR


def _normalize_path(path: str) -> str:
    p = PurePosixPath(path)
    if not p.is_absolute():
        p = PurePosixPath(_VIRTUAL_ROOT) / p
    parts = []
    for part in p.parts:
        if part == "/":
            continue
        parts.append(part)
    result = "/" + "/".join(parts) if parts else "/"
    if not result.startswith(_VIRTUAL_ROOT):
        result = _VIRTUAL_ROOT + result
    return result


def _path_to_node_id(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]


def _numbered_lines(text: str, start: int = 1) -> str:
    lines = text.split("\n")
    width = len(str(start + len(lines) - 1))
    return "\n".join(
        f"{i:>{width}}\t{line}" for i, line in enumerate(lines, start=start)
    )


class _FileCache:
    """Thread-safe in-memory cache for virtual memory files."""

    def __init__(self) -> None:
        self._files: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, path: str) -> str | None:
        with self._lock:
            return self._files.get(path)

    def set(self, path: str, content: str) -> None:
        with self._lock:
            self._files[path] = content

    def delete(self, path: str) -> bool:
        with self._lock:
            return self._files.pop(path, None) is not None

    def rename(self, old_path: str, new_path: str) -> bool:
        with self._lock:
            content = self._files.pop(old_path, None)
            if content is None:
                return False
            self._files[new_path] = content
            return True

    def list_dir(self, dir_path: str) -> list[str]:
        with self._lock:
            prefix = dir_path.rstrip("/") + "/"
            children: set[str] = set()
            for p in self._files:
                if p.startswith(prefix):
                    remainder = p[len(prefix):]
                    child = remainder.split("/")[0]
                    children.add(child)
            return sorted(children)

    def list_all(self) -> list[str]:
        with self._lock:
            return sorted(self._files.keys())

    def clear(self) -> None:
        with self._lock:
            self._files.clear()


class _RecallBridge:
    """Bridges virtual file operations to recall's knowledge store."""

    def save(self, path: str, content: str) -> str:
        from synapt.recall.server import recall_save

        category = "memory"
        parts = PurePosixPath(path).parts
        if len(parts) >= 3 and parts[1] == "memories":
            category = parts[2]

        return recall_save(
            content=content,
            category=category,
            confidence=0.85,
            tags=["anthropic-memory", PurePosixPath(path).name],
            node_id=_path_to_node_id(path),
        )

    def retract(self, path: str) -> str:
        from synapt.recall.server import recall_save

        return recall_save(
            node_id=_path_to_node_id(path),
            retract=True,
        )

    def search(self, query: str, max_chunks: int = 3) -> str:
        from synapt.recall.server import recall_search

        return recall_search(
            query=query,
            max_chunks=max_chunks,
            max_tokens=800,
        )

    def load_knowledge_files(self, cache: _FileCache) -> None:
        """Seed the cache with existing recall knowledge nodes."""
        try:
            from synapt.recall.core import project_index_dir
            from synapt.recall.storage import RecallDB
            from pathlib import Path

            project = Path.cwd().resolve()
            db_path = project_index_dir(project) / "recall.db"
            if not db_path.exists():
                return
            db = RecallDB(db_path)
            try:
                nodes = db.list_knowledge_nodes(limit=200)
                for node in nodes:
                    if node.get("status") == "retracted":
                        continue
                    content = node.get("content", "")
                    node_id = node.get("id", "")
                    category = node.get("category", "knowledge")
                    if category not in _VIRTUAL_DIRS:
                        category = "knowledge"
                    slug = re.sub(r"[^a-z0-9]+", "-", content[:60].lower()).strip("-")
                    filename = f"{slug}-{node_id[:6]}.md"
                    path = f"{_VIRTUAL_ROOT}/{category}/{filename}"
                    cache.set(path, content)
            finally:
                db.close()
        except Exception as e:
            log.debug("Could not seed knowledge files: %s", e)


class _MemoryToolCore:
    """Shared implementation for sync and async memory tools."""

    def __init__(self) -> None:
        self._cache = _FileCache()
        self._bridge = _RecallBridge()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self._bridge.load_knowledge_files(self._cache)
            self._initialized = True

    def do_view(self, command: Any) -> str:
        self._ensure_initialized()
        path = _normalize_path(command.path)

        if path == _VIRTUAL_ROOT or path == _VIRTUAL_ROOT + "/":
            entries = list(_VIRTUAL_DIRS)
            cached_top = set()
            for p in self._cache.list_all():
                rel = p[len(_VIRTUAL_ROOT) + 1:] if p.startswith(_VIRTUAL_ROOT + "/") else p
                top = rel.split("/")[0]
                cached_top.add(top)
            entries = sorted(set(entries) | cached_top)
            listing = "\n".join(f"  {e}/" for e in entries)
            return f"Here's the result of running `view` on {path}:\n{listing}\n"

        content = self._cache.get(path)
        if content is None:
            children = self._cache.list_dir(path)
            if children:
                listing = "\n".join(f"  {c}" for c in children)
                return f"Here's the result of running `view` on {path}:\n{listing}\n"

            if path.rstrip("/").split("/")[-1] in _VIRTUAL_DIRS:
                return f"Here's the result of running `view` on {path}:\n  (empty directory)\n"

            raise Exception(f"path '{path}' does not exist. Use `view` on '{_VIRTUAL_ROOT}' to see available files.")

        view_range = getattr(command, "view_range", None)
        lines = content.split("\n")

        if view_range and len(view_range) == 2:
            start, end = view_range
            start = max(1, start)
            end = min(len(lines), end)
            selected = lines[start - 1:end]
            numbered = _numbered_lines("\n".join(selected), start=start)
        else:
            numbered = _numbered_lines(content)

        return f"Here's the result of running `view` on {path}:\n{numbered}\n"

    def do_create(self, command: Any) -> str:
        self._ensure_initialized()
        path = _normalize_path(command.path)

        if self._cache.get(path) is not None:
            raise Exception(f"file '{path}' already exists. Use `str_replace` to edit or `delete` first.")

        self._cache.set(path, command.file_text)

        try:
            self._bridge.save(path, command.file_text)
        except Exception as e:
            log.warning("Recall save failed for %s: %s", path, e)

        return f"File created successfully at {path}."

    def do_str_replace(self, command: Any) -> str:
        self._ensure_initialized()
        path = _normalize_path(command.path)
        content = self._cache.get(path)

        if content is None:
            raise Exception(f"file '{path}' does not exist.")

        occurrences = content.count(command.old_str)
        if occurrences == 0:
            raise Exception(f"`old_str` not found in {path}. No changes made.")
        if occurrences > 1:
            raise Exception(f"`old_str` found {occurrences} times in {path}. Please provide a more unique string.")

        new_content = content.replace(command.old_str, command.new_str, 1)
        self._cache.set(path, new_content)

        try:
            self._bridge.save(path, new_content)
        except Exception as e:
            log.warning("Recall update failed for %s: %s", path, e)

        return f"The file {path} has been edited successfully."

    def do_insert(self, command: Any) -> str:
        self._ensure_initialized()
        path = _normalize_path(command.path)
        content = self._cache.get(path)

        if content is None:
            raise Exception(f"file '{path}' does not exist.")

        lines = content.split("\n")
        insert_line = command.insert_line

        if insert_line < 0 or insert_line > len(lines):
            raise Exception(f"insert_line {insert_line} is out of range [0, {len(lines)}].")

        new_lines = command.insert_text.split("\n")
        lines[insert_line:insert_line] = new_lines
        new_content = "\n".join(lines)
        self._cache.set(path, new_content)

        try:
            self._bridge.save(path, new_content)
        except Exception as e:
            log.warning("Recall update failed for %s: %s", path, e)

        return f"The file {path} has been edited successfully."

    def do_delete(self, command: Any) -> str:
        self._ensure_initialized()
        path = _normalize_path(command.path)

        children = self._cache.list_dir(path)
        if children:
            for child in children:
                child_path = path.rstrip("/") + "/" + child
                self._cache.delete(child_path)
                try:
                    self._bridge.retract(child_path)
                except Exception:
                    pass
            return f"Directory '{path}' and {len(children)} file(s) deleted."

        if not self._cache.delete(path):
            raise Exception(f"path '{path}' does not exist.")

        try:
            self._bridge.retract(path)
        except Exception as e:
            log.warning("Recall retract failed for %s: %s", path, e)

        return f"File '{path}' has been deleted."

    def do_rename(self, command: Any) -> str:
        self._ensure_initialized()
        old_path = _normalize_path(command.old_path)
        new_path = _normalize_path(command.new_path)

        if self._cache.get(new_path) is not None:
            raise Exception(f"destination '{new_path}' already exists.")

        if not self._cache.rename(old_path, new_path):
            raise Exception(f"source '{old_path}' does not exist.")

        try:
            content = self._cache.get(new_path)
            if content is not None:
                self._bridge.retract(old_path)
                self._bridge.save(new_path, content)
        except Exception as e:
            log.warning("Recall rename failed: %s", e)

        return f"Renamed '{old_path}' to '{new_path}'."

    def do_clear_all(self) -> str:
        self._cache.clear()
        self._initialized = False
        return "All memory cleared."


if BetaAbstractMemoryTool is not None:

    class SynaptMemoryTool(BetaAbstractMemoryTool):
        """Anthropic Memory Tool backed by synapt recall.

        Drop-in replacement for BetaAbstractMemoryTool. Routes all file
        operations through a virtual filesystem facade backed by recall's
        hybrid search and knowledge persistence.

        Args:
            cache_control: Optional Anthropic cache control parameter.
        """

        def __init__(
            self,
            *,
            cache_control: BetaCacheControlEphemeralParam | None = None,
        ) -> None:
            _require_anthropic()
            super().__init__(cache_control=cache_control)
            self._core = _MemoryToolCore()

        def view(self, command: BetaMemoryTool20250818ViewCommand) -> BetaFunctionToolResultType:
            return self._core.do_view(command)

        def create(self, command: BetaMemoryTool20250818CreateCommand) -> BetaFunctionToolResultType:
            return self._core.do_create(command)

        def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> BetaFunctionToolResultType:
            return self._core.do_str_replace(command)

        def insert(self, command: BetaMemoryTool20250818InsertCommand) -> BetaFunctionToolResultType:
            return self._core.do_insert(command)

        def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> BetaFunctionToolResultType:
            return self._core.do_delete(command)

        def rename(self, command: BetaMemoryTool20250818RenameCommand) -> BetaFunctionToolResultType:
            return self._core.do_rename(command)

        def clear_all_memory(self) -> BetaFunctionToolResultType:
            return self._core.do_clear_all()

        def get_context(self, query: str, *, max_chunks: int = 3) -> str:
            """Search recall for context related to a query.

            Use this outside the memory tool loop to retrieve relevant
            recall context without polluting view() responses.
            """
            try:
                result = self._core._bridge.search(query, max_chunks=max_chunks)
                if result and "No results" not in result:
                    return result
                return ""
            except Exception:
                return ""


if BetaAsyncAbstractMemoryTool is not None:

    class SynaptAsyncMemoryTool(BetaAsyncAbstractMemoryTool):
        """Async variant of SynaptMemoryTool.

        Same behavior as SynaptMemoryTool but implements the async interface
        for use with AsyncAnthropic clients.
        """

        def __init__(
            self,
            *,
            cache_control: BetaCacheControlEphemeralParam | None = None,
        ) -> None:
            _require_anthropic()
            super().__init__(cache_control=cache_control)
            self._core = _MemoryToolCore()

        async def view(self, command: BetaMemoryTool20250818ViewCommand) -> BetaFunctionToolResultType:
            return self._core.do_view(command)

        async def create(self, command: BetaMemoryTool20250818CreateCommand) -> BetaFunctionToolResultType:
            return self._core.do_create(command)

        async def str_replace(self, command: BetaMemoryTool20250818StrReplaceCommand) -> BetaFunctionToolResultType:
            return self._core.do_str_replace(command)

        async def insert(self, command: BetaMemoryTool20250818InsertCommand) -> BetaFunctionToolResultType:
            return self._core.do_insert(command)

        async def delete(self, command: BetaMemoryTool20250818DeleteCommand) -> BetaFunctionToolResultType:
            return self._core.do_delete(command)

        async def rename(self, command: BetaMemoryTool20250818RenameCommand) -> BetaFunctionToolResultType:
            return self._core.do_rename(command)

        async def clear_all_memory(self) -> BetaFunctionToolResultType:
            return self._core.do_clear_all()

        def get_context(self, query: str, *, max_chunks: int = 3) -> str:
            """Search recall for context related to a query."""
            try:
                result = self._core._bridge.search(query, max_chunks=max_chunks)
                if result and "No results" not in result:
                    return result
                return ""
            except Exception:
                return ""

else:

    class SynaptMemoryTool:  # type: ignore[no-redef]
        """Stub when anthropic SDK is not installed."""

        def __init__(self, **kwargs: Any) -> None:
            _require_anthropic()

    class SynaptAsyncMemoryTool:  # type: ignore[no-redef]
        """Stub when anthropic SDK is not installed."""

        def __init__(self, **kwargs: Any) -> None:
            _require_anthropic()
