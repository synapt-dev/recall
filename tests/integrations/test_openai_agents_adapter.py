"""Tests for the OpenAI Agents SDK SynaptSession adapter."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def session(tmp_path: Path):
    from synapt.integrations.openai_agents import SynaptSession

    s = SynaptSession(session_id="test-session-1", db_path=tmp_path / "test.db")
    yield s
    s.close()


@pytest.fixture
def session_b(tmp_path: Path):
    from synapt.integrations.openai_agents import SynaptSession

    s = SynaptSession(session_id="test-session-2", db_path=tmp_path / "test.db")
    yield s
    s.close()


class TestSessionProtocolContract:

    @pytest.mark.asyncio
    async def test_empty_on_init(self, session):
        assert await session.get_items() == []

    @pytest.mark.asyncio
    async def test_add_and_retrieve_items(self, session):
        items = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        await session.add_items(items)
        retrieved = await session.get_items()
        assert len(retrieved) == 2
        assert retrieved[0]["role"] == "user"
        assert retrieved[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_get_items_with_limit(self, session):
        await session.add_items([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ])
        retrieved = await session.get_items(limit=2)
        assert len(retrieved) == 2
        assert retrieved[0]["content"] == "second"
        assert retrieved[1]["content"] == "third"

    @pytest.mark.asyncio
    async def test_add_items_across_calls(self, session):
        await session.add_items([{"role": "user", "content": "a"}])
        await session.add_items([{"role": "assistant", "content": "b"}])
        retrieved = await session.get_items()
        assert len(retrieved) == 2
        assert retrieved[0]["content"] == "a"
        assert retrieved[1]["content"] == "b"

    @pytest.mark.asyncio
    async def test_pop_item(self, session):
        await session.add_items([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ])
        popped = await session.pop_item()
        assert popped["content"] == "second"
        remaining = await session.get_items()
        assert len(remaining) == 1
        assert remaining[0]["content"] == "first"

    @pytest.mark.asyncio
    async def test_pop_item_empty(self, session):
        assert await session.pop_item() is None

    @pytest.mark.asyncio
    async def test_clear_session(self, session):
        await session.add_items([
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ])
        assert len(await session.get_items()) == 2
        await session.clear_session()
        assert await session.get_items() == []

    @pytest.mark.asyncio
    async def test_clear_is_idempotent(self, session):
        await session.clear_session()
        await session.clear_session()
        assert await session.get_items() == []


class TestSessionIsolation:

    @pytest.mark.asyncio
    async def test_sessions_are_isolated(self, session, session_b):
        await session.add_items([{"role": "user", "content": "session 1"}])
        await session_b.add_items([{"role": "user", "content": "session 2"}])
        items_1 = await session.get_items()
        items_2 = await session_b.get_items()
        assert len(items_1) == 1
        assert items_1[0]["content"] == "session 1"
        assert len(items_2) == 1
        assert items_2[0]["content"] == "session 2"

    @pytest.mark.asyncio
    async def test_clear_only_affects_own_session(self, session, session_b):
        await session.add_items([{"role": "user", "content": "keep"}])
        await session_b.add_items([{"role": "user", "content": "delete"}])
        await session_b.clear_session()
        assert len(await session.get_items()) == 1
        assert await session_b.get_items() == []

    @pytest.mark.asyncio
    async def test_item_count(self, session):
        assert session.item_count == 0
        await session.add_items([{"role": "user", "content": "one"}])
        assert session.item_count == 1

    @pytest.mark.asyncio
    async def test_get_session_ids(self, session, session_b):
        await session.add_items([{"role": "user", "content": "a"}])
        await session_b.add_items([{"role": "user", "content": "b"}])
        ids = session.get_session_ids()
        assert "test-session-1" in ids
        assert "test-session-2" in ids


class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_nested_json_item(self, session):
        item = {
            "role": "assistant",
            "content": "result",
            "tool_calls": [{"id": "call_1", "function": {"name": "search", "arguments": "{}"}}],
            "metadata": {"model": "gpt-5.4", "tokens": 42},
        }
        await session.add_items([item])
        retrieved = await session.get_items()
        assert retrieved[0]["tool_calls"][0]["id"] == "call_1"
        assert retrieved[0]["metadata"]["model"] == "gpt-5.4"

    @pytest.mark.asyncio
    async def test_empty_content(self, session):
        await session.add_items([{"role": "assistant", "content": ""}])
        assert (await session.get_items())[0]["content"] == ""

    @pytest.mark.asyncio
    async def test_large_item(self, session):
        big = {"role": "user", "content": "x" * 100_000}
        await session.add_items([big])
        assert (await session.get_items())[0]["content"] == big["content"]

    def test_repr(self, session):
        r = repr(session)
        assert "test-session-1" in r
        assert "items=0" in r

    def test_importable(self):
        from synapt.integrations.openai_agents import SynaptSession
        assert SynaptSession is not None


class TestRecallIntegration:

    def test_search_method_exists(self, session):
        assert callable(session.search)

    def test_save_to_recall_method_exists(self, session):
        assert callable(session.save_to_recall)
