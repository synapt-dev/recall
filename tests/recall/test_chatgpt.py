"""Tests for ChatGPT archive parser."""

import json
import tempfile
import zipfile
from pathlib import Path

from synapt.recall import parse_chatgpt_archive, parse_chatgpt_conversation

from conftest import chatgpt_message


def test_chatgpt_store_time_truncation():
    """ChatGPT parser truncates long user/assistant text with '...' indicator."""
    conv = {
        "id": "conv-trunc-test",
        "title": "Truncation Test",
        "create_time": 1769000000.0,
        "update_time": 1769000600.0,
        "current_node": "n2",
        "mapping": {
            "n0": {"id": "n0", "parent": None, "children": ["n1"], "message": None},
            "n1": {
                "id": "n1",
                "parent": "n0",
                "children": ["n2"],
                "message": chatgpt_message("user", "Q" * 2000),
            },
            "n2": {
                "id": "n2",
                "parent": "n1",
                "children": [],
                "message": chatgpt_message("assistant", "A" * 6000),
            },
        },
    }
    chunks = parse_chatgpt_conversation(conv)
    assert len(chunks) == 1
    # user_text: 1500 + "..."
    assert len(chunks[0].user_text) == 1503
    assert chunks[0].user_text.endswith("...")
    # assistant_text: 5000 + "..."
    assert len(chunks[0].assistant_text) == 5003
    assert chunks[0].assistant_text.endswith("...")


def test_parse_chatgpt_conversation_active_path_and_tools():
    """Parse current_node path and include tool role data."""
    conv = {
        "id": "conv-1234567890",
        "title": "Test Conversation",
        "create_time": 1769000000.0,
        "update_time": 1769000600.0,
        "current_node": "n8",
        "mapping": {
            "n0": {"id": "n0", "parent": None, "children": ["n1"], "message": None},
            "n1": {
                "id": "n1",
                "parent": "n0",
                "children": ["n2"],
                "message": chatgpt_message(
                    "user",
                    "persistent profile text",
                    content_type="user_editable_context",
                    create_time=1769000001.0,
                ),
            },
            "n2": {
                "id": "n2",
                "parent": "n1",
                "children": ["n3"],
                "message": chatgpt_message("user", "first question", create_time=1769000002.0),
            },
            "n3": {
                "id": "n3",
                "parent": "n2",
                "children": ["n4"],
                "message": chatgpt_message("assistant", "first answer", create_time=1769000003.0),
            },
            "n4": {
                "id": "n4",
                "parent": "n3",
                "children": ["n5"],
                "message": chatgpt_message("user", "search this topic", create_time=1769000004.0),
            },
            "n5": {
                "id": "n5",
                "parent": "n4",
                "children": ["n6"],
                "message": chatgpt_message(
                    "assistant",
                    'search("quality curve")',
                    content_type="code",
                    recipient="browser.search",
                    create_time=1769000005.0,
                ),
            },
            "n6": {
                "id": "n6",
                "parent": "n5",
                "children": ["n8"],
                "message": chatgpt_message(
                    "tool",
                    "search results payload",
                    content_type="text",
                    author_name="web.run",
                    create_time=1769000006.0,
                ),
            },
            "n7": {
                "id": "n7",
                "parent": "n4",
                "children": [],
                "message": chatgpt_message(
                    "assistant",
                    "alternate answer not on active branch",
                    create_time=1769000007.0,
                ),
            },
            "n8": {
                "id": "n8",
                "parent": "n6",
                "children": [],
                "message": chatgpt_message(
                    "assistant",
                    "final grounded answer",
                    content_type="multimodal_text",
                    create_time=1769000008.0,
                ),
            },
        },
    }

    chunks = parse_chatgpt_conversation(conv)
    assert len(chunks) == 2

    assert chunks[0].user_text == "first question"
    assert "first answer" in chunks[0].assistant_text

    assert chunks[1].user_text == "search this topic"
    assert "search results payload" in chunks[1].assistant_text
    assert "final grounded answer" in chunks[1].assistant_text
    assert "alternate answer not on active branch" not in chunks[1].assistant_text
    assert "browser.search" in chunks[1].tools_used
    assert "web.run" in chunks[1].tools_used


def test_parse_chatgpt_archive_zip():
    """Parse ChatGPT export from a zip archive containing conversations.json."""
    convs = [
        {
            "id": "conv-aaaaaaaa",
            "create_time": 1769001000.0,
            "update_time": 1769001200.0,
            "current_node": "n2",
            "mapping": {
                "n0": {"id": "n0", "parent": None, "children": ["n1"], "message": None},
                "n1": {
                    "id": "n1",
                    "parent": "n0",
                    "children": ["n2"],
                    "message": chatgpt_message("user", "hello"),
                },
                "n2": {
                    "id": "n2",
                    "parent": "n1",
                    "children": [],
                    "message": chatgpt_message("assistant", "hi there"),
                },
            },
        }
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        zip_path = tmpdir / "chatgpt-export.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("conversations.json", json.dumps(convs))

        chunks = parse_chatgpt_archive(zip_path)
        assert len(chunks) == 1
        assert chunks[0].session_id == "conv-aaaaaaaa"
        assert chunks[0].user_text == "hello"
        assert "hi there" in chunks[0].assistant_text
