"""Tests for LLM response parsing utilities."""

from synapt.recall._llm_util import parse_llm_json


def test_parse_clean_json():
    result = parse_llm_json('{"nodes": [{"action": "create", "content": "test"}]}')
    assert result == {"nodes": [{"action": "create", "content": "test"}]}


def test_parse_markdown_fenced_json():
    result = parse_llm_json('```json\n{"nodes": []}\n```')
    assert result == {"nodes": []}


def test_parse_json_with_preamble():
    result = parse_llm_json('Here is the result:\n{"nodes": []}')
    assert result == {"nodes": []}


def test_parse_markdown_numbered_list():
    """Small models output markdown lists instead of JSON."""
    response = (
        'After analyzing, I found:\n\n'
        '1. "Always run tests before deploying" (convention)\n'
        '\t* Category: convention\n'
        '\t* Confidence: 0.9\n'
        '\t* Tags: ["testing", "deploy"]\n'
        '2. "Use Python 3.12 for all projects" (preference)\n'
        '\t* Category: preference\n'
        '\t* Confidence: 0.7\n'
        '\t* Tags: ["python"]\n'
    )
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 2
    assert result["nodes"][0]["content"] == "Always run tests before deploying"
    assert result["nodes"][0]["category"] == "convention"
    assert result["nodes"][0]["confidence"] == 0.9
    assert result["nodes"][0]["tags"] == ["testing", "deploy"]
    assert result["nodes"][1]["content"] == "Use Python 3.12 for all projects"
    assert result["nodes"][1]["category"] == "preference"
    assert result["nodes"][1]["confidence"] == 0.7


def test_parse_markdown_without_metadata():
    """Numbered list with content but no Category/Confidence fields."""
    response = (
        'Knowledge:\n\n'
        '1. "Database uses PostgreSQL 15"\n'
        '2. "API rate limit is 100 req/min"\n'
    )
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 2
    assert result["nodes"][0]["content"] == "Database uses PostgreSQL 15"
    assert result["nodes"][0]["category"] == "workflow"  # default
    assert result["nodes"][0]["confidence"] == 0.6  # default


def test_parse_empty_response():
    assert parse_llm_json("") is None


def test_parse_gibberish():
    assert parse_llm_json("no structured output here at all") is None
