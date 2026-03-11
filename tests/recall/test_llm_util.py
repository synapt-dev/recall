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


def test_parse_json_array():
    """LLM outputs a JSON array instead of {"nodes": [...]}."""
    response = '[{"action": "create", "content": "fact one"}, {"action": "create", "content": "fact two"}]'
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 2
    assert result["nodes"][0]["content"] == "fact one"


def test_parse_json_array_with_preamble():
    """JSON array embedded in surrounding text — multi-element so inner-object
    extraction can't grab a single valid dict spanning first-{ to last-}."""
    response = 'Here are the nodes:\n[{"action": "create", "content": "one"}, {"action": "create", "content": "two"}]\nDone.'
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 2
    assert result["nodes"][0]["content"] == "one"


def test_parse_json_array_in_markdown_fences():
    """JSON array wrapped in markdown code fences."""
    response = '```json\n[{"action": "create", "content": "test"}]\n```'
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 1


def test_parse_truncated_json():
    """Model ran out of tokens mid-JSON — salvage complete nodes."""
    response = (
        'Here are the nodes:\n'
        '{"nodes": [{"action": "create", "content": "fact one", "category": "workflow"}, '
        '{"action": "create", "content": "fact two", "category": "preference"}, '
        '{"action": "cre'  # truncated mid-node
    )
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 2
    assert result["nodes"][0]["content"] == "fact one"
    assert result["nodes"][1]["content"] == "fact two"


def test_parse_truncated_json_single_complete_node():
    """Only one node completed before truncation."""
    response = '{"nodes": [{"action": "create", "content": "the only fact", "category": "fact"}, {"ac'
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 1
    assert result["nodes"][0]["content"] == "the only fact"


def test_repair_truncated_dict_string_values():
    """Truncated enrichment dict — salvage complete key-value pairs."""
    response = '{"focus": "Fixed authentication", "done": ["Added OAuth", "Fixed redirect"], "deci'
    result = parse_llm_json(response)
    assert result is not None
    assert result["focus"] == "Fixed authentication"
    assert result["done"] == ["Added OAuth", "Fixed redirect"]
    assert "deci" not in result  # truncated key not included


def test_repair_truncated_dict_focus_only():
    """Truncated after focus value — still usable."""
    response = '{"focus": "Set up the database schema", "done": ["Migra'
    result = parse_llm_json(response)
    assert result is not None
    assert result["focus"] == "Set up the database schema"


def test_repair_truncated_dict_no_useful_keys():
    """Truncated dict without any enrichment keys → None."""
    response = '{"random_key": "value", "other'
    result = parse_llm_json(response)
    assert result is None
