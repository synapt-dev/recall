"""Tests for LLM response parsing utilities."""

from synapt.recall._llm_util import parse_llm_json, truncate_at_word


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
        '2. "Use Python 3.12 for all projects" (preference)\n'
    )
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 2
    assert result["nodes"][0]["content"] == "Always run tests before deploying"
    assert result["nodes"][1]["content"] == "Use Python 3.12 for all projects"


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
    assert result["nodes"][0]["category"] == "fact"  # default
    assert result["nodes"][0]["confidence"] == 0.6  # default


def test_parse_bullet_list_with_section_headers():
    """3B models output bullet lists under section headers like **Facts**."""
    response = (
        'After analyzing the session summaries:\n\n'
        '**Facts**\n\n'
        '* Caroline adopted a rescue dog named Rex in April 2025.\n'
        '* Caroline signed up for pottery class on July 2nd.\n'
        '* Caroline grew up in Dublin, Ireland.\n\n'
        '**Preferences**\n\n'
        '* Caroline prefers pottery class over other activities.\n'
    )
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 4
    assert result["nodes"][0]["content"] == "Caroline adopted a rescue dog named Rex in April 2025"
    assert result["nodes"][0]["category"] == "fact"
    assert result["nodes"][1]["content"] == "Caroline signed up for pottery class on July 2nd"
    assert result["nodes"][1]["category"] == "fact"
    assert result["nodes"][2]["content"] == "Caroline grew up in Dublin, Ireland"
    assert result["nodes"][2]["category"] == "fact"
    assert result["nodes"][3]["content"] == "Caroline prefers pottery class over other activities"
    assert result["nodes"][3]["category"] == "preference"


def test_parse_bullet_list_with_source_turns():
    """Bullet items with source turn references at end."""
    response = (
        '**Facts**\n\n'
        '* Caroline attended a council meeting for adoption (s008c00:10)\n'
        '* Caroline signed up for pottery class (s001c00:5, s008c00:10)\n'
    )
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 2
    assert result["nodes"][0]["content"] == "Caroline attended a council meeting for adoption"
    assert result["nodes"][0]["source_turns"] == ["s008c00:10"]
    assert result["nodes"][1]["source_turns"] == ["s001c00:5", "s008c00:10"]


def test_parse_bullet_list_with_inline_structured_fields():
    """Bullet items with inline category/confidence/tags fields."""
    response = (
        '**Facts**\n\n'
        '* Caroline adopted a rescue dog named Rex in April 2025. '
        '(category: fact, existing_id: null, content: "Caroline adopted a rescue dog named Rex in April 2025", '
        'confidence: 0.9, tags: ["adoption", "dog"], '
        'source_turns: ["s008c00:5", "s012c00:10"])\n'
    )
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 1
    node = result["nodes"][0]
    assert node["content"] == "Caroline adopted a rescue dog named Rex in April 2025"
    assert node["category"] == "fact"
    assert node["confidence"] == 0.9
    assert node["tags"] == ["adoption", "dog"]
    assert node["source_turns"] == ["s008c00:5", "s012c00:10"]


def test_parse_bullet_list_with_category_annotation():
    """Bullet items with category annotation in parentheses."""
    response = (
        'Knowledge:\n\n'
        '* Caroline discussed her transition and received support (fact)\n'
        '* Caroline prefers hiking over camping (preference)\n'
    )
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 2
    assert result["nodes"][0]["content"] == "Caroline discussed her transition and received support"
    assert result["nodes"][1]["content"] == "Caroline prefers hiking over camping"


def test_parse_dash_bullet_list():
    """Dash-prefixed bullet items."""
    response = (
        '**Facts**\n\n'
        '- Caroline joined a new LGBTQ activist group last Tuesday.\n'
        '- Caroline passed adoption agency interviews last Friday.\n'
    )
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 2
    assert result["nodes"][0]["content"] == "Caroline joined a new LGBTQ activist group last Tuesday"
    assert result["nodes"][1]["content"] == "Caroline passed adoption agency interviews last Friday"


def test_parse_real_3b_failure_response():
    """Actual 3B model response that was going to consolidation_failures.jsonl."""
    response = (
        'After analyzing the session summaries, I extracted the following '
        'durable knowledge nodes:\n\n'
        '**Facts**\n\n'
        '* Caroline attended a council meeting for adoption (s008c00:10)\n'
        '* Caroline realized she could be herself without fear and transitioned (s008c00:10)\n'
        '* Caroline signed up for pottery class on July 2nd (s001c00:5, s008c00:10)\n'
        '* Melanie got peace through creativity and family (s008c00:10)\n'
    )
    result = parse_llm_json(response)
    assert result is not None
    assert len(result["nodes"]) == 4
    assert result["nodes"][0]["category"] == "fact"
    assert result["nodes"][2]["source_turns"] == ["s001c00:5", "s008c00:10"]


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


# ---------------------------------------------------------------------------
# truncate_at_word
# ---------------------------------------------------------------------------

def test_truncate_short_text_unchanged():
    assert truncate_at_word("hello world", 50) == "hello world"

def test_truncate_at_word_boundary():
    text = "Configure xcconfig file with build targets"
    result = truncate_at_word(text, 25)
    assert result == "Configure xcconfig file"
    # Should not end mid-word
    assert not result.endswith("fi") and not result.endswith("wit")

def test_truncate_no_mid_word_cut():
    text = "a" * 50 + " " + "b" * 50
    result = truncate_at_word(text, 55)
    assert result == "a" * 50
    assert not result.endswith("b")

def test_truncate_exact_limit():
    text = "hello world"
    assert truncate_at_word(text, 11) == "hello world"

def test_truncate_single_long_word():
    """If no space found in reasonable range, truncate hard."""
    text = "x" * 400
    result = truncate_at_word(text, 300)
    assert len(result) == 300

def test_truncate_preserves_content():
    text = "Use --iters 500 for cloud training to prevent truncation artifacts"
    result = truncate_at_word(text, 40)
    # Should end at a word boundary within limit
    assert result[-1] != " "
    assert len(result) <= 40
