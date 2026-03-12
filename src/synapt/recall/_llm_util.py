"""Shared LLM response parsing utilities for recall modules."""

from __future__ import annotations

import json
import re


def truncate_at_word(text: str, max_chars: int) -> str:
    """Truncate text at a word boundary to avoid mid-word cuts.

    Returns the original text if it's already within the limit.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > max_chars * 0.7:
        return truncated[:last_space]
    return truncated


def _is_node_array(data: object) -> bool:
    """Check if data is a non-empty list of dicts (i.e. a knowledge node array)."""
    return isinstance(data, list) and bool(data) and isinstance(data[0], dict)


def parse_llm_json(response: str) -> dict | None:
    """Parse an LLM's JSON response, handling common formatting issues.

    Handles:
    - Clean JSON
    - JSON wrapped in markdown code fences
    - JSON embedded in surrounding text
    - Markdown-formatted lists (fallback for small models that ignore JSON instructions)

    Returns the parsed dict, or None if parsing fails.
    """
    response = response.strip()
    # Strip markdown code fences if present (only first/last fence lines)
    if response.startswith("```"):
        lines = response.split("\n")
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        response = "\n".join(lines).strip()

    try:
        data = json.loads(response)
        if isinstance(data, dict):
            return data
        if _is_node_array(data):
            return {"nodes": data}
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the response
    start = response.find("{")
    end = response.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(response[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Try to find JSON array in the response
    arr_start = response.find("[")
    arr_end = response.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        try:
            data = json.loads(response[arr_start : arr_end + 1])
            if _is_node_array(data):
                return {"nodes": data}
        except json.JSONDecodeError:
            pass

    # Repair truncated JSON — small models often run out of tokens mid-response.
    # Try to extract complete node objects from a truncated {"nodes": [...]}
    if start >= 0:
        repaired = _repair_truncated_json(response[start:])
        if repaired is not None:
            return repaired

        # Try to salvage key-value pairs from a truncated dict
        # (e.g. {"focus": "Did stuff", "done": ["Thing 1"], "deci...)
        repaired = _repair_truncated_dict(response[start:])
        if repaired is not None:
            return repaired

    # Fallback: extract nodes from markdown-formatted lists.
    # Small models sometimes output numbered lists instead of JSON:
    #   1. "content here" (category)
    #      * Category: workflow
    #      * Confidence: 0.8
    nodes = _parse_markdown_nodes(response)
    if nodes:
        return {"nodes": nodes}

    return None


def _find_balanced_end(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Find the position of the matching close delimiter, respecting strings.

    Returns the index of the closing delimiter, or -1 if the text is truncated.
    ``start`` should point to the opening delimiter character.
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _repair_truncated_json(text: str) -> dict | None:
    """Try to salvage nodes from truncated JSON output.

    When a small model runs out of tokens, the JSON is cut mid-response:
        {"nodes": [{"action": "create", "content": "fact1", ...}, {"action": "cre

    Strategy: find the "nodes" array start, then extract each complete
    {...} object at array depth (skipping the outer wrapper).
    """
    # Find the start of the nodes array
    nodes_idx = text.find('"nodes"')
    if nodes_idx < 0:
        return None
    arr_start = text.find("[", nodes_idx)
    if arr_start < 0:
        return None

    # Walk through the array, extracting complete objects
    nodes = []
    i = arr_start + 1
    while i < len(text):
        ch = text[i]
        if ch == '{':
            end = _find_balanced_end(text, i, '{', '}')
            if end < 0:
                break  # Truncated
            candidate = text[i : end + 1]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict) and "content" in obj:
                    nodes.append(obj)
            except json.JSONDecodeError:
                pass
            i = end + 1
        elif ch == ']':
            break  # End of array
        else:
            i += 1

    if nodes:
        return {"nodes": nodes}
    return None


def _repair_truncated_dict(text: str) -> dict | None:
    """Salvage complete key-value pairs from a truncated JSON dict.

    When a small model runs out of tokens, the JSON is cut mid-value::

        {"focus": "Did stuff", "done": ["Thing 1", "Thing 2"], "deci

    Strategy: extract each complete ``"key": value`` pair (string or
    array values) by walking braces/brackets with string awareness.
    Returns the dict of all complete pairs, or None if fewer than one
    useful key was recovered.
    """
    if not text.startswith("{"):
        return None

    result: dict = {}

    # Extract complete "key": "value" string pairs
    for m in re.finditer(r'"(\w+)"\s*:\s*"((?:[^"\\]|\\.)*)"', text):
        result[m.group(1)] = m.group(2)

    # Extract complete "key": [...] array pairs via bracket balancing
    for m in re.finditer(r'"(\w+)"\s*:\s*\[', text):
        key = m.group(1)
        arr_start = m.end() - 1  # position of [
        arr_end = _find_balanced_end(text, arr_start, '[', ']')
        if arr_end > 0:
            try:
                arr = json.loads(text[arr_start : arr_end + 1])
                result[key] = arr
            except json.JSONDecodeError:
                pass

    # Only return if we got at least one useful enrichment key
    if any(k in result for k in ("focus", "done", "content", "nodes")):
        return result
    return None


def _parse_markdown_nodes(text: str) -> list[dict]:
    """Extract knowledge nodes from markdown-formatted LLM output.

    Handles patterns like:
        1. "Train on Alfred, test on Batman" (convention)
           * Category: convention
           * Confidence: 0.8
           * Tags: ["training", "eval"]
    """
    nodes = []
    # Split into numbered items (first element is text before item 1 — skip it)
    items = re.split(r'\n\s*\d+\.\s+', text)
    for item in items[1:]:
        if not item.strip():
            continue
        # Extract quoted content
        content_match = re.search(r'"([^"]+)"', item)
        if not content_match:
            # Try unquoted: first line up to parenthetical
            first_line = item.split('\n')[0].strip()
            content_match = re.match(r'(.+?)(?:\s*\(|$)', first_line)
            if not content_match or len(content_match.group(1)) < 5:
                continue
            content = content_match.group(1).strip(' "')
        else:
            content = content_match.group(1)

        if len(content) < 5:
            continue

        node: dict = {
            "action": "create",
            "existing_id": None,
            "content": content,
            "category": "workflow",
            "confidence": 0.6,
            "tags": [],
            "contradiction_note": "",
        }

        # Extract category
        cat_match = re.search(
            r'[Cc]ategory:\s*(\w[\w\s_-]*)', item,
        )
        if cat_match:
            node["category"] = cat_match.group(1).strip().lower()

        # Extract confidence
        conf_match = re.search(r'[Cc]onfidence:\s*([\d.]+)', item)
        if conf_match:
            try:
                node["confidence"] = min(1.0, max(0.0, float(conf_match.group(1))))
            except ValueError:
                pass

        # Extract tags
        tags_match = re.search(r'[Tt]ags:\s*\[([^\]]*)\]', item)
        if tags_match:
            tags_str = tags_match.group(1)
            node["tags"] = [
                t.strip().strip('"\'')
                for t in tags_str.split(",")
                if t.strip().strip('"\'')
            ]

        # Extract action (corroborate/contradict)
        action_match = re.search(
            r'[Aa]ction:\s*(create|corroborate|contradict)', item,
        )
        if action_match:
            node["action"] = action_match.group(1).lower()

        nodes.append(node)

    return nodes
