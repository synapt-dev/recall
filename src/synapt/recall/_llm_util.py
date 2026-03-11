"""Shared LLM response parsing utilities for recall modules."""

from __future__ import annotations

import json
import re


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

    # Fallback: extract nodes from markdown-formatted lists.
    # Small models sometimes output numbered lists instead of JSON:
    #   1. "content here" (category)
    #      * Category: workflow
    #      * Confidence: 0.8
    nodes = _parse_markdown_nodes(response)
    if nodes:
        return {"nodes": nodes}

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
