"""Shared LLM response parsing utilities for recall modules."""

from __future__ import annotations

import json


def parse_llm_json(response: str) -> dict | None:
    """Parse an LLM's JSON response, handling common formatting issues.

    Handles:
    - Clean JSON
    - JSON wrapped in markdown code fences
    - JSON embedded in surrounding text

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

    return None
