"""
utils.py
--------
Pure helper functions shared across modules.
"""

import json
import re
from typing import Any


def extract_json(text: str) -> dict | list | None:
    """
    Try to parse JSON from an LLM response.

    First attempts a direct parse; falls back to extracting the first
    {...} block found in the string.
    """
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


def clean_id(text: str, index: int) -> str:
    """
    Convert an arbitrary string into a safe node identifier.

    e.g. ``clean_id("Define the Problem!", 0)`` → ``"define_the_problem_0"``
    """
    cleaned = re.sub(r"[^a-z0-9 ]", "", text.lower())
    words = cleaned.split()[:3]
    return "_".join(words) + f"_{index}"


def summarize_spec(spec: dict) -> list[str]:
    """Return a flat list of node purposes from a graph spec."""
    return [n["purpose"] for n in spec.get("nodes", [])]


def truncate(obj: Any, max_chars: int = 1500) -> str:
    """JSON-serialise *obj* and truncate to *max_chars* characters."""
    return json.dumps(obj)[:max_chars]
