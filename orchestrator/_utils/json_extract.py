"""Robust JSON extraction from messy LLM text.

Browser-LLM shims have no native structured-output / tool-calling (see project
memory project_bridge_tool_calling_upgrade), so agents ask the model to emit
JSON in prose and we recover it here. Handles: fenced ```json blocks, a JSON
object embedded in surrounding text, and trailing commentary.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL | re.IGNORECASE)


def _find_balanced(text: str, open_ch: str, close_ch: str) -> str | None:
    """Return the first balanced {...} / [...] span in *text*, or None."""
    start = text.find(open_ch)
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> Any:
    """Extract and parse the first JSON value from *text*.

    Raises ValueError if nothing parseable is found.
    """
    if not text or not text.strip():
        raise ValueError("empty text, no JSON to extract")

    # 1) fenced code block
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2) whole string is JSON
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 3) first balanced object, then array
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        span = _find_balanced(text, open_ch, close_ch)
        if span:
            try:
                return json.loads(span)
            except json.JSONDecodeError:
                continue

    raise ValueError(f"no parseable JSON found in text: {text[:200]!r}")
