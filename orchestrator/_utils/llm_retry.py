"""Shared LLM-call helper with transient-error retry + content extraction.

Every agent that talks to the browser bridge should go through this. The bridge
is flaky at the substrate level (Playwright/asyncio races, CHATGPT_ERROR blips,
navigation hiccups, 5xx); a short backoff clears most of them. Centralizing the
retry here means requirement/design/review/deploy/direct_coder all get the same
resilience instead of each doing a bare .ainvoke() that dies on the first blip.
"""
from __future__ import annotations

import asyncio
import logging

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_TRANSIENT_MARKERS = (
    "sync api inside", "asyncio loop", "chatgpt_error", "deepseek",
    "service_unavailable", "timed out", "timeout", "navigat", "target closed",
    "cdp", "net::", "503", "500", "502", "504",
)


def is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


def extract_text(response) -> str:
    """Flatten a model response's content to a plain string."""
    content = getattr(response, "content", str(response))
    if isinstance(content, list):
        return " ".join(
            str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content
        )
    return content


async def ainvoke_text(
    model: BaseChatModel, prompt, *, max_attempts: int = 3, base_delay: float = 3.0
) -> str:
    """Invoke *model* with *prompt*, retrying transient bridge errors, return text."""
    last: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return extract_text(await model.ainvoke(prompt))
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < max_attempts and is_transient(exc):
                logger.warning("[llm_retry] transient error (attempt %d): %s",
                               attempt, str(exc)[:120])
                await asyncio.sleep(base_delay * attempt)
                continue
            raise
    raise last  # unreachable
