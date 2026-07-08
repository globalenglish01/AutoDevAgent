"""LLM token usage tracking callback — local JSONL, no MongoDB, no cost table.

Adapted from TestAgent's backend/app/agents/_callbacks/usage_tracker.py.
AutoDevAgent has no database yet, and — per docs/PoC设计书 section 7 — never
calls a paid API, so there is no real dollar cost to estimate (cost is always
0.0). This tracker exists to answer one question during development: "how
many tokens did this run burn, and on which bridge (code vs review role)?" —
appended as one JSON line per LLM call to AUTODEV_LOG_DIR/usage_logs.jsonl.

Usage:
    from orchestrator._callbacks import attach_usage_tracker
    model = attach_usage_tracker(chat_model, agent_name="hello_world_code")
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

_LOG_DIR = Path(os.environ.get("AUTODEV_LOG_DIR", "./.autodev_logs"))
_USAGE_LOG_FILE = "usage_logs.jsonl"


def _extract_token_usage(response: LLMResult) -> dict[str, int]:
    """Best-effort token usage extraction from an OpenAI-compatible response,
    as returned by the llm_bridge proxy."""
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    llm_output = response.llm_output or {}
    token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
    usage["input_tokens"] = token_usage.get("prompt_tokens", 0) or 0
    usage["output_tokens"] = token_usage.get("completion_tokens", 0) or 0
    usage["total_tokens"] = token_usage.get("total_tokens", 0) or (
        usage["input_tokens"] + usage["output_tokens"]
    )
    return usage


def _append_jsonl(filename: str, entry: dict[str, Any]) -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_LOG_DIR / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class UsageTrackerCallbackHandler(AsyncCallbackHandler):
    """Appends one JSON line per LLM call to AUTODEV_LOG_DIR/usage_logs.jsonl."""

    raise_error: bool = False  # 回调出错不打断 agent 主流程

    def __init__(self, agent_name: Optional[str] = None) -> None:
        super().__init__()
        self.agent_name = agent_name
        self._starts: dict[UUID, float] = {}

    async def on_llm_start(
        self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._starts[run_id] = time.monotonic()

    async def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            started = self._starts.pop(run_id, None)
            duration_ms = int((time.monotonic() - started) * 1000) if started else None
            usage = _extract_token_usage(response)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "agent_name": self.agent_name,
                "run_id": str(run_id),
                "duration_ms": duration_ms,
                "cost_usd": 0.0,  # AutoDevAgent never calls a paid API
                **usage,
            }
            _append_jsonl(_USAGE_LOG_FILE, entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning("UsageTracker: failed to record usage: %s", exc)


def attach_usage_tracker(model: Any, agent_name: Optional[str] = None) -> Any:
    """Attach a UsageTrackerCallbackHandler to a chat model (mutates model.callbacks)."""
    handler = UsageTrackerCallbackHandler(agent_name=agent_name)
    existing_callbacks = list(getattr(model, "callbacks", None) or [])
    existing_callbacks.append(handler)
    try:
        model.callbacks = existing_callbacks
    except Exception:  # noqa: BLE001
        logger.warning("UsageTracker: cannot mutate model.callbacks, tracking skipped")
    return model
