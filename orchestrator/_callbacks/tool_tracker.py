"""MCP/LangChain 工具调用追踪回调 — 本地 JSONL，无 MongoDB。

Adapted from TestAgent's backend/app/agents/_callbacks/tool_tracker.py: 去掉
MongoDB 持久化和 run_id/thread_id 关联的复杂 metadata 解析（AutoDevAgent 还没有
数据库，也没有 TestAgent 那种多用户/多项目上下文），只保留"记下每次工具调用的
入参/出参/耗时"这个核心能力，写到 AUTODEV_LOG_DIR/tool_calls.jsonl。

Usage:
    from orchestrator._callbacks import attach_tool_tracker
    model = attach_tool_tracker(model, agent_name="hello_world_code")
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

logger = logging.getLogger(__name__)

_LOG_DIR = Path(os.environ.get("AUTODEV_LOG_DIR", "./.autodev_logs"))
_TOOL_LOG_FILE = "tool_calls.jsonl"
_MAX_OUTPUT_LEN = 2000


def _append_jsonl(filename: str, entry: dict[str, Any]) -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(_LOG_DIR / filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class ToolCallTrackerCallbackHandler(AsyncCallbackHandler):
    """异步回调：在每次工具调用前后写一行 JSON 到 tool_calls.jsonl。

    一个 handler 实例可被多个 agent 共享。
    """

    raise_error: bool = False  # 回调出错不打断 agent 主流程

    def __init__(self, agent_name: Optional[str] = None) -> None:
        super().__init__()
        self.agent_name = agent_name
        self._starts: dict[UUID, dict[str, Any]] = {}

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        self._starts[run_id] = {
            "t": time.monotonic(),
            "name": serialized.get("name", "unknown_tool"),
            "input": inputs or {},
        }

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        self._record(run_id, output=output, error=None)

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._record(run_id, output=None, error=error)

    def _record(self, run_id: UUID, output: Any, error: Optional[BaseException]) -> None:
        entry_start = self._starts.pop(run_id, None)
        if not entry_start:
            return
        try:
            duration_ms = int((time.monotonic() - entry_start["t"]) * 1000)
            output_str = str(output) if output is not None else None
            if output_str and len(output_str) > _MAX_OUTPUT_LEN:
                output_str = output_str[:_MAX_OUTPUT_LEN] + "...[truncated]"
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "agent_name": self.agent_name,
                "run_id": str(run_id),
                "tool_name": entry_start["name"],
                "tool_input": entry_start.get("input"),
                "tool_output": output_str,
                "tool_error": f"{type(error).__name__}: {str(error)[:500]}" if error else None,
                "status": "error" if error else "success",
                "duration_ms": duration_ms,
            }
            _append_jsonl(_TOOL_LOG_FILE, entry)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ToolCallTracker: failed to record tool call: %s", exc)


_tool_handlers: dict[str, ToolCallTrackerCallbackHandler] = {}


def get_tool_handler(agent_name: Optional[str] = None) -> ToolCallTrackerCallbackHandler:
    """返回指定 agent 的 ToolCallTrackerCallbackHandler 单例。"""
    key = agent_name or "_default_"
    if key not in _tool_handlers:
        _tool_handlers[key] = ToolCallTrackerCallbackHandler(agent_name=agent_name)
    return _tool_handlers[key]


def attach_tool_tracker(model: Any, agent_name: Optional[str] = None) -> Any:
    """给 chat model 挂上 tool call tracker callback（直接 mutate model.callbacks）。"""
    handler = get_tool_handler(agent_name)
    existing_callbacks = list(getattr(model, "callbacks", None) or [])
    if handler not in existing_callbacks:
        existing_callbacks.append(handler)
    try:
        model.callbacks = existing_callbacks
    except Exception:  # noqa: BLE001
        logger.warning("ToolCallTracker: cannot mutate model.callbacks, tool tracking skipped")
    return model
