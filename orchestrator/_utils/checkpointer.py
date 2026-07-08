"""LangGraph checkpointer factory for AutoDevAgent.

Adapted from TestAgent's app/agents/_utils/checkpointer.py: that version backs
onto AsyncPostgresSaver against infra (Postgres + Alembic migrations 042/044)
this greenfield project doesn't have yet. Phase 1 uses an in-process
InMemorySaver instead — enough to prove resume-within-a-run works. Swap in a
persistent backend (Postgres or langgraph-checkpoint-sqlite) once cross-process
/ cross-restart resume is actually needed; call sites only use get_checkpointer(),
so the swap is local to this file.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("AGENT_CHECKPOINTER_ENABLED", "true").lower() != "false"

_checkpointer: InMemorySaver | None = None


async def get_checkpointer() -> Any | None:
    """Return (or lazily create) the module-level InMemorySaver singleton.

    Returns None when AGENT_CHECKPOINTER_ENABLED=false. Kept `async def` (even
    though InMemorySaver needs no awaited setup) so swapping in an async
    Postgres/SQLite saver later doesn't change any call site.
    """
    global _checkpointer
    if not _ENABLED:
        logger.info("[Checkpointer] disabled via AGENT_CHECKPOINTER_ENABLED=false")
        return None
    if _checkpointer is None:
        _checkpointer = InMemorySaver()
        logger.info("[Checkpointer] InMemorySaver initialized (Phase 1 — not persistent across process restarts)")
    return _checkpointer


def reset_checkpointer() -> None:
    """Reset singleton for testing."""
    global _checkpointer
    _checkpointer = None
