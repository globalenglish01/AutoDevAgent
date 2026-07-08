"""Provider factory for AutoDevAgent — HTTP bridge only, no paid API path.

Differs from TestAgent's app/agents/_utils/model_factory.py on purpose:
that file's "browser:" provider does an in-process `sys.path` import of
TestAgent's backend/local_llm/langchain_browser_llm.py. AutoDevAgent must stay
independent of TestAgent's file layout (docs/PoC设计书 section 2), so instead
this module talks over HTTP to the llm_bridge OpenAI-compatible proxy
(backend/llm_bridge/start_llm_proxy.py) — the two projects share a *running
service*, not Python source.

Dual-role routing (docs/PoC设计书 section 3.1 — "DeepSeek 主写 + ChatGPT 专职审"):
  ModelRole.CODE   -> DeepSeek bridge. Long-context primary generation model,
                      used by RequirementAnalyzer / DesignAgent / CodeAgent / DeployAgent.
  ModelRole.REVIEW -> ChatGPT bridge. Dedicated review model, used only by
                      ReviewAgent to catch issues CodeAgent's own model would miss.

There is no paid-API branch here — per docs/PoC设计书 section 7 (confirmed
2026-07-08), AutoDevAgent never calls a paid LLM API. Both roles resolve to a
local llm_bridge proxy instance you must start yourself:

    cd D:\\TestAgentPythonProject\\backend\\llm_bridge
    python start_llm_proxy.py --provider deepseek --account 1 --port 8765
    python start_llm_proxy.py --provider chatgpt  --account 1 --port 8766
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from orchestrator._callbacks import attach_tool_tracker, attach_usage_tracker

logger = logging.getLogger(__name__)


class ModelRole(str, Enum):
    """Which job a model call is doing — decides which bridge instance it hits."""

    CODE = "code"
    REVIEW = "review"


_DEFAULT_BRIDGE_URL: dict[ModelRole, str] = {
    ModelRole.CODE: "http://127.0.0.1:8765/v1",
    ModelRole.REVIEW: "http://127.0.0.1:8766/v1",
}
_BRIDGE_URL_ENV: dict[ModelRole, str] = {
    ModelRole.CODE: "AUTODEV_CODE_BRIDGE_URL",
    ModelRole.REVIEW: "AUTODEV_REVIEW_BRIDGE_URL",
}
# The bridge does not check the API key, but ChatOpenAI refuses an empty string.
_PLACEHOLDER_API_KEY = "browser-llm"
_BRIDGE_MODEL_NAME = "browser-llm"

# Browser LLM round-trips run ~5-6 minutes per turn (see project memory
# project_llm_bridge_direction_a) — a short timeout would kill every real call.
DEFAULT_BRIDGE_TIMEOUT_SECONDS = 600


def bridge_url_for(role: ModelRole) -> str:
    """Resolve the bridge base URL for a role, overridable via env var."""
    return os.environ.get(_BRIDGE_URL_ENV[role], _DEFAULT_BRIDGE_URL[role])


def build_model(
    agent_name: str,
    *,
    role: ModelRole = ModelRole.CODE,
    timeout: int = DEFAULT_BRIDGE_TIMEOUT_SECONDS,
    extra_kwargs: dict[str, Any] | None = None,
) -> BaseChatModel:
    """Build a chat model bound to the llm_bridge proxy for the given role.

    Usage/tool tracking (local JSONL, see _callbacks) is attached automatically.
    """
    base_url = bridge_url_for(role)
    kwargs: dict[str, Any] = extra_kwargs or {}
    logger.info(
        "[ModelFactory] %s -> role=%s bridge=%s", agent_name, role.value, base_url
    )
    model: BaseChatModel = ChatOpenAI(
        base_url=base_url,
        api_key=_PLACEHOLDER_API_KEY,
        model=_BRIDGE_MODEL_NAME,
        timeout=timeout,
        max_retries=1,
        **kwargs,
    )
    tracker_name = f"{agent_name}_{role.value}"
    model = attach_usage_tracker(model, agent_name=tracker_name)
    model = attach_tool_tracker(model, agent_name=tracker_name)
    return model


def build_code_model(agent_name: str) -> BaseChatModel:
    """Convenience wrapper — DeepSeek bridge, primary generation role."""
    return build_model(agent_name, role=ModelRole.CODE)


def build_review_model(agent_name: str) -> BaseChatModel:
    """Convenience wrapper — ChatGPT bridge, dedicated review role."""
    return build_model(agent_name, role=ModelRole.REVIEW)
