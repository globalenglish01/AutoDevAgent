"""Phase 1 proof-of-life agent.

Not a real capability from docs/PoC设计书 section 3.1 (RequirementAnalyzer/
DesignAgent/CodeAgent/ReviewAgent/VerifyAgent/DeployAgent all come in later
phases). Its only job is to prove the ported skeleton wires together end to
end through deepagents.create_deep_agent:

  - orchestrator._utils.model_factory  -> a chat model bound to the llm_bridge
  - orchestrator._utils.workspace      -> an isolated per-run FilesystemBackend
  - orchestrator._utils.checkpointer   -> resumable graph state
  - orchestrator._callbacks            -> usage/tool-call logging

It asks the agent to read a file through the isolated workspace using
deepagents' own built-in `read_file` tool (FilesystemMiddleware, wired
automatically whenever a `backend` is passed to create_deep_agent) — the same
primitive CodeAgent will build on later, so this is a real exercise of the
workspace-isolation piece, not just a "did it import" check.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain_core.language_models import BaseChatModel

from orchestrator._callbacks import get_tool_handler
from orchestrator._utils.checkpointer import get_checkpointer
from orchestrator._utils.model_factory import ModelRole, build_model
from orchestrator._utils.workspace import create_run_workspace

AGENT_NAME = "hello_world"

SYSTEM_PROMPT = (
    "你是 AutoDevAgent Phase 1 的骨架验证智能体。"
    "用户会告诉你一个文件名，你必须调用 read_file 工具读出它的内容，"
    "然后原样复述给用户，不要额外加工或解释。"
)

WORKSPACE_ROOT = Path(__file__).resolve().parent / "workspace_runs"


async def build_hello_world_agent(
    model: BaseChatModel | None = None,
) -> tuple[Any, str, Path]:
    """组装最小 deep agent：真实模型工厂 + 隔离工作区 + checkpointer。

    Returns:
        (agent, run_id, workspace_path) — 调用方跑完后应调用
        cleanup_run_workspace(workspace_path) 清理隔离目录。
    """
    model = model or build_model(AGENT_NAME, role=ModelRole.CODE)
    run_id, workspace_path, backend = create_run_workspace(
        WORKSPACE_ROOT, agent_name=AGENT_NAME
    )
    checkpointer = await get_checkpointer()
    agent = create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        checkpointer=checkpointer,
        name="hello_world_agent",
    )
    # Tool-call tracking must be registered at the graph level: LangGraph's
    # ToolNode does not consult model-level callbacks (see
    # orchestrator._callbacks.tool_tracker.get_tool_handler docstring).
    agent = agent.with_config({"callbacks": [get_tool_handler(AGENT_NAME)]})
    return agent, run_id, workspace_path
