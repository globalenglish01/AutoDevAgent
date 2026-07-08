"""CodeAgent skeleton (Phase 2).

The first agent that operates on a *real target repository* rather than a
throwaway workspace. It wires together:
  - deepagents' built-in fs tools (read_file/write_file/edit_file/ls/glob/grep),
    scoped to the target repo by FilesystemBackend(virtual_mode=True)
  - the Phase 2 StaticCheck tools (syntax/lint/typecheck) and git tools
    (status/diff/create_branch/commit) from orchestrator.tools

Phase 2 scope: prove the agent can read code, write code, run static checks,
and commit on a feature branch. It is NOT yet the full CodeAgent from
docs/PoC设计书 section 3.1 — there is no DesignAgent feeding it a change plan
and no ReviewAgent (ChatGPT) downstream yet. Those arrive in Phase 3+.

Unlike hello_world, this does not create an isolated empty workspace: the
backend points straight at the target repo so the agent edits real files.
Path safety comes from virtual_mode (blocks traversal) plus the path_guard
inside the git/staticcheck tools (blocks sensitive filenames).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.language_models import BaseChatModel

from orchestrator._callbacks import get_tool_handler
from orchestrator._utils.checkpointer import get_checkpointer
from orchestrator._utils.model_factory import build_code_model
from orchestrator.tools import build_git_tools, build_staticcheck_tools

AGENT_NAME = "code_agent"

SYSTEM_PROMPT = """你是 AutoDevAgent 的编码智能体（CodeAgent）。你在一个真实的代码仓库里工作。

工作纪律：
1. 动手改代码前，先用 ls / read_file / grep 了解相关文件，不要凭空猜测。
2. 开始编码前，先用 git_create_branch 切到一个 feature 分支（禁止直接在 main/master 上改）。
3. 用 write_file / edit_file 修改代码，按文件或逻辑单元小步改，不要一次性大改。
4. 每改完一个文件，立刻按顺序自检：先 syntax_check（最快），过了再 lint_check，最后 typecheck_run。
   任何一步 FAIL，都要先根据报错把代码改对，再继续，不要留着错误往下走。
5. 全部静态检查通过后，用 git_commit 提交，commit message 用中文简述改了什么。
6. 绝不读取或写入 .env、*.pem、私钥等敏感文件——工具会拒绝，你也不要尝试。
"""


async def build_code_agent(
    target_repo: Path | str,
    model: BaseChatModel | None = None,
) -> Any:
    """Assemble the CodeAgent bound to *target_repo* (must be an existing git repo).

    Returns the compiled agent (ready for .ainvoke). The agent edits files in
    target_repo directly, so pass a repo you are willing to have modified — in
    real use this is a fresh clone / worktree of the target project.
    """
    root = Path(target_repo).resolve()
    model = model or build_code_model(AGENT_NAME)
    backend = FilesystemBackend(root_dir=root, virtual_mode=True)

    tools = [*build_staticcheck_tools(root), *build_git_tools(root)]

    checkpointer = await get_checkpointer()
    agent = create_deep_agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
        backend=backend,
        checkpointer=checkpointer,
        name="code_agent",
    )
    # Tool-call tracking must be registered at the graph level (see
    # orchestrator._callbacks.tool_tracker.get_tool_handler docstring).
    agent = agent.with_config({"callbacks": [get_tool_handler(AGENT_NAME)]})
    return agent
