"""Lean code agent — browser-LLM-optimized (real-run finding, 2026-07-08).

Why this exists: the deepagents-based CodeAgent (agents/code/agent.py) injects
write_todos + subagent middleware + the full verbose fs tool suite. Against the
real browser bridge, ChatGPT free-tier choked on that oversized tool prompt
(503 CHATGPT_ERROR). This lean agent uses langgraph's create_react_agent with a
*minimal, small-schema* tool set and a short prompt, sized to fit a weak browser
model's window.

Tool set (5): list_dir, read_file, write_file (fs_tools) + syntax_check,
lint_check (staticcheck). No git here — the pipeline captures the diff and
commits; the coder's job is just read → write → self-check syntax/lint. No mypy
(slow/noisy for a weak model), no glob/grep/edit/todos/subagents.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from orchestrator._callbacks import get_tool_handler
from orchestrator._utils.model_factory import build_code_model
from orchestrator.tools.fs_tools import build_fs_tools
from orchestrator.tools.staticcheck import build_staticcheck_tools

AGENT_NAME = "lean_code_agent"

SYSTEM_PROMPT = """你是编码助手，在一个 Python 代码仓库里工作。用工具完成用户的编码任务。

规则：
1. 先用 list_dir / read_file 看清相关文件，再动手。
2. 用 write_file 写完整文件内容（不是片段）来新增或修改文件。
3. 每次 write_file 后调用 syntax_check 确认语法没错；有错就改。
4. 需要就调用 lint_check。全部通过后，用一句话说明你做了什么，结束。
保持简洁，不要空谈。"""


def build_lean_code_agent(repo: Path | str, model: BaseChatModel | None = None):
    """Build the lean ReAct code agent bound to *repo*."""
    root = Path(repo).resolve()
    model = model or build_code_model(AGENT_NAME)
    staticcheck = {t.name: t for t in build_staticcheck_tools(root)}
    tools = [
        *build_fs_tools(root),
        staticcheck["syntax_check"],
        staticcheck["lint_check"],
    ]
    agent = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
    agent = agent.with_config({"callbacks": [get_tool_handler(AGENT_NAME)]})
    return agent
