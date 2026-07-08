"""Phase 2 integration test — CodeAgent wiring.

Does NOT hit a real llm_bridge (see tests/test_hello_world.py for why). Uses a
fake model that plays back a tool_call to the StaticCheck `syntax_check` tool,
proving our custom Phase 2 tools are actually reachable as agent tools through
create_deep_agent alongside deepagents' built-in fs tools. Real-bridge runs
happen manually once the DeepSeek bridge is up.
"""
from __future__ import annotations

import subprocess

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from orchestrator._utils.checkpointer import reset_checkpointer
from orchestrator.agents.code.agent import build_code_agent


class _ToolCapableFakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel doesn't implement bind_tools; deepagents always
    calls it. Our canned messages already carry the tool_calls, so this is a
    no-op passthrough."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@pytest.fixture(autouse=True)
def _reset_checkpointer():
    reset_checkpointer()
    yield
    reset_checkpointer()


@pytest.fixture
def sample_repo(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "sample.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    return tmp_path


@pytest.mark.asyncio
async def test_code_agent_can_call_staticcheck_tool(sample_repo):
    fake = _ToolCapableFakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "syntax_check", "args": {"path": "sample.py"}, "id": "c1"}
                    ],
                ),
                AIMessage(content="syntax_check 通过，代码语法正确。"),
            ]
        )
    )

    agent = await build_code_agent(sample_repo, model=fake)
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "检查 sample.py 的语法"}]},
        config={"configurable": {"thread_id": "test-code-1"}},
    )

    # The syntax_check ToolMessage should be in the transcript and report PASS.
    tool_messages = [m for m in result["messages"] if getattr(m, "name", None) == "syntax_check"]
    assert tool_messages, "expected a syntax_check ToolMessage in the transcript"
    assert tool_messages[0].content.startswith("PASS")
    assert "语法正确" in result["messages"][-1].content
