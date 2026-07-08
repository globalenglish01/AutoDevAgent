"""Phase 1 wiring test.

Does NOT hit a real llm_bridge (those round-trips take 5-6 minutes and require
a logged-in browser session — see docs/PoC设计书). Instead swaps in a
GenericFakeChatModel that plays back a canned tool_call + final answer, so we
can prove the actual skeleton wiring (deepagents.create_deep_agent + isolated
FilesystemBackend workspace + InMemorySaver checkpointer + usage/tool
tracking callbacks) works end to end. Real-bridge verification is a separate,
manual step via run_hello_world.py once DeepSeek/ChatGPT bridges are running.
"""
from __future__ import annotations

import json

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from orchestrator._utils.checkpointer import reset_checkpointer
from orchestrator._utils.workspace import cleanup_run_workspace
from orchestrator.agents.hello_world.agent import build_hello_world_agent


class _ToolCapableFakeChatModel(GenericFakeChatModel):
    """GenericFakeChatModel doesn't implement bind_tools (raises
    NotImplementedError), but deepagents always calls it to attach the
    filesystem/todo tool schemas. Since our canned messages already carry the
    tool_calls we want, bind_tools here is a no-op passthrough."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@pytest.fixture(autouse=True)
def _reset_checkpointer():
    reset_checkpointer()
    yield
    reset_checkpointer()


def _fake_model() -> _ToolCapableFakeChatModel:
    """Plays back: 1) a tool_call to read_file, 2) a final text answer."""
    canned = iter(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "/hello.txt"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="Hello from AutoDevAgent Phase 1!"),
        ]
    )
    return _ToolCapableFakeChatModel(messages=canned)


@pytest.mark.asyncio
async def test_hello_world_agent_reads_isolated_workspace_file():
    agent, run_id, workspace_path = await build_hello_world_agent(model=_fake_model())
    try:
        (workspace_path / "hello.txt").write_text(
            "Hello from AutoDevAgent Phase 1!\n", encoding="utf-8"
        )

        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "读一下 hello.txt 的内容"}]},
            config={"configurable": {"thread_id": run_id}},
        )

        final_message = result["messages"][-1]
        assert "Hello from AutoDevAgent Phase 1!" in final_message.content

        # Prove the read_file tool actually touched the isolated workspace file
        # (not some other path) by checking the ToolMessage that preceded the
        # final answer came back successful.
        tool_messages = [m for m in result["messages"] if getattr(m, "name", None) == "read_file"]
        assert tool_messages, "expected a read_file ToolMessage in the transcript"
        assert "Hello from AutoDevAgent Phase 1!" in tool_messages[0].content
    finally:
        cleanup_run_workspace(workspace_path)


@pytest.mark.asyncio
async def test_tool_call_is_logged_to_local_jsonl(tmp_path, monkeypatch):
    # Point the tool tracker's log dir at a temp dir so this test doesn't
    # pollute .autodev_logs/ and can assert on exact content.
    import orchestrator._callbacks.tool_tracker as tool_tracker_module

    monkeypatch.setattr(tool_tracker_module, "_LOG_DIR", tmp_path)

    agent, run_id, workspace_path = await build_hello_world_agent(model=_fake_model())
    try:
        (workspace_path / "hello.txt").write_text("content\n", encoding="utf-8")
        await agent.ainvoke(
            {"messages": [{"role": "user", "content": "读一下 hello.txt 的内容"}]},
            config={"configurable": {"thread_id": run_id}},
        )
    finally:
        cleanup_run_workspace(workspace_path)

    log_file = tmp_path / "tool_calls.jsonl"
    assert log_file.exists(), "tool_tracker should have written tool_calls.jsonl"
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "tool_calls.jsonl should have at least one entry"
    entry = json.loads(lines[-1])
    assert entry["tool_name"] == "read_file"
    assert entry["status"] == "success"
