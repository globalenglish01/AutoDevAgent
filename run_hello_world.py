"""Phase 1 手动验证入口 — 对着真实 llm_bridge 跑一次 hello world agent。

前置条件：先在 D:\\TestAgentPythonProject\\backend\\llm_bridge 下启动 DeepSeek 垫片
（ModelRole.CODE 默认打 http://127.0.0.1:8765/v1）：

    cd D:\\TestAgentPythonProject\\backend\\llm_bridge
    python start_llm_proxy.py --provider deepseek --account 1 --port 8765

用法：
    python run_hello_world.py
"""
from __future__ import annotations

import asyncio

from orchestrator._utils.workspace import cleanup_run_workspace
from orchestrator.agents.hello_world.agent import build_hello_world_agent


async def main() -> None:
    agent, run_id, workspace_path = await build_hello_world_agent()
    (workspace_path / "hello.txt").write_text(
        "Hello from AutoDevAgent Phase 1!\n", encoding="utf-8"
    )
    try:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "读一下 hello.txt 的内容"}]},
            config={"configurable": {"thread_id": run_id}},
        )
        print(result["messages"][-1].content)
    finally:
        cleanup_run_workspace(workspace_path)


if __name__ == "__main__":
    asyncio.run(main())
