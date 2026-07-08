"""Human-in-the-loop 中断检测辅助函数

Ported verbatim from TestAgent's backend/app/agents/_utils/hitl.py — 供
DeployAgent（部署前必须人工审批，见 docs/PoC设计书 第5节）等未来 agent 共用：
`agent.ainvoke()` 命中 `HumanInTheLoopMiddleware` 的 interrupt() 时不会抛异常，
而是在返回结果里带上 `__interrupt__` 键。调用方必须显式检查这个键，否则会把
"等待人工审批"误判为"执行完成"。
"""

from typing import Any


def extract_interrupt(result: dict[str, Any]) -> dict[str, Any] | None:
    """从 `agent.ainvoke()` 的返回结果中提取 HITLRequest（未命中 interrupt 则返回 None）。"""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    return interrupts[0].value


def count_pending_decisions(interrupt_payload: dict[str, Any] | None) -> int:
    """HITLRequest 里等待人工提交的 decision 数量（等于 action_requests 长度）。"""
    if not interrupt_payload:
        return 0
    return len(interrupt_payload.get("action_requests", []))
