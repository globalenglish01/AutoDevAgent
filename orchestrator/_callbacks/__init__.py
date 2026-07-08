"""Agent 通用回调（usage tracking、tool call tracking），本地 JSONL 落盘，无 MongoDB。

对应 TestAgent 的 backend/app/agents/_callbacks/，公开接口保持一致
（attach_usage_tracker / attach_tool_tracker / get_tool_handler），但持久化目标
从 MongoDB 换成本地 JSONL 文件——AutoDevAgent 目前没有数据库基础设施，且永远不
调用付费 API（见 docs/PoC设计书 第7节），没有真实美元成本需要估算。
"""

from .usage_tracker import UsageTrackerCallbackHandler, attach_usage_tracker
from .tool_tracker import ToolCallTrackerCallbackHandler, attach_tool_tracker, get_tool_handler

__all__ = [
    "UsageTrackerCallbackHandler",
    "attach_usage_tracker",
    "ToolCallTrackerCallbackHandler",
    "attach_tool_tracker",
    "get_tool_handler",
]
