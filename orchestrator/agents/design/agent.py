"""DesignAgent — second stage (DeepSeek/code role).

Turns structured requirements + a repo map into a concrete change plan: which
files to touch and the ordered steps to implement. The plan is injected into
the CodeAgent's prompt so it doesn't have to (re)derive the approach itself —
important with a weak model. Fail-safe to a minimal plan.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from orchestrator._utils.json_extract import extract_json
from orchestrator._utils.model_factory import build_code_model

AGENT_NAME = "design_agent"

_PROMPT = """你是软件设计师。根据需求和仓库概览，给出一个具体、可执行的改动方案。

明确后的需求：
{clarified_task}

验收标准：
{acceptance_criteria}

目标仓库概览：
{repo_summary}

请**只输出一个 JSON 对象**：
{{"summary": "方案总述（一两句话）",
  "files": ["预计要新增/修改的文件路径"],
  "steps": ["实现步骤1", "实现步骤2"]}}
"""


@dataclass
class Design:
    summary: str
    files: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    raw: str = ""

    def as_prompt_block(self) -> str:
        """Render the design as a block to inject into the CodeAgent prompt."""
        lines = [f"改动方案：{self.summary}"]
        if self.files:
            lines.append("涉及文件：\n- " + "\n- ".join(self.files))
        if self.steps:
            lines.append("实现步骤：\n- " + "\n- ".join(self.steps))
        return "\n".join(lines)


def _as_str_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value:
        return [str(value)]
    return []


async def make_design(
    clarified_task: str,
    acceptance_criteria: list[str] | None = None,
    repo_summary: str = "",
    model: BaseChatModel | None = None,
) -> Design:
    model = model or build_code_model(AGENT_NAME)
    criteria = "\n- ".join(acceptance_criteria or []) or "(无)"
    prompt = _PROMPT.format(
        clarified_task=clarified_task,
        acceptance_criteria=criteria,
        repo_summary=repo_summary or "(无)",
    )
    response = await model.ainvoke(prompt)
    content = getattr(response, "content", str(response))
    if isinstance(content, list):
        content = " ".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)
    try:
        data = extract_json(content)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except ValueError:
        return Design(summary=clarified_task, raw=content)
    return Design(
        summary=str(data.get("summary") or clarified_task),
        files=_as_str_list(data.get("files")),
        steps=_as_str_list(data.get("steps")),
        raw=content,
    )
