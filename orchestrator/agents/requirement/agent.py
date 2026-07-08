"""RequirementAnalyzer — first stage of the pipeline (DeepSeek/code role).

Turns a one-line natural-language ask into a structured requirement: a
clarified restatement, explicit acceptance criteria, and any ambiguities the
downstream agents (and possibly a human) should be aware of. Structured output
via json_extract (no native structured output on the shim); fail-safe to the
raw task so the pipeline never hard-stops here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from orchestrator._utils.json_extract import extract_json
from orchestrator._utils.llm_retry import ainvoke_text
from orchestrator._utils.model_factory import build_code_model

AGENT_NAME = "requirement_analyzer"

_PROMPT = """你是需求分析师。把下面这句自然语言开发需求拆解清楚。

需求：
{task}

目标仓库概览：
{repo_summary}

请**只输出一个 JSON 对象**：
{{"clarified_task": "把需求重述得更明确、可执行",
  "acceptance_criteria": ["验收标准1", "验收标准2"],
  "ambiguities": ["需要澄清的歧义点（如果没有就空数组）"]}}
"""


@dataclass
class Requirements:
    clarified_task: str
    acceptance_criteria: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)
    raw: str = ""


def _as_str_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if value:
        return [str(value)]
    return []


async def analyze_requirement(
    task: str, repo_summary: str = "", model: BaseChatModel | None = None
) -> Requirements:
    model = model or build_code_model(AGENT_NAME)
    prompt = _PROMPT.format(task=task, repo_summary=repo_summary or "(无)")
    content = await ainvoke_text(model, prompt)
    try:
        data = extract_json(content)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except ValueError:
        return Requirements(clarified_task=task, raw=content)
    return Requirements(
        clarified_task=str(data.get("clarified_task") or task),
        acceptance_criteria=_as_str_list(data.get("acceptance_criteria")),
        ambiguities=_as_str_list(data.get("ambiguities")),
        raw=content,
    )
