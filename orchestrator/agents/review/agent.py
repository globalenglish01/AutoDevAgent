"""ReviewAgent — dedicated code reviewer on the ChatGPT bridge role.

Design section 3.1: a *different* model from the one that wrote the code, so
two independent viewpoints check each other. It reviews a diff against the task
intent and returns a structured verdict (approved + concrete issues). Runs
*after* StaticCheck, so it never wastes a 5-6 minute browser round on syntax/
type errors the deterministic tools already catch — it looks only for the
semantic/logic/design problems that need understanding.

Not a deep agent with tools: weak browser models are far more reliable asked to
read a diff and emit a JSON verdict than to orchestrate tool calls. Structured
output is recovered with json_extract (no native structured output on the shim).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel

from orchestrator._utils.json_extract import extract_json
from orchestrator._utils.llm_retry import ainvoke_text
from orchestrator._utils.model_factory import build_review_model

AGENT_NAME = "review_agent"

_REVIEW_PROMPT = """你是一名严格的代码审查员。下面是一次针对某个开发任务的代码改动（unified diff）。
你的任务：只找**真正的问题**——逻辑错误、遗漏的边界情况、与任务意图不符、明显的安全隐患、
明显的设计不一致。不要吹毛求疵纯风格问题（那些已由静态检查处理）。

开发任务：
{task}

代码改动（diff）：
{diff}

请**只输出一个 JSON 对象**，不要有其他文字，格式：
{{"approved": true/false, "issues": ["具体问题1", "具体问题2"]}}
- approved=true 表示这次改动可以通过审查（issues 为空数组）。
- approved=false 表示有必须修复的问题，issues 列出每个问题（要具体，能让编码者据此定位修改）。
"""


@dataclass
class ReviewVerdict:
    approved: bool
    issues: list[str] = field(default_factory=list)
    raw: str = ""  # raw model text, kept for debugging

    @property
    def summary(self) -> str:
        if self.approved:
            return "APPROVED"
        return "CHANGES_REQUESTED: " + "; ".join(self.issues)


def _parse_verdict(text: str) -> ReviewVerdict:
    try:
        data = extract_json(text)
    except ValueError:
        # If the model didn't emit parseable JSON, fail safe: treat as
        # changes-requested so a human/loop notices rather than silently passing.
        return ReviewVerdict(
            approved=False,
            issues=["审查模型未返回可解析的 JSON 结论，无法确认改动安全"],
            raw=text,
        )
    if not isinstance(data, dict):
        return ReviewVerdict(approved=False, issues=["审查结论格式非对象"], raw=text)
    approved = bool(data.get("approved", False))
    issues_raw = data.get("issues", []) or []
    issues = [str(i) for i in issues_raw] if isinstance(issues_raw, list) else [str(issues_raw)]
    return ReviewVerdict(approved=approved, issues=issues, raw=text)


async def review_diff(diff: str, task: str, model: BaseChatModel | None = None) -> ReviewVerdict:
    """Review *diff* against *task* and return a structured verdict.

    Uses the ChatGPT bridge role by default; pass *model* to inject a fake in tests.
    """
    if not diff or not diff.strip():
        return ReviewVerdict(approved=True, issues=[], raw="(empty diff)")
    model = model or build_review_model(AGENT_NAME)
    prompt = _REVIEW_PROMPT.format(task=task, diff=diff)
    content = await ainvoke_text(model, prompt)
    return _parse_verdict(content)
