"""ReviewAgent — dedicated code reviewer on the ChatGPT bridge role.

Design section 3.1: a *different* model from the one that wrote the code, so
two independent viewpoints check each other. It reviews a diff against the task
intent and returns a structured verdict (approved + concrete issues). Runs
*after* StaticCheck, so it never wastes a 5-6 minute browser round on syntax/
type errors the deterministic tools already catch — it looks only for the
semantic/logic/design problems that need understanding.

Not a deep agent with tools: weak browser models are far more reliable asked to
read a diff and give a one-word verdict than to orchestrate tool calls.

Verdict protocol (real-run finding 2026-07-09): asking for a JSON object was too
fragile — ChatGPT frequently returned prose the parser couldn't read, and the
conservative fail-safe then blocked correct, test-passing code. Same lesson as
the code step: don't demand structured JSON from a weak browser model. So the
reviewer answers with a bare first-line keyword APPROVE / REJECT; we parse that
(with a JSON fallback for backward compatibility).
"""
from __future__ import annotations

import re
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

【输出格式 — 必须严格遵守，二选一】
- 若改动可以通过审查：**第一行只写一个词** APPROVE （后面不要再写别的）
- 若有必须修复的问题：**第一行只写一个词** REJECT ，从第二行起逐条列出问题（每行一条，要具体）
不要输出 JSON、表格、代码块或其他任何格式；第一行必须是 APPROVE 或 REJECT。
"""

_VERDICT_RE = re.compile(r"\b(APPROVE|REJECT)\b", re.IGNORECASE)
# Chinese/informal verdict signals (checked only if no keyword/JSON). Reject
# signals win over approve signals when both appear.
_ZH_REJECT = ("驳回", "拒绝", "不通过", "不能通过", "有问题", "存在问题", "需要修改", "需修改", "必须修复")
_ZH_APPROVE = ("通过审查", "可以通过", "审查通过", "没有问题", "无问题", "没问题", "同意", "批准", "lgtm")


@dataclass
class ReviewVerdict:
    approved: bool
    issues: list[str] = field(default_factory=list)
    raw: str = ""  # raw model text, kept for debugging
    # True when the reviewer produced no readable verdict (not an explicit
    # rejection). The pipeline defers these to the deterministic test gate
    # rather than blocking correct code on a flaky review.
    inconclusive: bool = False

    @property
    def summary(self) -> str:
        if self.approved:
            return "APPROVED"
        if self.inconclusive:
            return "INCONCLUSIVE: " + "; ".join(self.issues)
        return "CHANGES_REQUESTED: " + "; ".join(self.issues)


def _parse_verdict(text: str) -> ReviewVerdict:
    stripped = (text or "").strip()
    if not stripped:
        return ReviewVerdict(approved=False, issues=["审查无输出"], raw=text)

    # 1) Primary: bare APPROVE / REJECT keyword (robust for weak browser models).
    m = _VERDICT_RE.search(stripped)
    if m:
        if m.group(1).upper() == "APPROVE":
            return ReviewVerdict(approved=True, issues=[], raw=text)
        # REJECT — everything after the keyword line is the issue list.
        after = stripped[m.end():].strip()
        issues = [ln.strip("-*• \t") for ln in after.splitlines() if ln.strip()]
        return ReviewVerdict(approved=False, issues=issues or ["审查驳回（未给出具体原因）"], raw=text)

    # 2) Fallback: a JSON object (older prompt / model habit).
    try:
        data = extract_json(stripped)
        if isinstance(data, dict):
            approved = bool(data.get("approved", False))
            issues_raw = data.get("issues", []) or []
            issues = [str(i) for i in issues_raw] if isinstance(issues_raw, list) else [str(issues_raw)]
            return ReviewVerdict(approved=approved, issues=issues, raw=text)
    except ValueError:
        pass

    # 3) Chinese/informal prose signals (reject wins over approve).
    low = stripped.lower()
    if any(sig in stripped for sig in _ZH_REJECT):
        return ReviewVerdict(approved=False, issues=[stripped[:300]], raw=text)
    if any(sig in stripped or sig in low for sig in _ZH_APPROVE):
        return ReviewVerdict(approved=True, issues=[], raw=text)

    # 4) Inconclusive: no readable verdict. NOT an explicit rejection — mark
    # inconclusive so the pipeline defers to the test gate instead of blocking
    # correct code on a flaky review (real-run finding 2026-07-09).
    return ReviewVerdict(
        approved=False,
        inconclusive=True,
        issues=["审查结论无法解析（无关键词/JSON/中文判定词），交由测试环节兜底"],
        raw=text,
    )


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
