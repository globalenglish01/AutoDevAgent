"""Pipeline state machine: init → code → staticcheck → review → verify → END.

Implements the closed loop of docs/PoC设计书 section 3.1. Every gate that fails
routes back to `code` with concrete feedback, up to max_code_retries; when
retries are exhausted the run stops with status="needs_human" (design section
5.5 — escalate rather than ship something unverified).

The `code` step is dependency-injected as a `coder` callable so the control
flow can be tested deterministically without driving a 5-6 minute browser LLM.
In production the coder wraps the CodeAgent deep agent (see
orchestrator.pipeline.production.make_code_agent_coder); tests inject a scripted
coder that edits files directly. Either way the staticcheck / review / verify
gate nodes are the real thing.

Ordering rationale (design section 3.1): StaticCheck (free, deterministic) runs
before ReviewAgent (expensive ChatGPT round) so syntax/type errors never burn a
review round.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, StateGraph

from orchestrator.agents.review.agent import ReviewVerdict, review_diff
from orchestrator.agents.verify.agent import run_verification
from orchestrator.tools.sandbox_exec import run_command
from orchestrator.tools.staticcheck import build_staticcheck_tools

# coder(task, repo, feedback) -> short summary of what it did (side effect:
# edits + commits files in `repo`).
Coder = Callable[[str, str, str], Awaitable[str]]
# reviewer(diff, task) -> ReviewVerdict.
Reviewer = Callable[[str, str], Awaitable[ReviewVerdict]]


class PipelineState(TypedDict, total=False):
    task: str
    repo: str
    base_sha: str
    feedback: str
    diff: str
    code_summary: str
    staticcheck_ok: bool
    staticcheck_report: str
    review_approved: bool
    review_issues: list[str]
    verify_ok: bool
    verify_report: str
    code_retries: int
    status: str  # running / needs_human / done
    log: list[str]


def _git(args: list[str], repo: Path) -> str:
    return run_command(["git", *args], cwd=repo).output


def _run_staticcheck(repo: Path, path: str = ".") -> tuple[bool, str]:
    """Run all three static checks; return (all_ok, combined_report)."""
    tools = {t.name: t for t in build_staticcheck_tools(repo)}
    reports = []
    all_ok = True
    for name in ("syntax_check", "lint_check", "typecheck_run"):
        result = tools[name].invoke({"path": path})
        reports.append(result)
        if not result.startswith("PASS"):
            all_ok = False
    return all_ok, "\n".join(reports)


def build_pipeline(
    repo: Path | str,
    coder: Coder,
    reviewer: Reviewer | None = None,
    *,
    max_code_retries: int = 2,
    staticcheck_path: str = ".",
):
    """Compile the pipeline graph bound to *repo* and *coder*.

    reviewer defaults to the real ReviewAgent (ChatGPT bridge); inject a fake in
    tests. max_code_retries bounds how many times a failed gate can bounce back
    to the coder before the run escalates to needs_human.
    """
    root = Path(repo).resolve()

    async def _default_reviewer(diff: str, task: str) -> ReviewVerdict:
        return await review_diff(diff=diff, task=task)

    review_fn: Reviewer = reviewer or _default_reviewer

    async def init_node(state: PipelineState) -> PipelineState:
        base = _git(["rev-parse", "HEAD"], root).strip()
        return {
            "base_sha": base,
            "feedback": "",
            "code_retries": 0,
            "status": "running",
            "log": [f"init: base={base[:8] or '(no commits yet)'}"],
        }

    async def code_node(state: PipelineState) -> PipelineState:
        feedback = state.get("feedback", "")
        summary = await coder(state["task"], str(root), feedback)
        base = state.get("base_sha", "")
        diff = _git(["diff", base], root) if base else _git(["diff"], root)
        log = [*state.get("log", []), f"code: {summary[:120]}"]
        return {"code_summary": summary, "diff": diff, "log": log}

    async def staticcheck_node(state: PipelineState) -> PipelineState:
        ok, report = _run_staticcheck(root, staticcheck_path)
        log = [*state.get("log", []), f"staticcheck: {'PASS' if ok else 'FAIL'}"]
        return {"staticcheck_ok": ok, "staticcheck_report": report, "log": log}

    async def review_node(state: PipelineState) -> PipelineState:
        verdict = await review_fn(state.get("diff", ""), state["task"])
        log = [*state.get("log", []), f"review: {verdict.summary[:120]}"]
        return {
            "review_approved": verdict.approved,
            "review_issues": verdict.issues,
            "log": log,
        }

    async def verify_node(state: PipelineState) -> PipelineState:
        result = run_verification(root)
        log = [*state.get("log", []), f"verify: {result.summary}"]
        return {"verify_ok": result.ok, "verify_report": result.output, "log": log}

    async def done_node(state: PipelineState) -> PipelineState:
        return {"status": "done", "log": [*state.get("log", []), "done: all gates passed"]}

    async def escalate_node(state: PipelineState) -> PipelineState:
        return {
            "status": "needs_human",
            "log": [*state.get("log", []), "escalate: retries exhausted, human needed"],
        }

    def _bounce_or_advance(state: PipelineState, advance: str) -> str:
        """Route a failed gate: back to code if retries remain, else escalate."""
        if state.get("code_retries", 0) >= max_code_retries:
            return "escalate"
        return "retry"

    async def bump_retry_node(state: PipelineState) -> PipelineState:
        # Build feedback from whichever gate failed most recently.
        parts = []
        if not state.get("staticcheck_ok", True):
            parts.append("静态检查未通过：\n" + state.get("staticcheck_report", ""))
        if not state.get("review_approved", True) and state.get("review_issues"):
            parts.append("审查发现问题：\n- " + "\n- ".join(state["review_issues"]))
        if state.get("verify_ok") is False:
            parts.append("测试未通过：\n" + state.get("verify_report", ""))
        feedback = "\n\n".join(parts) or "上一轮未通过，请修正。"
        return {
            "feedback": feedback,
            "code_retries": state.get("code_retries", 0) + 1,
            "log": [*state.get("log", []), f"retry #{state.get('code_retries', 0) + 1}"],
        }

    def after_staticcheck(state: PipelineState) -> str:
        if state.get("staticcheck_ok"):
            return "review"
        return _bounce_or_advance(state, "review")

    def after_review(state: PipelineState) -> str:
        if state.get("review_approved"):
            return "verify"
        return _bounce_or_advance(state, "verify")

    def after_verify(state: PipelineState) -> str:
        if state.get("verify_ok"):
            return "done"
        return _bounce_or_advance(state, "done")

    graph = StateGraph(PipelineState)
    graph.add_node("init", init_node)
    graph.add_node("code", code_node)
    graph.add_node("staticcheck", staticcheck_node)
    graph.add_node("review", review_node)
    graph.add_node("verify", verify_node)
    graph.add_node("bump_retry", bump_retry_node)
    graph.add_node("done", done_node)
    graph.add_node("escalate", escalate_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "code")
    graph.add_edge("code", "staticcheck")
    graph.add_conditional_edges(
        "staticcheck", after_staticcheck,
        {"review": "review", "retry": "bump_retry", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "review", after_review,
        {"verify": "verify", "retry": "bump_retry", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "verify", after_verify,
        {"done": "done", "retry": "bump_retry", "escalate": "escalate"},
    )
    graph.add_edge("bump_retry", "code")
    graph.add_edge("done", END)
    graph.add_edge("escalate", END)

    return graph.compile()
