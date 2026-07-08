"""Pipeline state machine implementing docs/PoC设计书 section 3.1.

Full shape (all stages enabled):
    init → requirement → design → code → staticcheck → review → verify → deploy → END

- requirement / design (front) run only if an analyzer / designer is provided.
- deploy (back) runs only if a deployer is provided; it prepares a PR draft and
  ends with status="awaiting_approval" — the pipeline NEVER auto-pushes (design
  section 5.5). Without a deployer the run ends "done" after verify.
- Every gate that fails (staticcheck / review / verify) routes back to `code`
  with concrete feedback, up to max_code_retries, then escalates to
  status="needs_human".

The `code` step is dependency-injected (`coder`) so control flow is testable
without a 5-6 minute browser LLM. requirement/design/deploy are likewise
injectable. The staticcheck / review / verify gates are always the real thing.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from orchestrator.agents.review.agent import ReviewVerdict, review_diff
from orchestrator.agents.verify.agent import run_verification
from orchestrator.tools.repo_index import build_repo_index
from orchestrator.tools.sandbox_exec import run_command
from orchestrator.tools.staticcheck import build_staticcheck_tools

Coder = Callable[[str, str, str], Awaitable[str]]
Reviewer = Callable[[str, str], Awaitable[ReviewVerdict]]
# analyzer(task, repo_summary) -> object with .clarified_task / .acceptance_criteria
Analyzer = Callable[[str, str], Awaitable[Any]]
# designer(clarified_task, acceptance_criteria, repo_summary) -> object with .as_prompt_block()
Designer = Callable[[str, list, str], Awaitable[Any]]
# deployer(task, diff, branch) -> object with .pr_title / .pr_body
Deployer = Callable[[str, str, str], Awaitable[Any]]


class PipelineState(TypedDict, total=False):
    task: str
    repo: str
    repo_summary: str
    base_sha: str
    clarified_task: str
    acceptance_criteria: list[str]
    design_block: str
    enriched_task: str
    feedback: str
    diff: str
    code_summary: str
    code_made_changes: bool
    staticcheck_ok: bool
    staticcheck_report: str
    review_approved: bool
    review_issues: list[str]
    verify_ok: bool
    verify_report: str
    branch: str
    pr_title: str
    pr_body: str
    code_retries: int
    status: str  # running / needs_human / done / awaiting_approval
    log: list[str]


def _git(args: list[str], repo: Path) -> str:
    return run_command(["git", *args], cwd=repo).output


def _capture_diff(repo: Path, base_sha: str) -> str:
    """Diff of everything the coder changed since *base_sha*, INCLUDING new
    untracked files (plain `git diff` omits those). We briefly stage all with
    `git add -A`, diff the index against base, then `git reset` to leave the
    working tree exactly as the coder left it (untracked stays untracked).
    """
    _git(["add", "-A"], repo)
    diff = _git(["diff", "--cached", base_sha], repo) if base_sha else _git(["diff", "--cached"], repo)
    _git(["reset", "-q"], repo)
    return diff


def _run_staticcheck(repo: Path, path: str = ".") -> tuple[bool, str]:
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
    analyzer: Analyzer | None = None,
    designer: Designer | None = None,
    deployer: Deployer | None = None,
    max_code_retries: int = 2,
    staticcheck_path: str = ".",
):
    """Compile the pipeline graph bound to *repo* and *coder*.

    Provide analyzer/designer to enable the planning front-end, deployer to
    enable the PR-draft back-end. All default to the real agents when the
    corresponding stage is requested via build_full_pipeline(); pass fakes in tests.
    """
    root = Path(repo).resolve()

    async def _default_reviewer(diff: str, task: str) -> ReviewVerdict:
        return await review_diff(diff=diff, task=task)

    review_fn: Reviewer = reviewer or _default_reviewer

    # ── nodes ────────────────────────────────────────────────────────────────
    async def init_node(state: PipelineState) -> PipelineState:
        base = _git(["rev-parse", "HEAD"], root).strip()
        return {
            "repo": str(root),
            "repo_summary": build_repo_index(root),
            "base_sha": base,
            "feedback": "",
            "code_retries": 0,
            "status": "running",
            "log": [f"init: base={base[:8] or '(no commits)'}"],
        }

    async def requirement_node(state: PipelineState) -> PipelineState:
        req = await analyzer(state["task"], state.get("repo_summary", ""))
        clarified = getattr(req, "clarified_task", state["task"])
        criteria = list(getattr(req, "acceptance_criteria", []) or [])
        return {
            "clarified_task": clarified,
            "acceptance_criteria": criteria,
            "log": [*state.get("log", []), f"requirement: {clarified[:100]}"],
        }

    async def design_node(state: PipelineState) -> PipelineState:
        clarified = state.get("clarified_task", state["task"])
        design = await designer(
            clarified, state.get("acceptance_criteria", []), state.get("repo_summary", "")
        )
        block = design.as_prompt_block() if hasattr(design, "as_prompt_block") else str(design)
        enriched = f"{clarified}\n\n{block}"
        return {
            "design_block": block,
            "enriched_task": enriched,
            "log": [*state.get("log", []), f"design: {getattr(design, 'summary', '')[:100]}"],
        }

    async def code_node(state: PipelineState) -> PipelineState:
        task = state.get("enriched_task") or state["task"]
        summary = await coder(task, str(root), state.get("feedback", ""))
        base = state.get("base_sha", "")
        diff = _capture_diff(root, base)
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], root).strip()
        return {
            "code_summary": summary,
            "diff": diff,
            "branch": branch,
            "code_made_changes": bool(diff.strip()),
            "log": [*state.get("log", []), f"code: {summary[:120]}"],
        }

    async def staticcheck_node(state: PipelineState) -> PipelineState:
        ok, report = _run_staticcheck(root, staticcheck_path)
        return {
            "staticcheck_ok": ok,
            "staticcheck_report": report,
            "log": [*state.get("log", []), f"staticcheck: {'PASS' if ok else 'FAIL'}"],
        }

    async def review_node(state: PipelineState) -> PipelineState:
        task = state.get("clarified_task") or state["task"]
        verdict = await review_fn(state.get("diff", ""), task)
        return {
            "review_approved": verdict.approved,
            "review_issues": verdict.issues,
            "log": [*state.get("log", []), f"review: {verdict.summary[:120]}"],
        }

    async def verify_node(state: PipelineState) -> PipelineState:
        result = run_verification(root)
        return {
            "verify_ok": result.ok,
            "verify_report": result.output,
            "log": [*state.get("log", []), f"verify: {result.summary}"],
        }

    async def deploy_node(state: PipelineState) -> PipelineState:
        task = state.get("clarified_task") or state["task"]
        deployment = await deployer(task, state.get("diff", ""), state.get("branch", ""))
        return {
            "pr_title": getattr(deployment, "pr_title", ""),
            "pr_body": getattr(deployment, "pr_body", ""),
            "status": "awaiting_approval",
            "log": [
                *state.get("log", []),
                f"deploy: PR draft ready, awaiting human approval ({getattr(deployment, 'pr_title', '')[:80]})",
            ],
        }

    async def done_node(state: PipelineState) -> PipelineState:
        return {"status": "done", "log": [*state.get("log", []), "done: all gates passed"]}

    async def escalate_node(state: PipelineState) -> PipelineState:
        return {
            "status": "needs_human",
            "log": [*state.get("log", []), "escalate: retries exhausted, human needed"],
        }

    async def bump_retry_node(state: PipelineState) -> PipelineState:
        parts = []
        if not state.get("code_made_changes", True):
            parts.append(
                "上一轮没有产生任何代码改动。请务必用 <<<FILE 相对路径>>> ... <<<END>>> 的格式"
                "输出需要新增/修改文件的**完整内容**，不要只用文字描述。"
            )
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

    # ── routing ──────────────────────────────────────────────────────────────
    def _bounce(state: PipelineState) -> str:
        return "escalate" if state.get("code_retries", 0) >= max_code_retries else "retry"

    def after_code(state: PipelineState) -> str:
        # A code step that changed nothing is a failure (weak model replied in
        # prose instead of emitting file blocks) — never let an empty diff sail
        # through staticcheck/review/verify as a false success.
        if not state.get("code_made_changes", False):
            return _bounce(state)
        return "advance"

    def after_staticcheck(state: PipelineState) -> str:
        return "advance" if state.get("staticcheck_ok") else _bounce(state)

    def after_review(state: PipelineState) -> str:
        return "advance" if state.get("review_approved") else _bounce(state)

    def after_verify(state: PipelineState) -> str:
        return "advance" if state.get("verify_ok") else _bounce(state)

    # ── graph assembly ─────────────────────────────────────────────────────────
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

    # Front-end: optional requirement → design before code.
    first_after_init = "code"
    if analyzer is not None:
        graph.add_node("requirement", requirement_node)
        first_after_init = "requirement"
        if designer is not None:
            graph.add_node("design", design_node)
            graph.add_edge("requirement", "design")
            graph.add_edge("design", "code")
        else:
            graph.add_edge("requirement", "code")
    elif designer is not None:
        # designer without analyzer: design straight off the raw task
        graph.add_node("design", design_node)
        first_after_init = "design"
        graph.add_edge("design", "code")
    graph.add_edge("init", first_after_init)

    graph.add_conditional_edges(
        "code", after_code,
        {"advance": "staticcheck", "retry": "bump_retry", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "staticcheck", after_staticcheck,
        {"advance": "review", "retry": "bump_retry", "escalate": "escalate"},
    )
    graph.add_conditional_edges(
        "review", after_review,
        {"advance": "verify", "retry": "bump_retry", "escalate": "escalate"},
    )

    # Back-end: optional deploy after verify passes.
    if deployer is not None:
        graph.add_node("deploy", deploy_node)
        verify_target = "deploy"
        graph.add_edge("deploy", END)
    else:
        verify_target = "done"
    graph.add_conditional_edges(
        "verify", after_verify,
        {"advance": verify_target, "retry": "bump_retry", "escalate": "escalate"},
    )

    graph.add_edge("bump_retry", "code")
    graph.add_edge("done", END)
    graph.add_edge("escalate", END)

    return graph.compile()
