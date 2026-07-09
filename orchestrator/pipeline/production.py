"""Production wiring: adapt the CodeAgent deep agent into a pipeline `Coder`.

Kept separate from graph.py so the graph stays testable with scripted coders.
This is what the CLI (run_autodev.py) uses against the real DeepSeek bridge.
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.language_models import BaseChatModel

from orchestrator.agents.code.agent import build_code_agent
from orchestrator.agents.code.direct_coder import generate_code_change
from orchestrator.agents.deploy.agent import prepare_deployment
from orchestrator.agents.design.agent import make_design
from orchestrator.agents.requirement.agent import analyze_requirement
from orchestrator.agents.review.agent import review_diff
from orchestrator.pipeline.graph import build_pipeline

_TASK_TEMPLATE = """开发任务：
{task}

{feedback_block}请在当前仓库中完成上述任务。记得：先切 feature 分支，小步修改，
每改完用 syntax_check/lint_check/typecheck_run 自检，全部通过后再 git_commit。
"""

_FEEDBACK_TEMPLATE = """⚠️ 上一轮没有通过，请根据以下反馈修正后再提交：
{feedback}

"""


def make_code_agent_coder(model: BaseChatModel | None = None):
    """Return a pipeline Coder backed by the CodeAgent deep agent.

    A fresh agent conversation is built per attempt; prior-round feedback (static
    check errors / review issues / test failures) is injected into the task
    prompt so the agent knows what to fix.
    """
    _thread_counter = {"n": 0}

    async def coder(task: str, repo: str, feedback: str) -> str:
        agent = await build_code_agent(Path(repo), model=model)
        feedback_block = _FEEDBACK_TEMPLATE.format(feedback=feedback) if feedback else ""
        prompt = _TASK_TEMPLATE.format(task=task, feedback_block=feedback_block)
        _thread_counter["n"] += 1
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": f"code-{_thread_counter['n']}"}},
        )
        last = result["messages"][-1]
        content = getattr(last, "content", "")
        if isinstance(content, list):
            content = " ".join(
                str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content
            )
        return content or "(code agent produced no summary)"

    return coder


def make_direct_coder():
    """Return a pipeline Coder backed by the direct whole-file generator.

    This is the DEFAULT coder — real-bridge runs (2026-07-08) showed the
    deepagents tool-calling coder is unreliable on the browser shim (weak-model
    JSON, CHATGPT_ERROR), while direct whole-file generation works (verified:
    ChatGPT generated a correct `multiply` that passed tests). See
    orchestrator/agents/code/direct_coder.py.
    """
    async def coder(task: str, repo: str, feedback: str) -> str:
        result = await generate_code_change(task, repo, feedback=feedback)
        return result.summary

    return coder


def build_lite_pipeline(repo, *, max_code_retries: int = 2):
    """Planning-free pipeline for weak/free browser bridges: code → staticcheck
    → review → verify → deploy. No requirement/design front-end.

    Real-run finding (2026-07-08): on the free ChatGPT bridge the planning stages
    hurt — they (a) burn 2 calls that degrade the session before code even runs
    (the bridge's clean copy-button extraction works best on early calls), and
    (b) feed the weak model a verbose enriched task it tends to answer in prose
    instead of emitting file blocks. Going straight to code as the FIRST call
    matches the configuration that succeeded (a correct, test-passing `multiply`).
    Prefer this over build_full_pipeline when driving a free browser bridge.
    """
    async def _reviewer(diff, task):
        return await review_diff(diff=diff, task=task)

    async def _deployer(task, diff, branch):
        return await prepare_deployment(task, diff, branch)

    return build_pipeline(
        repo,
        coder=make_direct_coder(),
        reviewer=_reviewer,
        deployer=_deployer,
        max_code_retries=max_code_retries,
    )


def build_full_pipeline(repo, *, max_code_retries: int = 2, use_deep_agent: bool = False):
    """Wire the complete real pipeline against the llm_bridge:
    RequirementAnalyzer + DesignAgent (DeepSeek/CODE role) → coder →
    StaticCheck → ReviewAgent (ChatGPT/REVIEW role) → VerifyAgent → DeployAgent.

    coder defaults to the direct whole-file generator (make_direct_coder) which
    is reliable on the browser shim. Pass use_deep_agent=True to use the
    tool-calling deepagents CodeAgent instead (needs a stronger model/bridge).

    Requires bridge instances running for the CODE role (:8765) and, for review,
    the REVIEW role (:8766) — or point both at one via env vars. This is what
    run_autodev.py invokes.
    """
    async def _analyzer(task, repo_summary):
        return await analyze_requirement(task, repo_summary)

    async def _designer(clarified, criteria, repo_summary):
        return await make_design(clarified, criteria, repo_summary)

    async def _reviewer(diff, task):
        return await review_diff(diff=diff, task=task)

    async def _deployer(task, diff, branch):
        return await prepare_deployment(task, diff, branch)

    coder = make_code_agent_coder() if use_deep_agent else make_direct_coder()
    return build_pipeline(
        repo,
        coder=coder,
        reviewer=_reviewer,
        analyzer=_analyzer,
        designer=_designer,
        deployer=_deployer,
        max_code_retries=max_code_retries,
    )
