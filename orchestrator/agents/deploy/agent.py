"""DeployAgent — final stage, and the hard human boundary (design section 5.5).

The most important rule in the whole system: **automate up to the PR, never
auto-merge or auto-deploy to production.** So this stage does two separable things:

  1. prepare_deployment(): generate a PR title + body from the task and diff.
     Pure preparation — touches nothing outside producing text.
  2. push_feature_branch(): the ONLY function that pushes, and it is never
     called automatically by the pipeline. The pipeline stops after step 1 with
     status="awaiting_approval"; a human must explicitly call this (via the CLI
     approval path) to push. It refuses main/master and requires a real remote.

This mirrors the HITL intent of orchestrator/_utils/hitl.py without needing a
mid-graph interrupt: the graph simply ends at the approval boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from orchestrator._utils.json_extract import extract_json
from orchestrator._utils.model_factory import build_code_model
from orchestrator.tools.sandbox_exec import run_command

AGENT_NAME = "deploy_agent"
_PROTECTED_BRANCHES = {"main", "master", "head"}

_PROMPT = """根据开发任务和代码改动，写一个 Pull Request 的标题和正文。

开发任务：
{task}

代码改动（diff）：
{diff}

请**只输出一个 JSON 对象**：
{{"pr_title": "简洁的PR标题", "pr_body": "PR正文：说明改了什么、为什么、如何验证"}}
"""


@dataclass
class Deployment:
    pr_title: str
    pr_body: str
    branch: str = ""
    raw: str = ""


class DeployError(RuntimeError):
    pass


async def prepare_deployment(
    task: str, diff: str, branch: str = "", model: BaseChatModel | None = None
) -> Deployment:
    """Generate a PR draft. Does NOT push anything."""
    fallback_title = f"AutoDev: {task[:60]}"
    if not diff or not diff.strip():
        return Deployment(pr_title=fallback_title, pr_body="(无代码改动)", branch=branch)
    model = model or build_code_model(AGENT_NAME)
    response = await model.ainvoke(_PROMPT.format(task=task, diff=diff))
    content = getattr(response, "content", str(response))
    if isinstance(content, list):
        content = " ".join(str(c.get("text", c)) if isinstance(c, dict) else str(c) for c in content)
    try:
        data = extract_json(content)
        if not isinstance(data, dict):
            raise ValueError("not an object")
    except ValueError:
        return Deployment(pr_title=fallback_title, pr_body=task, branch=branch, raw=content)
    return Deployment(
        pr_title=str(data.get("pr_title") or fallback_title),
        pr_body=str(data.get("pr_body") or task),
        branch=branch,
        raw=content,
    )


def push_feature_branch(repo: Path | str, branch: str, remote: str = "origin") -> str:
    """Push *branch* to *remote*. HUMAN-APPROVED ONLY — never called by the graph.

    Refuses protected branches (design section 3.2). Requires a real remote.
    """
    if branch.strip().lower() in _PROTECTED_BRANCHES:
        raise DeployError(f"refusing to push protected branch {branch!r}")
    if not branch.strip():
        raise DeployError("no branch specified")

    root = Path(repo).resolve()
    remotes = run_command(["git", "remote"], cwd=root)
    if remote not in remotes.output.split():
        raise DeployError(
            f"remote {remote!r} not configured; add it before deploying "
            f"(available: {remotes.output.strip() or 'none'})"
        )

    result = run_command(["git", "push", "-u", remote, branch], cwd=root)
    if not result.ok:
        raise DeployError(f"git push failed: {result.output}")
    return f"pushed {branch!r} to {remote!r}"
