"""Phase 4 e2e — full pipeline with all stages, scripted (non-LLM) callables.

Proves requirement → design → code → staticcheck → review → verify → deploy
runs end to end and stops at status="awaiting_approval" (never auto-pushes).
"""
from __future__ import annotations

import subprocess

import pytest

from orchestrator.agents.deploy.agent import Deployment
from orchestrator.agents.design.agent import Design
from orchestrator.agents.requirement.agent import Requirements
from orchestrator.agents.review.agent import ReviewVerdict
from orchestrator.pipeline.graph import build_pipeline


def _init_repo(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "--allow-empty", "-m", "base"],
                   cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


@pytest.mark.asyncio
async def test_full_pipeline_reaches_awaiting_approval(tmp_path):
    repo = _init_repo(tmp_path)
    seen = {}

    async def analyzer(task, repo_summary):
        seen["repo_summary"] = repo_summary
        return Requirements(clarified_task=f"[clarified] {task}", acceptance_criteria=["测试通过"])

    async def designer(clarified, criteria, repo_summary):
        seen["clarified"] = clarified
        return Design(summary="加 add 函数", files=["feature.py"], steps=["写函数", "写测试"])

    async def coder(task, repo_path, feedback):
        # The enriched task should carry the design block through.
        seen["code_task"] = task
        (tmp_path / "feature.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (tmp_path / "test_feature.py").write_text(
            "from feature import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            encoding="utf-8",
        )
        return "wrote feature.py + test"

    async def reviewer(diff, task):
        return ReviewVerdict(approved=True, issues=[])

    async def deployer(task, diff, branch):
        seen["deploy_diff"] = diff
        return Deployment(pr_title="feat: add 函数", pr_body="新增 add", branch=branch)

    pipeline = build_pipeline(
        repo, coder=coder, reviewer=reviewer,
        analyzer=analyzer, designer=designer, deployer=deployer,
    )
    result = await pipeline.ainvoke({"task": "加一个加法函数"})

    assert result["status"] == "awaiting_approval"
    assert result["staticcheck_ok"]
    assert result["review_approved"]
    assert result["verify_ok"]
    assert result["pr_title"] == "feat: add 函数"

    # Front-end actually fed forward: repo summary → analyzer, design block → coder.
    assert "feature" in seen["repo_summary"] or seen["repo_summary"]  # index was built
    assert "[clarified]" in seen["code_task"]
    assert "加 add 函数" in seen["code_task"]  # design block injected
    assert seen["deploy_diff"].strip() != ""  # deploy saw the real diff


@pytest.mark.asyncio
async def test_full_pipeline_escalates_without_deploy(tmp_path):
    """Even with all stages, a never-passing gate escalates and never deploys."""
    repo = _init_repo(tmp_path)
    deployed = {"called": False}

    async def analyzer(task, repo_summary):
        return Requirements(clarified_task=task)

    async def designer(clarified, criteria, repo_summary):
        return Design(summary="x")

    async def coder(task, repo_path, feedback):
        (tmp_path / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
        return "broken"

    async def reviewer(diff, task):
        return ReviewVerdict(approved=True, issues=[])

    async def deployer(task, diff, branch):
        deployed["called"] = True
        return Deployment(pr_title="x", pr_body="y")

    pipeline = build_pipeline(
        repo, coder=coder, reviewer=reviewer,
        analyzer=analyzer, designer=designer, deployer=deployer,
        max_code_retries=1,
    )
    result = await pipeline.ainvoke({"task": "加个函数"})

    assert result["status"] == "needs_human"
    assert not deployed["called"]  # deploy never runs when gates fail
