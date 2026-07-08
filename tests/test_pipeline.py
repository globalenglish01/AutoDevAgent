"""Phase 3d tests — pipeline control flow with a scripted coder.

The coder is injected (not a real LLM) so we can deterministically drive the
graph and assert the retry loops fire against the *real* staticcheck / review /
verify gate nodes. Proves:
  - happy path: code → staticcheck → review → verify → done
  - staticcheck failure bounces back to code, which fixes it, then proceeds
  - retries exhausted → needs_human
"""
from __future__ import annotations

import subprocess

import pytest

from orchestrator.agents.review.agent import ReviewVerdict
from orchestrator.pipeline.graph import build_pipeline


def _init_repo(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "--allow-empty", "-m", "base"],
                   cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


async def _approve(diff, task):
    return ReviewVerdict(approved=True, issues=[])


async def _reject(diff, task):
    return ReviewVerdict(approved=False, issues=["构造的审查问题"])


@pytest.mark.asyncio
async def test_happy_path(tmp_path):
    repo = _init_repo(tmp_path)

    async def coder(task, repo_path, feedback):
        (tmp_path / "feature.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (tmp_path / "test_feature.py").write_text(
            "from feature import add\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            encoding="utf-8",
        )
        return "wrote feature.py + test"

    pipeline = build_pipeline(repo, coder=coder, reviewer=_approve)
    result = await pipeline.ainvoke({"task": "加一个 add 函数"})

    assert result["status"] == "done"
    assert result["staticcheck_ok"]
    assert result["review_approved"]
    assert result["verify_ok"]


@pytest.mark.asyncio
async def test_staticcheck_failure_then_fix(tmp_path):
    repo = _init_repo(tmp_path)
    calls = {"n": 0}

    async def coder(task, repo_path, feedback):
        calls["n"] += 1
        if calls["n"] == 1:
            # First attempt: broken syntax -> staticcheck fails.
            (tmp_path / "feature.py").write_text("def add(a, b:\n    return a + b\n", encoding="utf-8")
            return "attempt 1 (broken)"
        # Second attempt: fixed. Feedback should carry the static-check failure.
        assert "静态检查" in feedback
        (tmp_path / "feature.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        return "attempt 2 (fixed)"

    pipeline = build_pipeline(repo, coder=coder, reviewer=_approve, max_code_retries=2)
    result = await pipeline.ainvoke({"task": "加一个 add 函数"})

    assert calls["n"] == 2
    assert result["status"] == "done"
    assert result["staticcheck_ok"]


@pytest.mark.asyncio
async def test_retries_exhausted_escalates(tmp_path):
    repo = _init_repo(tmp_path)

    async def coder(task, repo_path, feedback):
        # Always writes broken syntax -> staticcheck never passes.
        (tmp_path / "feature.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
        return "always broken"

    pipeline = build_pipeline(repo, coder=coder, reviewer=_approve, max_code_retries=2)
    result = await pipeline.ainvoke({"task": "加个函数"})

    assert result["status"] == "needs_human"
    assert result["code_retries"] == 2


@pytest.mark.asyncio
async def test_review_rejection_bounces_back(tmp_path):
    repo = _init_repo(tmp_path)
    calls = {"n": 0}
    reviews = iter([_reject, _approve])

    async def coder(task, repo_path, feedback):
        calls["n"] += 1
        (tmp_path / "feature.py").write_text("x = 1\n", encoding="utf-8")
        return f"attempt {calls['n']}"

    async def reviewer(diff, task):
        fn = next(reviews)
        return await fn(diff, task)

    pipeline = build_pipeline(repo, coder=coder, reviewer=reviewer, max_code_retries=2)
    result = await pipeline.ainvoke({"task": "加个变量"})

    assert calls["n"] == 2  # rejected once, then approved
    assert result["status"] == "done"
    assert result["review_approved"]
