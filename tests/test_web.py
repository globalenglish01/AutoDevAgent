"""Web UI tests — no browser bridge needed.

Uses FastAPI TestClient with an injected fake-model pipeline (scripted coder/
reviewer) so the full HTTP flow (start → poll → awaiting_approval → approve)
is exercised deterministically. execute_run / apply_and_push are also tested
directly since TestClient's background-task scheduling is awkward to await.
"""
from __future__ import annotations

import subprocess

import pytest
from fastapi.testclient import TestClient

from orchestrator.agents.deploy.agent import Deployment
from orchestrator.agents.review.agent import ReviewVerdict
from orchestrator.pipeline.graph import build_pipeline
from orchestrator.web.app import apply_and_push, create_app, execute_run


def _init_repo(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "--allow-empty", "-m", "base"],
                   cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


def _fake_pipeline(repo):
    async def coder(task, repo_path, feedback):
        from pathlib import Path
        (Path(repo_path) / "feature.py").write_text("def sq(n):\n    return n * n\n", encoding="utf-8")
        (Path(repo_path) / "test_feature.py").write_text(
            "from feature import sq\n\ndef test_sq():\n    assert sq(4) == 16\n", encoding="utf-8")
        return "wrote sq"

    async def reviewer(diff, task):
        return ReviewVerdict(approved=True, issues=[])

    async def deployer(task, diff, branch):
        return Deployment(pr_title="feat: add sq", pr_body="新增平方函数", branch=branch)

    return build_pipeline(repo, coder=coder, reviewer=reviewer, deployer=deployer)


# ── HTTP surface ──────────────────────────────────────────────────────────────

def test_index_served():
    client = TestClient(create_app())
    r = client.get("/")
    assert r.status_code == 200
    assert "AutoDevAgent" in r.text
    assert "startRun" in r.text  # the page JS is present


def test_run_validation():
    client = TestClient(create_app())
    assert client.post("/api/run", json={"repo": "", "task": ""}).status_code == 400
    assert client.post("/api/run", json={"repo": "/no/such/dir", "task": "x"}).status_code == 400


def test_get_unknown_run_404():
    client = TestClient(create_app())
    assert client.get("/api/run/deadbeef").status_code == 404


# ── run execution (direct, deterministic) ──────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_run_reaches_awaiting_approval(tmp_path):
    repo = _init_repo(tmp_path)
    record: dict = {"status": "running", "log": [], "repo": str(repo), "done": False}
    await execute_run(record, _fake_pipeline(repo), task="加平方函数", run_id="testrun1")
    assert record["done"]
    assert record["status"] == "awaiting_approval"
    assert record["pr_title"] == "feat: add sq"
    assert record["verify_ok"] is True
    assert record["log"]  # progress was streamed


# ── approve / apply ─────────────────────────────────────────────────────────

def test_apply_and_push_commits_on_feature_branch(tmp_path):
    repo = _init_repo(tmp_path)
    (tmp_path / "feature.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    result = apply_and_push(str(repo), "abc123", pr_title="feat: f")
    assert "abc123" in result["branch"] or "feature" in result["branch"].lower()
    assert "committed" in result["commit"]
    # No remote configured → push reports the local-commit fallback, not a crash.
    assert "未推送" in result["push"] or "pushed" in result["push"]
    # The commit really landed on the feature branch.
    log = subprocess.run(["git", "log", "--oneline"], cwd=repo, capture_output=True, text=True)
    assert "feat: f" in log.stdout


def test_apply_refuses_sensitive_file(tmp_path):
    repo = _init_repo(tmp_path)
    (tmp_path / ".env").write_text("SECRET=x\n", encoding="utf-8")
    result = apply_and_push(str(repo), "sec1", pr_title="oops")
    # git_commit refuses to stage .env → nothing committed with the secret.
    assert "REJECTED" in result["commit"] or "nothing" in result["commit"].lower()
