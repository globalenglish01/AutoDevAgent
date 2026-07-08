"""Phase 4 tests — requirement / design / deploy agents + repo_index."""
from __future__ import annotations

import subprocess

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from orchestrator.agents.deploy.agent import (
    DeployError,
    prepare_deployment,
    push_feature_branch,
)
from orchestrator.agents.design.agent import make_design
from orchestrator.agents.requirement.agent import analyze_requirement
from orchestrator.tools.repo_index import build_repo_index


def _model(text: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


# ── repo_index ────────────────────────────────────────────────────────────────

def test_repo_index_lists_files_skips_noise(tmp_path):
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("x=1\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("junk\n", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "c.pyc").write_text("x\n", encoding="utf-8")

    index = build_repo_index(tmp_path)
    assert "app/main.py" in index
    assert ".git" not in index
    assert "__pycache__" not in index


# ── requirement ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_requirement_parses_structured():
    req = await analyze_requirement(
        "加登录",
        model=_model('{"clarified_task":"实现用户名密码登录接口","acceptance_criteria":["返回token"],"ambiguities":["是否需要验证码"]}'),
    )
    assert req.clarified_task == "实现用户名密码登录接口"
    assert req.acceptance_criteria == ["返回token"]
    assert req.ambiguities == ["是否需要验证码"]


@pytest.mark.asyncio
async def test_requirement_failsafe_on_garbage():
    req = await analyze_requirement("加登录", model=_model("没有JSON"))
    assert req.clarified_task == "加登录"  # falls back to raw task
    assert req.acceptance_criteria == []


# ── design ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_design_parses_and_renders_block():
    design = await make_design(
        "实现登录接口",
        ["返回token"],
        model=_model('{"summary":"加一个POST /login","files":["app/auth.py"],"steps":["写handler","写测试"]}'),
    )
    assert design.summary == "加一个POST /login"
    block = design.as_prompt_block()
    assert "app/auth.py" in block
    assert "写handler" in block


# ── deploy ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_prepare_deployment_generates_pr():
    dep = await prepare_deployment(
        "加登录",
        diff="+def login(): ...\n",
        branch="feature/login",
        model=_model('{"pr_title":"feat: 登录接口","pr_body":"新增 POST /login"}'),
    )
    assert dep.pr_title == "feat: 登录接口"
    assert "POST /login" in dep.pr_body
    assert dep.branch == "feature/login"


@pytest.mark.asyncio
async def test_prepare_deployment_empty_diff():
    dep = await prepare_deployment("无改动", diff="", model=_model("unused"))
    assert dep.pr_body == "(无代码改动)"


def test_push_refuses_protected_branch(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    with pytest.raises(DeployError, match="protected"):
        push_feature_branch(tmp_path, "main")


def test_push_requires_remote(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    with pytest.raises(DeployError, match="remote"):
        push_feature_branch(tmp_path, "feature/x")
