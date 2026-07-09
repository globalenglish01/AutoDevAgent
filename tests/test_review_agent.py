"""Phase 3b tests — ReviewAgent verdict parsing + json_extract robustness."""
from __future__ import annotations

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from orchestrator._utils.json_extract import extract_json
from orchestrator.agents.review.agent import review_diff


# ── json_extract ──────────────────────────────────────────────────────────────

def test_extract_plain_json():
    assert extract_json('{"approved": true, "issues": []}') == {"approved": True, "issues": []}


def test_extract_fenced_json():
    text = "分析如下：\n```json\n{\"approved\": false, \"issues\": [\"x\"]}\n```\n完毕"
    assert extract_json(text) == {"approved": False, "issues": ["x"]}


def test_extract_embedded_json():
    text = '这是结论 {"approved": true, "issues": []} 以上。'
    assert extract_json(text) == {"approved": True, "issues": []}


def test_extract_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_json("完全没有 JSON 的一段话")


# ── review_diff ───────────────────────────────────────────────────────────────

def _model(text: str) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([AIMessage(content=text)]))


@pytest.mark.asyncio
async def test_review_approved_keyword():
    verdict = await review_diff(
        diff="+def f():\n+    return 1\n", task="加一个函数 f",
        model=_model("APPROVE"),
    )
    assert verdict.approved
    assert verdict.issues == []


@pytest.mark.asyncio
async def test_review_rejected_keyword():
    verdict = await review_diff(
        diff="+def div(a,b):\n+    return a/b\n", task="加一个除法函数",
        model=_model("REJECT\n未处理除零\n缺少类型注解"),
    )
    assert not verdict.approved
    assert any("除零" in i for i in verdict.issues)
    assert len(verdict.issues) == 2


@pytest.mark.asyncio
async def test_review_approved_json_fallback():
    # Older habit: model still emits JSON — must still work.
    verdict = await review_diff(
        diff="+x = 1\n", task="加变量",
        model=_model('{"approved": true, "issues": []}'),
    )
    assert verdict.approved


@pytest.mark.asyncio
async def test_review_rejected_json_fallback():
    verdict = await review_diff(
        diff="+def div(a,b):\n+    return a/b\n", task="除法",
        model=_model('{"approved": false, "issues": ["未处理除零"]}'),
    )
    assert not verdict.approved
    assert "未处理除零" in verdict.issues[0]


@pytest.mark.asyncio
async def test_review_chinese_approve():
    verdict = await review_diff(
        diff="+x = 1\n", task="加变量", model=_model("这段代码没问题，可以通过。"),
    )
    assert verdict.approved


@pytest.mark.asyncio
async def test_review_chinese_reject():
    verdict = await review_diff(
        diff="+def d(a,b):\n+    return a/b\n", task="除法", model=_model("有问题：未处理除零。"),
    )
    assert not verdict.approved
    assert not verdict.inconclusive  # explicit rejection, not inconclusive


@pytest.mark.asyncio
async def test_review_unparseable_is_inconclusive_not_reject():
    verdict = await review_diff(
        diff="+x = 1\n",
        task="加个变量",
        model=_model("嗯……让我想想这段代码。"),  # no keyword, no JSON, no zh verdict
    )
    assert not verdict.approved       # not approved
    assert verdict.inconclusive       # but flagged inconclusive (defer to tests), not a hard reject


@pytest.mark.asyncio
async def test_review_empty_diff_auto_approves():
    verdict = await review_diff(diff="", task="无改动", model=_model("unused"))
    assert verdict.approved
