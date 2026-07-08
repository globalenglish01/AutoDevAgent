"""Tests for the direct whole-file coder (parser + write + static-check fix loop).

Uses fake models — no tool-calling, so no bind_tools patch needed. This is the
weak-browser-LLM-robust code path; these tests lock in its deterministic parts.
"""
from __future__ import annotations

import pytest
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from orchestrator.agents.code.direct_coder import generate_code_change, parse_file_blocks


def _model(*texts):
    return GenericFakeChatModel(messages=iter([AIMessage(content=t) for t in texts]))


def test_parse_plain_block():
    text = "说明\n<<<FILE calc.py>>>\ndef f():\n    return 1\n<<<END>>>\n"
    blocks = parse_file_blocks(text)
    assert blocks == [("calc.py", "def f():\n    return 1")]


def test_parse_fenced_block():
    text = "<<<FILE a.py>>>\n```python\nx = 1\n```\n<<<END>>>"
    blocks = parse_file_blocks(text)
    assert blocks == [("a.py", "x = 1")]


def test_parse_multiple_blocks():
    text = "<<<FILE a.py>>>\nx=1\n<<<END>>>\n<<<FILE b.py>>>\ny=2\n<<<END>>>"
    blocks = parse_file_blocks(text)
    assert [p for p, _ in blocks] == ["a.py", "b.py"]


@pytest.mark.asyncio
async def test_generate_success(tmp_path):
    out = ("好的\n<<<FILE calc.py>>>\n"
           "def add(a: int, b: int) -> int:\n    return a + b\n<<<END>>>\n")
    result = await generate_code_change("加 add", tmp_path, model=_model(out))
    assert result.ok
    assert "calc.py" in result.changed_files
    assert (tmp_path / "calc.py").read_text(encoding="utf-8").startswith("def add")


@pytest.mark.asyncio
async def test_fix_loop_recovers(tmp_path):
    bad = "<<<FILE calc.py>>>\ndef add(a, b:\n    return a+b\n<<<END>>>"   # syntax error
    good = ("<<<FILE calc.py>>>\ndef add(a: int, b: int) -> int:\n    return a + b\n<<<END>>>")
    result = await generate_code_change("加 add", tmp_path, model=_model(bad, good), max_fix_rounds=2)
    assert result.ok  # recovered on round 2
    assert "def add(a: int" in (tmp_path / "calc.py").read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_no_blocks_fails(tmp_path):
    result = await generate_code_change("随便", tmp_path, model=_model("我不知道怎么做"))
    assert not result.ok
    assert "未按格式" in result.summary


@pytest.mark.asyncio
async def test_rejects_sensitive_path(tmp_path):
    out = "<<<FILE .env>>>\nSECRET=x\n<<<END>>>"
    result = await generate_code_change("写env", tmp_path, model=_model(out, out), max_fix_rounds=1)
    assert not result.ok
    assert not (tmp_path / ".env").exists()  # path_guard blocked the write
