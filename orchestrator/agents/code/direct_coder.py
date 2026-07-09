"""Direct whole-file code generation — the robust path for weak browser LLMs.

Real-run finding (2026-07-08): JSON-emulated tool-calling (ReAct / deepagents)
is too fragile on the browser shim — the weak model emits slightly malformed
tool-call JSON and the whole round fails (CHATGPT_ERROR / JSON parse errors).

This coder avoids tool-calling entirely:
  1. WE gather context deterministically (read the repo's files) — the model
     never has to call a read tool.
  2. The model only has to output full file contents in a simple delimited
     format — something even a weak model does reliably.
  3. WE parse the blocks and write the files (through path_guard), then run
     StaticCheck. On failure we feed the errors back for a bounded number of
     fix rounds.

This is the "whole-file edit" strategy (like Aider's whole format), chosen
because it degrades gracefully with model quality where tool-calling does not.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models import BaseChatModel

from orchestrator._utils.llm_retry import ainvoke_text
from orchestrator._utils.model_factory import build_code_model
from orchestrator.tools.path_guard import PathGuardError, resolve_safe_path
from orchestrator.tools.staticcheck import build_staticcheck_tools

logger = logging.getLogger(__name__)

AGENT_NAME = "direct_coder"

_MAX_CONTEXT_FILES = 12
_MAX_FILE_CHARS = 6000
_SKIP = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".mypy_cache"}

# Delimited block the model must emit per file. Chosen to be unambiguous and
# easy for a weak model to reproduce.
_BLOCK_RE = re.compile(
    r"<<<FILE\s+(?P<path>[^\n>]+?)>>>\s*\n(?P<body>.*?)\n?<<<END>>>",
    re.DOTALL,
)
# Strip an optional ```lang fence the model may wrap the body in.
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```\s*$", re.DOTALL)

_PROMPT = """你是编码助手。你的回复**必须**包含至少一个"文件块"，否则本次完全无效。

【输出格式 — 最重要，务必遵守】
对每个要新增或修改的文件，输出它的**完整内容**（不是片段、不是 diff、不是文字描述），
用如下格式，可多个：
<<<FILE 相对路径>>>
（这里是该文件从头到尾的完整内容）
<<<END>>>
禁止只回复"好的/明白/我会…"这类文字而不给文件块。文字解释请压到最少。

【任务】
{task}
{feedback}
【当前相关文件内容（供你参考，保留其中不该改动的部分）】
{context}

现在，直接输出文件块（以 <<<FILE 开头）："""


@dataclass
class CodeGenResult:
    ok: bool
    summary: str
    changed_files: list[str] = field(default_factory=list)
    staticcheck_report: str = ""
    raw: str = ""


def _gather_context(repo: Path, context_files: list[str] | None) -> str:
    files: list[Path] = []
    if context_files:
        for rel in context_files:
            p = repo / rel
            if p.is_file():
                files.append(p)
    else:
        for p in sorted(repo.rglob("*.py")):
            if any(part in _SKIP for part in p.relative_to(repo).parts):
                continue
            files.append(p)
            if len(files) >= _MAX_CONTEXT_FILES:
                break
    if not files:
        return "(仓库暂无 Python 文件)"
    parts = []
    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")[:_MAX_FILE_CHARS]
        parts.append(f"--- {p.relative_to(repo).as_posix()} ---\n{text}")
    return "\n\n".join(parts)


def _normalize_code(text: str) -> str:
    """Clean up browser-scraping artifacts in generated code.

    ChatGPT's answer is scraped from rendered DOM, which double-spaces lines and
    can insert a blank line right after a `:`-terminated header. Both are
    cosmetically ugly (though Python-valid). We strip trailing whitespace, drop a
    blank line immediately following a colon-header, and collapse 3+ consecutive
    blank lines to at most 2 (PEP8). Conservative — never touches indented code
    content, so it can't break structure.
    """
    lines = [ln.rstrip() for ln in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    out: list[str] = []
    blank_run = 0
    for i, ln in enumerate(lines):
        if ln == "":
            prev = out[-1] if out else ""
            # Drop a blank immediately after a colon-header (def/if/for/class/...).
            if prev.rstrip().endswith(":"):
                continue
            blank_run += 1
            if blank_run > 2:
                continue
        else:
            blank_run = 0
        out.append(ln)
    return "\n".join(out).strip("\n") + "\n"


def parse_file_blocks(text: str) -> list[tuple[str, str]]:
    """Extract (relative_path, content) pairs from the model's output."""
    blocks = []
    for m in _BLOCK_RE.finditer(text):
        path = m.group("path").strip().strip('"').strip("`")
        body = m.group("body")
        fence = _FENCE_RE.match(body.strip())
        if fence:
            body = fence.group(1)
        blocks.append((path, body))
    return blocks


def _write_blocks(repo: Path, blocks: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
    written, rejected = [], []
    for path, content in blocks:
        try:
            target = resolve_safe_path(repo, path)
        except PathGuardError as exc:
            rejected.append(f"{path}: {exc}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_normalize_code(content), encoding="utf-8")
        written.append(path)
    return written, rejected


def _staticcheck(repo: Path, files: list[str]) -> tuple[bool, str]:
    tools = {t.name: t for t in build_staticcheck_tools(repo)}
    reports, ok = [], True
    for f in files:
        for name in ("syntax_check", "lint_check"):
            r = tools[name].invoke({"path": f})
            if not r.startswith("PASS"):
                ok = False
                reports.append(f"[{f}] {r}")
    return ok, "\n".join(reports)


async def _ask(model: BaseChatModel, prompt: str) -> str:
    """Invoke the model with shared transient-error retry (see llm_retry)."""
    return await ainvoke_text(model, prompt)


async def generate_code_change(
    task: str,
    repo: Path | str,
    model: BaseChatModel | None = None,
    *,
    context_files: list[str] | None = None,
    feedback: str = "",
    max_fix_rounds: int = 2,
) -> CodeGenResult:
    """Generate code for *task* by whole-file output; write + static-check; fix-loop."""
    root = Path(repo).resolve()
    model = model or build_code_model(AGENT_NAME)

    fb_block = f"\n上一轮的问题（请修正）：\n{feedback}\n" if feedback else ""
    prompt = _PROMPT.format(task=task, feedback=fb_block, context=_gather_context(root, context_files))

    raw_all = []
    all_changed: list[str] = []
    for round_no in range(1, max_fix_rounds + 1):
        raw = await _ask(model, prompt)
        raw_all.append(raw)
        blocks = parse_file_blocks(raw)
        if not blocks:
            return CodeGenResult(
                ok=False,
                summary="模型未按格式输出文件块（无 <<<FILE...>>> 块）",
                raw="\n---\n".join(raw_all),
            )
        written, rejected = _write_blocks(root, blocks)
        all_changed = sorted(set(all_changed) | set(written))
        ok, report = _staticcheck(root, written)
        if ok and not rejected:
            return CodeGenResult(
                ok=True,
                summary=f"生成/修改 {len(written)} 个文件，静态检查通过：{', '.join(written)}",
                changed_files=all_changed,
                staticcheck_report=report,
                raw="\n---\n".join(raw_all),
            )
        # Not ok — build a fix prompt for the next round.
        problems = report + ("\n" + "\n".join(rejected) if rejected else "")
        prompt = _PROMPT.format(
            task=task,
            feedback=f"\n上一轮静态检查未通过，请给出修正后的完整文件：\n{problems}\n",
            context=_gather_context(root, context_files),
        )

    return CodeGenResult(
        ok=False,
        summary=f"经过 {max_fix_rounds} 轮仍未通过静态检查",
        changed_files=all_changed,
        staticcheck_report=report,
        raw="\n---\n".join(raw_all),
    )
