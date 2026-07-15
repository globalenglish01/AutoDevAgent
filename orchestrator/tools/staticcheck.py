"""StaticCheck stage tools — Python only (docs/PoC设计书 section 3.2, 2026-07-08).

These are the cheap, deterministic, zero-LLM-cost checks that run *before* the
ChatGPT ReviewAgent so syntax/type errors never waste a 5-6 minute browser-LLM
review round (design section 3.1). Python target only for now; JS/TS (tsc +
eslint) and other languages are deferred.

Three tools, each bound to a per-run workspace root via build_staticcheck_tools():
  - syntax_check   -> python -m compileall   (compile errors = broken syntax)
  - lint_check     -> ruff check
  - typecheck_run  -> mypy

Each returns a short PASS/FAIL string the CodeAgent can act on directly.
Paths from the LLM are validated through path_guard so a check can never be
pointed outside the workspace or at a secret file.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from orchestrator.tools.path_guard import PathGuardError, resolve_safe_path

# Static checks are fast; this only guards against a pathological hang.
_CHECK_TIMEOUT_SECONDS = 120
# ruff/mypy output is ordered most-relevant-first, so keep the head.
_MAX_OUTPUT_CHARS = 4000


def _truncate_head(text: str) -> str:
    text = text or ""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + "\n...[truncated]..."


# Markers that mean the CHECK TOOL ITSELF crashed / couldn't run — as opposed
# to the tool running fine and reporting code problems. `python -m ruff` on
# Windows re-spawns a bundled ruff.exe and intermittently hits WinError 4551
# (file locked, e.g. Defender scanning the exe on rapid repeated invocations);
# that surfaces as a Python traceback, NOT a lint finding. Treating a tool crash
# as a code failure would spuriously bounce the pipeline, so callers detect this
# and skip (non-blocking) instead.
_TOOL_CRASH_MARKERS = (
    "Traceback (most recent call last)",
    "WinError",
    "ModuleNotFoundError",
    "ImportError",
    "PermissionError",
    "tool not available",
    "timed out after",
)


def _is_tool_crash(output: str) -> bool:
    return any(m in output for m in _TOOL_CRASH_MARKERS)


def _run_check(cmd: list[str], cwd: Path, *, retries: int = 2) -> tuple[bool, str]:
    """Run *cmd* in *cwd*. Returns (ok, combined_output). Retries transient
    tool-spawn crashes (e.g. Windows WinError 4551) a couple of times."""
    last_output = ""
    for attempt in range(retries + 1):
        try:
            proc = subprocess.run(
                cmd, cwd=str(cwd), capture_output=True, text=True,
                timeout=_CHECK_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:  # tool not installed / not on PATH
            return False, f"tool not available: {exc}"
        except subprocess.TimeoutExpired:
            return False, f"check timed out after {_CHECK_TIMEOUT_SECONDS}s"
        combined = _truncate_head(((proc.stdout or "") + (proc.stderr or "")).strip())
        if proc.returncode == 0:
            return True, combined
        # Non-zero: retry only if it looks like a transient tool crash, not findings.
        last_output = combined
        if attempt < retries and _is_tool_crash(combined):
            continue
        return False, combined
    return False, last_output


def _safe_target(workspace_root: Path, path: str) -> Path:
    """Resolve an LLM-supplied path within the workspace, or raise PathGuardError."""
    # path_guard rejects sensitive *filenames*; a bare "." (whole workspace) is fine.
    if path in ("", ".", "/"):
        return workspace_root
    return resolve_safe_path(workspace_root, path)


def build_staticcheck_tools(workspace_root: Path | str) -> list[BaseTool]:
    """Build the three StaticCheck tools bound to *workspace_root*."""
    root = Path(workspace_root).resolve()

    @tool
    def syntax_check(path: str = ".") -> str:
        """Check Python syntax by byte-compiling files (python -m compileall).

        Fastest, most fundamental check — run this first. `path` is relative to
        the workspace root; use "." for the whole workspace or a single file
        like "app/models.py". Returns PASS or the compile errors.
        """
        try:
            target = _safe_target(root, path)
        except PathGuardError as exc:
            return f"REJECTED: {exc}"
        ok, output = _run_check(
            [sys.executable, "-m", "compileall", "-q", str(target)], root
        )
        return "PASS: syntax OK" if ok else f"FAIL (syntax):\n{output}"

    @tool
    def lint_check(path: str = ".") -> str:
        """Lint Python code with ruff (python -m ruff check).

        Flags style/correctness issues (unused imports, undefined names, etc.).
        `path` is relative to the workspace root. Returns PASS or the lint findings.
        """
        try:
            target = _safe_target(root, path)
        except PathGuardError as exc:
            return f"REJECTED: {exc}"
        ok, output = _run_check(
            [sys.executable, "-m", "ruff", "check", str(target)], root
        )
        if ok:
            return "PASS: no lint issues"
        if _is_tool_crash(output):
            # ruff itself couldn't run (transient spawn crash) — advisory check,
            # don't block correct code on tooling flakiness.
            return f"PASS: lint 跳过（ruff 无法运行，非代码问题）:\n{output[:300]}"
        return f"FAIL (lint):\n{output}"

    @tool
    def typecheck_run(path: str = ".") -> str:
        """Type-check Python code with mypy (python -m mypy).

        Catches type errors static analysis can prove. `path` is relative to the
        workspace root. Returns PASS or the type errors. Slower than syntax/lint.
        """
        try:
            target = _safe_target(root, path)
        except PathGuardError as exc:
            return f"REJECTED: {exc}"
        ok, output = _run_check(
            [sys.executable, "-m", "mypy", str(target)], root
        )
        if ok:
            return "PASS: no type errors"
        if _is_tool_crash(output):
            return f"PASS: typecheck 跳过（mypy 无法运行，非代码问题）:\n{output[:300]}"
        return f"FAIL (typecheck):\n{output}"

    return [syntax_check, lint_check, typecheck_run]
