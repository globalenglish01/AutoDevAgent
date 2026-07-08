"""Deterministic repository indexer (docs/PoC设计书 section 3.3).

Feeding a whole repo's source to a weak browser model blows the context budget,
so we build a compact map — directory tree + file inventory — for the planning
agents (RequirementAnalyzer, DesignAgent). No LLM, no cost. Deliberately simple
for now (tree + sizes); richer summaries / dependency graphs come later.
"""
from __future__ import annotations

from pathlib import Path

_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".next",
              "dist", "build", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_MAX_FILES = 400
_MAX_CHARS = 6000


def build_repo_index(repo: Path | str) -> str:
    """Return a compact text index of *repo* (relative paths, capped)."""
    root = Path(repo).resolve()
    if not root.is_dir():
        return f"(not a directory: {root})"

    lines: list[str] = []
    count = 0
    truncated = False
    for path in sorted(root.rglob("*")):
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.is_dir():
            continue
        count += 1
        if count > _MAX_FILES:
            truncated = True
            break
        rel = path.relative_to(root).as_posix()
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        lines.append(f"{rel} ({size}B)")

    body = "\n".join(lines)
    if len(body) > _MAX_CHARS:
        body = body[:_MAX_CHARS] + "\n...[truncated]..."
    header = f"仓库文件清单（根目录 {root.name}，共 {count}{'+' if truncated else ''} 个文件）:\n"
    return header + (body or "(空仓库)")
