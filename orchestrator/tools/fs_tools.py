"""Minimal workspace-scoped filesystem tools for the lean code agent.

deepagents' built-in fs tools carry long descriptions and come bundled with
write_todos / subagent middleware — together that tool prompt is too big for a
weak browser LLM (ChatGPT free choked on it, returning CHATGPT_ERROR). These
are deliberately tiny (one-line docstrings, few tools) so the whole tool schema
fits comfortably in the browser model's window. All paths go through path_guard
(no traversal, no secret files).
"""
from __future__ import annotations

from pathlib import Path

from langchain_core.tools import BaseTool, tool

from orchestrator.tools.path_guard import PathGuardError, resolve_safe_path

_MAX_READ_CHARS = 8000
_SKIP = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache", ".mypy_cache"}


def build_fs_tools(workspace_root: Path | str) -> list[BaseTool]:
    """Build list_dir / read_file / write_file bound to *workspace_root*."""
    root = Path(workspace_root).resolve()

    @tool
    def list_dir(path: str = ".") -> str:
        """List files under a workspace directory (relative path)."""
        try:
            target = root if path in ("", ".", "/") else resolve_safe_path(root, path)
        except PathGuardError as exc:
            return f"REJECTED: {exc}"
        if not target.exists():
            return f"(not found: {path})"
        names = []
        for p in sorted(target.rglob("*")):
            if any(part in _SKIP for part in p.relative_to(root).parts):
                continue
            if p.is_file():
                names.append(p.relative_to(root).as_posix())
        return "\n".join(names) or "(empty)"

    @tool
    def read_file(path: str) -> str:
        """Read a text file in the workspace (relative path)."""
        try:
            target = resolve_safe_path(root, path)
        except PathGuardError as exc:
            return f"REJECTED: {exc}"
        if not target.is_file():
            return f"(not found: {path})"
        text = target.read_text(encoding="utf-8", errors="replace")
        return text[:_MAX_READ_CHARS] + ("\n...[truncated]..." if len(text) > _MAX_READ_CHARS else "")

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file in the workspace (relative path)."""
        try:
            target = resolve_safe_path(root, path)
        except PathGuardError as exc:
            return f"REJECTED: {exc}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {path} ({len(content)} chars)"

    return [list_dir, read_file, write_file]
