"""Git tools for the code agent — status / diff / branch / commit.

Bound to a per-run workspace root via build_git_tools(). Two safety rules from
docs/PoC设计书:
  - create_branch refuses main/master/HEAD, so the agent always works on a
    feature branch (section 3.2 — never commit directly on the mainline).
  - commit refuses to stage a sensitive file (.env/*.pem/...) even if the agent
    somehow created one, reusing the path_guard filename patterns (section 5.4).

git.push is intentionally NOT exposed in Phase 2: the isolated test workspace
has no remote, and pushing needs the feature-branch-only guard + credential
handling that belong with the Phase 3+ DeployAgent.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from orchestrator.tools.path_guard import is_sensitive_filename

_GIT_TIMEOUT_SECONDS = 60
_MAX_OUTPUT_CHARS = 4000
_PROTECTED_BRANCHES = {"main", "master", "head"}

# Commit identity passed per-invocation so commits work without relying on the
# workspace repo (or global) git config being set.
_COMMIT_IDENTITY = ["-c", "user.name=AutoDevAgent", "-c", "user.email=autodev@localhost"]


def _truncate(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + "\n...[truncated]..."


def _run_git(args: list[str], cwd: Path) -> tuple[bool, str]:
    """Run `git <args>` in *cwd*. Returns (ok, combined_output)."""
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError:
        return False, "git not available on PATH"
    except subprocess.TimeoutExpired:
        return False, f"git timed out after {_GIT_TIMEOUT_SECONDS}s"
    combined = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, _truncate(combined)


def build_git_tools(workspace_root: Path | str) -> list[BaseTool]:
    """Build the git tools bound to *workspace_root* (must be a git repo)."""
    root = Path(workspace_root).resolve()

    @tool
    def git_status() -> str:
        """Show the working-tree status of the workspace repo (git status --short)."""
        ok, output = _run_git(["status", "--short"], root)
        if not ok:
            return f"FAIL: {output}"
        return output or "(clean working tree)"

    @tool
    def git_diff(staged: bool = False) -> str:
        """Show uncommitted changes as a unified diff.

        Set staged=True to show only staged changes (git diff --cached),
        otherwise shows unstaged working-tree changes.
        """
        args = ["diff", "--cached"] if staged else ["diff"]
        ok, output = _run_git(args, root)
        if not ok:
            return f"FAIL: {output}"
        return output or "(no changes)"

    @tool
    def git_create_branch(name: str) -> str:
        """Create and switch to a new feature branch (git checkout -b <name>).

        Refuses main/master/HEAD — the agent must always work on a feature
        branch, never directly on the mainline.
        """
        if name.strip().lower() in _PROTECTED_BRANCHES:
            return f"REJECTED: refusing to create/switch to protected branch {name!r}"
        ok, output = _run_git(["checkout", "-b", name], root)
        return f"created and switched to branch {name!r}" if ok else f"FAIL: {output}"

    @tool
    def git_commit(message: str) -> str:
        """Stage all changes and commit them with *message*.

        Refuses the commit if any staged path is a sensitive file
        (.env / *.pem / id_rsa / ...), unstaging everything so nothing leaks.
        """
        ok, output = _run_git(["add", "-A"], root)
        if not ok:
            return f"FAIL (stage): {output}"

        ok, staged = _run_git(["diff", "--cached", "--name-only"], root)
        if not ok:
            return f"FAIL (inspect staged): {staged}"

        staged_files = [line.strip() for line in staged.splitlines() if line.strip()]
        offending = [f for f in staged_files if is_sensitive_filename(Path(f).name)]
        if offending:
            _run_git(["reset"], root)  # unstage everything so the secret never gets committed
            return (
                "REJECTED: refusing to commit sensitive file(s): "
                f"{', '.join(offending)}. Nothing was committed."
            )

        if not staged_files:
            return "nothing to commit (working tree clean)"

        ok, output = _run_git([*_COMMIT_IDENTITY, "commit", "-m", message], root)
        return f"committed {len(staged_files)} file(s)" if ok else f"FAIL (commit): {output}"

    return [git_status, git_diff, git_create_branch, git_commit]
