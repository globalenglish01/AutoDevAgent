"""Phase 2 tool tests — exercise the real subprocess tools (ruff/mypy/git run
for real), not mocks. These prove StaticCheck actually catches bad code and
passes good code, that git tools operate on a real repo, and that the path
guard blocks escapes and secrets.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from orchestrator.tools.git_tools import build_git_tools
from orchestrator.tools.path_guard import (
    PathGuardError,
    is_sensitive_filename,
    resolve_safe_path,
)
from orchestrator.tools.staticcheck import build_staticcheck_tools


# ── path_guard ──────────────────────────────────────────────────────────────

def test_path_guard_accepts_normal_file(tmp_path):
    resolved = resolve_safe_path(tmp_path, "app/models.py")
    assert resolved == (tmp_path / "app" / "models.py").resolve()


def test_path_guard_rejects_traversal(tmp_path):
    with pytest.raises(PathGuardError):
        resolve_safe_path(tmp_path, "../../etc/passwd")


def test_path_guard_rejects_absolute_escape(tmp_path):
    # A real absolute path outside the workspace (leading-slash virtual paths are
    # treated as workspace-relative, so use a Windows drive / other-root path).
    outside = Path(tmp_path.anchor) / "somewhere_else" / "secret.txt"
    with pytest.raises(PathGuardError):
        resolve_safe_path(tmp_path, str(outside))


@pytest.mark.parametrize(
    "name",
    [".env", ".env.local", "server.pem", "private.key", "id_rsa", "id_rsa.pub", ".git-credentials"],
)
def test_path_guard_rejects_sensitive_names(name):
    assert is_sensitive_filename(name)


def test_path_guard_rejects_sensitive_file_in_workspace(tmp_path):
    with pytest.raises(PathGuardError):
        resolve_safe_path(tmp_path, ".env")


# ── staticcheck ───────────────────────────────────────────────────────────────

def _tools_by_name(tools):
    return {t.name: t for t in tools}


def test_syntax_check_catches_broken_then_passes_fixed(tmp_path):
    tools = _tools_by_name(build_staticcheck_tools(tmp_path))
    syntax_check = tools["syntax_check"]

    bad = tmp_path / "broken.py"
    bad.write_text("def f(:\n    pass\n", encoding="utf-8")  # invalid syntax
    result = syntax_check.invoke({"path": "broken.py"})
    assert result.startswith("FAIL")

    bad.write_text("def f():\n    pass\n", encoding="utf-8")  # valid now
    result = syntax_check.invoke({"path": "broken.py"})
    assert result.startswith("PASS")


def test_lint_check_flags_unused_import(tmp_path):
    tools = _tools_by_name(build_staticcheck_tools(tmp_path))
    lint_check = tools["lint_check"]

    f = tmp_path / "lint_me.py"
    f.write_text("import os\n", encoding="utf-8")  # F401 unused import
    result = lint_check.invoke({"path": "lint_me.py"})
    assert result.startswith("FAIL")

    f.write_text("x = 1\n", encoding="utf-8")
    result = lint_check.invoke({"path": "lint_me.py"})
    assert result.startswith("PASS")


def test_typecheck_flags_type_error(tmp_path):
    tools = _tools_by_name(build_staticcheck_tools(tmp_path))
    typecheck_run = tools["typecheck_run"]

    f = tmp_path / "typed.py"
    f.write_text("def add(a: int, b: int) -> int:\n    return a + b\n\nadd('x', 1)\n", encoding="utf-8")
    result = typecheck_run.invoke({"path": "typed.py"})
    assert result.startswith("FAIL")


def test_staticcheck_rejects_sensitive_path(tmp_path):
    tools = _tools_by_name(build_staticcheck_tools(tmp_path))
    result = tools["syntax_check"].invoke({"path": ".env"})
    assert result.startswith("REJECTED")


# ── git tools ─────────────────────────────────────────────────────────────────

@pytest.fixture
def git_repo(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "README.md").write_text("# sample\n", encoding="utf-8")
    return tmp_path


def test_git_status_and_commit_flow(git_repo):
    tools = _tools_by_name(build_git_tools(git_repo))

    status = tools["git_status"].invoke({})
    assert "README.md" in status  # untracked

    commit_result = tools["git_commit"].invoke({"message": "initial commit"})
    assert "committed" in commit_result

    # After commit the tree is clean.
    assert tools["git_status"].invoke({}) == "(clean working tree)"


def test_git_create_branch_rejects_protected(git_repo):
    tools = _tools_by_name(build_git_tools(git_repo))
    # Need at least one commit before branching.
    (git_repo / "f.txt").write_text("x\n", encoding="utf-8")
    tools["git_commit"].invoke({"message": "init"})

    assert tools["git_create_branch"].invoke({"name": "main"}).startswith("REJECTED")
    assert tools["git_create_branch"].invoke({"name": "master"}).startswith("REJECTED")

    ok = tools["git_create_branch"].invoke({"name": "feature/add-x"})
    assert "feature/add-x" in ok


def test_git_commit_refuses_sensitive_file(git_repo):
    tools = _tools_by_name(build_git_tools(git_repo))
    (git_repo / ".env").write_text("SECRET=hunter2\n", encoding="utf-8")

    result = tools["git_commit"].invoke({"message": "oops"})
    assert result.startswith("REJECTED")
    assert ".env" in result

    # Nothing was committed, and .env was unstaged.
    log = subprocess.run(["git", "log", "--oneline"], cwd=git_repo, capture_output=True, text=True)
    assert log.stdout.strip() == ""  # no commits
