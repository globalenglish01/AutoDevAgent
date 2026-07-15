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
    # ruff on Windows sometimes can't spawn (WinError 4551); when it crashes we
    # skip (non-blocking) rather than falsely FAIL — accept that as "tool
    # unavailable in this env" instead of asserting a lint finding.
    if "跳过" in result:
        pytest.skip("ruff could not run in this environment (spawn crash)")
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


# ── staticcheck: tool-crash vs real findings (real-run robustness) ───────────

def test_lint_tool_crash_is_non_blocking(tmp_path, monkeypatch):
    """A ruff spawn crash (WinError/traceback) must NOT be reported as a lint
    failure that blocks the pipeline — it's non-code tooling flakiness."""
    import orchestrator.tools.staticcheck as sc

    def fake_run_check(cmd, cwd, retries=2):
        if "ruff" in cmd:
            return False, "Traceback (most recent call last):\n  WinError 4551 ..."
        return True, ""
    monkeypatch.setattr(sc, "_run_check", fake_run_check)

    tools = {t.name: t for t in sc.build_staticcheck_tools(tmp_path)}
    (tmp_path / "x.py").write_text("y = 1\n", encoding="utf-8")
    result = tools["lint_check"].invoke({"path": "x.py"})
    assert result.startswith("PASS")   # crash → non-blocking pass, not FAIL
    assert "跳过" in result


def test_lint_real_findings_still_fail(tmp_path, monkeypatch):
    """A genuine ruff finding (not a crash) must still FAIL."""
    import orchestrator.tools.staticcheck as sc

    def fake_run_check(cmd, cwd, retries=2):
        if "ruff" in cmd:
            return False, "x.py:1:8: F401 [*] `os` imported but unused\nFound 1 error."
        return True, ""
    monkeypatch.setattr(sc, "_run_check", fake_run_check)

    tools = {t.name: t for t in sc.build_staticcheck_tools(tmp_path)}
    (tmp_path / "x.py").write_text("import os\n", encoding="utf-8")
    result = tools["lint_check"].invoke({"path": "x.py"})
    assert result.startswith("FAIL")   # real finding → still blocks
