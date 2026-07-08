"""Phase 3a tests — sandbox_exec whitelist + real execution."""
from __future__ import annotations

import sys

import pytest

from orchestrator.tools.sandbox_exec import SandboxError, run_command


def test_runs_whitelisted_python(tmp_path):
    result = run_command([sys.executable, "-c", "print('hi')"], cwd=tmp_path)
    assert result.ok
    assert result.exit_code == 0
    assert "hi" in result.output


def test_rejects_non_whitelisted_executable(tmp_path):
    with pytest.raises(SandboxError):
        run_command(["curl", "http://evil.example"], cwd=tmp_path)


def test_rejects_empty_command(tmp_path):
    with pytest.raises(SandboxError):
        run_command([], cwd=tmp_path)


def test_nonzero_exit_is_not_ok(tmp_path):
    result = run_command([sys.executable, "-c", "import sys; sys.exit(3)"], cwd=tmp_path)
    assert not result.ok
    assert result.exit_code == 3


def test_full_path_python_is_normalized_and_allowed(tmp_path):
    # argv[0] with a directory + .exe should still match the "python" whitelist entry.
    result = run_command([sys.executable, "-c", "print(1)"], cwd=tmp_path)
    assert result.ok


def test_timeout(tmp_path):
    result = run_command(
        [sys.executable, "-c", "import time; time.sleep(5)"], cwd=tmp_path, timeout=1
    )
    assert not result.ok
    assert "timed out" in result.output
