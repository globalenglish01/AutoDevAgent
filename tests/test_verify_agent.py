"""Phase 3c tests — VerifyAgent runs real pytest in a sample repo."""
from __future__ import annotations

from orchestrator.agents.verify.agent import run_verification


def _write(repo, name, content):
    (repo / name).write_text(content, encoding="utf-8")


def test_passing_tests(tmp_path):
    _write(tmp_path, "test_ok.py", "def test_ok():\n    assert 1 + 1 == 2\n")
    result = run_verification(tmp_path)
    assert result.ran
    assert result.ok
    assert result.summary == "TESTS_PASSED"


def test_failing_tests(tmp_path):
    _write(tmp_path, "test_bad.py", "def test_bad():\n    assert 1 == 2\n")
    result = run_verification(tmp_path)
    assert result.ran
    assert not result.ok
    assert result.summary == "TESTS_FAILED"
    assert "assert" in result.output.lower() or "fail" in result.output.lower()


def test_no_tests_detected(tmp_path):
    # Bare dir with a non-test file: nothing to verify.
    _write(tmp_path, "main.py", "print('hi')\n")
    result = run_verification(tmp_path)
    assert not result.ran
    assert result.ok  # nothing to run is not a failure
    assert result.summary.startswith("NO_TESTS")
