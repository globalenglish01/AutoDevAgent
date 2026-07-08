"""VerifyAgent — the real-execution backstop of the pipeline.

Design section 3.1: after StaticCheck (deterministic) and ReviewAgent (ChatGPT
judgment), VerifyAgent actually *runs the target project's tests*. This is the
ground truth — code that compiles, lints, type-checks and reads well can still
fail its tests. Runs through sandbox_exec (whitelist + timeout + no shell), so
it never executes anything outside the allowed set.

Deterministic on purpose: it does not need an LLM. On failure it returns the raw
test output, which the pipeline feeds straight back to CodeAgent (DeepSeek) to
fix — no extra browser round is spent just to "interpret" the failure.

Python-focused for now (pytest). Detection is intentionally simple; broaden
per-language later alongside StaticCheck's JS/TS support.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from orchestrator.tools.sandbox_exec import DEFAULT_TIMEOUT_SECONDS, run_command

# pytest's exit code for "no tests were collected".
_PYTEST_NO_TESTS = 5


@dataclass
class VerifyResult:
    ok: bool
    output: str
    ran: bool  # False when there was nothing to run (no tests detected)

    @property
    def summary(self) -> str:
        if not self.ran:
            return "NO_TESTS (nothing to verify)"
        return "TESTS_PASSED" if self.ok else "TESTS_FAILED"


def _has_pytest_target(repo: Path) -> bool:
    if (repo / "pytest.ini").exists() or (repo / "conftest.py").exists():
        return True
    if (repo / "tests").is_dir() or (repo / "test").is_dir():
        return True
    if (repo / "pyproject.toml").exists() or (repo / "setup.cfg").exists():
        return True
    # Any test_*.py / *_test.py at the top level.
    return any(repo.glob("test_*.py")) or any(repo.glob("*_test.py"))


def run_verification(
    repo: Path | str,
    test_command: list[str] | None = None,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> VerifyResult:
    """Run the target repo's tests. Returns a VerifyResult.

    *test_command* overrides auto-detection (must start with a whitelisted
    executable — sandbox_exec enforces this).
    """
    root = Path(repo).resolve()

    if test_command is None:
        if not _has_pytest_target(root):
            return VerifyResult(ok=True, output="(no test target detected)", ran=False)
        test_command = [sys.executable, "-m", "pytest", "-q"]

    result = run_command(test_command, cwd=root, timeout=timeout)

    if result.exit_code == _PYTEST_NO_TESTS:
        return VerifyResult(ok=True, output="(no tests collected)", ran=False)

    return VerifyResult(ok=result.ok, output=result.output, ran=True)
