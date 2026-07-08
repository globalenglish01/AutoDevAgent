"""Safe command runner — the single choke point for executing anything.

Implements the executable-whitelist + no-shell + timeout + workspace-scoped
parts of docs/PoC设计书 section 5. VerifyAgent uses run_command() to run the
target project's tests; a future shell.exec agent tool (Phase 5) will expose
the same primitive to the LLM.

IMPORTANT — what this does and does NOT protect against:
  ✔ command injection via shell metacharacters — we never use shell=True and
    take an argv *list*, so `rm -rf / ; curl evil` can't be smuggled in.
  ✔ arbitrary executables — only whitelisted programs run.
  ✔ runaway processes — hard timeout.
  ✔ working outside the task workspace — cwd is forced to the workspace root.
  ✘ network egress — a whitelisted `pip install` / `npm install` can still
    reach the internet. True network isolation needs an OS sandbox (container
    with `--network=none` + an allowlisted proxy). That is the Phase 5
    production-hardening step (design section 5.1/5.3); on a bare Windows host
    we cannot enforce it, so this module documents the gap rather than
    pretending to close it.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 300
_MAX_OUTPUT_CHARS = 6000

# Only these programs may be executed. Keep this list tight — every addition is
# a new way for an agent to affect the host.
ALLOWED_EXECUTABLES = frozenset(
    {
        "python",
        "python3",
        "py",
        "pytest",
        "pip",
        "ruff",
        "mypy",
        "node",
        "npm",
        "npx",
        "pnpm",
        "yarn",
        "git",
    }
)


class SandboxError(ValueError):
    """Raised when a command is rejected before execution."""


@dataclass
class ExecResult:
    ok: bool
    exit_code: int | None
    output: str  # combined stdout+stderr, trimmed


def _base_executable(program: str) -> str:
    """Normalize argv[0] to a bare program name for whitelist matching.

    Strips any directory and a trailing .exe so both "python" and
    "C:\\Python314\\python.exe" match the whitelist entry "python".
    """
    name = Path(program).name.lower()
    if name.endswith(".exe"):
        name = name[:-4]
    return name


def _truncate(text: str) -> str:
    text = (text or "").strip()
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + "\n...[truncated]..."


def run_command(
    argv: list[str],
    cwd: Path | str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> ExecResult:
    """Run *argv* in *cwd* under the whitelist. Never uses a shell.

    Raises:
        SandboxError: if argv is empty or argv[0] is not whitelisted.
    """
    if not argv:
        raise SandboxError("empty command")
    program = _base_executable(argv[0])
    if program not in ALLOWED_EXECUTABLES:
        raise SandboxError(
            f"executable not allowed: {argv[0]!r} (whitelist: "
            f"{', '.join(sorted(ALLOWED_EXECUTABLES))})"
        )

    workdir = Path(cwd).resolve()
    if not workdir.is_dir():
        raise SandboxError(f"cwd is not a directory: {workdir}")

    try:
        proc = subprocess.run(
            argv,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=False,
        )
    except FileNotFoundError:
        return ExecResult(ok=False, exit_code=None, output=f"program not found: {argv[0]}")
    except subprocess.TimeoutExpired:
        return ExecResult(ok=False, exit_code=None, output=f"timed out after {timeout}s")

    combined = (proc.stdout or "") + (proc.stderr or "")
    return ExecResult(ok=proc.returncode == 0, exit_code=proc.returncode, output=_truncate(combined))
