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

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TIMEOUT_SECONDS = 300
_MAX_OUTPUT_CHARS = 6000

# Shell control tokens. We never use shell=True, so these are already inert
# (passed as literal argv to the program), but we reject them as standalone
# tokens so an agent can't believe it chained commands — and as defense-in-depth.
# Checked per-token AFTER shlex.split, so a ";" inside a quoted string (e.g.
# python -c "import sys; sys.exit(1)") is fine — only a bare ";" token is not.
_SHELL_CONTROL_TOKENS = {";", "|", "&", "&&", "||", ">", ">>", "<", "`"}


def _has_control_token(argv: list[str]) -> bool:
    return any(
        tok in _SHELL_CONTROL_TOKENS or tok.startswith((">", "<", "|", "&", "`"))
        for tok in argv
    )

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


def build_exec_tool(workspace_root: Path | str, timeout: int = DEFAULT_TIMEOUT_SECONDS):
    """Build a guarded shell.exec agent tool bound to *workspace_root*.

    This is the design-section-5 `shell.exec`: controlled command execution
    exposed to an LLM. It splits the command string, rejects shell
    metacharacters, and runs through run_command (whitelist + no-shell +
    timeout + workspace cwd). Returns a LangChain tool.
    """
    from langchain_core.tools import tool

    root = Path(workspace_root).resolve()

    @tool
    def run_shell(command: str) -> str:
        """Run a build/test/package command in the project workspace.

        Only whitelisted programs are allowed (python, pytest, pip, npm, node,
        git, ruff, mypy, ...). Command chaining / redirection (; | & > <) is
        rejected — run one command per call. Returns exit status + output.
        """
        try:
            # posix=True gives correct quote handling; commands are simple
            # build/test invocations (`python -m pytest`, `npm test`) without
            # backslash exe paths, so backslash-as-escape is not a concern here.
            argv = shlex.split(command)
        except ValueError as exc:
            return f"REJECTED: could not parse command: {exc}"
        if not argv:
            return "REJECTED: empty command"
        if _has_control_token(argv):
            return (
                "REJECTED: command chaining/redirection not allowed "
                "(one command per call, no ; | & > < ` as separate tokens)."
            )
        try:
            result = run_command(argv, cwd=root, timeout=timeout)
        except SandboxError as exc:
            return f"REJECTED: {exc}"
        status = "OK" if result.ok else f"FAILED (exit {result.exit_code})"
        return f"{status}\n{result.output}"

    return run_shell
