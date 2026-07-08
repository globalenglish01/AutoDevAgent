"""Output trimming helpers — keep ToolMessage payloads small so they don't
sink into agent history and cost tokens forever.

Ported from TestAgent's backend/app/agents/_utils/output_trim.py, no logic
changes — only the log-dir env var is renamed (AGENT_LOG_DIR -> AUTODEV_LOG_DIR)
so the two projects don't collide if both happen to run on the same machine.

The pattern: subprocess produces big stdout/stderr (build/test output can be
10-50 KB), we dump the full log to disk, then return only a short summary +
last N chars to the LLM. The agent can read the full file later if needed.
"""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime
from pathlib import Path

DEFAULT_TAIL_CHARS = 1000
DEFAULT_LOG_DIR = Path(os.environ.get("AUTODEV_LOG_DIR", "./.autodev_logs"))

# Patterns that leak internal server paths to the LLM via stderr.
# Replace absolute paths with short placeholders so the LLM cannot discover
# workspace layout or server filesystem structure.
_PATH_SANITIZE_PATTERNS = [
    # Unix absolute paths containing "agent", "workspace", "app", "tmp", "home"
    (re.compile(r"/(?:home|app|tmp|var|opt|root|usr)[^\s:\"']+"), "<path>"),
    # Windows-style absolute paths (drive letter prefix)
    (re.compile(r"[A-Za-z]:\\[^\s\"']*"), "<path>"),
]


def _sanitize_paths(text: str) -> str:
    """Remove absolute filesystem paths from text returned to the LLM."""
    for pattern, replacement in _PATH_SANITIZE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# Common build/test-runner result lines we care about, in priority order
_SUMMARY_PATTERNS = [
    re.compile(r"^\s*\d+\s+(?:passed|failed|skipped|flaky|interrupted)\b.*", re.M | re.I),
    re.compile(r"^\s*Error:.*", re.M),
    re.compile(r"^\s*✘\s+\d+\s+.+", re.M),
    re.compile(r"^\s*✓\s+\d+\s+.+", re.M),
    re.compile(r"^\s*at\s+.+:\d+:\d+.*", re.M),  # stack trace top frames
]


def _strip_ansi(text: str) -> str:
    """Remove ANSI color codes that test/build tools emit."""
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text or "")


def _extract_summary(stdout: str, max_lines: int = 12) -> str:
    """Pull the highest-signal lines (pass/fail counts + first errors)."""
    text = _strip_ansi(stdout or "")
    found: list[str] = []
    for pat in _SUMMARY_PATTERNS:
        for m in pat.finditer(text):
            line = m.group(0).strip()
            if line and line not in found:
                found.append(line)
                if len(found) >= max_lines:
                    return "\n".join(found)
    return "\n".join(found) if found else "(no recognizable test summary lines)"


def _tail(text: str, max_chars: int = DEFAULT_TAIL_CHARS) -> str:
    text = _strip_ansi(text or "")
    if len(text) <= max_chars:
        return text
    return "...[truncated]...\n" + text[-max_chars:]


def trim_test_output(
    stdout: str,
    stderr: str,
    max_chars: int = DEFAULT_TAIL_CHARS,
) -> dict:
    """Return a compact dict suitable for embedding in a ToolMessage."""
    return {
        "summary": _extract_summary(stdout),
        "stdout_tail": _tail(stdout, max_chars),
        "stderr_tail": _sanitize_paths(_tail(stderr, max_chars)),
        "stdout_full_chars": len(stdout or ""),
        "stderr_full_chars": len(stderr or ""),
    }


def dump_full_log(stdout: str, stderr: str, prefix: str = "run") -> str:
    """Write the full output to a log file and return its absolute path.

    The LLM can request the file via read_file if it really needs detail.
    """
    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}_{uuid.uuid4().hex[:8]}.log"
    path = DEFAULT_LOG_DIR / fname
    body = (
        f"===== STDOUT ({len(stdout or '')} chars) =====\n"
        f"{stdout or ''}\n"
        f"===== STDERR ({len(stderr or '')} chars) =====\n"
        f"{stderr or ''}\n"
    )
    path.write_text(body, encoding="utf-8", errors="replace")
    return str(path)
