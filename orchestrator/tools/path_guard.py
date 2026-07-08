"""Write-path + sensitive-filename guard.

Implements docs/PoC设计书 section 5.4: file operations must reject any path
outside the per-run workspace root (traversal via ``..`` or an absolute path
that escapes the root) and must reject sensitive filenames (``.env``, ``*.pem``,
``id_rsa`` …) so an agent can never read or clobber credentials.

deepagents' FilesystemBackend(virtual_mode=True) already blocks traversal for
its *built-in* fs tools. This module is the reusable guard for the tools we
build ourselves (git, staticcheck, and the Phase 3 write path) and adds the
sensitive-filename layer that virtual_mode does not cover.
"""
from __future__ import annotations

import re
from pathlib import Path

# Filenames / patterns that must never be read or written by an agent.
_SENSITIVE_PATTERNS = [
    re.compile(r"^\.env($|\.)", re.IGNORECASE),        # .env, .env.local, .env.prod ...
    re.compile(r"\.pem$", re.IGNORECASE),               # *.pem
    re.compile(r"\.key$", re.IGNORECASE),               # *.key
    re.compile(r"^id_rsa($|\.)", re.IGNORECASE),        # id_rsa, id_rsa.pub
    re.compile(r"^id_ed25519($|\.)", re.IGNORECASE),    # id_ed25519, id_ed25519.pub
    re.compile(r"^\.git-credentials$", re.IGNORECASE),
    re.compile(r"^\.npmrc$", re.IGNORECASE),            # may hold registry auth tokens
    re.compile(r"^\.pypirc$", re.IGNORECASE),           # may hold PyPI credentials
    re.compile(r"\.p12$", re.IGNORECASE),
    re.compile(r"\.pfx$", re.IGNORECASE),
]


class PathGuardError(ValueError):
    """Raised when a candidate path escapes the workspace or targets a secret."""


def is_sensitive_filename(name: str) -> bool:
    """True if *name* (a bare filename, not a path) matches a secret pattern."""
    return any(pat.search(name) for pat in _SENSITIVE_PATTERNS)


def resolve_safe_path(workspace_root: Path | str, candidate: str) -> Path:
    """Resolve *candidate* against *workspace_root* and validate it.

    *candidate* may be given relative to the workspace root or as a virtual
    absolute path (a leading ``/`` is treated as "workspace root", matching the
    virtual-path convention deepagents' built-in fs tools use).

    Returns the resolved absolute Path on success.

    Raises:
        PathGuardError: if the resolved path escapes *workspace_root*, or the
            filename matches a sensitive pattern.
    """
    root = Path(workspace_root).resolve()

    # Treat a leading "/" as workspace-root-relative (virtual absolute path),
    # so the guard agrees with how the built-in fs tools present paths to the LLM.
    cleaned = candidate.lstrip("/\\") if candidate.startswith(("/", "\\")) else candidate
    resolved = (root / cleaned).resolve()

    if resolved != root and root not in resolved.parents:
        raise PathGuardError(
            f"path escapes workspace root: {candidate!r} resolved to {resolved} "
            f"(root: {root})"
        )

    if is_sensitive_filename(resolved.name):
        raise PathGuardError(f"refusing to touch sensitive file: {resolved.name!r}")

    return resolved
