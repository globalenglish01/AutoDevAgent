"""Per-run workspace isolation helpers for agent factories.

Ported verbatim from TestAgent's backend/app/agents/_utils/workspace.py — no
TestAgent-specific dependencies, only stdlib + deepagents.backends.
"""
from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path

from deepagents.backends import FilesystemBackend

_log = logging.getLogger(__name__)


def create_run_workspace(
    root: Path, org_id: str = "", agent_name: str = "agent"
) -> tuple[str, Path, FilesystemBackend]:
    """Create an isolated per-run workspace directory.

    Returns (run_id, workspace_path, filesystem_backend).
    Call cleanup_run_workspace() in a finally block to remove the directory.
    """
    run_id = str(uuid.uuid4())
    run_workspace = (root / org_id / run_id) if org_id else (root / run_id)
    run_workspace.mkdir(parents=True, exist_ok=True)
    backend = FilesystemBackend(root_dir=run_workspace, virtual_mode=True)
    _log.debug("[%s] Isolated workspace: %s", agent_name, run_workspace)
    return run_id, run_workspace, backend


def cleanup_run_workspace(path: Path, agent_name: str = "agent") -> None:
    """Remove the workspace directory created by create_run_workspace."""
    try:
        shutil.rmtree(path, ignore_errors=True)
        _log.debug("[%s] Cleaned up workspace: %s", agent_name, path)
    except Exception as _e:
        _log.warning("[%s] Failed to clean up workspace %s: %s", agent_name, path, _e)
