"""AutoDevAgent code tools (Phase 2).

Native Python LangChain tools the deep agent calls to operate on a target
repository: static checks (ruff/mypy/py_compile), git, and a shared path guard.

Deviation from docs/PoC设计书 section 3.2 (documented there too): the design
proposed these as a Node.js MCP server ("code-mcp"). They ship as in-process
Python tools instead because (a) deepagents already provides fs.read/write/
edit/ls/glob/grep for free via FilesystemMiddleware when a backend is passed
(proven in Phase 1), and (b) git/ruff/mypy/py_compile are Python/CLI tools —
wrapping them in a Node process that shells out to Python would add a runtime
and IPC hop for zero benefit. The separate-sandboxed-process pattern from the
design is reserved for shell.exec (Phase 5), where process isolation actually
matters.
"""

from orchestrator.tools.git_tools import build_git_tools
from orchestrator.tools.path_guard import PathGuardError, is_sensitive_filename, resolve_safe_path
from orchestrator.tools.staticcheck import build_staticcheck_tools

__all__ = [
    "build_git_tools",
    "build_staticcheck_tools",
    "PathGuardError",
    "is_sensitive_filename",
    "resolve_safe_path",
]
