"""Web UI for AutoDevAgent — a thin FastAPI layer over the pipeline.

Submit a task in the browser, watch the pipeline stream its progress, and
approve the result (commit on a feature branch + push if a remote exists).
The heavy logic stays in orchestrator.pipeline / agents; this is just a driver.
"""

from orchestrator.web.app import apply_and_push, create_app, execute_run

__all__ = ["create_app", "execute_run", "apply_and_push"]
