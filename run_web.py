"""Launch the AutoDevAgent web UI.

    python run_web.py            # http://127.0.0.1:8770
    python run_web.py --port 9000

Prerequisite: start a ChatGPT browser bridge (llm_bridge) on port 8765 first,
same as the CLI. Then open the page, enter a repo path + task, and run.
"""
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoDevAgent Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    print(f"AutoDevAgent Web UI: http://{args.host}:{args.port}")
    uvicorn.run("orchestrator.web.app:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
