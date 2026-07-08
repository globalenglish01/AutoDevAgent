"""Phase 6 — production wiring smoke test.

build_full_pipeline() imports and assembles every real agent against the
llm_bridge. This test compiles the graph (no invocation, so no bridge needed)
to prove the full wiring has no import/assembly errors — the thing most likely
to break silently until someone tries a real run.
"""
from __future__ import annotations

import subprocess

from orchestrator.pipeline.production import build_full_pipeline


def test_full_pipeline_compiles(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    pipeline = build_full_pipeline(tmp_path, max_code_retries=2)
    # Compiled LangGraph exposes ainvoke; all nodes wired without error.
    assert hasattr(pipeline, "ainvoke")
    graph = pipeline.get_graph()
    node_names = set(graph.nodes.keys())
    for expected in {"init", "requirement", "design", "code", "staticcheck",
                     "review", "verify", "deploy"}:
        assert expected in node_names, f"missing node: {expected}"
