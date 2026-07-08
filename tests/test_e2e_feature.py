"""Phase 6 — realistic end-to-end: implement a real feature through the full
pipeline. LLM stages are scripted (deterministic), but StaticCheck and Verify
run FOR REAL (ruff/mypy/pytest actually execute against the sample project).
This is the closest we get to a real run without driving the 5-6 min browser LLM.

Scenario: the sample calculator has add/subtract; the task is to add multiply.
The scripted coder implements it; the real gates confirm it's correct.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from orchestrator.agents.deploy.agent import Deployment
from orchestrator.agents.design.agent import Design
from orchestrator.agents.requirement.agent import Requirements
from orchestrator.agents.review.agent import ReviewVerdict
from orchestrator.pipeline.graph import build_pipeline

_SAMPLE = Path(__file__).resolve().parent.parent / "examples" / "sample_project"


def _seed_repo(tmp_path):
    for f in _SAMPLE.glob("*.py"):
        shutil.copy(f, tmp_path / f.name)
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A"],
                   cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    return tmp_path


@pytest.mark.asyncio
async def test_implement_multiply_end_to_end(tmp_path):
    repo = _seed_repo(tmp_path)

    async def analyzer(task, repo_summary):
        return Requirements(
            clarified_task="给 calculator.py 增加 multiply(a, b) 并补测试",
            acceptance_criteria=["multiply(2,3)==6", "现有测试仍通过"],
        )

    async def designer(clarified, criteria, repo_summary):
        return Design(
            summary="在 calculator.py 加 multiply，test_calculator.py 加用例",
            files=["calculator.py", "test_calculator.py"],
            steps=["实现 multiply", "加 test_multiply"],
        )

    async def coder(task, repo_path, feedback):
        # Real implementation edit (this is what the DeepSeek CodeAgent would do).
        calc = Path(repo_path) / "calculator.py"
        calc.write_text(
            calc.read_text(encoding="utf-8")
            + "\n\ndef multiply(a: float, b: float) -> float:\n    return a * b\n",
            encoding="utf-8",
        )
        test = Path(repo_path) / "test_calculator.py"
        test.write_text(
            test.read_text(encoding="utf-8").replace(
                "from calculator import add, subtract",
                "from calculator import add, multiply, subtract",
            )
            + "\n\ndef test_multiply():\n    assert multiply(2, 3) == 6\n",
            encoding="utf-8",
        )
        return "implemented multiply + test"

    async def reviewer(diff, task):
        # A real ChatGPT reviewer would read the diff; here we approve a correct one.
        assert "multiply" in diff
        return ReviewVerdict(approved=True, issues=[])

    async def deployer(task, diff, branch):
        return Deployment(pr_title="feat: calculator.multiply", pr_body="新增乘法函数与测试")

    pipeline = build_pipeline(
        repo, coder=coder, reviewer=reviewer,
        analyzer=analyzer, designer=designer, deployer=deployer,
    )
    result = await pipeline.ainvoke({"task": "给 calculator 加乘法"})

    # All REAL gates passed on REAL code, pipeline stopped at the human boundary.
    assert result["staticcheck_ok"], result.get("staticcheck_report")
    assert result["verify_ok"], result.get("verify_report")
    assert result["status"] == "awaiting_approval"
    assert result["pr_title"] == "feat: calculator.multiply"

    # The feature really exists in the repo now.
    assert "def multiply" in (tmp_path / "calculator.py").read_text(encoding="utf-8")
