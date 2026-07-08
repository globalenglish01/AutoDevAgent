"""AutoDevAgent CLI — run the full pipeline against a target repo.

    python run_autodev.py --repo D:\\path\\to\\repo --task "给用户模块加软删除接口"

Prerequisites (both llm_bridge instances must be running — see README):
    # DeepSeek bridge (code role)
    cd D:\\TestAgentPythonProject\\backend\\llm_bridge
    python start_llm_proxy.py --provider deepseek --account 1 --port 8765
    # ChatGPT bridge (review role)
    python start_llm_proxy.py --provider chatgpt  --account 1 --port 8766

The pipeline runs requirement → design → code → staticcheck → review → verify →
deploy(PR draft) and STOPS at status="awaiting_approval". It never pushes; to
actually push the feature branch after you've reviewed the PR draft, use
--approve (which calls the guarded push_feature_branch).

Because each browser-LLM round takes ~5-6 minutes, a full real run can take
quite a while. Watch progress in the printed log.
"""
from __future__ import annotations

import argparse
import asyncio

from orchestrator.agents.deploy.agent import DeployError, push_feature_branch
from orchestrator.pipeline.production import build_full_pipeline


def _print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print(f"状态: {result.get('status')}")
    print("=" * 60)
    print("\n执行日志:")
    for line in result.get("log", []):
        print(f"  • {line}")
    if result.get("status") == "awaiting_approval":
        print("\n--- PR 草稿（等待人工审批）---")
        print(f"分支: {result.get('branch')}")
        print(f"标题: {result.get('pr_title')}")
        print(f"正文:\n{result.get('pr_body')}")
        print("\n审阅无误后，用 --approve 推送该 feature 分支。")
    elif result.get("status") == "needs_human":
        print("\n⚠️ 重试次数用尽，需要人工介入。最后的反馈：")
        print(result.get("feedback", ""))


async def _amain() -> None:
    parser = argparse.ArgumentParser(description="AutoDevAgent 全自动编程流水线")
    parser.add_argument("--repo", required=True, help="目标 git 仓库路径")
    parser.add_argument("--task", required=True, help="自然语言开发需求")
    parser.add_argument("--max-retries", type=int, default=2, help="编码重试上限")
    parser.add_argument(
        "--approve", metavar="BRANCH", default=None,
        help="人工审批后推送指定 feature 分支（跳过流水线，仅执行 push）",
    )
    parser.add_argument("--remote", default="origin", help="推送的远程名（配合 --approve）")
    args = parser.parse_args()

    if args.approve:
        try:
            print(push_feature_branch(args.repo, args.approve, remote=args.remote))
        except DeployError as exc:
            print(f"推送失败: {exc}")
        return

    pipeline = build_full_pipeline(args.repo, max_code_retries=args.max_retries)
    result = await pipeline.ainvoke({"task": args.task})
    _print_result(result)


if __name__ == "__main__":
    asyncio.run(_amain())
