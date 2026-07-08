# AutoDevAgent

自动编程 Agent —— PoC / Phase 1。完整设计见 [`docs/PoC设计书_自动编程Agent.md`](docs/PoC设计书_自动编程Agent.md)。

与 `D:\TestAgentPythonProject` **完全独立**的新项目：不同代码库、不同产品目标（通用编码 Agent，而非测试管理 SaaS）。部分底层能力（agent 编排范式、模型工厂、工作区隔离、checkpointer、HITL、成本/工具调用追踪）从 TestAgent 移植/改造而来，两个项目之间只共享**正在运行的 llm_bridge 服务**，不共享 Python 源码或 import 路径。

## 当前进度：Phase 1（地基）

已完成：
- `orchestrator/_utils/` —— 从 TestAgent 移植/改造的编排工具：
  - `workspace.py`、`hitl.py`：原样移植，无改动
  - `output_trim.py`：原样移植，仅把日志目录环境变量从 `AGENT_LOG_DIR` 改名为 `AUTODEV_LOG_DIR`
  - `checkpointer.py`：**改造**——用 `InMemorySaver` 替代 TestAgent 的 `AsyncPostgresSaver`（本项目还没有 Postgres 基础设施），接口保持一致，以后要上持久化只需换这一个文件
  - `model_factory.py`：**改造**——见下方"关键架构差异"
- `orchestrator/_callbacks/` —— 从 TestAgent 移植/改造的 usage/tool 调用追踪：逻辑结构一致，但持久化目标从 MongoDB 换成本地 JSONL（`.autodev_logs/usage_logs.jsonl`、`.autodev_logs/tool_calls.jsonl`），因为本项目没有数据库，且永远不调用付费 API，没有真实成本需要算
- `orchestrator/agents/hello_world/` —— Phase 1 骨架验证 agent：证明"模型工厂 → 隔离工作区 → checkpointer → 工具调用追踪"这条链路能通过 `deepagents.create_deep_agent` 真正跑起来
- `tests/test_hello_world.py` —— 用 `GenericFakeChatModel`（打了 `bind_tools` 补丁）跑通全链路，**不需要真实 llm_bridge**，已验证通过（`python -m pytest tests/`，2 passed）

未开始：StaticCheck 工具、code-mcp、RequirementAnalyzer/DesignAgent/CodeAgent/ReviewAgent/VerifyAgent/DeployAgent 五个真实能力 agent —— 均属于 Phase 2 及以后，见设计文档第 6 节。

## 关键架构差异：为什么 model_factory.py 没有照抄 TestAgent

TestAgent 的 `backend/app/agents/_utils/model_factory.py` 里 `browser:` provider 分支是**进程内 `sys.path` 注入**，直接 `import` TestAgent 的 `backend/local_llm/langchain_browser_llm.py`。这要求 AutoDevAgent 知道 TestAgent 的目录结构，违反"完全独立"的前提。

本项目改成走 **HTTP**：`backend/llm_bridge/start_llm_proxy.py` 本身就对外暴露 OpenAI 兼容的 `/v1` 接口（这是它给 TestAgent backend 也在用的同一个集成点，见该文件的启动说明）。`orchestrator/_utils/model_factory.py` 用 `langchain_openai.ChatOpenAI` 指向这个 HTTP 地址即可，两个项目之间只共享一个**正在运行的服务**，不共享代码。

## 双模型分工（docs/PoC设计书 3.1 节，2026-07-08 确认不用付费 API 后的缓解方案）

| 角色 | Bridge | 用途 |
|---|---|---|
| `ModelRole.CODE` | DeepSeek（默认 `http://127.0.0.1:8765/v1`） | 主力生成：需求分析/设计/编码/部署文案，长上下文优势 |
| `ModelRole.REVIEW` | ChatGPT（默认 `http://127.0.0.1:8766/v1`） | 专职审查：只挑错，不生成，两个不同模型互相制衡 |

可通过环境变量 `AUTODEV_CODE_BRIDGE_URL` / `AUTODEV_REVIEW_BRIDGE_URL` 覆盖端口。

## 如何运行

### 1. 跑骨架测试（不需要真实 LLM，几秒钟出结果）

```bash
cd D:\AutoDevAgent
python -m pytest tests/ -v
```

### 2. 对着真实 llm_bridge 跑一次 hello world（需要先启动两个 bridge 实例）

```powershell
# 终端 1：DeepSeek 垫片（ModelRole.CODE）
cd D:\TestAgentPythonProject\backend\llm_bridge
python start_llm_proxy.py --provider deepseek --account 1 --port 8765

# 终端 2（如果要跑到真正用 ChatGPT 审查的阶段才需要，Phase 1 hello world 用不到）
python start_llm_proxy.py --provider chatgpt --account 1 --port 8766
```

```bash
# 终端 3
cd D:\AutoDevAgent
python run_hello_world.py
```

## 已确认的硬约束

- **绝不调用任何付费 LLM API**（Anthropic/OpenAI/DeepSeek 官方 API 一律禁止），只用本地免费的浏览器垫片方案。这是有意的产品决策，不是技术限制——`model_factory.py` 里没有付费 provider 分支。
- 这意味着"理解任意代码库 + 多文件重构"类任务的效果上限明显低于用付费 API 的方案，需要如实向用户说明，不能承诺达到 Claude Code 的水准。详见设计文档第 7 节。
