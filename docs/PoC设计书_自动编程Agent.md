# 自动编程 Agent（AutoDevAgent）PoC 设计书

- 状态：PoC 设计阶段，未写代码
- 定位：与 `TestAgentPythonProject` **完全独立**的新项目（不同代码库、不同产品），部分底层能力从 TestAgent 迁移复用
- 目标：客户/开发者提一句自然语言需求 → 系统自动完成 **需求分析 → 方案设计 → 编码 → 测试 → 部署**，类似 Claude Code 的全自动编程工具
- 来源：从 TestAgentPythonProject 会话中衍生的调研结论，日期 2026-07-08

---

## 1. 结论先行

TestAgent 的 agent 编排骨架（`deepagents.create_deep_agent` + LangGraph + MCP 工具服务器 + 多 provider 模型工厂）本身是**领域无关**的，可以原样搬过来当 AutoDevAgent 的地基。但 TestAgent 现有 4 个 agent（api/ui/perf/testcase）的**工具集和 prompt 全部是"生成测试产物"专用的**，没有一个是"读写任意代码仓库、跑 shell、判断代码对不对"的通用编码能力。这部分必须从零建。

也就是说：**编排层（怎么调 LLM、怎么管上下文、怎么做人机确认、怎么做多轮重试）可以复用 70%；能力层（LLM 具体能做什么事）需要新建 90%**。

---

## 2. 从 TestAgent 直接复用的部分

| 复用项 | 来源路径 | 复用方式 |
|---|---|---|
| Deep Agent 编排范式 | `backend/app/agents/{api,ui,perf,testcase}/agent.py` | 作为模板抄一份骨架，`create_deep_agent(model, tools, backend, ...)` 这套调用方式直接搬 |
| 多 Provider 模型工厂 | `backend/app/agents/_utils/model_factory.py` | 整文件复制；关键是 `build_model(agent_name, model_id="browser:deepseek")` / `build_model(agent_name, model_id="browser:chatgpt")` 这种**显式指定 provider** 的用法——不是 fallback 关系，而是给不同 Agent **固定绑定**不同垫片，落地第 3.1 节的"DeepSeek 主写 + ChatGPT 专职审"分工 |
| 单次运行隔离工作区 | `backend/app/agents/_utils/workspace.py` | `create_run_workspace()` 的"每次运行开一个独立目录 + FilesystemBackend virtual_mode"思路直接复用——这正好是"不能让 agent 改到不该改的文件"的第一道防线 |
| Checkpointer / 断点续跑 | `backend/app/agents/_utils/checkpointer.py` | 长任务（分析→设计→编码→测试是个长链路）需要能中断恢复，直接复用 |
| Human-in-the-loop 中断 | `backend/app/agents/_utils/hitl.py` | 复用为"部署前必须人工审批"的机制（见第 5 节安全设计） |
| Token/成本追踪 | `backend/app/agents/_callbacks/usage_tracker.py`、`tool_tracker` | 直接复用，跑起来就知道一次"自动编程"任务花了多少 token |
| Budget 中间件 | `backend/app/agents/_middleware/` | 复用"超预算自动降级/中止"逻辑 |
| MCP 工具服务器骨架 | `backend/mcp/api/src/`（config/tools/index.js 结构） | 作为新建 `code-mcp` 服务器的项目结构模板，不复用其内容（内容是 API 测试专用工具） |
| 本地免费 LLM 垫片（可选） | `backend/llm_bridge/`（gitignore 独立仓库） | 如果 AutoDevAgent 也要"零成本本地跑"，可以指向同一个 bridge（`browser:deepseek` provider），但天花板问题见第 7 节 |
| CapabilityOS 文档纪律（可选） | `capability_os/knowledge/` 的 capability-index + incidents 模式 | 建议 AutoDevAgent 从第一天就采用同样的"能力先查后写、bug 先查 incident"纪律，避免重蹈覆辙 |

**不复用**：`app/api/v2/*`、`app/models/*`、`app/schemas/*`、UI 的 Next.js 页面——这些都是 TestAgent 的业务代码（测试用例管理），跟"自动编程"无关。

---

## 3. 需要新建的部分

### 3.1 新增 Agent（能力层）

**2026-07-08 已确认：AutoDevAgent 同样遵守 [[feedback_no_paid_api]]，不使用任何付费 API。** 为缓解免费本地方案的天花板问题，采用 **DeepSeek + ChatGPT 双浏览器垫片分工**，而不是单一模型自己写自己审：

| Agent | 职责 | LLM 分工 | 关键输入/输出 |
|---|---|---|---|
| **RequirementAnalyzer** | 把自然语言需求拆解为结构化需求点、澄清歧义、识别验收标准 | DeepSeek 垫片（主力） | in: 需求文本 + 目标仓库摘要；out: 结构化需求 JSON |
| **DesignAgent** | 基于需求 + 现有代码库结构，产出改动方案（涉及哪些文件、新增哪些模块、接口设计） | DeepSeek 垫片（主力） | in: 结构化需求 + 仓库索引；out: 设计文档 + 改动清单 |
| **CodeAgent** | 按设计方案实际编辑代码（新建/修改文件、写测试） | DeepSeek 垫片（主力） | in: 改动清单；out: 代码 diff |
| **StaticCheck**（非 LLM） | 语法/类型/lint 检查，秒级，不占用任何 LLM 轮次 | 无（纯工具） | in: 代码 diff；out: 通过/报错清单 |
| **ReviewAgent**（新增） | 专职代码审查：找逻辑问题、遗漏边界、设计不一致，发现问题立刻打回 CodeAgent | ChatGPT 垫片（专职审查） | in: 代码 diff + 设计方案；out: 审查意见（通过/问题清单） |
| **VerifyAgent** | 跑 build/test 真实执行，失败则把报错喂回 CodeAgent 自纠正，循环直到通过或达到重试上限 | 无（纯工具执行）+ DeepSeek 解读报错 | in: 代码 diff；out: 通过/失败 + 报错详情 |
| **DeployAgent** | 生成 PR / commit，触发部署流水线（不同项目的部署脚本不同，需做成可插拔） | DeepSeek 垫片（写 PR 描述） | in: 验证通过的 diff；out: PR 链接 / 部署结果 |

**为什么是"DeepSeek 主写 + ChatGPT 专职审"，不是反过来或者两个都写**：依据 [[project_bridge_tool_calling_upgrade]]，DeepSeek 垫片已经升级为高上限模式（单次输入 48000 字符、20 轮历史、48 个工具，超限自动分批发送），适合放"仓库摘要 + 改动清单 + 多文件 diff"这种长上下文的生成类工作；ChatGPT 垫片走的是原始低上限档位，不适合承担长上下文生成，但足够胜任"盯着一个 diff 挑错"这种短上下文、专注力要求高的审查工作。让两个不同的模型互相审查，也比同一个模型自己审自己更容易抓出问题——这是应对"免费模型智商有限"的关键工程手段之一。

编排关系（LangGraph 状态机，非线性——失败可回退，且每一步都比单模型直给方案多一层"免费但有效"的过滤）：

```
需求文本
   │
   ▼
RequirementAnalyzer(DeepSeek) ──(歧义)──▶ 人工澄清（HITL）
   │
   ▼
DesignAgent(DeepSeek) ──(方案确认，可选人工审核)──▶
   │
   ▼
CodeAgent(DeepSeek) 生成/修改代码（建议按文件/逻辑单元小批量提交，而非一次性甩一个大 diff）
   │
   ▼
StaticCheck（lint/类型检查/语法检查，秒级，零 LLM 成本）──失败──▶ CodeAgent（直接回传报错，不占用 ChatGPT 轮次）
   │ 通过
   ▼
ReviewAgent(ChatGPT，专职审查) ──发现问题──▶ CodeAgent（带着具体问题描述回去改，回到 StaticCheck 重新走一遍）
   │ 通过
   ▼
VerifyAgent 跑 build/test（真实执行结果兜底）──失败──▶ CodeAgent（重试，上限 N 次）
   │ 通过                              │ 超过重试上限
   ▼                                   ▼
DeployAgent（需 HITL 审批才能真正部署）      上报人工介入
```

**为什么 StaticCheck 要放在 ReviewAgent 前面**：浏览器 LLM 垫片每轮往返约 5-6 分钟（见 [[project_llm_bridge_direction_a]]），是这套系统里最贵、最慢的资源。语法错误、类型错误这类"工具能确定性判断"的问题如果也丢给 ChatGPT 去发现，纯属浪费一轮 5-6 分钟的往返。先用免费的静态工具把这类问题挡掉，ChatGPT 的审查轮次只花在真正需要"理解语义/设计意图"的问题上。

### 3.2 新增 MCP 工具服务器：`code-mcp`

TestAgent 现有 3 个 MCP 服务器（api/ui/perf）都是"调用被测系统"的工具，没有一个是"操作代码仓库本身"的工具。AutoDevAgent 需要新建一套：

- `fs.read` / `fs.write` / `fs.edit`（限定在 workspace 根目录内，禁止越权路径）
- `fs.glob` / `fs.grep`（代码检索，复用 ripgrep）
- `git.status` / `git.diff` / `git.commit` / `git.branch` / `git.push`（`git.push` 默认只能推 feature 分支，禁止直推 main/master）
- `shell.exec`（**最高风险项**，见第 5 节，必须白名单+沙箱）
- `pkg.install`（按目标项目的包管理器，装依赖）
- `lint.check` / `typecheck.run` / `syntax.check`（**StaticCheck 阶段专用**，不经过 LLM，纯工具调用，按目标仓库语言自动选型：Python 用 `python -m py_compile` + `ruff check`（+ 可选 `mypy`）；JS/TS 用 `tsc --noEmit` + `eslint`；其他语言按需扩展。这是 3.1 节双 LLM 分工能省钱的前提——没有这层，语法错误也要靠 ChatGPT 去发现）

### 3.3 仓库索引/上下文构建

给 CodeAgent 塞整个仓库源码不现实（token 爆炸）。需要新建一个"仓库索引器"：扫描目标仓库 → 生成目录树 + 关键文件摘要 + 依赖关系图，类似 TestAgent 里 OpenAPI 解析给 API Agent 用摘要而非全量文档的思路（`app/agents/api/` 里已有类似"抽取摘要喂给模型"的模式，可以参考其做法但要重写，因为解析对象从 OpenAPI spec 换成任意语言的源代码）。

---

## 4. 目录结构提案

```
D:\AutoDevAgent\
├── docs/                        # 设计文档（本文件所在目录）
├── orchestrator/                # Python，LangGraph 状态机 + 5 个 Agent
│   ├── agents/
│   │   ├── requirement_analyzer/
│   │   ├── design/
│   │   ├── code/
│   │   ├── verify/
│   │   └── deploy/
│   ├── _utils/                  # 从 TestAgent 搬运：model_factory.py / workspace.py / checkpointer.py / hitl.py
│   ├── _callbacks/               # usage_tracker / tool_tracker（搬运）
│   └── graph.py                  # 顶层状态机装配
├── mcp/
│   └── code-mcp/                 # 新建，Node.js，结构参照 backend/mcp/api
├── sandbox/                       # shell.exec 的隔离执行环境（容器化，见第5节）
└── cli/ 或 api/                    # 触发入口：CLI 命令行 或 一个简单的 FastAPI（后续再定）
```

---

## 5. 安全 / 沙箱设计（相对 TestAgent 新增的最大风险点）

TestAgent 现有的 MCP 工具再危险也只是"调用被测系统的 API / 操作 Playwright 浏览器"，爆炸半径有限。AutoDevAgent 一旦有 `shell.exec` 和任意文件读写，等于给 LLM 一台能执行任意命令的机器，必须新增以下防线（现有系统里都没有对应机制）：

1. **执行环境隔离**：`shell.exec` 绝不能在宿主机进程里跑，必须在容器（Docker）或至少受限用户权限的子进程里跑，且挂载卷仅限当前任务的 workspace 目录（复用 `create_run_workspace` 的隔离目录思路，但要从"目录隔离"升级到"进程/容器隔离"）。
2. **命令白名单**：默认只允许构建/测试/包管理类命令（`npm`, `pytest`, `git`, `pip`, 等的固定子命令集），拒绝任意 shell 元字符拼接，防止命令注入。
3. **网络出口限制**：沙箱默认无出网，仅放行包管理源和目标 git 远程仓库，防止数据外泄或下载恶意脚本。
4. **写路径守卫**：`fs.write`/`fs.edit` 拒绝写 workspace 根目录之外的任何路径，拒绝 `.env`、`*.pem`、`id_rsa` 等敏感文件名模式。
5. **部署强制人工审批**：`DeployAgent` 的"真正推送/上线"动作必须经过 HITL 中断（复用 `hitl.py`），不能全自动直接部署到生产——这是与"自动编码"最本质的边界：**自动到 PR 为止，合并/上线保留人工确认**，除非用户后续明确要求连这一步也自动化。
6. **单次运行资源/时间上限**：复用 Budget 中间件思路，给 shell 执行加超时和重试上限，防止死循环/失控消耗。

---

## 6. 阶段划分与工作量估算

| 阶段 | 内容 | 粗估工作量 |
|---|---|---|
| Phase 0（已完成）| 本设计文档 | - |
| Phase 1：地基 | 从 TestAgent 搬运编排骨架（model_factory/workspace/checkpointer/hitl/callbacks），跑通一个最简单的"hello world" LangGraph agent | 0.5–1 天 |
| Phase 2：code-mcp | 实现 `fs.*`/`git.*`/`lint.check`/`typecheck.run` 工具（不含 shell.exec），先在一个小型示例仓库上验证"AI 能读写代码 + 静态检查能跑通" | 1–2 天 |
| Phase 3：单点切片 | 实现 CodeAgent(DeepSeek) → StaticCheck → ReviewAgent(ChatGPT) → VerifyAgent 的最小闭环：给定一个足够具体的小需求（如"给某仓库加一个 CRUD 接口"），能自动写代码 + 静态检查 + 双模型审查 + 跑测试 + 报告通过/失败，不含 DesignAgent 和 DeployAgent | 3–5 天（比单模型方案多了双垫片调度和 StaticCheck 联调） |
| Phase 4：补全设计与部署 | 加 RequirementAnalyzer + DesignAgent（需求→方案）与 DeployAgent（PR 生成），加 HITL 审批点 | 3–5 天 |
| Phase 5：shell.exec 沙箱化 | 容器化执行环境、白名单、网络隔离——安全关键，不能跳过 | 2–3 天 |
| Phase 6：真实项目试跑 + 打磨 | 挑一个真实的小仓库，跑几轮真实需求，根据失败案例迭代 prompt/工具 | 视效果而定，通常是最长的阶段 |

以上是"能跑通"的工作量，不是"好用到接近 Claude Code"的工作量——后者的差距主要不在工程量，而在底层模型能力，见第 7 节。

---

## 7. 已确认：仅用免费本地方案 + 天花板缓解措施

**2026-07-08 已确认**：AutoDevAgent 同样遵守 [[feedback_no_paid_api]]，绝不调用任何付费 LLM API（Anthropic/OpenAI/DeepSeek 官方 API 一律禁止），只用浏览器 ChatGPT/DeepSeek 垫片 + 本地 Ollama。这条规则的适用范围已从"TestAgent 与 anything-chat-rag"扩展到 AutoDevAgent（见 [[feedback_no_paid_api]] 更新记录）。

天花板问题依然存在，这是模型能力 + 硬件的物理限制，工程手段改善不了根本——垫片没有原生 function-calling（靠 prompt 拼 JSON 校验重试逼近）、本机 15GB 内存无独立 GPU、单轮 5-6 分钟延迟。**但可以缓解**，本设计文档采用的三个具体手段：

1. **DeepSeek 主写 + ChatGPT 专职审的双垫片分工**（第 3.1 节）：不让同一个模型自己审自己，用两个不同视角互相制衡，比单模型方案更容易在生成阶段就抓出问题，而不是等真实执行才暴露。
2. **StaticCheck 静态检查前置**（第 3.1/3.2 节）：把语法、类型这类工具能确定性判断的错误挡在 LLM 审查之前，宝贵的 ChatGPT 审查轮次只花在真正需要"理解"的问题上，同时减少无谓的 5-6 分钟往返次数。
3. **小批量增量提交**：CodeAgent 按文件/逻辑单元小批量产出，而不是一次性甩一个大 diff——出问题时改动范围小、定位快、纠正成本低，也避免大 diff 让本就上限有限的模型判断力被进一步稀释。

**如实说明给最终用户/客户的边界**：即便有以上缓解手段，"理解任意代码库 + 多文件重构 + 自纠代码 bug"这类任务的成功率，预计仍明显低于使用付费 API（如 Claude/GPT 正式 API）的效果。这是免费本地方案的能力上限，不能承诺达到 Claude Code 的水准，只能承诺"在这个上限内尽量把工程手段做到位"。`model_factory.py` 的多 provider 抽象仍然保留（切换只需改环境变量），万一未来决策变化，接入付费 API 不需要改编排骨架。

---

## 8. 下一步

等你确认：
1. 目录结构和阶段划分（第 4、6 节）是否认可
2. StaticCheck 具体检查工具的语言范围（先支持 Python/TS，还是需要覆盖更多语言）
3. Phase 1（地基搬运）是否现在就开始写代码

我再进入实现阶段。
