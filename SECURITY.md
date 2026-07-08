# AutoDevAgent 安全模型

AutoDevAgent 会让一个 LLM 读写代码、执行命令、操作 git。这比 TestAgent（只调用被测系统的 API / 操作 Playwright 浏览器）多出一个本质更大的攻击面。本文件如实说明**当前实现挡得住什么、挡不住什么**，对应设计文档第 5 节。

## 已实现的防线

| 防线 | 实现位置 | 说明 |
|---|---|---|
| 可执行文件白名单 | `orchestrator/tools/sandbox_exec.py` `ALLOWED_EXECUTABLES` | 只允许 python/pytest/pip/ruff/mypy/node/npm/npx/pnpm/yarn/git，其余一律拒绝 |
| 绝不用 shell | `sandbox_exec.run_command`（`shell=False` + argv 列表） | 从根上杜绝 `; | & $()` 命令注入——元字符只会被当普通参数传给程序 |
| 命令链/重定向拒绝 | `build_exec_tool` 的 `_has_control_token` | shell.exec 工具再加一层：拆词后若出现独立的 `;`/`|`/`>` 等控制 token 直接拒绝（引号内的 `;` 不误伤） |
| 执行超时 | `run_command(timeout=...)` 默认 300s | 防死循环/失控进程 |
| 工作目录限定 | `run_command` 强制 `cwd=workspace` | 命令在任务工作区内执行 |
| 写路径守卫 | `orchestrator/tools/path_guard.py` | 拒绝路径穿越（`..`/绝对路径逃逸），拒绝 `.env`/`*.pem`/`id_rsa`/`*.key` 等敏感文件名 |
| 内置 fs 工具隔离 | deepagents `FilesystemBackend(virtual_mode=True)` | read/write/edit 限定在工作区根内，拦截穿越 |
| 提交敏感文件拒绝 | `orchestrator/tools/git_tools.py` `git_commit` | 暂存区出现敏感文件名时拒绝提交并撤回暂存 |
| 禁止直推主干 | `git_create_branch` / `push_feature_branch` | 拒绝 main/master/HEAD，强制 feature 分支 |
| 部署人工审批 | `orchestrator/agents/deploy/agent.py` + 流水线 `awaiting_approval` | **流水线绝不自动 push/合并/上线**；只生成 PR 草稿后停下，等人工调用 `push_feature_branch` |
| 单次运行重试上限 | `pipeline/graph.py` `max_code_retries` | 用尽则升级 `needs_human`，不无限烧 |

## 当前实现挡不住的（生产必须补，见设计第 5.1/5.3 节）

- **网络出口**：白名单里的 `pip install` / `npm install` 仍能联网，可能下载恶意依赖或外泄数据。裸 Windows 主机进程**无法**强制断网。
  - 生产硬化：把 `run_command` 的执行搬进容器，`docker run --network=none`（或仅放行一个受控代理），非 root 用户，除工作区外只读挂载，加 cgroup CPU/内存上限。
- **进程/内核隔离**：目前是宿主机子进程，白名单程序自身若有漏洞仍在宿主机权限下运行。生产需容器/gVisor/受限用户。
- **磁盘/资源**：只限了输出大小和超时，未限磁盘写入量。生产靠容器配额。

## 结论

当前实现对"LLM 误操作 / 提示注入诱导执行意外命令"这类**非恶意但危险**的情况有实质防护（白名单 + 无 shell + 路径守卫 + 部署人工审批）。但对"运行明确恶意代码"没有内核级隔离——这需要容器化执行环境（Phase 5 的生产硬化目标）。因此当前实现适合在**受信任的目标仓库 + 开发者本机**使用，不适合直接对不受信任的外部输入开放。
