# AutoDevAgent 演示

`sample_project/` 是一个最小 Python 项目（`calculator.py` 有 `add`/`subtract`，配套测试）。
用它体验一次真实的全自动流水线。

## 步骤

1. 把样例拷到一个独立目录并初始化 git（不要在本仓库里直接跑，避免污染）：

   ```powershell
   Copy-Item -Recurse D:\AutoDevAgent\examples\sample_project C:\tmp\autodev_demo
   cd C:\tmp\autodev_demo
   git init; git add -A; git -c user.name=you -c user.email=you@x commit -m "init"
   ```

2. 启动两个 llm_bridge 实例（见项目根 README）：DeepSeek :8765、ChatGPT :8766。

3. 跑流水线，给一个具体需求：

   ```powershell
   cd D:\AutoDevAgent
   python run_autodev.py --repo C:\tmp\autodev_demo --task "给 calculator 加一个 multiply(a, b) 乘法函数，并补充对应测试"
   ```

4. 流水线会依次跑 需求分析 → 设计 → 编码(DeepSeek) → 静态检查 → 审查(ChatGPT) → 测试，
   最后停在 `awaiting_approval` 并打印 PR 草稿。确认无误后推送：

   ```powershell
   python run_autodev.py --repo C:\tmp\autodev_demo --approve <流水线打印的分支名>
   ```

> 注意：浏览器垫片每轮约 5-6 分钟，一次完整跑可能要十几到几十分钟。这是免费本地方案的物理限制。
