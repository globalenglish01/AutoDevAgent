"""FastAPI app + single-page UI for driving the AutoDevAgent pipeline.

Endpoints:
  GET  /                      -> the single-page UI
  POST /api/run               -> start a pipeline run (background), returns {run_id}
  GET  /api/run/{run_id}      -> current status + streamed log (poll this)
  POST /api/run/{run_id}/approve -> HITL apply: feature branch + commit (+ push if remote)

Run state is kept in an in-memory registry (RUNS) — this is a local, single-user
dev tool, not a multi-tenant service. execute_run() and apply_and_push() are
module-level so they can be unit-tested without a live browser bridge.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from orchestrator.agents.deploy.agent import DeployError, push_feature_branch
from orchestrator.pipeline.production import build_full_pipeline, build_lite_pipeline
from orchestrator.tools.git_tools import build_git_tools

# run_id -> mutable record the UI polls.
RUNS: dict[str, dict[str, Any]] = {}

_STREAM_KEYS = (
    "status", "log", "pr_title", "pr_body", "branch", "diff",
    "staticcheck_ok", "review_approved", "review_inconclusive", "verify_ok",
    "code_summary", "review_issues", "feedback",
)


async def execute_run(record: dict[str, Any], pipeline, task: str, run_id: str) -> None:
    """Stream *pipeline* to completion, mirroring pipeline state into *record*.

    Module-level + pipeline-injected so tests can drive it with a fake-model
    pipeline (no browser bridge needed).
    """
    try:
        config = {"configurable": {"thread_id": run_id}, "recursion_limit": 60}
        async for state in pipeline.astream({"task": task}, config, stream_mode="values"):
            for k in _STREAM_KEYS:
                if k in state:
                    record[k] = state[k]
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = str(exc)[:800]
    finally:
        record["done"] = True
        record.setdefault("status", "error")


def apply_and_push(repo: str, run_id: str, pr_title: str) -> dict[str, str]:
    """HITL apply: create a feature branch, commit the working-tree changes, and
    push if a remote is configured. Never touches main/master; refuses to commit
    secrets (both enforced by the git tools). Returns a per-step result dict.
    """
    branch = f"autodev/{run_id}"
    tools = {t.name: t for t in build_git_tools(repo)}
    result = {"branch": tools["git_create_branch"].invoke({"name": branch})}
    result["commit"] = tools["git_commit"].invoke({"message": pr_title or "AutoDev change"})
    try:
        result["push"] = push_feature_branch(repo, branch)
    except DeployError as exc:
        result["push"] = f"未推送（{exc}）——改动已在本地分支 {branch!r} 提交，可手动推送"
    return result


def _build_pipeline(repo: str, *, lite: bool, max_retries: int):
    builder = build_lite_pipeline if lite else build_full_pipeline
    return builder(repo, max_code_retries=max_retries)


def create_app(pipeline_builder: Callable[..., Any] | None = None) -> FastAPI:
    """Build the FastAPI app. *pipeline_builder(repo, lite, max_retries)* can be
    injected in tests to avoid the real bridge."""
    app = FastAPI(title="AutoDevAgent")

    def _make_pipeline(repo: str, lite: bool, max_retries: int):
        if pipeline_builder is not None:
            return pipeline_builder(repo, lite=lite, max_retries=max_retries)
        return _build_pipeline(repo, lite=lite, max_retries=max_retries)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    @app.post("/api/run")
    async def start_run(body: dict) -> dict:
        repo = str((body or {}).get("repo", "")).strip()
        task = str((body or {}).get("task", "")).strip()
        lite = bool((body or {}).get("lite", True))
        max_retries = int((body or {}).get("max_retries", 2))
        if not repo or not task:
            raise HTTPException(status_code=400, detail="repo 和 task 均为必填")
        if not Path(repo).is_dir():
            raise HTTPException(status_code=400, detail=f"仓库路径不存在: {repo}")
        run_id = uuid.uuid4().hex[:12]
        RUNS[run_id] = {
            "run_id": run_id, "repo": repo, "task": task, "lite": lite,
            "status": "running", "log": [], "done": False,
        }
        pipeline = _make_pipeline(repo, lite, max_retries)
        asyncio.create_task(execute_run(RUNS[run_id], pipeline, task, run_id))
        return {"run_id": run_id}

    @app.get("/api/run/{run_id}")
    def get_run(run_id: str) -> dict:
        rec = RUNS.get(run_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="run 不存在")
        return rec

    @app.post("/api/run/{run_id}/approve")
    def approve(run_id: str) -> dict:
        rec = RUNS.get(run_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="run 不存在")
        if rec.get("status") != "awaiting_approval":
            raise HTTPException(
                status_code=400,
                detail=f"当前状态为 {rec.get('status')!r}，只有 awaiting_approval 才能审批",
            )
        applied = apply_and_push(rec["repo"], run_id, rec.get("pr_title") or "")
        rec["approved"] = True
        rec["apply_result"] = applied
        return applied

    return app


# ── Single-page UI (vanilla HTML/JS, no build step) ──────────────────────────
_INDEX_HTML = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AutoDevAgent</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { font-family: system-ui, "Segoe UI", sans-serif; margin: 0; padding: 24px;
         max-width: 900px; margin-inline: auto; line-height: 1.5; }
  h1 { font-size: 20px; margin: 0 0 4px; }
  .sub { opacity: .65; font-size: 13px; margin-bottom: 20px; }
  label { display:block; font-size: 13px; font-weight: 600; margin: 12px 0 4px; }
  input[type=text], textarea { width: 100%; padding: 8px 10px; border-radius: 8px;
         border: 1px solid #8884; background: transparent; color: inherit; font: inherit; }
  textarea { min-height: 70px; resize: vertical; }
  .row { display:flex; gap: 16px; align-items:center; flex-wrap: wrap; }
  button { padding: 8px 16px; border-radius: 8px; border: 0; cursor: pointer;
           font: inherit; font-weight: 600; background: #2563eb; color: #fff; }
  button:disabled { opacity: .5; cursor: default; }
  button.approve { background: #16a34a; }
  .card { border: 1px solid #8883; border-radius: 12px; padding: 16px; margin-top: 20px; }
  .badge { display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 12px;
           font-weight: 700; }
  .b-running { background:#3b82f633; } .b-awaiting_approval { background:#f59e0b33; }
  .b-done { background:#16a34a33; } .b-needs_human { background:#ef444433; } .b-error { background:#ef444455; }
  #log { list-style: none; padding: 0; margin: 8px 0 0; font-size: 13px; }
  #log li { padding: 3px 0; border-bottom: 1px dashed #8882; }
  pre { white-space: pre-wrap; word-break: break-word; background: #8881; padding: 10px;
        border-radius: 8px; font-size: 12px; max-height: 260px; overflow:auto; }
  .muted { opacity:.6; font-size:12px; }
</style>
</head>
<body>
  <h1>AutoDevAgent</h1>
  <div class="sub">自然语言需求 → 编码 → 静态检查 → 审查 → 测试 → PR 草稿（停在人工审批）。免费浏览器垫片驱动。</div>

  <div class="card">
    <label>目标仓库路径（本机 git 仓库）</label>
    <input id="repo" type="text" placeholder="C:\\tmp\\autodev_demo"/>
    <label>开发需求（自然语言）</label>
    <textarea id="task" placeholder="在 calculator.py 增加 multiply(a, b) 乘法函数，并补测试"></textarea>
    <div class="row" style="margin-top:12px">
      <label style="margin:0"><input id="lite" type="checkbox" checked/> 精简模式（推荐，免费垫片更稳）</label>
      <button id="runBtn" onclick="startRun()">运行</button>
      <span id="hint" class="muted"></span>
    </div>
    <div class="muted" style="margin-top:8px">前置：先启动 ChatGPT 浏览器垫片（llm_bridge，端口 8765）。每轮约需数十秒。</div>
  </div>

  <div class="card" id="statusCard" style="display:none">
    <div class="row" style="justify-content:space-between">
      <div>状态：<span id="badge" class="badge">—</span></div>
      <div class="muted" id="runId"></div>
    </div>
    <ul id="log"></ul>
    <div id="prPanel" style="display:none; margin-top:12px">
      <div><b>PR 草稿</b>（分支将为 <code id="prBranch"></code>）</div>
      <label>标题</label><div id="prTitle"></div>
      <label>正文</label><pre id="prBody"></pre>
      <button class="approve" id="approveBtn" onclick="approve()">✔ 审批并应用（建 feature 分支 + 提交，有远程则推送）</button>
      <div id="applyResult" class="muted" style="margin-top:8px"></div>
    </div>
    <div id="errBox" class="muted" style="display:none; color:#ef4444; margin-top:10px"></div>
  </div>

<script>
let runId = null, timer = null;
async function startRun() {
  const repo = document.getElementById('repo').value.trim();
  const task = document.getElementById('task').value.trim();
  const lite = document.getElementById('lite').checked;
  if (!repo || !task) { setHint('请填写仓库路径和需求'); return; }
  setHint('');
  const btn = document.getElementById('runBtn'); btn.disabled = true;
  try {
    const r = await fetch('/api/run', {method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({repo, task, lite})});
    const j = await r.json();
    if (!r.ok) { setHint(j.detail || '启动失败'); btn.disabled = false; return; }
    runId = j.run_id;
    document.getElementById('statusCard').style.display = 'block';
    document.getElementById('runId').textContent = 'run ' + runId;
    poll();
    timer = setInterval(poll, 2000);
  } catch (e) { setHint('' + e); btn.disabled = false; }
}
function setHint(t){ document.getElementById('hint').textContent = t; }
function setBadge(s){ const b=document.getElementById('badge'); b.textContent=s; b.className='badge b-'+s; }
async function poll() {
  if (!runId) return;
  const r = await fetch('/api/run/' + runId);
  const s = await r.json();
  setBadge(s.status || '—');
  const ul = document.getElementById('log'); ul.innerHTML='';
  (s.log||[]).forEach(line => { const li=document.createElement('li'); li.textContent='• '+line; ul.appendChild(li); });
  if (s.error) { const e=document.getElementById('errBox'); e.style.display='block'; e.textContent='错误：'+s.error; }
  if (s.status === 'awaiting_approval') {
    document.getElementById('prPanel').style.display = 'block';
    document.getElementById('prBranch').textContent = 'autodev/' + runId;
    document.getElementById('prTitle').textContent = s.pr_title || '(无标题)';
    document.getElementById('prBody').textContent = s.pr_body || '';
  }
  if (s.done) {
    clearInterval(timer); timer=null;
    document.getElementById('runBtn').disabled = false;
  }
}
async function approve() {
  const btn = document.getElementById('approveBtn'); btn.disabled = true;
  const r = await fetch('/api/run/' + runId + '/approve', {method:'POST'});
  const j = await r.json();
  const box = document.getElementById('applyResult');
  box.textContent = r.ok ? ('已应用：'+ (j.branch||'') +' | '+ (j.commit||'') +' | '+ (j.push||''))
                         : ('应用失败：'+ (j.detail||''));
  if (!r.ok) btn.disabled = false;
}
</script>
</body>
</html>
"""

# Module-level default app for `uvicorn orchestrator.web.app:app`.
app = create_app()
