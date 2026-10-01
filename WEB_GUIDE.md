# 网页版：从浏览器到 Agent

## 日常启动

已安装依赖并构建前端后，在项目目录执行：

```powershell
.\.venv\Scripts\python.exe web.py
```

打开 http://127.0.0.1:8000 。窗口运行期间保留终端，Ctrl+C 停止服务。
接口文档：http://127.0.0.1:8000/docs 。本地版使用一个后端进程；不要同时运行另一个 Web/CLI 研究进程，也不要开启多 worker。

网页复用项目 `.env` 中的模型和搜索配置，以及 `data/research.db`。密钥只在后端加载。
没有模型密钥时仍能查看历史，提交按钮禁用；修改 `.env` 后需重启服务。
页面每两秒轮询。关掉网页不停止后台研究；关闭服务会取消正在执行的任务并尽力保存状态，重新启动后可在任务详情点击“继续研究”。强制退出可能保留 running，但网页会将其标为待继续。

## 首次安装 / 前端修改后构建

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
.\.venv\Scripts\python.exe web.py
```

Node 需要符合 Vite 的要求，本次验证版本为 Node 22.17.1。`package-lock.json` 固定前端依赖。
构建只把前端代码输出到 `frontend/dist`，API 不提供项目目录、数据库文件或 `.env` 的下载。

开发前端时可另开终端，在 frontend 目录运行 `npm run dev`，访问 http://127.0.0.1:5173 。
Vite 会把 `/api` 请求代理到 8000；后端仍需运行。修改 JSX/CSS 会自动更新；正式构建页面需重跑 build 并刷新。

## 一次请求的完整路径

1. `frontend/src/main.jsx`：收集问题，调用 `frontend/src/api.js`。
2. `POST /api/runs`：`api.py` 校验问题，调用 `RunService.submit()`。
3. `web_service.py`：检查执行名额，调用 `runtime.create_research_run()`，将任务和 planning 状态保存到 SQLite。
4. 登记 `asyncio.Task` 执行 `execute_created_research()`，HTTP 立即返回 202 和任务编号。
5. Runtime 调用 Planner、保存计划，再运行现有 Agent 工具循环；事件和检查点照常写入数据库。
6. 浏览器轮询 `GET /api/runs/{id}`，把计划、回答、来源、事件显示出来。
7. 点击继续研究：`POST /api/runs/{id}/resume`，先校验可恢复性，再调度原来的 `resume_research()`。

`create_task` 是当前 Python 进程内的后台协程，不是独立服务器或持久化任务队列。恢复依据仍是 SQLite。
CLI 的 `execute_research()` 也复用“先创建再执行”的入口，原 CLI 用法保持不变。

## 接口

| 方法和路径 | 作用 |
|---|---|
| GET /api/health | 服务、模型配置与执行占用情况 |
| GET /api/runs?limit=50&offset=0 | 任务摘要，支持分页 |
| POST /api/runs | JSON 请求体 `{"question":"问题"}`，创建任务并返回 202 |
| GET /api/runs/{id} | 任务详情、最近 100 条事件、计划和证据 |
| POST /api/runs/{id}/resume | 恢复原任务，返回 202 |

4000 字上限；空问题拒绝；编号格式错误返回 422，不存在返回 404，忙碌或不可恢复返回 409，缺密钥或数据库故障返回 503。
前端禁止重复点击，后端限制同时一个任务。POST 尚未加入持久化幂等键：提交响应丢失时，应先检查任务列表，不要自动重试提交。

## 验证

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

API 测试覆盖后台执行、失败恢复、服务关闭与重启、参数校验、跨站写入拒绝、缺密钥和保存失败。
如需手动演示故障恢复而不调用真实模型：

```powershell
.\.venv\Scripts\python.exe tests\serve_web_fixture.py
```

仅访问 http://127.0.0.1:8001 测试。此入口不读取 `.env`，使用临时数据库和模拟搜索，第一次任务会故意中断，点击继续后完成。退出即删除临时测试数据库。
正常入口为 8000；模拟回答不是真实研究结果。

## 当前边界

- 这是本机单用户学习版，无账号系统，启动入口只绑定 127.0.0.1，尚未做公网部署。
- 同时一个研究；没有独立任务队列、跨进程锁或多用户隔离。
- 事件使用轮询，不是 SSE，也不是逐 token 输出；详情展示最近 100 条，历史列表展示最近最多 100 项，完整数据仍在 SQLite。
- 没有网页取消按钮；停止服务会请求取消，硬退出依赖之前已保存的恢复状态。
- 证据是最近完整检查点或最终记录的快照，搜索进行中不会逐条立即更新。
- Runtime 原有“最终写入失败时告警”的策略保留：数据库故障可能使内存结果未保存，页面只展示已持久化结果。
- 下一阶段再学习 SSE、Git/CI、Docker 和部署；本次没有购买服务器、创建远程仓库或发布公网。

参考：[FastAPI 生命周期](https://fastapi.tiangolo.com/advanced/events/)、[静态文件](https://fastapi.tiangolo.com/tutorial/static-files/)、[Vite](https://vite.dev/guide/)。
