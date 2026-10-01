"""网页 HTTP 入口。python web.py 启动；/docs 可交互查看接口。"""
import os
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

import database
from web_service import ModelUnavailable, RunService, ServiceConflict

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "frontend" / "dist"


class CreateRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=4000)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("问题不能为空")
        return value.strip()


def create_app(*, client_factory=None, load_environment=True):
    @asynccontextmanager
    async def lifespan(app):
        if load_environment:
            load_dotenv(ROOT / ".env")
        database.init_db()
        client = None
        if client_factory is not None:
            client = client_factory()
        elif os.getenv("DEEPSEEK_API_KEY"):
            client = AsyncOpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                timeout=60.0, max_retries=0,
            )
        app.state.service = RunService(client)
        try:
            yield
        finally:
            await app.state.service.shutdown()
            if client is not None and client_factory is None:
                await client.close()

    app = FastAPI(title="研究助手 · 本地 API", lifespan=lifespan)

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        # 本地学习版不开放跨站浏览器写入，也不使用通配 CORS。
        origin = request.headers.get("origin")
        if request.method in {"POST", "PUT", "PATCH", "DELETE"} and origin:
            parsed = urlsplit(origin)
            if parsed.scheme not in {"http", "https"} or parsed.netloc not in {
                request.headers.get("host"), "localhost:5173", "127.0.0.1:5173",
            }:
                return JSONResponse({"detail": "不允许跨站提交任务"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(sqlite3.Error)
    async def database_error(request, error):
        return JSONResponse({"detail": "数据库暂时不可用，请稍后重试"}, status_code=503)

    @app.exception_handler(ServiceConflict)
    async def conflict(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(ModelUnavailable)
    async def unavailable(request, error):
        return JSONResponse({"detail": str(error)}, status_code=503)

    def check_id(run_id):
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise HTTPException(422, "任务编号格式不正确")

    @app.get("/api/health")
    async def health(request: Request):
        service = request.app.state.service
        return {"status": "ok", "model_configured": service.client is not None,
                "busy": any(not t.done() for t in service.tasks.values())}

    @app.get("/api/runs")
    async def runs(request: Request, limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)):
        items = database.list_runs(limit, offset)
        for item in items:
            item["active"] = request.app.state.service.is_active(item["id"])
        return {"items": items}

    @app.post("/api/runs", status_code=202)
    async def submit(body: CreateRun, request: Request):
        run_id = request.app.state.service.submit(body.question)
        return {"run_id": run_id, "status": "running"}

    @app.get("/api/runs/{run_id}")
    async def detail(run_id: str, request: Request):
        check_id(run_id)
        try:
            return request.app.state.service.detail(run_id)
        except LookupError:
            raise HTTPException(404, "没有找到这个任务")
        except (ValueError, TypeError):
            raise HTTPException(409, "任务记录格式不正确，无法读取")

    @app.post("/api/runs/{run_id}/resume", status_code=202)
    async def resume(run_id: str, request: Request):
        check_id(run_id)
        try:
            request.app.state.service.resume(run_id)
        except LookupError:
            raise HTTPException(404, "没有找到这个任务")
        return {"run_id": run_id, "status": "running"}

    @app.get("/")
    async def index():
        if not (DIST / "index.html").is_file():
            return JSONResponse({"detail": "请先在 frontend 目录运行 npm install 和 npm run build"}, status_code=503)
        return FileResponse(DIST / "index.html", headers={"Cache-Control": "no-cache"})

    # 只挂载构建后的静态资源，不暴露项目根目录、.env 或数据库。
    app.mount("/assets", StaticFiles(directory=DIST / "assets", check_dir=False), name="assets")
    return app


app = create_app()
