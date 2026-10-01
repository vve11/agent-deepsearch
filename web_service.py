"""单进程网页任务调度；不处理 HTTP，不在浏览器连接断开时取消研究。"""
import asyncio
import logging
from dataclasses import asdict

import database
import runtime

logger = logging.getLogger(__name__)


class ServiceConflict(Exception):
    pass


class ModelUnavailable(Exception):
    pass


class RunService:
    def __init__(self, client):
        self.client = client
        self.tasks: dict[str, asyncio.Task] = {}
        self.closing = False

    def is_active(self, run_id):
        task = self.tasks.get(run_id)
        return task is not None and not task.done()

    def check_available(self):
        if self.closing:
            raise ServiceConflict("服务正在关闭，请稍后重试")
        if self.client is None:
            raise ModelUnavailable("尚未配置 DEEPSEEK_API_KEY；配置项目 .env 后重启网页服务")
        if any(not task.done() for task in self.tasks.values()):
            raise ServiceConflict("已有研究正在运行，请等待它完成后再开始或恢复任务")

    def submit(self, question):
        # 检查、保存、登记任务之间没有 await，避免同一事件循环内重复占用名额。
        self.check_available()
        run = runtime.create_research_run(question)
        self.schedule(run.id, lambda: runtime.execute_created_research(self.client, run))
        return run.id

    def resume(self, run_id):
        self.check_available()
        run = database.load_run(run_id)
        if run.status == "completed":
            raise ServiceConflict("任务已经完成，无需恢复")
        try:
            runtime._recovery_checkpoint(run)
        except (ValueError, TypeError, LookupError) as error:
            raise ServiceConflict(str(error)) from error
        self.schedule(run_id, lambda: runtime.resume_research(self.client, run_id))
        return run_id

    def schedule(self, run_id, operation):
        async def drive():
            try:
                await operation()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                # Runtime 已负责保存失败；这里收取后台异常，避免无人处理的 Task。
                logger.warning("[网页任务失败] %s: %s", run_id, type(error).__name__)
        task = asyncio.create_task(drive(), name=f"research:{run_id}")
        self.tasks[run_id] = task
        def discard(done):
            if self.tasks.get(run_id) is done:
                self.tasks.pop(run_id, None)
        task.add_done_callback(discard)

    def detail(self, run_id):
        run = database.load_run(run_id)
        result = asdict(run)
        result["active"] = self.is_active(run_id)
        result["phase"] = database.load_workflow_phase(run_id)
        result["plan"] = None
        result["warning"] = None
        try:
            result["plan"] = asdict(database.load_plan(run_id))
        except LookupError:
            if result["phase"] in {"planned", "research"}:
                result["warning"] = "已保存计划丢失，无法安全恢复"
        except (ValueError, TypeError):
            result["warning"] = "研究计划数据损坏，无法安全恢复"
        try:
            checkpoint = database.load_checkpoint(run_id)
            # 活跃研究的证据优先取最近完整检查点；结束后用最终保存的证据。
            if result["active"]:
                result["evidence"] = checkpoint.evidence
        except LookupError:
            pass
        except (ValueError, TypeError):
            result["warning"] = "检查点数据损坏，无法安全恢复"
        result["can_resume"] = False
        if not result["active"] and run.status != "completed":
            try:
                runtime._recovery_checkpoint(run)
                result["can_resume"] = True
            except (ValueError, TypeError, LookupError) as error:
                result["warning"] = str(error)
        events = database.load_recent_events(run_id)
        result["events"] = [{k: v for k, v in event.items() if k != "payload_json"} for event in events]
        return result

    async def shutdown(self):
        self.closing = True
        tasks = list(self.tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.tasks.clear()
