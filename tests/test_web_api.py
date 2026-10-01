"""真实 API + Runtime + 临时数据库；仅模型响应被模拟，不消耗真实 API 额度。"""
import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
import database
from api import create_app
from test_checkpoints import client_with, response

PLAN = json.dumps({"tasks": [{"question": "调查测试问题"}]})


class WebApiTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(database, "DB_PATH", Path(directory.name) / "test.db")
        patcher.start()
        self.addCleanup(patcher.stop)

    def browser(self, app):
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost:8000")

    async def finished(self, app):
        tasks = list(app.state.service.tasks.values())
        await asyncio.wait_for(asyncio.gather(*tasks), 3)

    async def test_submit_returns_persisted_id_before_answer_then_history_is_available(self):
        release = asyncio.Event()
        client = client_with()
        calls = 0
        async def model(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                await release.wait()
                return response(PLAN)
            return response("## 测试回答\n完整研究已完成。")
        client.chat.completions.create.side_effect = model
        app = create_app(client_factory=lambda: client, load_environment=False)
        async with app.router.lifespan_context(app), self.browser(app) as browser:
            submitted = await browser.post("/api/runs", json={"question": "  测试问题  "})
            self.assertEqual(submitted.status_code, 202)
            run_id = submitted.json()["run_id"]
            self.assertEqual(database.load_run(run_id).question, "测试问题")
            self.assertIsNone(database.load_run(run_id).answer)
            self.assertTrue((await browser.get(f"/api/runs/{run_id}")).json()["active"])
            conflict = await browser.post("/api/runs", json={"question": "重复提交"})
            self.assertEqual(conflict.status_code, 409)
            self.assertEqual((await browser.post(f"/api/runs/{run_id}/resume")).status_code, 409)
            release.set()
            await self.finished(app)
            detail = (await browser.get(f"/api/runs/{run_id}")).json()
            self.assertEqual(detail["status"], "completed")
            self.assertIn("测试回答", detail["answer"])
            self.assertFalse(detail["active"])
            self.assertFalse(detail["can_resume"])
            self.assertEqual(detail["plan"]["tasks"][0]["id"], "T1")
            self.assertEqual(detail["events"][-1]["event_type"], "run_completed")
            self.assertEqual((await browser.get("/api/runs")).json()["items"][0]["id"], run_id)
            self.assertEqual((await browser.post(f"/api/runs/{run_id}/resume")).status_code, 409)

    async def test_failed_research_resumes_same_id_without_replanning(self):
        client = client_with(response(PLAN), ValueError("模拟研究失败"), response("恢复成功"))
        app = create_app(client_factory=lambda: client, load_environment=False)
        async with app.router.lifespan_context(app), self.browser(app) as browser:
            run_id = (await browser.post("/api/runs", json={"question": "测试问题"})).json()["run_id"]
            await self.finished(app)
            detail = (await browser.get(f"/api/runs/{run_id}")).json()
            self.assertEqual(detail["status"], "failed")
            self.assertTrue(detail["can_resume"])
            resumed = await browser.post(f"/api/runs/{run_id}/resume")
            self.assertEqual(resumed.status_code, 202)
            self.assertEqual(resumed.json()["run_id"], run_id)
            await self.finished(app)
            self.assertEqual(database.load_run(run_id).answer, "恢复成功")
            self.assertEqual(client.chat.completions.create.await_count, 3)

    async def test_shutdown_cancels_model_and_new_server_can_resume(self):
        started = asyncio.Event()
        client = client_with()
        async def wait_model(**kwargs):
            started.set()
            await asyncio.Event().wait()
        client.chat.completions.create.side_effect = wait_model
        app = create_app(client_factory=lambda: client, load_environment=False)
        async with app.router.lifespan_context(app), self.browser(app) as browser:
            run_id = (await browser.post("/api/runs", json={"question": "中断测试"})).json()["run_id"]
            await asyncio.wait_for(started.wait(), 2)
        self.assertEqual(database.load_run(run_id).status, "failed")
        resumed_client = client_with(response(PLAN), response("重启恢复成功"))
        new_app = create_app(client_factory=lambda: resumed_client, load_environment=False)
        async with new_app.router.lifespan_context(new_app), self.browser(new_app) as browser:
            detail = (await browser.get(f"/api/runs/{run_id}")).json()
            self.assertTrue(detail["can_resume"])
            self.assertEqual((await browser.post(f"/api/runs/{run_id}/resume")).status_code, 202)
            await self.finished(new_app)
            self.assertEqual(database.load_run(run_id).answer, "重启恢复成功")

    async def test_invalid_input_missing_id_and_cross_site_are_rejected(self):
        client = client_with()
        app = create_app(client_factory=lambda: client, load_environment=False)
        async with app.router.lifespan_context(app), self.browser(app) as browser:
            for body in ({"question": " "}, {"question": "x" * 4001}, {"question": 123},
                         {"question": "valid", "api_key": "should-not-be-accepted"}, {}):
                self.assertEqual((await browser.post("/api/runs", json=body)).status_code, 422)
            self.assertEqual((await browser.get("/api/runs/invalid")).status_code, 422)
            self.assertEqual((await browser.get("/api/runs/" + "0" * 32)).status_code, 404)
            self.assertEqual((await browser.post("/api/runs/" + "0" * 32 + "/resume")).status_code, 404)
            self.assertEqual((await browser.get("/api/runs?limit=500")).status_code, 422)
            cross_site = await browser.post("/api/runs", json={"question": "bad"}, headers={"Origin": "https://example.org"})
            self.assertEqual(cross_site.status_code, 403)
            self.assertEqual((await browser.get("/.env")).status_code, 404)
            self.assertEqual((await browser.get("/data/research.db")).status_code, 404)
            client.chat.completions.create.assert_not_awaited()

    async def test_missing_key_keeps_history_readable_but_blocks_submission(self):
        app = create_app(client_factory=lambda: None, load_environment=False)
        async with app.router.lifespan_context(app), self.browser(app) as browser:
            self.assertFalse((await browser.get("/api/health")).json()["model_configured"])
            self.assertEqual((await browser.get("/api/runs")).status_code, 200)
            self.assertEqual((await browser.post("/api/runs", json={"question": "测试"})).status_code, 503)
            self.assertEqual(database.list_runs(), [])

    async def test_save_failure_does_not_schedule_model(self):
        client = client_with()
        app = create_app(client_factory=lambda: client, load_environment=False)
        async with app.router.lifespan_context(app), self.browser(app) as browser:
            with patch("runtime.save_run_with_event", side_effect=sqlite3.OperationalError("failed")):
                result = await browser.post("/api/runs", json={"question": "测试"})
            self.assertEqual(result.status_code, 503)
            self.assertEqual(app.state.service.tasks, {})
            client.chat.completions.create.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
