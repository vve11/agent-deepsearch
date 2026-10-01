"""真实临时 SQLite + 模拟模型，验证规划、研究、恢复之间的边界。"""
import asyncio
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import database
import runtime
from checkpoint import AgentCheckpoint
from plan import parse_plan, plan_context
from run import ResearchRun
from test_checkpoints import client_with, response


PLAN_REPLY = json.dumps({"tasks": [{"question": "调查回收技术"}, {"question": "比较成本"}]})


class PlanningRuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(database, "DB_PATH", Path(directory.name) / "test.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_db()

    def sql(self, statement, args=()):
        with closing(sqlite3.connect(database.DB_PATH)) as connection:
            with connection:
                return connection.execute(statement, args).fetchall()

    def saved_run(self):
        return database.load_run(self.sql("SELECT id FROM runs")[0][0])

    def pre_research(self, planned=False):
        run = ResearchRun(question="研究电池回收")
        database.save_run_with_event(run, "run_started", {}, start_planning=True)
        if planned:
            database.save_plan(run.id, parse_plan(run.question, json.loads(PLAN_REPLY)))
        return run

    async def test_new_run_plans_before_research_and_persists_context(self):
        client = client_with(response(PLAN_REPLY), response("研究答案"))
        run = await runtime.execute_research(client, "研究电池回收")
        self.assertEqual(run.status, "completed")
        calls = client.chat.completions.create.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertNotIn("tools", calls[0].kwargs)
        self.assertIn("tools", calls[1].kwargs)
        plan = database.load_plan(run.id)
        checkpoint = database.load_checkpoint(run.id)
        self.assertTrue(checkpoint.messages[0]["content"].endswith(plan_context(plan)))
        self.assertEqual(checkpoint.messages[1]["content"], run.question)
        self.assertEqual(database.load_workflow_phase(run.id), "research")
        self.assertEqual([e["event_type"] for e in database.load_events(run.id)], [
            "run_started", "plan_started", "plan_generated", "plan_saved",
            "checkpoint_saved", "model_started", "model_completed", "citation_check_passed", "run_completed",
        ])
        with self.assertRaisesRegex(ValueError, "已经完成"):
            await runtime.resume_research(client, run.id)
        self.assertEqual(client.chat.completions.create.await_count, 2)

    async def test_bad_planner_output_can_resume_with_same_id(self):
        with self.assertRaises(ValueError):
            await runtime.execute_research(client_with(response("invalid JSON")), "问题")
        run = self.saved_run()
        self.assertEqual(run.status, "failed")
        self.assertEqual(database.load_workflow_phase(run.id), "planning")
        self.assertEqual(runtime.get_resumable_runs()[0]["phase"], "planning")
        resumed = await runtime.resume_research(client_with(response(PLAN_REPLY), response("完成")), run.id)
        self.assertEqual(resumed.id, run.id)
        events = [e["event_type"] for e in database.load_events(run.id)]
        self.assertEqual(events.count("run_started"), 1)
        self.assertEqual(events.count("run_resumed"), 1)
        self.assertEqual(events.count("plan_started"), 2)

    async def test_saved_plan_before_checkpoint_is_reused(self):
        run = self.pre_research(planned=True)  # 模拟进程在保存计划后退出。
        self.assertEqual(runtime.get_resumable_runs()[0]["phase"], "planned")
        client = client_with(response("完成"))
        with patch.object(runtime, "create_plan") as planner:
            await runtime.resume_research(client, run.id)
            planner.assert_not_called()
        self.assertEqual(client.chat.completions.create.await_count, 1)
        self.assertIn("plan_reused", [e["event_type"] for e in database.load_events(run.id)])

    async def test_research_resume_preserves_messages_without_replanning(self):
        with self.assertRaises(ValueError):
            await runtime.execute_research(client_with(response(PLAN_REPLY), ValueError("offline")), "问题")
        run = self.saved_run()
        before = database.load_checkpoint(run.id)
        client = client_with(response("完成"))
        with patch.object(runtime, "create_plan") as planner:
            await runtime.resume_research(client, run.id)
            planner.assert_not_called()
        self.assertEqual(client.chat.completions.create.call_args.kwargs["messages"], before.messages)
        self.assertEqual(sum(e["event_type"] == "plan_saved" for e in database.load_events(run.id)), 1)

    async def test_plan_event_failure_rolls_back_phase_and_stops_research(self):
        self.sql("""CREATE TRIGGER fail_plan BEFORE INSERT ON run_events
            WHEN NEW.event_type = 'plan_saved'
            BEGIN SELECT RAISE(ABORT, 'plan event failed'); END""")
        client = client_with(response(PLAN_REPLY))
        with self.assertRaises(sqlite3.IntegrityError):
            await runtime.execute_research(client, "问题")
        run = self.saved_run()
        self.assertEqual(run.status, "failed")
        self.assertEqual(database.load_workflow_phase(run.id), "planning")
        self.assertEqual(client.chat.completions.create.await_count, 1)
        with self.assertRaises(LookupError):
            database.load_plan(run.id)
        with self.assertRaises(LookupError):
            database.load_checkpoint(run.id)

    async def test_checkpoint_event_failure_leaves_saved_plan_recoverable(self):
        self.sql("""CREATE TRIGGER fail_checkpoint BEFORE INSERT ON run_events
            WHEN NEW.event_type = 'checkpoint_saved'
            BEGIN SELECT RAISE(ABORT, 'checkpoint failed'); END""")
        client = client_with(response(PLAN_REPLY))
        with self.assertRaises(sqlite3.IntegrityError):
            await runtime.execute_research(client, "问题")
        run = self.saved_run()
        self.assertEqual(database.load_workflow_phase(run.id), "planned")
        with self.assertRaises(LookupError):
            database.load_checkpoint(run.id)
        self.assertEqual(client.chat.completions.create.await_count, 1)
        self.sql("DROP TRIGGER fail_checkpoint")
        with patch.object(runtime, "create_plan") as planner:
            await runtime.resume_research(client_with(response("完成")), run.id)
            planner.assert_not_called()

    async def test_missing_research_checkpoint_does_not_restart_budget(self):
        with self.assertRaises(ValueError):
            await runtime.execute_research(client_with(response(PLAN_REPLY), ValueError("offline")), "问题")
        run = self.saved_run()
        self.sql("DELETE FROM checkpoints WHERE run_id = ?", (run.id,))
        client = client_with()
        with self.assertRaises(LookupError):
            await runtime.resume_research(client, run.id)
        client.chat.completions.create.assert_not_awaited()
        self.assertEqual(runtime.get_resumable_runs(), [])

    async def test_missing_or_corrupt_plan_is_not_regenerated(self):
        run = self.pre_research(planned=True)
        original = self.sql("SELECT plan_json FROM research_plans")[0][0]
        for payload in ("not json", "{}", json.dumps({"question": run.question, "tasks": [
            {"id": "T9", "question": "调查"}]})):
            self.sql("UPDATE research_plans SET plan_json = ?", (payload,))
            client = client_with()
            with self.assertRaises(ValueError):
                await runtime.resume_research(client, run.id)
            client.chat.completions.create.assert_not_awaited()
            self.assertEqual(runtime.get_resumable_runs(), [])
        self.sql("UPDATE research_plans SET plan_json = ?", (original,))
        self.sql("DELETE FROM research_plans")
        with self.assertRaises(ValueError):
            await runtime.resume_research(client_with(), run.id)

    async def test_changed_plan_after_research_is_rejected(self):
        with self.assertRaises(ValueError):
            await runtime.execute_research(client_with(response(PLAN_REPLY), ValueError("offline")), "问题")
        run = self.saved_run()
        changed = {"question": run.question, "tasks": [{"id": "T1", "question": "不同的问题"}]}
        self.sql("UPDATE research_plans SET plan_json = ?", (json.dumps(changed),))
        with self.assertRaisesRegex(ValueError, "计划不一致"):
            await runtime.resume_research(client_with(), run.id)

    async def test_legacy_checkpoint_resumes_without_plan(self):
        run = ResearchRun(question="旧任务", status="failed")
        database.save_run(run)
        checkpoint = AgentCheckpoint(run.id, 0, 0, [
            {"role": "system", "content": "旧系统提示"}, {"role": "user", "content": run.question},
        ], [], "old-model")
        database.save_checkpoint(checkpoint)
        client = client_with(response("完成"))
        with patch.object(runtime, "create_plan") as planner:
            await runtime.resume_research(client, run.id)
            planner.assert_not_called()
        self.assertEqual(client.chat.completions.create.call_args.kwargs["messages"], checkpoint.messages)
        self.assertIsNone(database.load_workflow_phase(run.id))

    async def test_cancel_during_planning_stays_recoverable(self):
        started = asyncio.Event()
        async def blocked(**kwargs):
            started.set()
            await asyncio.Event().wait()
        client = client_with()
        client.chat.completions.create.side_effect = blocked
        task = asyncio.create_task(runtime.execute_research(client, "问题"))
        try:
            await asyncio.wait_for(started.wait(), 2)
            run = self.saved_run()
            self.assertEqual(runtime.get_resumable_runs(), [])  # 活跃任务不可重复恢复。
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(database.load_run(run.id).status, "failed")
        self.assertEqual(runtime.get_resumable_runs()[0]["id"], run.id)

    def test_run_start_failure_does_not_leave_workflow(self):
        self.sql("""CREATE TRIGGER fail_start BEFORE INSERT ON run_events
            WHEN NEW.event_type = 'run_started'
            BEGIN SELECT RAISE(ABORT, 'start failed'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.pre_research()
        self.assertEqual(self.sql("SELECT * FROM runs"), [])
        self.assertEqual(self.sql("SELECT * FROM run_workflow"), [])


if __name__ == "__main__":
    unittest.main()
