"""检查点的真实 SQLite 存取和模拟模型恢复测试，不访问网络。"""
import asyncio
import contextlib
import io
import json
import sqlite3
import tempfile
import subprocess
import sys
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import agent
import cli
import database
import runtime
from plan import parse_plan
from checkpoint import AgentCheckpoint
from evidence import EvidenceStore
from run import ResearchRun


def response(text=None, queries=()):
    calls = [
        {"id": f"call_{index}", "type": "function", "function": {
            "name": "search_web", "arguments": json.dumps({"query": query}),
        }} for index, query in enumerate(queries)
    ]
    message_data = {"role": "assistant", "content": text}
    if calls:
        message_data["tool_calls"] = calls
    message = SimpleNamespace(
        content=text,
        tool_calls=[SimpleNamespace(
            id=call["id"], function=SimpleNamespace(**call["function"]),
        ) for call in calls],
        model_dump=lambda **kwargs: deepcopy(message_data),
    )
    return SimpleNamespace(choices=[SimpleNamespace(
        message=message, finish_reason="tool_calls" if calls else "stop",
    )])


def client_with(*outcomes):
    create = AsyncMock(side_effect=list(outcomes))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class CheckpointTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db_patch = patch.object(database, "DB_PATH", Path(directory.name) / "test.db")
        db_patch.start()
        self.addCleanup(db_patch.stop)
        database.init_db()
        # These tests isolate research/recovery; planner has dedicated integration tests.
        async def planned(client, question, record):
            return parse_plan(question, {"tasks": [{"question": question}]})
        planner_patch = patch.object(runtime, "create_plan", side_effect=planned)
        planner_patch.start()
        self.addCleanup(planner_patch.stop)

    def saved_run(self):
        with contextlib.closing(sqlite3.connect(database.DB_PATH)) as connection:
            run_id = connection.execute("SELECT id FROM runs").fetchone()[0]
        return database.load_run(run_id)

    async def fail_after_search(self, count=2):
        client = client_with(response(queries=[f"q{i}" for i in range(count)]),
                             ValueError("模拟下一轮请求失败"))

        async def search(call):
            return {"result": [{"title": call.id, "url": f"https://example.com/{call.id}",
                                "content": "搜索资料"}]}

        with patch.object(agent, "execute_tool", side_effect=search) as tool:
            with self.assertRaisesRegex(ValueError, "模拟下一轮"):
                await runtime.execute_research(client, "研究电池回收")
            self.assertEqual(tool.await_count, count)
        return self.saved_run()

    async def test_resume_keeps_messages_evidence_and_does_not_repeat_search(self):
        run = await self.fail_after_search()
        checkpoint = database.load_checkpoint(run.id)
        self.assertEqual(checkpoint.next_step, 1)
        self.assertEqual(checkpoint.tool_count, 2)
        self.assertEqual([e["id"] for e in checkpoint.evidence], ["E1", "E2"])
        client = client_with(response("回答 [E1][E2]"))
        with patch.object(agent, "execute_tool", new_callable=AsyncMock) as tool:
            resumed = await runtime.resume_research(client, run.id)
            tool.assert_not_awaited()
        self.assertEqual(resumed.id, run.id)
        self.assertEqual(resumed.status, "completed")
        self.assertIsNone(resumed.error)
        self.assertEqual(resumed.evidence, checkpoint.evidence)
        self.assertEqual(database.load_run(run.id), resumed)
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["messages"], checkpoint.messages)
        self.assertEqual(kwargs["model"], checkpoint.model_name)
        events = database.load_events(run.id)
        self.assertEqual(sum(e["event_type"] == "run_started" for e in events), 1)
        self.assertEqual(sum(e["event_type"] == "run_resumed" for e in events), 1)
        model_events = [e for e in events if e["event_type"] == "model_started"]
        self.assertEqual(model_events[-1]["payload"]["step"], 2)

    async def test_exhausted_tool_budget_stays_none_after_resume(self):
        run = await self.fail_after_search(count=3)
        client = client_with(response("回答 [E1]"))
        await runtime.resume_research(client, run.id)
        self.assertEqual(client.chat.completions.create.call_args.kwargs["tool_choice"], "none")

    async def test_resume_adds_evidence_without_renumbering(self):
        run = await self.fail_after_search()
        next_response = response(queries=["补充资料"])
        next_response.choices[0].message.tool_calls[0].id = "new_call"
        next_response.choices[0].message.model_dump = lambda **kwargs: {
            "role": "assistant", "tool_calls": [{"id": "new_call", "type": "function",
            "function": {"name": "search_web", "arguments": '{"query":"补充资料"}'}}],
        }
        client = client_with(next_response, response("答案 [E3]"))
        with patch.object(agent, "execute_tool", return_value={"result": [
            {"title": "新资料", "url": "https://example.com/new", "content": "补充"},
        ]}) as tool:
            resumed = await runtime.resume_research(client, run.id)
        self.assertEqual(tool.await_count, 1)
        self.assertEqual([e["id"] for e in resumed.evidence], ["E1", "E2", "E3"])
        checkpoint = database.load_checkpoint(run.id)
        self.assertEqual((checkpoint.next_step, checkpoint.tool_count), (2, 3))

    async def test_initial_checkpoint_recovers_first_request_failure(self):
        with self.assertRaises(ValueError):
            await runtime.execute_research(client_with(ValueError("离线")), "问题")
        run = self.saved_run()
        self.assertEqual(database.load_checkpoint(run.id).next_step, 0)
        resumed = await runtime.resume_research(client_with(response("回答")), run.id)
        self.assertEqual(resumed.answer, "回答")

    async def test_checkpoint_save_failure_stops_before_model(self):
        client = client_with(response("不应调用"))
        with patch.object(runtime, "save_checkpoint", side_effect=sqlite3.OperationalError("失败")):
            with self.assertRaises(sqlite3.OperationalError):
                await runtime.execute_research(client, "问题")
        client.chat.completions.create.assert_not_awaited()
        self.assertEqual(self.saved_run().status, "failed")

    async def test_round_checkpoint_failure_keeps_previous_checkpoint(self):
        client = client_with(response(queries=["资料"]), response("不应请求"))
        writes = 0

        def save(checkpoint):
            nonlocal writes
            writes += 1
            if writes == 2:
                raise sqlite3.OperationalError("整轮保存失败")
            database.save_checkpoint(checkpoint)

        with patch.object(runtime, "save_checkpoint", side_effect=save), \
             patch.object(agent, "execute_tool", return_value={"result": [
                 {"title": "资料", "url": "https://example.com", "content": "内容"},
             ]}):
            with self.assertRaises(sqlite3.OperationalError):
                await runtime.execute_research(client, "问题")
        self.assertEqual(client.chat.completions.create.await_count, 1)
        run = self.saved_run()
        self.assertEqual(database.load_checkpoint(run.id).next_step, 0)
        self.assertEqual(len(run.evidence), 1)

    async def test_completed_and_missing_checkpoint_rejected(self):
        client = client_with(response("答案"))
        run = await runtime.execute_research(client, "问题")
        with self.assertRaisesRegex(ValueError, "已经完成"):
            await runtime.resume_research(client, run.id)
        old = ResearchRun(question="旧任务", status="failed")
        database.save_run(old)
        with self.assertRaises(LookupError):
            await runtime.resume_research(client, old.id)
        self.assertEqual(client.chat.completions.create.await_count, 1)

    async def test_event_failure_rolls_back_checkpoint_overwrite(self):
        run = await self.fail_after_search()
        before = database.load_checkpoint(run.id)
        old_events = database.load_events(run.id)
        with contextlib.closing(sqlite3.connect(database.DB_PATH)) as connection:
            connection.execute("""CREATE TRIGGER reject_checkpoint_event
                BEFORE INSERT ON run_events WHEN NEW.event_type = 'checkpoint_saved'
                BEGIN SELECT RAISE(ABORT, 'reject checkpoint event'); END""")
        changed = deepcopy(before)
        changed.model_name = "changed-model"
        with self.assertRaises(sqlite3.IntegrityError):
            database.save_checkpoint(changed)
        self.assertEqual(database.load_checkpoint(run.id), before)
        self.assertEqual(database.load_events(run.id), old_events)

    async def test_malformed_checkpoint_is_rejected_before_model(self):
        run = await self.fail_after_search()
        original = asdict(database.load_checkpoint(run.id))
        corruptions = []
        for key, value in [("version", 99), ("tool_count", True), ("tool_count", 0),
                           ("next_step", 0), ("messages", []), ("evidence", [{}]),
                           ("run_id", "0" * 32)]:
            data = deepcopy(original)
            data[key] = value
            corruptions.append(data)
        missing_result = deepcopy(original)
        missing_result["messages"].pop()
        corruptions.append(missing_result)
        for data in corruptions:
            with self.subTest(data=data):
                with contextlib.closing(sqlite3.connect(database.DB_PATH)) as connection:
                    with connection:
                        connection.execute("UPDATE checkpoints SET state_json = ? WHERE run_id = ?",
                                           (json.dumps(data), run.id))
                client = client_with(response("不应调用"))
                with self.assertRaises(ValueError):
                    await runtime.resume_research(client, run.id)
                client.chat.completions.create.assert_not_awaited()
        self.assertEqual(database.load_run(run.id).status, "failed")

    async def test_active_run_cannot_be_resumed_and_cancel_keeps_checkpoint(self):
        started = asyncio.Event()

        async def wait_response(**kwargs):
            started.set()
            await asyncio.Event().wait()

        client = client_with()
        client.chat.completions.create.side_effect = wait_response
        task = asyncio.create_task(runtime.execute_research(client, "阻塞任务"))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            run = self.saved_run()
            with self.assertRaisesRegex(ValueError, "正在本进程"):
                await runtime.resume_research(client, run.id)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(database.load_run(run.id).status, "failed")
        self.assertEqual(database.load_checkpoint(run.id).next_step, 0)
        self.assertNotIn(run.id, runtime._active_runs)

    async def test_cli_resume_command_and_usage(self):
        run = await self.fail_after_search()
        client = client_with(response("恢复后的答案 [E1]"))
        output = io.StringIO()
        with patch("builtins.input", side_effect=["/resume", "0", "/resume " + run.id, "/exit"]), \
             contextlib.redirect_stdout(output):
            await cli.run_cli(client)
        self.assertIn("已保存 2 条证据", output.getvalue())
        self.assertIn("恢复后的答案", output.getvalue())
        self.assertEqual(database.load_run(run.id).status, "completed")

    def test_evidence_restore_rejects_duplicates_and_preserves_next_id(self):
        records = [{"id": "E1", "title": "标题", "url": "https://example.com", "content": "内容"}]
        store = EvidenceStore.from_records(records)
        self.assertEqual(store.add(records[0])["id"], "E1")
        self.assertEqual(store.add({"title": "新", "url": "https://example.com/2", "content": "新"})["id"], "E2")
        with self.assertRaises(ValueError):
            EvidenceStore.from_records(records + records)

    async def test_new_python_process_resumes_running_task_without_old_memory(self):
        run = await self.fail_after_search()
        # 模拟硬退出留下的 running 状态；新解释器不共享任何内存变量。
        run.status = "running"
        run.error = None
        database.save_run(run)
        script = '''
import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import agent, database, runtime
database.DB_PATH = Path(sys.argv[1])
message = SimpleNamespace(content="Recovered [E1]", tool_calls=[])
reply = SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])
create = AsyncMock(return_value=reply)
client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
async def check():
    with patch.object(agent, "execute_tool", new_callable=AsyncMock) as tool:
        run = await runtime.resume_research(client, sys.argv[2])
        tool.assert_not_awaited()
    assert run.status == "completed"
    assert len(run.evidence) == 2
    assert len(create.call_args.kwargs["messages"]) == 5
asyncio.run(check())
'''
        result = await asyncio.to_thread(
            subprocess.run,
            [sys.executable, "-c", script, str(database.DB_PATH), run.id],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(database.load_run(run.id).status, "completed")


if __name__ == "__main__":
    unittest.main()
