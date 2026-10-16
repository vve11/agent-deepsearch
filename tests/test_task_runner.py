"""真实事务 + 模拟模型/搜索，覆盖新顺序研究流程；不使用真实研究数据。"""
import asyncio
import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch, AsyncMock

import agent
import database as db
import runtime
import pipeline_storage as storage
from budget import BudgetExceeded
from plan import parse_plan
from task_runner import inspect_pipeline
from test_checkpoints import client_with, response
from test_model_requests import status_error


PLAN = json.dumps({'tasks': [{'question': '调查技术'}, {'question': '比较成本'}]})


async def search(call):
    query = json.loads(call.function.arguments)['query']
    return {'result': [{'title': query, 'url': 'https://example.com/' + query, 'content': '资料' + query}]}


class TaskRunnerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # 此组专测原调度协议；质量门禁全链路见 test_quality_gate.py。
        gate = patch.dict(os.environ, {'QUALITY_GATE_ENABLED': 'false'})
        gate.start()
        self.addCleanup(gate.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(db, 'DB_PATH', Path(directory.name) / 'test.db')
        patcher.start()
        self.addCleanup(patcher.stop)
        db.init_db()

    def sql(self, statement, args=()):
        with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
            return conn.execute(statement, args).fetchall()

    def saved(self):
        return db.load_run(self.sql('SELECT id FROM runs')[0][0])

    def prepared(self, count=2):
        run = runtime.create_research_run('研究主题')
        db.save_plan(run.id, parse_plan(run.question, json.loads(PLAN) if count == 2 else
                                      {'tasks': [{'question': '调查技术'}]}))
        storage.begin_tasks(run.id)
        return run

    async def test_separate_conversations_shared_evidence_and_synthesis(self):
        client = client_with(response(PLAN), response(queries=['tech']), response('技术 [E1]'),
                             response(queries=['cost']), response('成本 [E2]'), response('综合 [E1][E2]'))
        with patch.object(agent, 'execute_tool', side_effect=search):
            run = await runtime.execute_research(client, '研究主题')
        self.assertEqual(run.status, 'completed')
        self.assertEqual([e['id'] for e in run.evidence], ['E1', 'E2'])
        self.assertEqual(run.answer.count('本次收集的证据'), 1)
        first = db.load_task_checkpoint(run.id, 'T1')
        second = db.load_task_checkpoint(run.id, 'T2')
        self.assertEqual(first.messages[1]['content'], '调查技术')
        self.assertEqual(second.messages[1]['content'], '比较成本')
        self.assertIn('E1', second.messages[0]['content'])
        self.assertEqual(db.load_task_execution(run.id, 'T2').evidence_ids, ['E2'])
        calls = client.chat.completions.create.call_args_list
        self.assertNotIn('tools', calls[-1].kwargs)
        self.assertIn('技术 [E1]', calls[-1].kwargs['messages'][1]['content'])
        state = storage.load_pipeline(run.id)
        self.assertEqual((state['model_used'], state['tool_used'], state['phase']), (6, 2, 'completed'))
        self.assertEqual(db.load_run(run.id), run)

    async def test_resume_skips_completed_task_and_saved_search(self):
        client = client_with(response(PLAN), response('技术答案'), response(queries=['cost']), ValueError('离线'))
        with patch.object(agent, 'execute_tool', side_effect=search) as tool:
            with self.assertRaisesRegex(ValueError, '离线'):
                await runtime.execute_research(client, '研究主题')
            run = self.saved()
            second_client = client_with(response('成本 [E1]'), response('报告 [E1]'))
            await runtime.resume_research(second_client, run.id)
            self.assertEqual(tool.await_count, 1)
        self.assertEqual(second_client.chat.completions.create.call_args_list[0].kwargs['messages'][1]['content'], '比较成本')
        self.assertEqual(storage.load_pipeline(run.id)['model_used'], 6)
        self.assertEqual(sum(e['event_type'] == 'task_started' and e['payload']['task_id'] == 'T1'
                             for e in db.load_events(run.id)), 1)

    async def test_running_without_checkpoint_is_safe_before_first_request(self):
        run = self.prepared(count=1)
        db.start_task_execution(run.id, 'T1')  # 模拟刚提交 running 后进程退出。
        result = await runtime.resume_research(client_with(response('答案'), response('报告')), run.id)
        self.assertEqual(result.status, 'completed')
        self.assertIn('task_failed', [e['event_type'] for e in db.load_events(run.id)])

    async def test_cancelled_tool_resumes_last_complete_round_and_keeps_spent_budget(self):
        run = self.prepared(count=1)
        entered = asyncio.Event()
        async def waiting(call):
            entered.set()
            await asyncio.Event().wait()
        with patch.object(agent, 'execute_tool', side_effect=waiting):
            task = asyncio.create_task(runtime.execute_created_research(client_with(response(queries=['x'])), run))
            await asyncio.wait_for(entered.wait(), 2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(db.load_task_execution(run.id, 'T1').status, 'failed')
        self.assertEqual(db.load_task_checkpoint(run.id, 'T1').next_step, 0)
        self.assertEqual(storage.load_pipeline(run.id)['tool_used'], 1)
        await runtime.resume_research(client_with(response('资料不足'), response('报告')), run.id)
        self.assertEqual(storage.load_pipeline(run.id)['tool_used'], 1)

    async def test_final_commit_failure_reuses_report_even_with_exhausted_budget(self):
        run = self.prepared(count=1)
        self.sql('UPDATE research_pipeline SET model_limit=2 WHERE run_id=?', (run.id,))
        with patch.object(storage, 'finish_run', side_effect=sqlite3.OperationalError('磁盘失败')):
            with self.assertRaises(sqlite3.Error):
                await runtime.execute_created_research(client_with(response('子答案'), response('已生成报告')), run)
        self.assertEqual(storage.load_pipeline(run.id)['report'], '已生成报告')
        unused = client_with()
        result = await runtime.resume_research(unused, run.id)
        self.assertEqual(result.answer, '已生成报告')
        unused.chat.completions.create.assert_not_awaited()

    async def test_task_completion_failure_does_not_start_next_task(self):
        run = self.prepared()
        with patch.object(db, 'complete_task_execution', side_effect=sqlite3.OperationalError('磁盘失败')):
            with self.assertRaises(sqlite3.Error):
                await runtime.execute_created_research(client_with(response('答案')), run)
        self.assertEqual(db.load_task_execution(run.id, 'T2').status, 'pending')
        self.assertEqual(db.load_task_execution(run.id, 'T1').status, 'failed')
        await runtime.resume_research(client_with(response('重做答案'), response('成本'), response('报告')), run.id)
        self.assertEqual(db.load_run(run.id).status, 'completed')

    async def test_missing_saved_checkpoint_is_rejected_before_model(self):
        run = self.prepared(count=1)
        with self.assertRaises(ValueError):
            await runtime.execute_created_research(client_with(ValueError('离线')), run)
        self.sql('DELETE FROM task_checkpoints WHERE run_id=?', (run.id,))
        client = client_with()
        with self.assertRaisesRegex(ValueError, '检查点丢失'):
            await runtime.resume_research(client, run.id)
        client.chat.completions.create.assert_not_awaited()

    async def test_missing_task_or_evidence_is_not_reinitialized(self):
        for table in ('task_executions', 'research_evidence'):
            with self.subTest(table=table):
                run = self.prepared(count=1)
                self.sql(f'DELETE FROM {table} WHERE run_id=?', (run.id,))
                client = client_with()
                with self.assertRaises((ValueError, LookupError)):
                    await runtime.resume_research(client, run.id)
                client.chat.completions.create.assert_not_awaited()

    async def test_model_budget_persists_and_blocks_repeated_resume(self):
        run = self.prepared(count=1)
        self.sql('UPDATE research_pipeline SET model_limit=1 WHERE run_id=?', (run.id,))
        with self.assertRaises(ValueError):
            await runtime.execute_created_research(client_with(ValueError('离线')), run)
        client = client_with()
        with self.assertRaises(BudgetExceeded):
            await runtime.resume_research(client, run.id)
        self.assertEqual(runtime.get_resumable_runs(), [])
        client.chat.completions.create.assert_not_awaited()
        self.assertEqual(storage.load_pipeline(run.id)['model_used'], 1)

    async def test_partial_tool_round_cannot_overspend_after_resume(self):
        run = self.prepared(count=1)
        self.sql('UPDATE research_pipeline SET tool_limit=1 WHERE run_id=?', (run.id,))
        with patch.object(agent, 'execute_tool', side_effect=search) as tool:
            with self.assertRaises(BudgetExceeded):
                await runtime.execute_created_research(client_with(response(queries=['a', 'b'])), run)
            self.assertEqual(tool.await_count, 1)
        self.assertEqual(db.load_research_evidence(run.id), [])
        client = client_with(response('证据不足'), response('证据不足的报告'))
        await runtime.resume_research(client, run.id)
        self.assertEqual(client.chat.completions.create.call_args_list[0].kwargs['tool_choice'], 'none')
        self.assertEqual(storage.load_pipeline(run.id)['tool_used'], 1)

    async def test_synthesis_failure_resumes_only_synthesis(self):
        run = self.prepared(count=1)
        with self.assertRaisesRegex(ValueError, '空正文'):
            await runtime.execute_created_research(client_with(response('子答案'), response(' ')), run)
        self.assertEqual(db.load_task_execution(run.id, 'T1').status, 'completed')
        self.assertEqual(runtime.get_resumable_runs()[0]['phase'], 'synthesis')
        client = client_with(response('汇总'))
        await runtime.resume_research(client, run.id)
        self.assertEqual(client.chat.completions.create.await_count, 1)
        self.assertNotIn('tools', client.chat.completions.create.call_args.kwargs)

    async def test_unknown_synthesis_citations_get_one_repair(self):
        run = self.prepared(count=1)
        await runtime.execute_created_research(client_with(response('答案'), response('错误 [E9]'), response('证据不足')), run)
        self.assertEqual(db.load_run(run.id).answer, '证据不足')
        self.assertEqual(storage.load_pipeline(run.id)['model_used'], 3)

    async def test_model_is_frozen_for_run_across_environment_changes(self):
        with patch.dict(os.environ, {'DEEPSEEK_MODEL': 'model-a'}):
            run = self.prepared(count=1)
        client = client_with(response('答案'), response('报告'))
        with patch.dict(os.environ, {'DEEPSEEK_MODEL': 'model-b'}):
            await runtime.execute_created_research(client, run)
        self.assertEqual({c.kwargs['model'] for c in client.chat.completions.create.call_args_list}, {'model-a'})

    def test_budget_and_event_roll_back_together(self):
        run = self.prepared(count=1)
        with patch.object(db, '_append_event', side_effect=sqlite3.OperationalError('event failed')):
            with self.assertRaises(sqlite3.Error):
                storage.reserve_budget(run.id, 'model')
        self.assertEqual(storage.load_pipeline(run.id)['model_used'], 0)

    def test_initialization_is_atomic(self):
        run = runtime.create_research_run('研究主题')
        db.save_plan(run.id, parse_plan(run.question, json.loads(PLAN)))
        with patch.object(db, '_append_event', side_effect=sqlite3.OperationalError('event failed')):
            with self.assertRaises(sqlite3.Error):
                storage.begin_tasks(run.id)
        self.assertEqual(storage.load_pipeline(run.id)['phase'], 'planning')
        self.assertEqual(self.sql('SELECT * FROM task_executions'), [])
        storage.begin_tasks(run.id)
        self.assertEqual(len(inspect_pipeline(run.id)[1]), 2)

    async def test_network_retry_consumes_persisted_model_budget(self):
        run = self.prepared(count=1)
        client = client_with(status_error(503), response('答案'), response('报告'))
        with patch('model_requests.asyncio.sleep', new_callable=AsyncMock):
            await runtime.execute_created_research(client, run)
        self.assertEqual(storage.load_pipeline(run.id)['model_used'], 3)

    async def test_budget_blocks_retry_before_another_network_request(self):
        run = self.prepared(count=1)
        self.sql('UPDATE research_pipeline SET model_limit=1 WHERE run_id=?', (run.id,))
        client = client_with(status_error(503), response('不能请求'))
        with patch('model_requests.asyncio.sleep', new_callable=AsyncMock):
            with self.assertRaises(BudgetExceeded):
                await runtime.execute_created_research(client, run)
        self.assertEqual(client.chat.completions.create.await_count, 1)

    async def test_failed_initial_checkpoint_prevents_model_request(self):
        run = self.prepared(count=1)
        client = client_with()
        with patch.object(db, 'save_task_checkpoint', side_effect=sqlite3.OperationalError('磁盘失败')):
            with self.assertRaises(sqlite3.Error):
                await runtime.execute_created_research(client, run)
        client.chat.completions.create.assert_not_awaited()
        self.assertEqual(storage.load_pipeline(run.id)['model_used'], 0)
        # 从未产生过检查点，与已保存检查点丢失不同，可以从首次请求开始。
        await runtime.resume_research(client_with(response('答案'), response('报告')), run.id)

    async def test_final_event_failure_rolls_back_completed_status(self):
        run = self.prepared(count=1)
        self.sql("""CREATE TRIGGER reject_completed BEFORE INSERT ON run_events
            WHEN NEW.event_type='run_completed'
            BEGIN SELECT RAISE(ABORT, 'event failed'); END""")
        with self.assertRaises(sqlite3.Error):
            await runtime.execute_created_research(client_with(response('答案'), response('报告')), run)
        self.assertEqual(storage.load_pipeline(run.id)['phase'], 'synthesis')
        self.assertEqual(db.load_run(run.id).status, 'failed')
        self.sql('DROP TRIGGER reject_completed')
        client = client_with()
        await runtime.resume_research(client, run.id)
        client.chat.completions.create.assert_not_awaited()

    async def test_same_run_cannot_be_executed_twice_concurrently(self):
        run = self.prepared(count=1)
        entered, release = asyncio.Event(), asyncio.Event()
        client = client_with()
        async def wait_response(**kwargs):
            entered.set()
            await release.wait()
            return response('答案')
        client.chat.completions.create.side_effect = wait_response
        task = asyncio.create_task(runtime.execute_created_research(client, run))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            with self.assertRaisesRegex(ValueError, '正在本进程执行'):
                await runtime.resume_research(client_with(), run.id)
        finally:
            release.set()
            await task

    async def test_malformed_checkpoint_event_blocks_resume_cleanly(self):
        run = self.prepared(count=1)
        with self.assertRaises(ValueError):
            await runtime.execute_created_research(client_with(ValueError('离线')), run)
        self.sql("UPDATE run_events SET payload_json='[]' WHERE run_id=? AND event_type='task_checkpoint_saved'", (run.id,))
        with self.assertRaisesRegex(ValueError, '事件数据损坏'):
            await runtime.resume_research(client_with(), run.id)


if __name__ == '__main__':
    unittest.main()
