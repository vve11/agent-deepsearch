"""完整记忆闭环：真实临时数据库，模拟研究模型与搜索，不联网。"""
import asyncio
from contextlib import closing, redirect_stdout
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import agent
import database as db
import runtime
from memory import MemoryScope, MemoryDraft, MemorySource, SemanticContent, MemoryQuery
from memory_storage import MemoryStore
from memory_codec import serialize_memory
from memory_context import load_context, MAX_CONTEXT_CHARS, current_memory_context
from memory_jobs import process_job, process_pending, retry_job, describe_memory
from memory_commands import handle_memory_command
from test_checkpoints import client_with, response

PLAN = json.dumps({'tasks': [{'question': '调查电池成本'}]})


class MemoryPipelineTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        gate = patch.dict(os.environ, {'QUALITY_GATE_ENABLED': 'false'})
        gate.start()
        self.addCleanup(gate.stop)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patcher = patch.object(db, 'DB_PATH', Path(folder.name) / 'research.db')
        patcher.start()
        self.addCleanup(patcher.stop)
        env = patch.dict(os.environ, {'MEMORY_SCOPE_ID': 'test-project', 'MEMORY_ENABLED': 'true'})
        env.start()
        self.addCleanup(env.stop)
        db.init_db()
        self.scope = MemoryScope('project', 'test-project')
        self.store = MemoryStore()

    def sql(self, statement, args=()):
        with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
            return conn.execute(statement, args).fetchall()

    async def research(self):
        client = client_with(response(PLAN), response(queries=['电池成本']), response('成本资料 [E1]'), response('电池成本报告 [E1]'))
        async def search(call):
            return {'result': [{'title': '电池成本报告', 'url': 'https://example.com/battery', 'content': '电池成本取决于材料和工艺。'}]}
        with patch.object(agent, 'execute_tool', side_effect=search):
            run = await runtime.execute_research(client, '研究电池成本')
        return run, client

    def preference(self, value='中文'):
        return MemoryDraft(kind='semantic', scope=self.scope,
            sources=[MemorySource('user_input', 'explicit-preference')],
            content=SemanticContent('preference', '用户', '报告语言', value, '未指定时'))

    async def test_completed_research_extracts_and_reviews_once_without_model_calls(self):
        run, client = await self.research()
        records = self.store.list(scope=self.scope)
        self.assertEqual({r.kind: r.status for r in records},
                         {'episodic': 'active', 'semantic': 'active', 'procedural': 'candidate'})
        self.assertEqual(client.chat.completions.create.await_count, 4)
        self.assertEqual(self.sql('SELECT status,attempts FROM memory_jobs WHERE run_id=?', (run.id,)), [('done', 1)])
        before = self.sql('SELECT * FROM memory_events')
        process_job(run.id)
        self.assertEqual(self.sql('SELECT * FROM memory_events'), before)
        self.assertTrue(all(self.store.reviews(r.id, scope=self.scope) for r in records))

    async def test_next_run_receives_memory_in_planner_agent_and_synthesis(self):
        await self.research()
        client = client_with(response(PLAN), response('需要重新核实资料'), response('本次报告'))
        run = await runtime.execute_research(client, '继续研究电池成本')
        for call in client.chat.completions.create.call_args_list:
            self.assertIn('以下是历史记忆', call.kwargs['messages'][0]['content'])
        self.assertIn('电池成本', load_context(run.id))
        self.assertEqual(run.evidence, [])  # 历史来源不自动成为本次证据。
        self.assertEqual(current_memory_context(), '')

    async def test_restore_keeps_snapshot_after_archive_and_new_memory(self):
        await self.research()
        run = runtime.create_research_run('研究电池成本')
        frozen = load_context(run.id)
        client = client_with(response(PLAN), ValueError('offline'))
        with self.assertRaises(ValueError):
            await runtime.execute_created_research(client, run)
        for memory in self.store.list(scope=self.scope):
            self.store.archive(memory.id, scope=self.scope)
        fresh = self.store.propose(self.preference('英语'))
        self.store.activate(fresh.id, scope=self.scope, reviewer='user', reason='用户明确要求')
        self.assertEqual(load_context(run.id), frozen)
        resumed = client_with(response('恢复答案'), response('恢复报告'))
        await runtime.resume_research(resumed, run.id)
        self.assertNotIn('英语', resumed.chat.completions.create.call_args_list[0].kwargs['messages'][0]['content'])

    async def test_extraction_failure_keeps_completed_answer_and_retries(self):
        with patch('memory_jobs.extract_drafts', side_effect=ValueError('bad extraction')):
            run, _ = await self.research()
        self.assertEqual(db.load_run(run.id).status, 'completed')
        self.assertEqual(self.sql('SELECT status FROM memory_jobs'), [('failed',)])
        process_pending()
        self.assertEqual(self.sql('SELECT status,attempts FROM memory_jobs'), [('done', 2)])
        self.assertEqual(len(self.store.list(scope=self.scope)), 3)

    async def test_failed_partial_research_does_not_extract(self):
        with self.assertRaises(ValueError):
            await runtime.execute_research(client_with(response(PLAN), ValueError('failed')), '研究电池成本')
        self.assertEqual(self.sql('SELECT * FROM memory_jobs'), [])
        self.assertEqual(self.store.list(scope=self.scope), [])

    async def test_manual_retry_checks_scope_and_completed_jobs_are_not_replayed(self):
        with patch('memory_jobs.extract_drafts', side_effect=ValueError('bad extraction')):
            run, _ = await self.research()
        info = describe_memory(run.id)
        self.assertTrue(info['enabled'])
        self.assertEqual(info['selected_ids'], [])
        self.assertEqual(info['extraction']['status'], 'failed')
        with self.assertRaises(LookupError):
            retry_job(run.id, scope=MemoryScope('project', 'other'))
        retry_job(run.id, scope=self.scope)
        self.assertEqual(describe_memory(run.id)['extraction']['status'], 'done')
        with self.assertRaises(ValueError):
            retry_job(run.id, scope=self.scope)

    async def test_job_retry_is_bounded(self):
        with patch('memory_jobs.extract_drafts', side_effect=ValueError('broken')):
            await self.research()
            for _ in range(5):
                process_pending()
        self.assertEqual(self.sql('SELECT status,attempts FROM memory_jobs'), [('failed', 3)])

    async def test_interrupted_job_recovered_on_startup(self):
        with patch('runtime.execute_pipeline', wraps=runtime.execute_pipeline), patch('memory_jobs.process_job'):
            run, _ = await self.research()
        self.sql("UPDATE memory_jobs SET status='running',attempts=1 WHERE run_id=?", (run.id,))
        process_pending(recover_interrupted=True)
        self.assertEqual(self.sql('SELECT status,attempts FROM memory_jobs'), [('done', 2)])

    async def test_partial_saved_candidates_retry_without_duplicates(self):
        real = MemoryStore.activate
        count = 0
        def fail_second(store, *args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise sqlite3.OperationalError('write failed')
            return real(store, *args, **kwargs)
        with patch.object(MemoryStore, 'activate', new=fail_second):
            await self.research()
        process_pending()
        self.assertEqual(len(self.store.list(scope=self.scope)), 3)
        self.assertEqual(self.sql('SELECT status FROM memory_jobs'), [('done',)])

    async def test_disabled_new_run_has_no_snapshot_content_or_extraction(self):
        await self.research()
        before = len(self.store.list(scope=self.scope))
        with patch.dict(os.environ, {'MEMORY_ENABLED': 'false'}):
            run, client = await self.research()
        self.assertEqual(load_context(run.id), '')
        self.assertEqual(len(self.store.list(scope=self.scope)), before)
        self.assertEqual(self.sql('SELECT status FROM memory_jobs WHERE run_id=?', (run.id,)), [])
        self.assertNotIn('以下是历史记忆', client.chat.completions.create.call_args_list[0].kwargs['messages'][0]['content'])

    async def test_missing_snapshot_rejected_before_resume_request(self):
        run = runtime.create_research_run('研究电池成本')
        self.sql('DELETE FROM run_memory_context WHERE run_id=?', (run.id,))
        client = client_with()
        with self.assertRaisesRegex(ValueError, '快照丢失'):
            await runtime.resume_research(client, run.id)
        client.chat.completions.create.assert_not_awaited()

    async def test_old_citations_are_not_current_citations_in_context(self):
        await self.research()
        run = runtime.create_research_run('电池成本')
        context = load_context(run.id)
        self.assertNotIn('[E1]', context)
        self.assertIn('历史引用(E1)', context)
        self.assertLessEqual(len(context), MAX_CONTEXT_CHARS)

    async def test_automatic_fact_review_rejects_rewritten_quote(self):
        await self.research()
        original = next(r for r in self.store.list(scope=self.scope) if r.kind == 'semantic')
        draft = MemoryDraft(kind=original.kind, scope=original.scope, content=deepcopy(original.content),
                            sources=original.sources, expires_at=original.expires_at)
        draft.content.value = '没有来源支持的结论'
        candidate = self.store.propose(draft)
        self.assertEqual(self.store.activate(candidate.id, scope=self.scope).status, 'candidate')
        self.assertEqual(self.store.reviews(candidate.id, scope=self.scope)[-1]['decision'], 'held')

    async def test_procedure_requires_explicit_review_and_then_is_searchable(self):
        await self.research()
        record = next(r for r in self.store.list(scope=self.scope) if r.kind == 'procedural')
        query = MemoryQuery(text='电池成本', scope=self.scope, kinds=['procedural'])
        self.assertEqual(self.store.search(query), [])
        with self.assertRaises(ValueError):
            self.store.activate(record.id, scope=self.scope, reviewer='user')
        self.store.activate(record.id, scope=self.scope, reviewer='user', reason='核对过调查步骤')
        self.assertEqual(self.store.search(query)[0].id, record.id)

    def test_preference_conflict_and_atomic_revision(self):
        first = self.store.propose(self.preference())
        self.assertEqual(self.store.activate(first.id, scope=self.scope).status, 'candidate')
        self.store.activate(first.id, scope=self.scope, reviewer='user', reason='明确偏好')
        conflicting = self.store.propose(self.preference('英文'))
        self.assertEqual(self.store.activate(conflicting.id, scope=self.scope, reviewer='user', reason='修改').status, 'candidate')
        revision = self.store.revise(first.id, self.preference('英文'), scope=self.scope)
        self.store.activate(revision.id, scope=self.scope, reviewer='user', reason='明确替代旧偏好')
        self.assertEqual(self.store.get(first.id, scope=self.scope).status, 'archived')
        self.assertEqual(self.store.search(MemoryQuery(text='任意主题', scope=self.scope))[0].content.value, '英文')

    def test_activation_event_failure_rolls_back_review_and_state(self):
        record = self.store.propose(self.preference())
        self.sql("""CREATE TRIGGER reject_activation BEFORE INSERT ON memory_events
            WHEN NEW.event_type='activated' BEGIN SELECT RAISE(ABORT,'failed'); END""")
        with self.assertRaises(sqlite3.Error):
            self.store.activate(record.id, scope=self.scope, reviewer='user', reason='核对')
        self.assertEqual(self.store.get(record.id, scope=self.scope).status, 'candidate')
        self.assertEqual(self.store.reviews(record.id, scope=self.scope), [])

    async def test_search_filters_scope_archived_expired_and_unrelated(self):
        await self.research()
        self.assertTrue(self.store.search(MemoryQuery(text='电池成本', scope=self.scope)))
        self.assertEqual(self.store.search(MemoryQuery(text='电池成本', scope=MemoryScope('project', 'other'))), [])
        self.assertEqual(self.store.search(MemoryQuery(text='astronomy nebula', scope=self.scope)), [])
        for record in self.store.list(scope=self.scope):
            if record.kind == 'semantic':
                record.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
                record.content.observed_at = record.created_at
                record.expires_at = record.created_at + timedelta(days=1)
                self.sql('UPDATE memories SET record_json=? WHERE id=?', (serialize_memory(record), record.id))
            else:
                self.store.archive(record.id, scope=self.scope)
        self.assertEqual(self.store.search(MemoryQuery(text='电池成本', scope=self.scope)), [])

    def test_cli_review_is_explicit_and_non_memory_text_is_not_consumed(self):
        record = self.store.propose(self.preference())
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertTrue(handle_memory_command(f'/approve-memory {record.id} 已核对用户原话'))
            self.assertFalse(handle_memory_command('普通研究问题'))
        self.assertEqual(self.store.get(record.id, scope=self.scope).status, 'active')

    async def test_changed_scope_does_not_change_existing_snapshot(self):
        await self.research()
        run = runtime.create_research_run('电池成本')
        context = load_context(run.id)
        with patch.dict(os.environ, {'MEMORY_SCOPE_ID': 'another-project'}):
            self.assertEqual(load_context(run.id), context)
            other = runtime.create_research_run('电池成本')
            self.assertEqual(load_context(other.id), '')


if __name__ == '__main__':
    unittest.main()
