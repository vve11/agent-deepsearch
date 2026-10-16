"""启用真实门禁的离线端到端测试；模型与搜索模拟，SQLite 使用临时文件。"""
import asyncio
import json
import os
import sqlite3
import tempfile
import subprocess
import sys
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch, AsyncMock

import agent
import database as db
import runtime
import pipeline_storage
import quality_storage
from budget import BudgetExceeded
from quality_gate import QualityRejected, deterministic_checks, parse_review
from test_checkpoints import client_with, response
from test_model_requests import status_error

PLAN = json.dumps({'tasks':[{'question':'调查电池成本'}]})


def verdict(action='none', ids=('E1',)):
    return response(json.dumps({'checks':[
        {'name':name,'passed':action=='none','reason':'资料支持结论' if action=='none' else '缺少成本依据',
         'evidence_ids':list(ids)} for name in ('task_coverage','claim_support','report_consistency')],
        'action':action,'query':'补充成本变化依据' if action=='research' else ''}))


async def search(call):
    query = json.loads(call.function.arguments)['query']
    return {'result':[{'title':'成本资料','url':'https://example.com/'+query,'content':'材料价格影响成本。'}]}


class QualityGateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        for p in (patch.object(db,'DB_PATH',Path(folder.name)/'test.db'),
                  patch.dict(os.environ, {'QUALITY_GATE_ENABLED':'true','MEMORY_ENABLED':'true'}),
                  patch.object(agent,'execute_tool',side_effect=search)):
            p.start()
            self.addCleanup(p.stop)
        db.init_db()

    def sql(self, command, values=()):
        with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
            return conn.execute(command, values).fetchall()

    def base(self, *tail, report='成本报告 [E1]'):
        return client_with(response(PLAN),response(queries=['base']),response('材料影响成本 [E1]'),response(report),*tail)

    def run_id(self):
        return self.sql('SELECT id FROM runs')[0][0]

    def state(self):
        return quality_storage.load(self.run_id())[1]

    async def test_passed_gate_is_required_before_completion_and_memory(self):
        client = self.base(verdict())
        run = await runtime.execute_research(client,'分析电池成本')
        self.assertEqual(run.status,'completed')
        self.assertEqual(self.state()['phase'],'passed')
        self.assertEqual(pipeline_storage.load_pipeline(run.id)['model_used'],5)
        self.assertEqual(self.sql('SELECT status FROM memory_jobs'),[('done',)])

    async def test_missing_evidence_triggers_supplement_then_rewrite(self):
        client = client_with(response(PLAN),response('尚无证据'),response('无法确定'),
            response(queries=['extra']),response('补充结果 [E1]'),response('修订报告 [E1]'),verdict())
        run = await runtime.execute_research(client,'分析电池成本')
        self.assertEqual(self.state()['used'],1)
        self.assertEqual(len(run.evidence),1)
        self.assertIn('修订报告',run.answer)
        self.assertEqual(db.load_research_evidence(run.id),[])  # 原子任务证据不被补救改写。

    async def test_semantic_gap_adds_evidence_without_changing_old_ids(self):
        client = self.base(verdict('research'),response(queries=['extra']),response('补充 [E2]'),
                           response('修订 [E1][E2]'),verdict(ids=('E1','E2')))
        run = await runtime.execute_research(client,'分析电池成本')
        self.assertEqual([e['id'] for e in run.evidence],['E1','E2'])
        self.assertEqual(run.evidence[0],db.load_research_evidence(run.id)[0])
        self.assertEqual(pipeline_storage.load_pipeline(run.id)['tool_used'],2)

    async def test_report_only_repair_does_not_search(self):
        client = self.base(verdict('repair'),response('修订 [E1]'),verdict())
        run = await runtime.execute_research(client,'分析电池成本')
        self.assertEqual(pipeline_storage.load_pipeline(run.id)['tool_used'],1)
        self.assertEqual(self.state()['used'],1)

    async def test_source_list_does_not_count_as_body_citation(self):
        client = self.base(response('修订 [E1]'),verdict(),report='没有正文引用')
        await runtime.execute_research(client,'分析电池成本')
        self.assertEqual(self.state()['used'],1)

    async def test_two_failed_remediations_stop_and_do_not_extract_memory(self):
        client = self.base(verdict('repair'),response('修订一 [E1]'),verdict('repair'),
                           response('修订二 [E1]'),verdict('repair'))
        with self.assertRaises(QualityRejected):
            await runtime.execute_research(client,'分析电池成本')
        self.assertEqual(self.state()['used'],2)
        self.assertEqual(self.state()['phase'],'exhausted')
        self.assertIn('修订二',self.state()['report'])
        self.assertEqual(self.sql('SELECT * FROM memory_jobs'),[])
        other = client_with()
        with self.assertRaises(ValueError):
            await runtime.resume_research(other,self.run_id())
        other.chat.completions.create.assert_not_called()

    async def test_cancelled_supplement_resumes_saved_tool_round(self):
        client = self.base(verdict('research'),response(queries=['extra']),asyncio.CancelledError())
        with self.assertRaises(asyncio.CancelledError):
            await runtime.execute_research(client,'分析电池成本')
        self.assertEqual(self.state()['phase'],'research')
        self.assertEqual(self.state()['checkpoint']['next_step'],1)
        used = pipeline_storage.load_pipeline(self.run_id())['model_used']
        resumed = client_with(response('补充 [E2]'),response('修订 [E1][E2]'),verdict(ids=('E1','E2')))
        with patch.object(agent,'execute_tool',new=AsyncMock(side_effect=AssertionError('不能重做搜索'))):
            await runtime.resume_research(resumed,self.run_id())
        self.assertEqual(self.state()['used'],1)
        self.assertEqual(pipeline_storage.load_pipeline(self.run_id())['model_used'],used+3)

    async def test_rewrite_failure_does_not_repeat_supplement(self):
        client = self.base(verdict('research'),response(queries=['extra']),response('补充 [E2]'),ValueError('offline'))
        with self.assertRaises(ValueError):
            await runtime.execute_research(client,'分析电池成本')
        self.assertEqual(self.state()['phase'],'rewrite')
        with patch.object(agent,'execute_tool',new=AsyncMock(side_effect=AssertionError('不能重做搜索'))):
            await runtime.resume_research(client_with(response('修订 [E2]'),verdict(ids=('E2',))),self.run_id())
        self.assertEqual(self.state()['used'],1)

    async def test_final_transaction_failure_reuses_passed_verdict_without_budget(self):
        self.sql("CREATE TRIGGER reject_finish BEFORE INSERT ON run_events WHEN NEW.event_type='run_completed' BEGIN SELECT RAISE(ABORT,'fail'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            await runtime.execute_research(self.base(verdict()),'分析电池成本')
        self.assertEqual(self.state()['phase'],'passed')
        self.sql('DROP TRIGGER reject_finish')
        self.sql('UPDATE research_pipeline SET model_limit=model_used')
        client = client_with()
        run = await runtime.resume_research(client,self.run_id())
        self.assertEqual(run.status,'completed')
        client.chat.completions.create.assert_not_called()

    async def test_model_budget_exhaustion_in_review_cannot_be_reset_by_resume(self):
        run = runtime.create_research_run('分析电池成本')
        self.sql('UPDATE research_pipeline SET model_limit=4')
        client = self.base(verdict())
        with self.assertRaises(BudgetExceeded):
            await runtime.execute_created_research(client,run)
        self.assertEqual(client.chat.completions.create.await_count,4)
        self.assertEqual(self.state()['phase'],'exhausted')
        with self.assertRaises(ValueError):
            await runtime.resume_research(client,run.id)

    async def test_tool_budget_exhaustion_stops_supplement_before_request(self):
        run = runtime.create_research_run('分析电池成本')
        self.sql('UPDATE research_pipeline SET tool_limit=1')
        client = self.base(verdict('research'))
        with self.assertRaises(BudgetExceeded):
            await runtime.execute_created_research(client,run)
        self.assertEqual(client.chat.completions.create.await_count,5)
        self.assertEqual(self.state()['used'],1)

    async def test_bad_review_is_not_a_pass_and_can_resume(self):
        with self.assertRaises(ValueError):
            await runtime.execute_research(self.base(response('not json')),'分析电池成本')
        self.assertEqual(self.state()['phase'],'checking')
        await runtime.resume_research(client_with(verdict()),self.run_id())
        self.assertEqual(self.state()['used'],0)

    async def test_missing_quality_state_blocks_resume(self):
        with self.assertRaises(ValueError):
            await runtime.execute_research(self.base(ValueError('offline')),'分析电池成本')
        self.sql('DELETE FROM research_quality')
        client = client_with()
        with self.assertRaises(ValueError):
            await runtime.resume_research(client,self.run_id())
        client.chat.completions.create.assert_not_called()

    async def test_remediation_transition_rollback_does_not_start_action(self):
        self.sql("CREATE TRIGGER reject_action BEFORE INSERT ON run_events WHEN NEW.event_type='quality_transition' AND json_extract(NEW.payload_json,'$.phase')='research' BEGIN SELECT RAISE(ABORT,'fail'); END")
        client = self.base(verdict('research'))
        with self.assertRaises(sqlite3.IntegrityError):
            await runtime.execute_research(client,'分析电池成本')
        self.assertEqual(self.state()['used'],0)
        self.assertEqual(self.state()['phase'],'checking')
        self.assertEqual(client.chat.completions.create.await_count,5)

    def test_review_cannot_reference_unknown_evidence(self):
        with self.assertRaises(ValueError):
            parse_review(verdict().choices[0].message.content,[])

    def test_deterministic_checks_reject_fake_citations(self):
        checks = deterministic_checks('报告 [E0]',[{'id':'E1','title':'x','url':'https://example.com','content':'data'}])
        self.assertFalse(checks[1]['passed'])

    async def test_model_retry_in_review_uses_shared_persistent_budget(self):
        run = runtime.create_research_run('分析电池成本')
        self.sql('UPDATE research_pipeline SET model_limit=5')
        client = self.base(status_error(503),verdict())
        with patch('model_requests.asyncio.sleep',new=AsyncMock()):
            with self.assertRaises(BudgetExceeded):
                await runtime.execute_created_research(client,run)
        self.assertEqual(client.chat.completions.create.await_count,5)
        self.assertEqual(pipeline_storage.load_pipeline(run.id)['model_used'],5)
        self.assertEqual(self.state()['phase'],'exhausted')

    async def test_modified_quality_snapshot_is_rejected(self):
        with self.assertRaises(ValueError):
            await runtime.execute_research(self.base(ValueError('offline')),'分析电池成本')
        self.sql("UPDATE research_quality SET state_json=json_set(state_json,'$.report','被改写的报告 [E1]')")
        with self.assertRaises(ValueError):
            await runtime.resume_research(client_with(),self.run_id())

    async def test_finish_cannot_bypass_pending_quality(self):
        with self.assertRaises(ValueError):
            await runtime.execute_research(self.base(ValueError('offline')),'分析电池成本')
        with self.assertRaises(ValueError):
            pipeline_storage.finish_run(db.load_run(self.run_id()))
        self.assertEqual(db.load_run(self.run_id()).status,'failed')

    async def test_new_process_resumes_saved_supplement_without_search(self):
        with self.assertRaises(asyncio.CancelledError):
            await runtime.execute_research(self.base(verdict('research'),response(queries=['extra']),asyncio.CancelledError()),'分析电池成本')
        code = '''import sys,asyncio
from pathlib import Path
from unittest.mock import patch,AsyncMock
sys.path.insert(0, 'tests')
import database as db, runtime, agent, quality_storage
from test_checkpoints import client_with,response
from test_quality_gate import verdict
db.DB_PATH=Path(sys.argv[1])
async def main():
    client=client_with(response('补充 [E2]'),response('修订 [E2]'),verdict(ids=('E2',)))
    with patch.object(agent,'execute_tool',new=AsyncMock(side_effect=AssertionError('must not search'))):
        run=await runtime.resume_research(client,sys.argv[2])
    assert run.status=='completed'
    assert quality_storage.load(run.id)[1]['used']==1
asyncio.run(main())
'''
        result = subprocess.run([sys.executable,'-c',code,str(db.DB_PATH),self.run_id()],
                                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(self.state()['phase'],'passed')
