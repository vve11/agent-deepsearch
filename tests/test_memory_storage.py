"""临时 SQLite 验证记忆往返、提交幂等、范围隔离和真实事务回滚。"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import database
from memory import (MemoryScope, MemorySource, MemoryDraft, EpisodicContent,
                    SemanticContent, ProceduralContent, create_memory_candidate)
from memory_codec import serialize_memory, deserialize_memory
from memory_storage import MemoryStore
from run import ResearchRun


class MemoryStorageTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / 'memory.db'
        patcher = patch.object(database, 'DB_PATH', self.path)
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_db()
        self.store = MemoryStore()
        self.scope = MemoryScope('project', 'test')
        self.run = ResearchRun(question='研究成本', status='completed', answer='结果 [E1]',
                               evidence=[{'id': 'E1', 'title': '来源', 'url': 'https://example.com', 'content': '资料'}])
        database.save_run(self.run)

    def draft(self, kind='episodic'):
        return MemoryDraft(kind=kind, scope=deepcopy(self.scope),
            sources=[MemorySource('research', self.run.id, ['E1'])], content={
                'episodic': EpisodicContent('成本研究', ['查找报告'], '完成比较', 'completed'),
                'semantic': SemanticContent('fact', '样本', '成本', '10 元', '该样本',
                                            datetime(2025, 1, 1, tzinfo=timezone.utc)),
                'procedural': ProceduralContent('比较成本', '跨来源', ['统一单位'], ['口径一致']),
            }[kind])

    def sql(self, statement, args=()):
        with closing(sqlite3.connect(self.path)) as conn, conn:
            return conn.execute(statement, args).fetchall()

    def test_all_types_roundtrip_after_reopening_store(self):
        for kind in ('episodic', 'semantic', 'procedural'):
            with self.subTest(kind=kind):
                draft = self.draft(kind)
                draft.expires_at = datetime.now(timezone.utc) + timedelta(days=30)
                saved = self.store.propose(draft)
                loaded = MemoryStore(self.path).get(saved.id, scope=self.scope)
                self.assertEqual(saved, loaded)
                self.assertEqual(type(loaded.content), type(draft.content))
                self.assertEqual(loaded.status, 'candidate')
                self.assertEqual(loaded.created_at.utcoffset(), timedelta(0))
        self.assertEqual(len(self.sql('SELECT * FROM memory_events')), 3)

    def test_exact_duplicate_returns_original_without_extra_event(self):
        first = self.store.propose(self.draft())
        second = self.store.propose(self.draft())
        self.assertEqual(first, second)
        self.assertEqual(len(self.sql('SELECT * FROM memories')), 1)
        self.assertEqual(len(self.sql('SELECT * FROM memory_events')), 1)

    def test_explicit_key_conflict_does_not_overwrite(self):
        first = self.store.propose(self.draft(), request_key='run-summary')
        self.assertEqual(self.store.propose(self.draft(), request_key='run-summary'), first)
        draft = self.draft()
        draft.content.outcome = '不同结果'
        with self.assertRaisesRegex(ValueError, '不同记忆'):
            self.store.propose(draft, request_key='run-summary')
        self.assertEqual(self.store.get(first.id, scope=self.scope), first)

    def test_concurrent_duplicate_submissions_create_one_record(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: self.store.propose(self.draft(), request_key='same'), range(2)))
        self.assertEqual(results[0], results[1])
        self.assertEqual(len(self.sql('SELECT * FROM memories')), 1)

    def test_scope_filter_and_missing_id_have_same_error(self):
        saved = self.store.propose(self.draft())
        errors = []
        for memory_id, scope in ((saved.id, MemoryScope('project', 'other')),
                                 ('f' * 32, self.scope), (saved.id, MemoryScope('user', 'test'))):
            with self.assertRaises(LookupError) as caught:
                self.store.get(memory_id, scope=scope)
            errors.append(str(caught.exception))
        self.assertEqual(len(set(errors)), 1)

    def test_request_keys_are_scoped(self):
        first = self.store.propose(self.draft(), request_key='same')
        draft = self.draft()
        draft.scope.id = 'other'
        second = self.store.propose(draft, request_key='same')
        self.assertNotEqual(first.id, second.id)

    def test_bad_sources_are_rejected_without_partial_writes(self):
        for source in (MemorySource('research', 'f' * 32), MemorySource('research', self.run.id, ['E2'])):
            draft = self.draft()
            draft.sources = [source]
            with self.subTest(source=source), self.assertRaises(ValueError):
                self.store.propose(draft)
        self.sql("UPDATE runs SET status='running' WHERE id=?", (self.run.id,))
        with self.assertRaises(ValueError):
            self.store.propose(self.draft())
        self.assertEqual(self.sql('SELECT * FROM memories'), [])

    def test_nonresearch_sources_remain_unverified_candidates(self):
        draft = self.draft('semantic')
        draft.content = SemanticContent('preference', '用户', '语言', '中文', '未指定时')
        draft.sources = [MemorySource('user_input', 'message-1')]
        saved = self.store.propose(draft)
        self.assertEqual(saved.status, 'candidate')
        self.assertEqual(self.store.get(saved.id, scope=self.scope), saved)

    def test_event_failure_rolls_back_every_write(self):
        self.sql("""CREATE TRIGGER reject_memory BEFORE INSERT ON memory_events
                    BEGIN SELECT RAISE(ABORT, 'failed'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.propose(self.draft(), request_key='retry')
        for table in ('memories', 'memory_proposals', 'memory_research_sources', 'memory_events'):
            self.assertEqual(self.sql(f'SELECT * FROM {table}'), [])
        self.sql('DROP TRIGGER reject_memory')
        self.store.propose(self.draft(), request_key='retry')
        self.assertEqual(len(self.sql('SELECT * FROM memories')), 1)

    def test_returned_objects_do_not_mutate_persisted_record(self):
        draft = self.draft()
        saved = self.store.propose(draft)
        saved.content.actions.append('修改返回值')
        draft.content.actions.append('修改草稿')
        loaded = self.store.get(saved.id, scope=self.scope)
        self.assertEqual(loaded.content.actions, ['查找报告'])

    def test_corrupted_json_and_index_are_rejected(self):
        saved = self.store.propose(self.draft())
        payload = serialize_memory(saved)
        for bad in ('{', payload.replace(saved.id, 'f' * 32)):
            self.sql('UPDATE memories SET record_json=? WHERE id=?', (bad, saved.id))
            with self.assertRaises(ValueError):
                self.store.get(saved.id, scope=self.scope)
        self.sql('UPDATE memories SET record_json=?, status=? WHERE id=?', (payload, 'active', saved.id))
        with self.assertRaises(ValueError):
            self.store.get(saved.id, scope=self.scope)

    def test_deleted_source_link_is_rejected(self):
        saved = self.store.propose(self.draft())
        self.sql('DELETE FROM memory_research_sources WHERE memory_id=?', (saved.id,))
        with self.assertRaisesRegex(ValueError, '关联损坏'):
            self.store.get(saved.id, scope=self.scope)

    def test_serialization_rejects_missing_extra_version_and_invalid_time(self):
        raw = json.loads(serialize_memory(create_memory_candidate(self.draft('semantic'))))
        cases = []
        for version in (2, True):
            changed = deepcopy(raw)
            changed['version'] = version
            cases.append(changed)
        changed = deepcopy(raw)
        del changed['record']['sources'][0]['evidence_ids']
        cases.append(changed)
        changed = deepcopy(raw)
        changed['record']['content']['unexpected'] = 1
        cases.append(changed)
        for bad_time in ('yesterday', '2025-01-01T00:00:00', 123):
            changed = deepcopy(raw)
            changed['record']['created_at'] = bad_time
            cases.append(changed)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                deserialize_memory(json.dumps(data))
        with self.assertRaises(ValueError):
            deserialize_memory('{"version":1,"version":1,"record":{}}')

    def test_revision_reference_requires_same_scope_and_type(self):
        original = self.store.propose(self.draft())
        draft = self.draft()
        draft.supersedes_id = original.id
        draft.content.outcome = '新结果'
        revised = self.store.propose(draft)
        self.assertEqual(revised.supersedes_id, original.id)
        self.assertEqual(self.store.get(original.id, scope=self.scope).status, 'candidate')
        draft.scope.id = 'other'
        with self.assertRaises(LookupError):
            self.store.propose(draft)
        draft = self.draft('semantic')
        draft.supersedes_id = original.id
        with self.assertRaises(ValueError):
            self.store.propose(draft)

    def test_reinitialization_preserves_memories(self):
        saved = self.store.propose(self.draft())
        database.init_db()
        self.assertEqual(self.store.get(saved.id, scope=self.scope), saved)

    def test_retry_after_expiry_returns_existing_historical_record(self):
        draft = self.draft()
        draft.expires_at = datetime(2025, 2, 1, tzinfo=timezone.utc)
        candidate = create_memory_candidate(self.draft())
        candidate.created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        candidate.expires_at = draft.expires_at
        with patch('memory_storage.create_memory_candidate', return_value=candidate):
            first = self.store.propose(draft)
        self.assertEqual(self.store.propose(draft), first)


if __name__ == '__main__':
    unittest.main()
