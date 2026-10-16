"""记忆契约测试：不调用模型，不读写研究数据库。"""
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import unittest

from memory import (
    EpisodicContent, SemanticContent, ProceduralContent, MemoryDraft,
    MemoryRecord, MemoryScope, MemorySource, create_memory_candidate,
    validate_memory_draft, validate_memory_record,
)


PAST = datetime(2025, 1, 1, tzinfo=timezone.utc)


class MemoryTests(unittest.TestCase):
    def draft(self, kind='episodic'):
        contents = {
            'episodic': EpisodicContent('研究成本', ['查找报告'], '获得数据', 'completed'),
            'semantic': SemanticContent('fact', '技术 A', '成本', '10 元/千克', '2024 年样本', PAST),
            'procedural': ProceduralContent('比较成本', '跨来源比较时', ['统一单位'], ['口径一致']),
        }
        return MemoryDraft(kind=kind, scope=MemoryScope('project', 'agent-learning'),
                           content=contents[kind], sources=[MemorySource('research', 'a' * 32, ['E1'])])

    def test_three_types_create_candidates_with_independent_ids(self):
        ids = set()
        for kind in ('episodic', 'semantic', 'procedural'):
            with self.subTest(kind=kind):
                draft = self.draft(kind)
                validate_memory_draft(draft)
                record = create_memory_candidate(draft)
                validate_memory_record(record)
                self.assertEqual(record.status, 'candidate')
                self.assertEqual(record.content, draft.content)
                ids.add(record.id)
        self.assertEqual(len(ids), 3)

    def test_type_and_content_must_match(self):
        for change in ({'kind': 'other'}, {'kind': []}, {'kind': 'semantic'}, {'content': {}},
                       {'scope': 'project'}, {'scope': MemoryScope('other', 'x')},
                       {'scope': MemoryScope('user', ' ')}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_memory_draft(replace(self.draft(), **change))
        with self.assertRaises(ValueError):
            validate_memory_draft({})

    def test_sources_are_required_typed_and_not_duplicated(self):
        original = self.draft().sources[0]
        cases = [[], None, [original, original], [{}],
                 [MemorySource('research', 'invalid')],
                 [MemorySource('research', 'a' * 32, ['E0'])],
                 [MemorySource('research', 'a' * 32, ['E1', 'E1'])],
                 [MemorySource('research', 'a' * 32, [1])],
                 [MemorySource('user_input', 'message-1', ['E1'])]]
        for sources in cases:
            with self.subTest(sources=sources), self.assertRaises(ValueError):
                validate_memory_draft(replace(self.draft(), sources=sources))

    def test_same_evidence_id_in_different_runs_is_not_a_duplicate(self):
        draft = self.draft()
        draft.sources.append(MemorySource('research', 'b' * 32, ['E1']))
        validate_memory_draft(draft)
        self.assertNotEqual(draft.sources[0].source_id, draft.sources[1].source_id)

    def test_episodic_requires_research_but_can_fail_before_any_action(self):
        draft = self.draft()
        draft.content = EpisodicContent('准备研究', [], '启动失败', 'failed')
        validate_memory_draft(draft)
        draft.sources = [MemorySource('user_input', 'message-1')]
        with self.assertRaises(ValueError):
            validate_memory_draft(draft)

    def test_fact_needs_timestamp_preference_does_not(self):
        draft = self.draft('semantic')
        draft.content.observed_at = None
        with self.assertRaises(ValueError):
            validate_memory_draft(draft)
        draft.content = SemanticContent('preference', '用户', '语言', '中文', '未指定时')
        draft.sources = [MemorySource('user_input', 'message-1')]
        self.assertEqual(create_memory_candidate(draft).status, 'candidate')

    def test_procedure_needs_steps_and_success_criteria(self):
        for changes in ({'steps': []}, {'steps': '一个步骤'}, {'steps': [' ']},
                        {'success_criteria': []}, {'applicability': ''}):
            draft = self.draft('procedural')
            draft.content = replace(draft.content, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_memory_draft(draft)

    def test_candidate_is_a_deep_snapshot(self):
        draft = self.draft()
        record = create_memory_candidate(draft)
        draft.scope.id = 'another-project'
        draft.content.actions.append('额外行动')
        draft.sources[0].evidence_ids.append('E2')
        self.assertEqual(record.scope.id, 'agent-learning')
        self.assertEqual(record.content.actions, ['查找报告'])
        self.assertEqual(record.sources[0].evidence_ids, ['E1'])

    def test_list_defaults_are_not_shared(self):
        first = EpisodicContent('问题', [], '失败', 'failed')
        second = EpisodicContent('问题', [], '失败', 'failed')
        first.lessons.append('可能原因')
        self.assertEqual(second.lessons, [])
        first_source = MemorySource('research', 'a' * 32)
        second_source = MemorySource('research', 'b' * 32)
        first_source.evidence_ids.append('E1')
        self.assertEqual(second_source.evidence_ids, [])

    def test_timestamp_and_revision_boundaries(self):
        record = create_memory_candidate(self.draft('semantic'))
        cases = [replace(record, id='invalid'), replace(record, status='approved'),
                 replace(record, supersedes_id=record.id),
                 replace(record, supersedes_id='invalid'),
                 replace(record, created_at=datetime(2025, 1, 1)),
                 replace(record, expires_at=record.created_at),
                 replace(record, expires_at=record.created_at - timedelta(seconds=1)),
                 replace(record, expires_at=datetime(2030, 1, 1, tzinfo=timezone(timedelta(hours=8))))]
        future_observation = deepcopy(record)
        future_observation.content.observed_at = record.created_at + timedelta(seconds=1)
        cases.append(future_observation)
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                validate_memory_record(candidate)
        with self.assertRaises(ValueError):
            validate_memory_record(self.draft())

    def test_expired_historical_record_can_still_be_read(self):
        # 读取历史与检索有效记忆不同；到期过滤属于后续检索层。
        record = create_memory_candidate(self.draft())
        record.created_at = PAST
        record.expires_at = PAST + timedelta(days=1)
        validate_memory_record(record)

    def test_validation_does_not_modify_input(self):
        draft = self.draft()
        before = deepcopy(draft)
        validate_memory_draft(draft)
        self.assertEqual(draft, before)

    def test_size_limits_reject_unbounded_model_output(self):
        for content in (EpisodicContent('x' * 4001, [], '失败', 'failed'),
                        EpisodicContent('问题', ['行动'] * 51, '失败', 'failed')):
            with self.subTest(content=content), self.assertRaises(ValueError):
                validate_memory_draft(replace(self.draft(), content=content))


if __name__ == '__main__':
    unittest.main()
