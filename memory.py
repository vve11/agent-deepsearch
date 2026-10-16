"""长期记忆的数据契约。只校验结构，不访问数据库、不判断事实真假。"""
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Literal
from uuid import uuid4


MemoryKind = Literal['episodic', 'semantic', 'procedural']


@dataclass
class MemoryScope:
    kind: Literal['user', 'project']
    id: str


@dataclass
class MemorySource:
    """research 的 source_id 是原 run_id；E1 必须与它一起解释。"""
    kind: Literal['research', 'user_input', 'procedure']
    source_id: str
    evidence_ids: list[str] = field(default_factory=list)


@dataclass
class EpisodicContent:
    situation: str
    actions: list[str]
    outcome: str
    outcome_status: Literal['completed', 'failed']
    # lessons 是待核实的经验推断，不等于实际发生的 actions/outcome。
    lessons: list[str] = field(default_factory=list)


@dataclass
class SemanticContent:
    category: Literal['fact', 'preference']
    subject: str
    predicate: str
    value: str
    conditions: str
    # 事实必须标明观察/获取时间；不表示它在此后永远成立。
    observed_at: datetime | None = None


@dataclass
class ProceduralContent:
    goal: str
    applicability: str
    steps: list[str]
    success_criteria: list[str]


MemoryContent = EpisodicContent | SemanticContent | ProceduralContent


@dataclass(kw_only=True)
class MemoryQuery:
    text: str
    scope: MemoryScope
    kinds: list[MemoryKind] = field(default_factory=lambda: ['episodic', 'semantic', 'procedural'])
    limit: int = 5


@dataclass(kw_only=True)
class MemoryDraft:
    kind: MemoryKind
    scope: MemoryScope
    content: MemoryContent
    sources: list[MemorySource]
    expires_at: datetime | None = None
    supersedes_id: str | None = None


@dataclass(kw_only=True)
class MemoryRecord(MemoryDraft):
    id: str
    created_at: datetime
    status: Literal['candidate', 'active', 'archived'] = 'candidate'


def _text(value, name, maximum=4000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f'{name} 必须是非空字符串，且不超过 {maximum} 字符')


def _texts(values, name, minimum=0):
    if not isinstance(values, list) or not minimum <= len(values) <= 50:
        raise ValueError(f'{name} 必须是包含 {minimum}～50 项的列表')
    for value in values:
        _text(value, name)


def _utc(value, name):
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f'{name} 必须是带时区的 datetime')
    if value.utcoffset().total_seconds() != 0:
        raise ValueError(f'{name} 必须使用 UTC 时间')


def _id(value, name):
    if not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{32}', value) is None:
        raise ValueError(f'{name} 必须是 32 位小写十六进制编号')


def validate_memory_draft(draft: MemoryDraft) -> None:
    """检查字段及相互关系；来源存在性、权限、事实核验由存储/审核层处理。"""
    if not isinstance(draft, MemoryDraft):
        raise ValueError('必须传入 MemoryDraft')
    content_types = {'episodic': EpisodicContent, 'semantic': SemanticContent,
                     'procedural': ProceduralContent}
    if not isinstance(draft.kind, str) or draft.kind not in content_types:
        raise ValueError('记忆类型不合法')
    if type(draft.content) is not content_types[draft.kind]:
        raise ValueError('记忆类型与 content 结构不一致')
    if not isinstance(draft.scope, MemoryScope) or draft.scope.kind not in ('user', 'project'):
        raise ValueError('记忆范围不合法')
    _text(draft.scope.id, '范围编号', 200)
    if not isinstance(draft.sources, list) or not 1 <= len(draft.sources) <= 50:
        raise ValueError('记忆必须提供 1～50 条来源')
    seen = set()
    for source in draft.sources:
        if not isinstance(source, MemorySource) or source.kind not in ('research', 'user_input', 'procedure'):
            raise ValueError('来源类型不合法')
        _text(source.source_id, '来源编号', 200)
        _texts(source.evidence_ids, '证据编号')
        if any(re.fullmatch(r'E[1-9][0-9]*', item) is None for item in source.evidence_ids):
            raise ValueError('证据编号必须是 E1、E2 这样的格式')
        if len(source.evidence_ids) != len(set(source.evidence_ids)):
            raise ValueError('来源中的证据编号不能重复')
        if source.kind == 'research':
            _id(source.source_id, '来源研究编号')
        elif source.evidence_ids:
            raise ValueError('只有研究来源可以携带证据编号')
        key = (source.kind, source.source_id)
        if key in seen:
            raise ValueError('同一来源应合并，不能重复')
        seen.add(key)
    if draft.expires_at is not None:
        _utc(draft.expires_at, '失效时间')
    if draft.supersedes_id is not None:
        _id(draft.supersedes_id, '被替代记忆编号')

    content = draft.content
    if isinstance(content, EpisodicContent):
        _text(content.situation, '情景')
        _texts(content.actions, '行动')  # 首次行动前失败，允许没有行动。
        _text(content.outcome, '结果')
        _texts(content.lessons, '经验')
        if content.outcome_status not in ('completed', 'failed'):
            raise ValueError('情景结果状态不合法')
        if not any(source.kind == 'research' for source in draft.sources):
            raise ValueError('情景记忆必须关联原始研究')
    elif isinstance(content, SemanticContent):
        if content.category not in ('fact', 'preference'):
            raise ValueError('语义记忆类别不合法')
        for name in ('subject', 'predicate', 'value', 'conditions'):
            _text(getattr(content, name), name)
        if content.category == 'fact' and content.observed_at is None:
            raise ValueError('事实必须带有观察时间')
        if content.observed_at is not None:
            _utc(content.observed_at, '观察时间')
    else:
        _text(content.goal, '方法目标')
        _text(content.applicability, '适用条件')
        _texts(content.steps, '操作步骤', minimum=1)
        _texts(content.success_criteria, '完成标准', minimum=1)


def validate_memory_record(record: MemoryRecord) -> None:
    if not isinstance(record, MemoryRecord):
        raise ValueError('必须传入 MemoryRecord')
    validate_memory_draft(record)
    _id(record.id, '记忆编号')
    _utc(record.created_at, '创建时间')
    if record.status not in ('candidate', 'active', 'archived'):
        raise ValueError('记忆状态不合法')
    if record.supersedes_id == record.id:
        raise ValueError('记忆不能替代自己')
    if record.expires_at is not None and record.expires_at <= record.created_at:
        raise ValueError('失效时间必须晚于创建时间')
    if isinstance(record.content, SemanticContent) and record.content.observed_at is not None:
        if record.content.observed_at > record.created_at:
            raise ValueError('观察时间不能晚于创建时间')


def create_memory_candidate(draft: MemoryDraft) -> MemoryRecord:
    """生成独立候选快照；只创建内存对象，不代表已保存或已通过审核。"""
    validate_memory_draft(draft)
    record = MemoryRecord(
        kind=draft.kind, scope=deepcopy(draft.scope), content=deepcopy(draft.content),
        sources=deepcopy(draft.sources), expires_at=draft.expires_at,
        supersedes_id=draft.supersedes_id, id=uuid4().hex,
        created_at=datetime.now(timezone.utc), status='candidate',
    )
    validate_memory_record(record)
    return record
