"""来源对齐审核不等于外部事实真伪验证；方法/偏好需人工明确审核。"""
from datetime import datetime, timedelta, timezone
from dataclasses import asdict
import json

from memory import EpisodicContent, SemanticContent
from memory_extraction import extract_drafts, FACT_CONDITIONS


def auto_review(connection, record):
    if isinstance(record.content, EpisodicContent):
        if len(record.sources) != 1 or record.sources[0].kind != 'research':
            return False, '情景记忆需要唯一的原研究来源'
        try:
            expected = extract_drafts(connection, record.sources[0].source_id, record.scope, record.created_at)[0]
        except ValueError:
            return False, '研究执行记录不支持这条情景记忆'
        if record.content == expected.content and record.sources == expected.sources:
            return True, '情景内容与已保存研究、已完成子任务逐项一致；结论仍为历史结果'
        return False, '情景内容与实际执行记录不一致'
    if isinstance(record.content, SemanticContent) and record.content.category == 'fact':
        content = record.content
        if (content.predicate != '来源原文摘录' or content.conditions != FACT_CONDITIONS
                or not record.expires_at or record.expires_at > content.observed_at + timedelta(days=30)):
            return False, '自动审核仅接受带时效限制的原文摘录'
        if len(record.sources) != 1 or record.sources[0].kind != 'research' or len(record.sources[0].evidence_ids) != 1:
            return False, '原文摘录需要唯一的研究证据来源'
        source = record.sources[0]
        row = connection.execute('SELECT evidence_json FROM runs WHERE id=?', (source.source_id,)).fetchone()
        evidence = next((e for e in json.loads(row[0]) if e['id'] == source.evidence_ids[0]), None)
        if evidence and content.value == evidence['content'].strip()[:700] and content.subject == (evidence['title'].strip()[:500] or evidence['url'][:500]):
            return True, '摘录与持久化来源一致；未验证来源真实性，使用时必须重新核实'
        return False, '摘录不符合来源原文'
    return False, '偏好和程序性方法需要人工核对来源及适用条件后明确批准'


def check_conflicts(connection, record):
    from memory_storage import _read
    rows = connection.execute(
        "SELECT id FROM memories WHERE scope_kind=? AND scope_id=? AND status='active'",
        (record.scope.kind, record.scope.id)).fetchall()
    for row in rows:
        other = _read(connection, row[0], record.scope)
        if other.id in (record.id, record.supersedes_id):
            continue
        if record.supersedes_id and other.supersedes_id == record.supersedes_id:
            return '旧版本已有生效修订，请基于当前版本修订'
        if other.expires_at and other.expires_at <= datetime.now(timezone.utc):
            continue
        if isinstance(record.content, SemanticContent) and isinstance(other.content, SemanticContent):
            a, b = asdict(record.content), asdict(other.content)
            keys = ('category', 'subject', 'predicate', 'conditions')
            if all(a[key].strip().casefold() == b[key].strip().casefold() for key in keys) and a['value'] != b['value']:
                return '同一主体、属性及条件存在不同值，请核对后使用 revise 显式替代'
    return None
