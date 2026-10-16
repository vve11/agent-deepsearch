"""从完整研究的持久化记录提取记忆；确定性、可重放，不新增模型调用。"""
from datetime import timedelta
import json

from evidence import EvidenceStore
from memory import (MemoryDraft, MemorySource, EpisodicContent,
                    SemanticContent, ProceduralContent)
from plan import restore_plan


FACT_CONDITIONS = '来源原文陈述，仅供历史参考；使用前核实时效与适用条件'


def extract_drafts(connection, run_id, scope, observed_at):
    row = connection.execute('SELECT question,status,answer,evidence_json FROM runs WHERE id=?', (run_id,)).fetchone()
    if row is None or row[1] != 'completed' or not isinstance(row[2], str) or not row[2].strip():
        raise ValueError('仅在完整研究成功保存后提取记忆')
    evidence = EvidenceStore.from_records(json.loads(row[3])).all()
    plan_row = connection.execute('SELECT plan_json FROM research_plans WHERE run_id=?', (run_id,)).fetchone()
    actions = []
    if plan_row:
        plan = restore_plan(json.loads(plan_row[0]))
        if plan.question != row[0]:
            raise ValueError('记忆提取计划与研究问题不一致')
        for task in plan.tasks:
            state = connection.execute('SELECT status FROM task_executions WHERE run_id=? AND task_id=?',
                                       (run_id, task.id)).fetchone()
            if state != ('completed',):
                raise ValueError('研究仍有未完成子任务，不能提取完成经验')
            actions.append('完成子问题：' + task.question)
    sources = [MemorySource('research', run_id, [e['id'] for e in evidence[:50]])]
    drafts = [MemoryDraft(
        kind='episodic', scope=scope, sources=sources,
        content=EpisodicContent(row[0][:4000], actions, row[2][:1800], 'completed'),
        expires_at=observed_at + timedelta(days=180),
    )]
    # 只摘录真实来源文字，不把最终模型答案自动提升为事实。
    for item in evidence[:3]:
        quote = item['content'].strip()[:700]
        if not quote:
            continue
        drafts.append(MemoryDraft(
            kind='semantic', scope=scope,
            sources=[MemorySource('research', run_id, [item['id']])],
            content=SemanticContent('fact', item['title'].strip()[:500] or item['url'][:500],
                                    '来源原文摘录', quote, FACT_CONDITIONS, observed_at),
            expires_at=observed_at + timedelta(days=30),
        ))
    if actions:
        drafts.append(MemoryDraft(
            kind='procedural', scope=scope, sources=sources,
            content=ProceduralContent(
                '调查：' + row[0][:1000], '仅适用于相似研究问题，需人工检查步骤是否可复用',
                actions, ['逐个子问题产生可追溯答案', '汇总时核对来源与信息缺口']),
            expires_at=observed_at + timedelta(days=180),
        ))
    return drafts
