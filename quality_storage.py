"""质量阶段独立日志：原子保存状态与事件，兼容升级前的研究。"""
import json
import hashlib
import os
import sqlite3
from contextlib import closing
from copy import deepcopy

import database as db
from checkpoint import AgentCheckpoint
from evidence import EvidenceStore

SCHEMA = '''CREATE TABLE IF NOT EXISTS research_quality (
    run_id TEXT PRIMARY KEY REFERENCES runs(id),
    revision INTEGER NOT NULL, state_json TEXT NOT NULL
)'''


def digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def initialize(conn, run_id):
    flag = os.getenv('QUALITY_GATE_ENABLED', 'true').lower()
    if flag not in {'true', 'false', '1', '0'}:
        raise ValueError('QUALITY_GATE_ENABLED 必须是 true 或 false')
    state = dict(version=1, phase='pending' if flag in {'true', '1'} else 'skipped',
                 used=0, limit=2, report=None, evidence=[], checkpoint=None,
                 query=None, supplement=None, checks=[], reason=None)
    conn.execute('INSERT INTO research_quality VALUES (?,0,?)', (run_id, json.dumps(state)))
    db._append_event(conn, run_id, 'quality_initialized', {'enabled': state['phase'] != 'skipped', 'digest': digest(state)})


def validate(state, run_id):
    keys = {'version','phase','used','limit','report','evidence','checkpoint','query','supplement','checks','reason'}
    if not isinstance(state, dict) or set(state) != keys or state['version'] != 1:
        raise ValueError('质量状态格式损坏')
    if state['phase'] not in {'pending','skipped','checking','research','rewrite','passed','exhausted'}:
        raise ValueError('质量阶段无效')
    if type(state['used']) is not int or state['limit'] != 2 or not 0 <= state['used'] <= 2:
        raise ValueError('补救次数无效')
    EvidenceStore.from_records(state['evidence'])
    if state['phase'] not in {'pending','skipped'} and (not isinstance(state['report'], str) or not state['report'].strip()):
        raise ValueError('质量阶段缺少报告')
    if not isinstance(state['checks'], list) or any(not isinstance(c, dict) for c in state['checks']):
        raise ValueError('质量检查记录损坏')
    required = {'minimum_evidence','citation_integrity','task_coverage','claim_support','report_consistency'}
    if state['phase'] == 'passed' and (len(state['checks']) != 5 or
            {c.get('name') for c in state['checks']} != required or any(c.get('passed') is not True for c in state['checks'])):
        raise ValueError('质量通过状态缺少有效检查结果')
    if state['phase'] in {'research','rewrite'} and (state['used'] == 0 or not isinstance(state['query'], str)):
        raise ValueError('补救阶段缺少已登记指令')
    if state['checkpoint'] is not None:
        cp = AgentCheckpoint.from_dict(state['checkpoint'])
        if cp.run_id != run_id or cp.evidence != state['evidence'] or cp.messages[1]['content'] != state['query']:
            raise ValueError('补充研究检查点不一致')


def read(conn, run_id):
    row = conn.execute('SELECT revision,state_json FROM research_quality WHERE run_id=?', (run_id,)).fetchone()
    if row is None:
        if conn.execute("SELECT 1 FROM run_events WHERE run_id=? AND event_type='quality_initialized'", (run_id,)).fetchone():
            raise ValueError('质量状态丢失，不能跳过门禁')
        return None  # 升级前任务不改变恢复协议。
    state = json.loads(row[1])
    validate(state, run_id)
    event = conn.execute("SELECT payload_json FROM run_events WHERE run_id=? AND event_type IN ('quality_transition','quality_initialized') ORDER BY id DESC LIMIT 1", (run_id,)).fetchone()
    audit = json.loads(event[0]) if event else {}
    if audit.get('revision', 0) != row[0] or audit.get('digest') != digest(state):
        raise ValueError('质量状态与事务事件不一致')
    return row[0], state


def load(run_id):
    with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
        conn.execute('BEGIN')
        return read(conn, run_id)


def save(run_id, revision, state, **changes):
    next_state = deepcopy(state)
    next_state.update(changes)
    validate(next_state, run_id)
    with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
        conn.execute('BEGIN IMMEDIATE')
        old = read(conn, run_id)
        if old != (revision, state):
            raise ValueError('质量状态已改变，拒绝覆盖')
        conn.execute('UPDATE research_quality SET revision=?,state_json=? WHERE run_id=?',
                     (revision + 1, json.dumps(next_state, ensure_ascii=False), run_id))
        db._append_event(conn, run_id, 'quality_transition', {
            'revision': revision + 1, 'phase': next_state['phase'], 'remediations_used': next_state['used'],
            'checks': next_state['checks'], 'reason': next_state['reason'], 'digest': digest(next_state),
        })
    return revision + 1, next_state


def checkpoint(run_id, snapshot):
    revision, state = load(run_id)
    cp = AgentCheckpoint(run_id=run_id, **snapshot)
    cp.validate()
    if state['phase'] != 'research':
        raise ValueError('只有补充研究可以保存检查点')
    from pipeline_storage import load_pipeline
    if cp.model_name != load_pipeline(run_id)['model_name']:
        raise ValueError('补充模型发生变化')
    previous = state['checkpoint']
    if previous and (cp.next_step < previous['next_step'] or cp.tool_count < previous['tool_count']
                     or cp.messages[:len(previous['messages'])] != previous['messages']):
        raise ValueError('补充研究检查点不能倒退')
    if cp.evidence[:len(state['evidence'])] != state['evidence']:
        raise ValueError('补充研究不能改写原证据')
    from dataclasses import asdict
    save(run_id, revision, state, checkpoint=asdict(cp), evidence=cp.evidence)


def final_material(conn, run_id, report, evidence):
    item = read(conn, run_id)
    if item is None or item[1]['phase'] == 'skipped':
        return report, evidence
    state = item[1]
    if state['phase'] != 'passed':
        raise ValueError('质量门禁尚未通过，不能完成研究')
    if state['evidence'][:len(evidence)] != evidence:
        raise ValueError('质量证据与原研究不一致')
    return state['report'], state['evidence']
