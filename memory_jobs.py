"""完成研究后的持久化记忆作业；失败不回滚研究，最多自动尝试三次。"""
from contextlib import closing
from datetime import datetime
import logging
import json
import sqlite3

import database
from memory import MemoryScope
from memory_storage import MemoryStore
from memory_extraction import extract_drafts
from memory_codec import deserialize_memory

logger = logging.getLogger(__name__)


def enqueue(connection, run_id):
    context = connection.execute('SELECT enabled FROM run_memory_context WHERE run_id=?', (run_id,)).fetchone()
    if context == (1,):
        connection.execute("""INSERT INTO memory_jobs(run_id,status,observed_at)
            VALUES (?, 'pending', strftime('%Y-%m-%dT%H:%M:%fZ','now'))
            ON CONFLICT(run_id) DO NOTHING""", (run_id,))


def process_job(run_id):
    try:
        with closing(sqlite3.connect(database.DB_PATH)) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            claimed = connection.execute("""UPDATE memory_jobs SET status='running', attempts=attempts+1,error=NULL
                WHERE run_id=? AND status IN ('pending','failed') AND attempts<3""", (run_id,))
            if claimed.rowcount != 1:
                return
            row = connection.execute('SELECT observed_at FROM memory_jobs WHERE run_id=?', (run_id,)).fetchone()
            scope = connection.execute('SELECT scope_kind,scope_id FROM run_memory_context WHERE run_id=?', (run_id,)).fetchone()
        if scope is None:
            raise ValueError('记忆作业缺少范围快照')
        with closing(sqlite3.connect(database.DB_PATH)) as connection, connection:
            connection.execute('BEGIN')
            drafts = extract_drafts(connection, run_id, MemoryScope(*scope), datetime.fromisoformat(row[0]))
        store = MemoryStore()
        for index, draft in enumerate(drafts):
            record = store.propose(draft, request_key=f'auto-v1:{run_id}:{index}')
            if record.status == 'candidate':
                store.activate(record.id, scope=record.scope)
        with closing(sqlite3.connect(database.DB_PATH)) as connection, connection:
            connection.execute("UPDATE memory_jobs SET status='done',error=NULL WHERE run_id=?", (run_id,))
    except Exception as error:
        logger.warning('记忆处理失败，研究结果已保留：%s (%s)', run_id, type(error).__name__)
        try:
            with closing(sqlite3.connect(database.DB_PATH)) as connection, connection:
                connection.execute("UPDATE memory_jobs SET status='failed',error=? WHERE run_id=?",
                                   (f'{type(error).__name__}: {str(error)[:500]}', run_id))
        except sqlite3.Error:
            logger.warning('记忆作业错误状态暂时无法保存：%s', run_id)


def process_pending(*, recover_interrupted=False):
    try:
        with closing(sqlite3.connect(database.DB_PATH)) as connection, connection:
            if recover_interrupted:
                # 应用启动时调用；沿用项目单进程运行约束。
                connection.execute("UPDATE memory_jobs SET status='failed',error='进程中断' WHERE status='running'")
            rows = connection.execute("SELECT run_id FROM memory_jobs WHERE status IN ('pending','failed') AND attempts<3 ORDER BY rowid LIMIT 20").fetchall()
        for row in rows:
            process_job(row[0])
    except Exception:
        logger.exception('暂时无法处理待办记忆；不阻止研究服务启动')


def describe_memory(run_id):
    from memory_context import load_context
    load_context(run_id)  # 先验证，不把损坏快照展示成正常结果。
    with closing(sqlite3.connect(database.DB_PATH)) as connection:
        context = connection.execute('SELECT enabled,snapshots_json FROM run_memory_context WHERE run_id=?', (run_id,)).fetchone()
        job = connection.execute('SELECT status,attempts,error FROM memory_jobs WHERE run_id=?', (run_id,)).fetchone()
    return {'enabled': bool(context[0]) if context else False,
            'selected_ids': [deserialize_memory(value).id for value in json.loads(context[1])] if context else [],
            'extraction': dict(zip(('status','attempts','error'), job)) if job else None}


def retry_job(run_id, *, scope):
    from memory import _id
    from memory_storage import _scope
    _id(run_id, '研究编号')
    _scope(scope)
    with closing(sqlite3.connect(database.DB_PATH)) as connection, connection:
        connection.execute('BEGIN IMMEDIATE')
        row = connection.execute('SELECT scope_kind,scope_id FROM run_memory_context WHERE run_id=?', (run_id,)).fetchone()
        if row != (scope.kind, scope.id):
            raise LookupError('当前范围没有这次研究的记忆作业')
        cursor = connection.execute("UPDATE memory_jobs SET status='pending',attempts=0,error=NULL WHERE run_id=? AND status='failed'", (run_id,))
        if cursor.rowcount != 1:
            raise ValueError('只有失败的记忆作业可以人工重试')
        database._append_event(connection, run_id, 'memory_retry_requested', {})
    process_job(run_id)
