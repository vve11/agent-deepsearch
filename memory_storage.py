"""SQLite 候选记忆存取。scope 来自可信调用方，不替代用户身份认证。"""
from contextlib import closing
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

import database
from evidence import EvidenceStore
from memory import MemoryDraft, MemoryScope, MemoryRecord, create_memory_candidate, _id, _text
from memory_codec import serialize_memory, deserialize_memory, draft_data


def init_memory_tables(connection):
    # 不使用 executescript，避免隐式提交调用方正在进行的事务。
    statements = (
        """CREATE TABLE IF NOT EXISTS memory_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL REFERENCES memories(id),
            reviewer TEXT NOT NULL, decision TEXT NOT NULL, reason TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )""",
        """CREATE TABLE IF NOT EXISTS run_memory_context (
            run_id TEXT PRIMARY KEY NOT NULL REFERENCES runs(id),
            scope_kind TEXT NOT NULL, scope_id TEXT NOT NULL,
            enabled INTEGER NOT NULL CHECK(enabled IN (0,1)),
            snapshots_json TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS memory_jobs (
            run_id TEXT PRIMARY KEY NOT NULL REFERENCES runs(id),
            status TEXT NOT NULL CHECK(status IN ('pending','running','done','failed')),
            attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts BETWEEN 0 AND 3),
            observed_at TEXT NOT NULL, error TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('episodic','semantic','procedural')),
            scope_kind TEXT NOT NULL CHECK(scope_kind IN ('user','project')),
            scope_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('candidate','active','archived')),
            record_json TEXT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_memories_scope
            ON memories(scope_kind, scope_id, kind, status)""",
        """CREATE TABLE IF NOT EXISTS memory_proposals (
            scope_kind TEXT NOT NULL,
            scope_id TEXT NOT NULL,
            request_key TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            memory_id TEXT NOT NULL REFERENCES memories(id),
            PRIMARY KEY(scope_kind, scope_id, request_key)
        )""",
        """CREATE TABLE IF NOT EXISTS memory_research_sources (
            memory_id TEXT NOT NULL REFERENCES memories(id),
            run_id TEXT NOT NULL REFERENCES runs(id),
            PRIMARY KEY(memory_id, run_id)
        )""",
        """CREATE TABLE IF NOT EXISTS memory_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL REFERENCES memories(id),
            event_type TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        )""",
    )
    for statement in statements:
        connection.execute(statement)


def _scope(scope):
    if not isinstance(scope, MemoryScope) or scope.kind not in ('user', 'project'):
        raise ValueError('必须指定合法的记忆范围')
    _text(scope.id, '范围编号', 200)


def _fingerprint(record):
    data = draft_data(record)
    # 来源顺序不改变含义；行动/步骤顺序改变含义，必须保留。
    for source in data['sources']:
        source['evidence_ids'].sort()
    data['sources'].sort(key=lambda source: (source['kind'], source['source_id']))
    encoded = json.dumps(data, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return sha256(encoded.encode('utf-8')).hexdigest()


def _validate_sources(connection, record):
    for source in record.sources:
        if source.kind != 'research':
            continue  # 用户消息/人工方法暂无持久化来源表，只保留引用，不宣称已核验。
        row = connection.execute('SELECT status, evidence_json FROM runs WHERE id=?',
                                 (source.source_id,)).fetchone()
        if row is None:
            raise ValueError('记忆引用的研究不存在')
        if row[0] not in ('completed', 'failed'):
            raise ValueError('只能从已结束的研究保存记忆')
        evidence = EvidenceStore.from_records(json.loads(row[1]))
        if not set(source.evidence_ids) <= {item['id'] for item in evidence.all()}:
            raise ValueError('来源研究中不存在指定证据')


def _read(connection, memory_id, scope):
    row = connection.execute(
        'SELECT id, kind, scope_kind, scope_id, status, record_json FROM memories '
        'WHERE id=? AND scope_kind=? AND scope_id=?',
        (memory_id, scope.kind, scope.id),
    ).fetchone()
    if row is None:
        # 不区分“不存在”和“属于其他范围”，不泄露其他范围的信息。
        raise LookupError('当前范围内没有这条记忆')
    record = deserialize_memory(row[5])
    if (record.id, record.kind, record.scope.kind, record.scope.id, record.status) != row[:5]:
        raise ValueError('记忆索引字段与内容不一致')
    links = {item[0] for item in connection.execute(
        'SELECT run_id FROM memory_research_sources WHERE memory_id=?', (memory_id,))}
    if links != {source.source_id for source in record.sources if source.kind == 'research'}:
        raise ValueError('记忆来源关联损坏')
    _validate_sources(connection, record)
    return record


class MemoryStore:
    """先调用 database.init_db() 创建表；可注入独立数据库路径供测试使用。"""
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else database.DB_PATH

    def propose(self, draft: MemoryDraft, *, request_key: str | None = None) -> MemoryRecord:
        draft = deepcopy(draft)
        fingerprint = _fingerprint(draft)
        if request_key is not None:
            _text(request_key, '提交编号', 200)
        # 两种编号分开命名，用户提交编号不会碰撞自动内容摘要。
        key = 'explicit:' + request_key if request_key is not None else 'content:' + fingerprint
        with closing(sqlite3.connect(self.db_path)) as connection:
            connection.execute('PRAGMA foreign_keys=ON')
            with connection:
                connection.execute('BEGIN IMMEDIATE')
                previous = connection.execute(
                    'SELECT fingerprint, memory_id FROM memory_proposals '
                    'WHERE scope_kind=? AND scope_id=? AND request_key=?',
                    (draft.scope.kind, draft.scope.id, key),
                ).fetchone()
                if previous is not None:
                    if previous[0] != fingerprint:
                        raise ValueError('同一提交编号不能用于不同记忆内容')
                    result = _read(connection, previous[1], draft.scope)
                    if _fingerprint(result) != fingerprint:
                        raise ValueError('提交记录与记忆内容不一致')
                else:
                    candidate = create_memory_candidate(draft)
                    payload = serialize_memory(candidate)
                    _validate_sources(connection, candidate)
                    if candidate.supersedes_id is not None:
                        old = _read(connection, candidate.supersedes_id, candidate.scope)
                        if old.kind != candidate.kind:
                            raise ValueError('修订记忆必须保持原记忆类型')
                    connection.execute('INSERT INTO memories VALUES (?, ?, ?, ?, ?, ?)',
                                       (candidate.id, candidate.kind, candidate.scope.kind,
                                        candidate.scope.id, candidate.status, payload))
                    for source in candidate.sources:
                        if source.kind == 'research':
                            connection.execute('INSERT INTO memory_research_sources VALUES (?, ?)',
                                               (candidate.id, source.source_id))
                    connection.execute('INSERT INTO memory_proposals VALUES (?, ?, ?, ?, ?)',
                                       (candidate.scope.kind, candidate.scope.id, key, fingerprint, candidate.id))
                    connection.execute("INSERT INTO memory_events(memory_id,event_type) VALUES (?, 'proposed')",
                                       (candidate.id,))
                    result = _read(connection, candidate.id, candidate.scope)
            # 离开事务块且提交成功之后才返回；提交失败会向调用方抛异常。
            return result

    def get(self, memory_id: str, *, scope: MemoryScope) -> MemoryRecord:
        _id(memory_id, '记忆编号')
        _scope(scope)
        with closing(sqlite3.connect(self.db_path)) as connection:
            with connection:
                connection.execute('BEGIN')  # 记忆、来源在同一个读快照中检查。
                return _read(connection, memory_id, scope)

    def list(self, *, scope: MemoryScope, status: str | None = None, limit: int = 50):
        _scope(scope)
        if status not in (None, 'candidate', 'active', 'archived'):
            raise ValueError('记忆状态不合法')
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('limit 必须为 1～100')
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute('BEGIN')
            rows = connection.execute(
                'SELECT id FROM memories WHERE scope_kind=? AND scope_id=? '
                'AND (? IS NULL OR status=?) ORDER BY rowid DESC LIMIT ?',
                (scope.kind, scope.id, status, status, limit))
            return [_read(connection, row[0], scope) for row in rows.fetchall()]

    def search(self, query):
        from memory_retrieval import search_memories
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute('BEGIN')
            return search_memories(connection, query)

    def activate(self, memory_id: str, *, scope: MemoryScope,
                 reviewer: str | None = None, reason: str | None = None):
        """无 reviewer 使用自动规则；人工调用必须明确提供审核人和理由。"""
        from memory_review import auto_review, check_conflicts
        _id(memory_id, '记忆编号')
        _scope(scope)
        if reviewer is not None:
            _text(reviewer, '审核人', 200)
            _text(reason, '审核理由', 1000)
            if reviewer == 'auto-v1':
                raise ValueError('审核人不能使用自动审核保留名称')
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute('PRAGMA foreign_keys=ON')
            connection.execute('BEGIN IMMEDIATE')
            record = _read(connection, memory_id, scope)
            if record.status == 'archived':
                raise ValueError('归档记忆不能直接重新生效，请创建修订')
            if record.status == 'active':
                return record
            if record.expires_at and record.expires_at <= datetime.now(timezone.utc):
                raise ValueError('过期记忆不能生效')
            if reviewer is None:
                allowed, explanation = auto_review(connection, record)
            else:
                allowed, explanation = True, reason
            if allowed:
                conflict = check_conflicts(connection, record)
                if conflict:
                    allowed, explanation = False, conflict
            connection.execute(
                'INSERT INTO memory_reviews(memory_id,reviewer,decision,reason) VALUES (?,?,?,?)',
                (memory_id, reviewer or 'auto-v1', 'approved' if allowed else 'held', explanation))
            if not allowed:
                return record  # 审核未通过仍为 candidate，可查看原因后人工处理。
            if record.supersedes_id:
                old = _read(connection, record.supersedes_id, scope)
                if old.kind != record.kind:
                    raise ValueError('修订类型不匹配')
                old = replace(old, status='archived')
                connection.execute("UPDATE memories SET status='archived', record_json=? WHERE id=?",
                                   (serialize_memory(old), old.id))
                connection.execute("INSERT INTO memory_events(memory_id,event_type) VALUES (?, 'superseded')", (old.id,))
            record = replace(record, status='active')
            connection.execute("UPDATE memories SET status='active', record_json=? WHERE id=?",
                               (serialize_memory(record), record.id))
            connection.execute("INSERT INTO memory_events(memory_id,event_type) VALUES (?, 'activated')", (record.id,))
            return record

    def archive(self, memory_id: str, *, scope: MemoryScope):
        _id(memory_id, '记忆编号')
        _scope(scope)
        with closing(sqlite3.connect(self.db_path)) as connection, connection:
            connection.execute('BEGIN IMMEDIATE')
            record = _read(connection, memory_id, scope)
            if record.status != 'archived':
                record = replace(record, status='archived')
                connection.execute("UPDATE memories SET status='archived',record_json=? WHERE id=?",
                                   (serialize_memory(record), record.id))
                connection.execute("INSERT INTO memory_events(memory_id,event_type) VALUES (?, 'archived')", (record.id,))
            return record

    def revise(self, memory_id: str, draft: MemoryDraft, *, scope: MemoryScope,
               request_key: str | None = None):
        old = self.get(memory_id, scope=scope)
        if draft.scope != scope or draft.kind != old.kind:
            raise ValueError('修订必须保持范围和类型')
        if draft.supersedes_id not in (None, memory_id):
            raise ValueError('修订目标不一致')
        return self.propose(replace(draft, supersedes_id=memory_id), request_key=request_key)

    def reviews(self, memory_id: str, *, scope: MemoryScope):
        self.get(memory_id, scope=scope)
        with closing(sqlite3.connect(self.db_path)) as connection:
            rows = connection.execute('SELECT reviewer,decision,reason,created_at FROM memory_reviews WHERE memory_id=? ORDER BY id', (memory_id,)).fetchall()
        return [dict(zip(('reviewer','decision','reason','created_at'), row)) for row in rows]
