"""范围内词项检索；英文词与中文双字片段，限定候选扫描与上下文大小。"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
import re

from memory import MemoryQuery
from memory_storage import _scope, _read

logger = logging.getLogger(__name__)


def terms(text):
    tokens = set(re.findall(r'[a-z0-9_]{2,}', text.casefold()))
    for segment in re.findall(r'[\u4e00-\u9fff]+', text):
        tokens.update(segment[i:i + 2] for i in range(len(segment) - 1))
        if len(segment) == 1:
            tokens.add(segment)
    return tokens


def search_memories(connection, query):
    if not isinstance(query, MemoryQuery):
        raise ValueError('必须提供 MemoryQuery')
    _scope(query.scope)
    if not isinstance(query.text, str) or not query.text.strip() or len(query.text) > 4000:
        raise ValueError('检索问题必须是 1～4000 字符')
    if type(query.limit) is not int or not 1 <= query.limit <= 20:
        raise ValueError('检索数量必须是 1～20')
    if not isinstance(query.kinds, list) or not query.kinds or any(k not in ('episodic','semantic','procedural') for k in query.kinds):
        raise ValueError('检索类型不合法')
    wanted = terms(query.text)
    ranked = []
    rows = connection.execute(
        "SELECT id FROM memories WHERE scope_kind=? AND scope_id=? AND status='active' ORDER BY rowid DESC LIMIT 1000",
        (query.scope.kind, query.scope.id)).fetchall()
    for row in rows:
        try:
            record = _read(connection, row[0], query.scope)
        except (ValueError, TypeError, LookupError):
            logger.warning('检索跳过损坏记忆：%s', row[0])
            continue
        if record.kind not in query.kinds or (record.expires_at and record.expires_at <= datetime.now(timezone.utc)):
            continue
        text = ' '.join(str(value) for value in asdict(record.content).values())
        score = len(wanted & terms(text))
        # 用户明确批准的偏好可跨主题适用，但仍受 scope 限制。
        if record.kind == 'semantic' and record.content.category == 'preference':
            score += 1
        if score:
            ranked.append((score, record.created_at, record.id, record))
    ranked.sort(key=lambda item: item[:3], reverse=True)
    selected, seen = [], set()
    for item in ranked:
        content = asdict(item[3].content)
        content.pop('observed_at', None)
        key = (item[3].kind, json.dumps(content, ensure_ascii=False, sort_keys=True))
        if key not in seen:
            selected.append(item[3])
            seen.add(key)
        if len(selected) == query.limit:
            break
    return selected
