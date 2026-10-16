"""记忆的版本化 JSON 编解码；严格还原嵌套类型，不修补损坏数据。"""
import json
from dataclasses import asdict, fields
from datetime import datetime

from memory import (
    MemoryRecord, MemoryScope, MemorySource, EpisodicContent, SemanticContent,
    ProceduralContent, validate_memory_record, validate_memory_draft, MemoryDraft,
)

MAX_JSON_CHARS = 2_000_000
CONTENT_TYPES = {'episodic': EpisodicContent, 'semantic': SemanticContent,
                 'procedural': ProceduralContent}


def draft_data(record: MemoryDraft) -> dict:
    validate_memory_draft(record)
    data = asdict(record)
    for key in ('id', 'created_at', 'status'):
        data.pop(key, None)
    data['expires_at'] = record.expires_at.isoformat() if record.expires_at else None
    if isinstance(record.content, SemanticContent):
        data['content']['observed_at'] = (
            record.content.observed_at.isoformat() if record.content.observed_at else None
        )
    return data


def record_data(record: MemoryRecord) -> dict:
    validate_memory_record(record)
    return {**draft_data(record), 'id': record.id, 'created_at': record.created_at.isoformat(),
            'status': record.status}


def serialize_memory(record: MemoryRecord) -> str:
    payload = json.dumps({'version': 1, 'record': record_data(record)},
                         ensure_ascii=False, sort_keys=True, allow_nan=False)
    if len(payload) > MAX_JSON_CHARS:
        raise ValueError('记忆 JSON 超过大小限制')
    return payload


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('记忆 JSON 包含重复字段')
        result[key] = value
    return result


def _shape(value, cls):
    if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
        raise ValueError(f'{cls.__name__} 字段不完整或包含未知字段')


def _time(value):
    if not isinstance(value, str):
        raise ValueError('时间必须编码为 ISO 字符串')
    try:
        return datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError('记忆时间格式损坏') from error


def deserialize_memory(payload: str) -> MemoryRecord:
    if not isinstance(payload, str) or len(payload) > MAX_JSON_CHARS:
        raise ValueError('记忆 JSON 类型或大小不合法')
    try:
        envelope = json.loads(payload, object_pairs_hook=_object)
    except (ValueError, RecursionError) as error:
        raise ValueError('记忆 JSON 无效') from error
    if not isinstance(envelope, dict) or set(envelope) != {'version', 'record'}:
        raise ValueError('记忆 JSON 外层字段不正确')
    if type(envelope['version']) is not int or envelope['version'] != 1:
        raise ValueError('不支持的记忆格式版本')
    data = envelope['record']
    _shape(data, MemoryRecord)
    kind = data['kind']
    if not isinstance(kind, str) or kind not in CONTENT_TYPES:
        raise ValueError('记忆类型不合法')
    _shape(data['content'], CONTENT_TYPES[kind])
    _shape(data['scope'], MemoryScope)
    if not isinstance(data['sources'], list):
        raise ValueError('来源必须是列表')
    for source in data['sources']:
        _shape(source, MemorySource)
    data['created_at'] = _time(data['created_at'])
    if data['expires_at'] is not None:
        data['expires_at'] = _time(data['expires_at'])
    if kind == 'semantic' and data['content']['observed_at'] is not None:
        data['content']['observed_at'] = _time(data['content']['observed_at'])
    record = MemoryRecord(
        **{key: value for key, value in data.items() if key not in ('scope', 'sources', 'content')},
        scope=MemoryScope(**data['scope']),
        sources=[MemorySource(**source) for source in data['sources']],
        content=CONTENT_TYPES[kind](**data['content']),
    )
    validate_memory_record(record)
    return record
