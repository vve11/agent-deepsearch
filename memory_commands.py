"""本地 CLI 的记忆查看、人工审核、检索和归档入口。"""
import json
import sqlite3

from memory import MemoryQuery
from memory_context import configured_scope
from memory_codec import record_data
from memory_storage import MemoryStore


def handle_memory_command(text):
    parts = text.split(maxsplit=2)
    command = parts[0] if parts else ''
    if command not in ('/memories', '/memory', '/approve-memory', '/archive-memory', '/search-memory', '/retry-memory'):
        return False
    try:
        store, scope = MemoryStore(), configured_scope()
        if command == '/retry-memory':
            if len(parts) != 2:
                raise ValueError('用法：/retry-memory 研究编号')
            from memory_jobs import retry_job, describe_memory
            retry_job(parts[1], scope=scope)
            print(json.dumps(describe_memory(parts[1]), ensure_ascii=False, indent=2))
        elif command == '/memories':
            if len(parts) > 2:
                raise ValueError('用法：/memories [candidate|active|archived]')
            for record in store.list(scope=scope, status=parts[1] if len(parts) == 2 else None):
                print(record.id, record.kind, record.status, str(record.content)[:180])
        elif command == '/search-memory':
            query = text.partition(' ')[2].strip()
            for record in store.search(MemoryQuery(text=query, scope=scope)):
                print(json.dumps(record_data(record), ensure_ascii=False, indent=2))
        elif command == '/approve-memory':
            if len(parts) != 3:
                raise ValueError('用法：/approve-memory 记忆编号 人工核对后的理由')
            record = store.activate(parts[1], scope=scope, reviewer='local-user', reason=parts[2])
            print('审核后状态：', record.status)
            print(json.dumps(store.reviews(record.id, scope=scope)[-1:], ensure_ascii=False))
        else:
            if len(parts) != 2:
                raise ValueError(f'用法：{command} 记忆编号')
            record = (store.archive(parts[1], scope=scope) if command == '/archive-memory'
                      else store.get(parts[1], scope=scope))
            print(json.dumps(record_data(record), ensure_ascii=False, indent=2))
            print(json.dumps(store.reviews(record.id, scope=scope), ensure_ascii=False, indent=2))
    except (ValueError, LookupError, sqlite3.Error) as error:
        print('记忆操作失败：', error)
    return True
