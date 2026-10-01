"""在真实临时 SQLite 中制造插入失败，验证事务回滚。"""
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import database
from run import ResearchRun


class TransactionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db_patch = patch.object(database, "DB_PATH", Path(directory.name) / "test.db")
        db_patch.start()
        self.addCleanup(db_patch.stop)
        database.init_db()

    def reject_events(self):
        # 触发器仅安装在测试数据库，强制事件 INSERT 报错。
        with closing(sqlite3.connect(database.DB_PATH)) as connection:
            with connection:
                connection.execute("""
                    CREATE TRIGGER reject_event BEFORE INSERT ON run_events
                    BEGIN
                        SELECT RAISE(ABORT, 'simulated event insert failure');
                    END
                """)

    def test_commit_start_and_completion(self):
        run = ResearchRun(question="事务验证")
        first = database.save_run_with_event(run, "run_started", {"question": run.question})
        run.status = "completed"
        run.answer = "结果"
        second = database.save_run_with_event(run, "run_completed", {"answer_chars": 2})
        self.assertEqual(database.load_run(run.id), run)
        events = database.load_events(run.id)
        self.assertEqual([e["id"] for e in events], [first, second])
        self.assertEqual([e["event_type"] for e in events], ["run_started", "run_completed"])
        self.assertEqual(events[1]["payload"], {"answer_chars": 2})

    def test_event_failure_rolls_back_new_run(self):
        self.reject_events()
        run = ResearchRun(question="不能留下半条启动记录")
        with self.assertRaises(sqlite3.IntegrityError):
            database.save_run_with_event(run, "run_started", {})
        with self.assertRaises(LookupError):
            database.load_run(run.id)
        self.assertEqual(database.load_events(run.id), [])

    def test_event_failure_restores_previous_run_and_keeps_old_events(self):
        run = ResearchRun(question="更新回滚")
        database.save_run_with_event(run, "run_started", {})
        before = database.load_run(run.id)
        old_events = database.load_events(run.id)
        self.reject_events()
        run.status = "completed"
        run.answer = "不应该提交的答案"
        run.evidence = [{"id": "E1", "content": "不应该提交的证据"}]
        with self.assertRaises(sqlite3.IntegrityError):
            database.save_run_with_event(run, "run_completed", {})
        self.assertEqual(database.load_run(run.id), before)
        self.assertEqual(database.load_events(run.id), old_events)
        # 回滚的是数据库，内存对象不会自动恢复。
        self.assertEqual(run.status, "completed")

    def test_task_failure_does_not_insert_event(self):
        run = ResearchRun(question="无效状态")
        run.status = "invalid"
        with self.assertRaises(sqlite3.IntegrityError):
            database.save_run_with_event(run, "run_started", {})
        self.assertEqual(database.load_events(run.id), [])

    def test_serialization_failure_also_rolls_back_task(self):
        run = ResearchRun(question="不能序列化的事件")
        with self.assertRaises(TypeError):
            database.save_run_with_event(run, "run_started", {"bad": object()})
        with self.assertRaises(LookupError):
            database.load_run(run.id)

    def test_independent_event_still_enforces_foreign_key(self):
        with self.assertRaises(sqlite3.IntegrityError):
            database.append_event("0" * 32, "model_started", {})


if __name__ == "__main__":
    unittest.main()
