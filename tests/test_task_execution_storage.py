import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import database
from plan import parse_plan
from run import ResearchRun
from task_execution import TaskExecution


class TaskExecutionStorageTests(unittest.TestCase):
    def setUp(self):
        # 每个测试使用自己的临时数据库，不接触真实研究记录。
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)

        patcher = patch.object(
            database,
            "DB_PATH",
            Path(directory.name) / "test.db",
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        database.init_db()

        self.research = ResearchRun(question="比较 Python 和 JavaScript")
        database.save_run(self.research)

        plan = parse_plan(
            self.research.question,
            {
                "tasks": [
                    {"question": "Python 适合哪些场景？"},
                    {"question": "JavaScript 适合哪些场景？"},
                ]
            },
        )
        database.save_plan(self.research.id, plan)

    def test_initialize_and_load(self):
        database.initialize_task_executions(self.research.id)

        for task_id in ("T1", "T2"):
            with self.subTest(task_id=task_id):
                loaded = database.load_task_execution(
                    self.research.id, task_id
                )
                expected = TaskExecution(
                    run_id=self.research.id,
                    task_id=task_id,
                )
                self.assertEqual(loaded, expected)

    def test_reinitialize_preserves_completed_result(self):
        database.initialize_task_executions(self.research.id)

        # 暂时直接用 SQL 模拟一个已完成任务。
        with closing(sqlite3.connect(database.DB_PATH)) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE task_executions
                    SET status = 'completed', answer = ?
                    WHERE run_id = ? AND task_id = ?
                    """,
                    ("已完成的研究结果", self.research.id, "T1"),
                )

        database.initialize_task_executions(self.research.id)

        loaded = database.load_task_execution(self.research.id, "T1")
        self.assertEqual(loaded.status, "completed")
        self.assertEqual(loaded.answer, "已完成的研究结果")

        with closing(sqlite3.connect(database.DB_PATH)) as connection:
            count = connection.execute(
                """
                SELECT COUNT(*) FROM task_executions
                WHERE run_id = ?
                """,
                (self.research.id,),
            ).fetchone()[0]

        self.assertEqual(count, 2)

    def test_missing_record_and_unknown_task(self):
        # 计划里有 T1，但还没初始化执行记录。
        with self.assertRaises(LookupError):
            database.load_task_execution(self.research.id, "T1")

        database.initialize_task_executions(self.research.id)

        # 计划里只有 T1、T2，没有 T3。
        with self.assertRaises(ValueError):
            database.load_task_execution(self.research.id, "T3")

    def test_inconsistent_saved_record_is_rejected(self):
        database.initialize_task_executions(self.research.id)

        # 故意制造“已完成却没有回答”的矛盾数据。
        with closing(sqlite3.connect(database.DB_PATH)) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE task_executions
                    SET status = 'completed'
                    WHERE run_id = ? AND task_id = ?
                    """,
                    (self.research.id, "T1"),
                )

        with self.assertRaises(ValueError):
            database.load_task_execution(self.research.id, "T1")


if __name__ == "__main__":
    unittest.main()