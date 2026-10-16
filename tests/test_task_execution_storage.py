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

    def test_start_pending_task_and_record_event(self):
        database.initialize_task_executions(self.research.id)

        database.start_task_execution(self.research.id, "T1")

        loaded = database.load_task_execution(self.research.id, "T1")
        self.assertEqual(loaded.status, "running")
        self.assertIsNone(loaded.answer)
        self.assertIsNone(loaded.error)

        event = database.load_events(self.research.id)[-1]
        self.assertEqual(event["event_type"], "task_started")
        self.assertEqual(
            event["payload"],
            {"task_id": "T1", "previous_status": "pending"},
        )

        # 启动 T1 不应改变 T2。
        other = database.load_task_execution(self.research.id, "T2")
        self.assertEqual(other.status, "pending")

    def test_running_task_cannot_start_twice(self):
        database.initialize_task_executions(self.research.id)
        database.start_task_execution(self.research.id, "T1")
        events_before = database.load_events(self.research.id)

        with self.assertRaises(ValueError):
            database.start_task_execution(self.research.id, "T1")

        # 拒绝重复启动，也不能多写一条启动事件。
        self.assertEqual(
            database.load_events(self.research.id),
            events_before,
        )

    def test_retry_failed_task_clears_error(self):
        database.initialize_task_executions(self.research.id)

        # 模拟之前执行失败的状态。
        with closing(sqlite3.connect(database.DB_PATH)) as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE task_executions
                    SET status = 'failed', error = ?
                    WHERE run_id = ? AND task_id = ?
                    """,
                    ("搜索超时", self.research.id, "T1"),
                )

        database.start_task_execution(self.research.id, "T1")

        loaded = database.load_task_execution(self.research.id, "T1")
        self.assertEqual(loaded.status, "running")
        self.assertIsNone(loaded.error)

        event = database.load_events(self.research.id)[-1]
        self.assertEqual(event["payload"]["previous_status"], "failed")
    def test_complete_round_trip_and_terminal_state_is_protected(self):
        database.initialize_task_executions(self.research.id)
        database.start_task_execution(self.research.id, "T1")
        database.complete_task_execution(
            self.research.id, "T1", "结果见 [E1]", ["E1"],
        )
        expected = TaskExecution(
            run_id=self.research.id, task_id="T1", status="completed",
            answer="结果见 [E1]", evidence_ids=["E1"],
        )
        self.assertEqual(database.load_task_execution(self.research.id, "T1"), expected)
        event = database.load_events(self.research.id)[-1]
        self.assertEqual(event["event_type"], "task_completed")
        self.assertEqual(event["payload"], {
            "task_id": "T1", "evidence_count": 1, "answer_chars": len(expected.answer),
        })
        events = database.load_events(self.research.id)
        for action in (
            lambda: database.start_task_execution(self.research.id, "T1"),
            lambda: database.complete_task_execution(self.research.id, "T1", "覆盖", []),
            lambda: database.fail_task_execution(self.research.id, "T1", "覆盖"),
        ):
            with self.assertRaises(ValueError):
                action()
        database.initialize_task_executions(self.research.id)
        self.assertEqual(database.load_task_execution(self.research.id, "T1"), expected)
        self.assertEqual(database.load_events(self.research.id), events)

    def test_fail_retry_preserves_evidence_and_clears_error(self):
        database.initialize_task_executions(self.research.id)
        database.start_task_execution(self.research.id, "T1")
        database.fail_task_execution(self.research.id, "T1", "搜索超时", ["E1"])
        failed = database.load_task_execution(self.research.id, "T1")
        self.assertEqual((failed.status, failed.error, failed.answer), ("failed", "搜索超时", None))
        self.assertEqual(failed.evidence_ids, ["E1"])
        event = database.load_events(self.research.id)[-1]
        self.assertEqual(event["event_type"], "task_failed")
        self.assertEqual(event["payload"]["error"], "搜索超时")
        database.start_task_execution(self.research.id, "T1")
        retried = database.load_task_execution(self.research.id, "T1")
        self.assertEqual(retried.status, "running")
        self.assertIsNone(retried.error)
        self.assertEqual(retried.evidence_ids, ["E1"])
        database.fail_task_execution(self.research.id, "T1", "再次超时")
        self.assertEqual(database.load_task_execution(self.research.id, "T1").evidence_ids, ["E1"])
        database.start_task_execution(self.research.id, "T1")
        database.complete_task_execution(self.research.id, "T1", "研究完成", ["E1"])
        self.assertEqual(database.load_task_execution(self.research.id, "T1").status, "completed")

    def test_cannot_finish_pending_or_failed_task(self):
        database.initialize_task_executions(self.research.id)
        for status in ("pending", "failed"):
            if status == "failed":
                database.start_task_execution(self.research.id, "T1")
                database.fail_task_execution(self.research.id, "T1", "错误")
            before = database.load_task_execution(self.research.id, "T1")
            events = database.load_events(self.research.id)
            with self.subTest(status=status):
                with self.assertRaises(ValueError):
                    database.complete_task_execution(self.research.id, "T1", "结果", [])
                with self.assertRaises(ValueError):
                    database.fail_task_execution(self.research.id, "T1", "错误")
                self.assertEqual(database.load_task_execution(self.research.id, "T1"), before)
                self.assertEqual(database.load_events(self.research.id), events)

    def test_invalid_result_does_not_change_record_or_events(self):
        database.initialize_task_executions(self.research.id)
        database.start_task_execution(self.research.id, "T1")
        before = database.load_task_execution(self.research.id, "T1")
        events = database.load_events(self.research.id)
        cases = [
            lambda: database.complete_task_execution(self.research.id, "T1", " ", []),
            lambda: database.complete_task_execution(self.research.id, "T1", "结果", ["E1", "E1"]),
            lambda: database.complete_task_execution(self.research.id, "T1", "结果", "E1"),
            lambda: database.fail_task_execution(self.research.id, "T1", ""),
            lambda: database.fail_task_execution(self.research.id, "T1", "错误", ["T1"]),
        ]
        for index, action in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                action()
            self.assertEqual(database.load_task_execution(self.research.id, "T1"), before)
            self.assertEqual(database.load_events(self.research.id), events)

    def test_event_failure_rolls_back_each_transition(self):
        database.initialize_task_executions(self.research.id)
        append_event = database._append_event

        def append_then_fail(*args):
            append_event(*args)
            raise sqlite3.OperationalError("模拟事件保存失败")

        actions = [
            lambda: database.start_task_execution(self.research.id, "T1"),
            lambda: database.complete_task_execution(self.research.id, "T1", "结果", []),
            lambda: database.fail_task_execution(self.research.id, "T1", "错误"),
        ]
        for index, action in enumerate(actions):
            if index == 1:
                database.start_task_execution(self.research.id, "T1")
            before = database.load_task_execution(self.research.id, "T1")
            events = database.load_events(self.research.id)
            with self.subTest(transition=index):
                with patch.object(database, "_append_event", side_effect=append_then_fail):
                    with self.assertRaises(sqlite3.OperationalError):
                        action()
                self.assertEqual(database.load_task_execution(self.research.id, "T1"), before)
                self.assertEqual(database.load_events(self.research.id), events)

    def test_same_task_id_in_different_runs_is_isolated(self):
        other = ResearchRun(question="另一个研究")
        database.save_run(other)
        database.save_plan(other.id, parse_plan(other.question, {"tasks": [{"question": "方向"}]}))
        for run_id in (self.research.id, other.id):
            database.initialize_task_executions(run_id)
        database.start_task_execution(self.research.id, "T1")
        database.complete_task_execution(self.research.id, "T1", "结果", [])
        self.assertEqual(database.load_task_execution(other.id, "T1").status, "pending")


if __name__ == "__main__":
    unittest.main()
