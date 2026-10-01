"""计划持久化：真实临时 SQLite，包含失败回滚和固定编号验证。"""
import json
import sqlite3
import tempfile
import unittest
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import database
from plan import ResearchTask, parse_plan
from run import ResearchRun


class PlanStorageTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(database, "DB_PATH", Path(directory.name) / "test.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_db()
        self.run = ResearchRun(question="电池回收研究")
        database.save_run_with_event(self.run, "run_started", {})
        self.plan = parse_plan(self.run.question, {"tasks": [
            {"question": "有哪些技术路线？"}, {"question": "成本如何？"},
        ]})

    def test_round_trip_restores_nested_objects(self):
        database.save_plan(self.run.id, self.plan)
        loaded = database.load_plan(self.run.id)
        self.assertEqual(loaded, self.plan)
        self.assertIsNot(loaded, self.plan)
        self.assertIsInstance(loaded.tasks[0], ResearchTask)
        self.assertEqual(loaded.tasks[1].id, "T2")
        self.assertEqual(database.load_run(self.run.id), self.run)
        event = database.load_events(self.run.id)[-1]
        self.assertEqual(event["event_type"], "plan_saved")
        self.assertEqual(event["payload"], {"task_count": 2, "task_ids": ["T1", "T2"]})

    def test_duplicate_save_is_noop_and_different_plan_cannot_overwrite(self):
        database.save_plan(self.run.id, self.plan)
        events = database.load_events(self.run.id)
        database.save_plan(self.run.id, self.plan)
        changed = parse_plan(self.run.question, {"tasks": [{"question": "新的子问题"}]})
        with self.assertRaisesRegex(ValueError, "不能覆盖"):
            database.save_plan(self.run.id, changed)
        self.assertEqual(database.load_events(self.run.id), events)
        self.assertEqual(database.load_plan(self.run.id), self.plan)

    def test_event_failure_rolls_back_plan_but_keeps_parent(self):
        with closing(sqlite3.connect(database.DB_PATH)) as connection:
            connection.execute("""CREATE TRIGGER reject_plan_event BEFORE INSERT ON run_events
                WHEN NEW.event_type = 'plan_saved'
                BEGIN SELECT RAISE(ABORT, 'event failure'); END""")
        with self.assertRaises(sqlite3.IntegrityError):
            database.save_plan(self.run.id, self.plan)
        with self.assertRaises(LookupError):
            database.load_plan(self.run.id)
        self.assertEqual(database.load_run(self.run.id), self.run)
        self.assertEqual([e["event_type"] for e in database.load_events(self.run.id)], ["run_started"])

    def test_missing_parent_and_missing_plan(self):
        with self.assertRaises(LookupError):
            database.save_plan("0" * 32, self.plan)
        with self.assertRaises(LookupError):
            database.load_plan(self.run.id)

    def test_mismatched_question_and_invalid_ids_rejected_on_save(self):
        other = parse_plan("其他问题", {"tasks": [{"question": "子问题"}]})
        with self.assertRaises(ValueError):
            database.save_plan(self.run.id, other)
        self.plan.tasks[0].id = "T99"
        with self.assertRaises(ValueError):
            database.save_plan(self.run.id, self.plan)
        with self.assertRaises(LookupError):
            database.load_plan(self.run.id)

    def test_corrupt_saved_data_is_rejected_not_renumbered(self):
        database.save_plan(self.run.id, self.plan)
        wrong_id = asdict(self.plan)
        wrong_id["tasks"][0]["id"] = "T99"
        wrong_question = asdict(self.plan)
        wrong_question["question"] = "其他研究"
        for text in ["invalid json", "null", "{}", json.dumps(wrong_id), json.dumps(wrong_question),
                     '{"question":"电池回收研究","tasks":[{"id":"T1","question":null}]}']:
            with self.subTest(text=text):
                with closing(sqlite3.connect(database.DB_PATH)) as connection:
                    with connection:
                        connection.execute("UPDATE research_plans SET plan_json = ? WHERE run_id = ?",
                                           (text, self.run.id))
                with self.assertRaises(ValueError):
                    database.load_plan(self.run.id)

    def test_schema_upgrade_preserves_existing_records(self):
        # 模拟旧库没有计划表，重跑初始化只添加新表。
        with closing(sqlite3.connect(database.DB_PATH)) as connection:
            connection.execute("DROP TABLE research_plans")
        database.init_db()
        database.init_db()
        self.assertEqual(database.load_run(self.run.id), self.run)
        self.assertEqual(len(database.load_events(self.run.id)), 1)
        database.save_plan(self.run.id, self.plan)
        self.assertEqual(database.load_plan(self.run.id), self.plan)

    def test_different_runs_may_each_have_t1(self):
        second = ResearchRun(question=self.run.question)
        database.save_run(second)
        database.save_plan(self.run.id, self.plan)
        database.save_plan(second.id, self.plan)
        self.assertEqual(database.load_plan(second.id).tasks[0].id, "T1")
        self.assertEqual(len(database.load_events(second.id)), 1)
