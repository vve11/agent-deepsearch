"""子任务检查点：独立消息、共享证据、原子保存以及旧接口兼容性。"""
import asyncio
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import agent
import database
from checkpoint import AgentCheckpoint, TaskCheckpoint
from evidence import EvidenceStore
from plan import parse_plan
from run import ResearchRun
from test_checkpoints import client_with, response


class TaskCheckpointTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(database, "DB_PATH", Path(directory.name) / "test.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_db()
        self.research = ResearchRun(question="比较语言")
        database.save_run(self.research)
        self.plan = parse_plan(self.research.question, {"tasks": [
            {"question": "研究 Python"}, {"question": "研究 JavaScript"},
        ]})
        database.save_plan(self.research.id, self.plan)
        database.initialize_task_executions(self.research.id)
        database.start_task_execution(self.research.id, "T1")

    def checkpoint(self, task_id="T1", evidence=None):
        question = next(task.question for task in self.plan.tasks if task.id == task_id)
        return TaskCheckpoint(
            run_id=self.research.id, task_id=task_id, next_step=0, tool_count=0,
            messages=[{"role": "system", "content": "研究助手"},
                      {"role": "user", "content": question}],
            evidence=deepcopy(evidence or []), model_name="test-model",
        )

    def evidence(self, count=1):
        return [{"id": f"E{i}", "title": f"资料{i}", "url": f"https://example.com/{i}",
                 "content": f"内容{i}"} for i in range(1, count + 1)]

    def after_search(self, checkpoint):
        messages = deepcopy(checkpoint.messages) + [
            {"role": "assistant", "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "search_web", "arguments": '{"query":"Python"}'},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "content": json.dumps(self.evidence())},
        ]
        return replace(checkpoint, next_step=1, tool_count=1, messages=messages, evidence=self.evidence())

    def execute_sql(self, sql, values=()):
        with closing(sqlite3.connect(database.DB_PATH)) as connection:
            with connection:
                connection.execute(sql, values)

    def test_round_trip_idempotence_and_input_snapshot(self):
        checkpoint = self.checkpoint(evidence=self.evidence())
        expected = deepcopy(checkpoint)
        database.save_task_checkpoint(checkpoint)
        events = database.load_events(self.research.id)
        database.save_task_checkpoint(checkpoint)
        self.assertEqual(database.load_events(self.research.id), events)
        checkpoint.messages[0]["content"] = "改变"
        checkpoint.evidence[0]["content"] = "改变"
        self.assertEqual(database.load_task_checkpoint(self.research.id, "T1"), expected)
        self.assertEqual(database.load_research_evidence(self.research.id), expected.evidence)

    def test_tasks_keep_independent_snapshots_and_global_ids(self):
        first = self.checkpoint(evidence=self.evidence())
        database.save_task_checkpoint(first)
        database.complete_task_execution(self.research.id, "T1", "结果 [E1]", ["E1"])
        database.start_task_execution(self.research.id, "T2")
        second = self.checkpoint("T2", database.load_research_evidence(self.research.id))
        database.save_task_checkpoint(second)
        second = replace(second, evidence=self.evidence(2))
        database.save_task_checkpoint(second)
        self.assertEqual(database.load_task_checkpoint(self.research.id, "T1"), first)
        self.assertEqual(database.load_task_checkpoint(self.research.id, "T2"), second)
        self.assertEqual(database.load_research_evidence(self.research.id), self.evidence(2))

    def test_shared_evidence_cannot_be_rewritten_or_truncated(self):
        saved = self.checkpoint(evidence=self.evidence())
        database.save_task_checkpoint(saved)
        changed = self.evidence()
        changed[0]["content"] = "改写已有资料"
        for records in ([], changed):
            with self.subTest(records=records), self.assertRaises(ValueError):
                database.save_task_checkpoint(replace(saved, evidence=records))
        self.assertEqual(database.load_research_evidence(self.research.id), self.evidence())

    def test_history_budget_and_model_cannot_be_reset(self):
        initial = self.checkpoint()
        database.save_task_checkpoint(initial)
        advanced = self.after_search(initial)
        database.save_task_checkpoint(advanced)
        rewritten = deepcopy(advanced.messages)
        rewritten[0]["content"] = "替换系统提示"
        for changed in (replace(initial, evidence=self.evidence()),
                        replace(advanced, messages=rewritten),
                        replace(advanced, model_name="other"),
                        replace(advanced, tool_count=0)):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                database.save_task_checkpoint(changed)
        self.assertEqual(database.load_task_checkpoint(self.research.id, "T1"), advanced)

    def test_wrong_owner_question_status_and_incomplete_round_rejected(self):
        checkpoint = self.checkpoint()
        wrong_messages = deepcopy(checkpoint.messages)
        wrong_messages[1]["content"] = "另一个问题"
        incomplete = self.after_search(checkpoint)
        incomplete.messages.pop()
        for changed in (replace(checkpoint, task_id="T99"),
                        replace(checkpoint, run_id="0" * 32),
                        replace(checkpoint, messages=wrong_messages), incomplete,
                        replace(checkpoint, version=2), replace(checkpoint, task_id="T0")):
            with self.subTest(changed=changed), self.assertRaises((ValueError, LookupError)):
                database.save_task_checkpoint(changed)
        with self.assertRaises(ValueError):
            database.save_task_checkpoint(self.checkpoint("T2"))  # pending
        database.fail_task_execution(self.research.id, "T1", "失败")
        with self.assertRaises(ValueError):
            database.save_task_checkpoint(checkpoint)

    def test_event_failure_rolls_back_first_save_and_overwrite(self):
        initial = self.checkpoint()
        append_event = database._append_event

        def fail(*args):
            append_event(*args)
            raise sqlite3.OperationalError("事件故障")

        events = database.load_events(self.research.id)
        with patch.object(database, "_append_event", side_effect=fail):
            with self.assertRaises(sqlite3.OperationalError):
                database.save_task_checkpoint(initial)
        with self.assertRaises(LookupError):
            database.load_task_checkpoint(self.research.id, "T1")
        with self.assertRaises(LookupError):
            database.load_research_evidence(self.research.id)
        self.assertEqual(database.load_events(self.research.id), events)
        database.save_task_checkpoint(initial)
        events = database.load_events(self.research.id)
        with patch.object(database, "_append_event", side_effect=fail):
            with self.assertRaises(sqlite3.OperationalError):
                database.save_task_checkpoint(self.after_search(initial))
        self.assertEqual(database.load_task_checkpoint(self.research.id, "T1"), initial)
        self.assertEqual(database.load_research_evidence(self.research.id), [])
        self.assertEqual(database.load_events(self.research.id), events)

    def test_corrupt_owner_and_missing_evidence_are_rejected(self):
        checkpoint = self.checkpoint()
        database.save_task_checkpoint(checkpoint)
        corrupted = asdict(checkpoint)
        corrupted["task_id"] = "T2"
        self.execute_sql("UPDATE task_checkpoints SET state_json = ?", (json.dumps(corrupted),))
        with self.assertRaises(ValueError):
            database.load_task_checkpoint(self.research.id, "T1")
        self.execute_sql("UPDATE task_checkpoints SET state_json = ?", (json.dumps(asdict(checkpoint)),))
        self.execute_sql("DELETE FROM research_evidence")
        with self.assertRaises(LookupError):
            database.load_task_checkpoint(self.research.id, "T1")
        with self.assertRaises(ValueError):
            database.save_task_checkpoint(checkpoint)

    def test_finish_rejects_unpersisted_evidence(self):
        database.save_task_checkpoint(self.checkpoint(evidence=self.evidence()))
        events = database.load_events(self.research.id)
        with self.assertRaises(ValueError):
            database.complete_task_execution(self.research.id, "T1", "结果", ["E2"])
        with self.assertRaises(ValueError):
            database.fail_task_execution(self.research.id, "T1", "错误", ["E2"])
        self.assertEqual(database.load_task_execution(self.research.id, "T1").status, "running")
        self.assertEqual(database.load_events(self.research.id), events)
        database.complete_task_execution(self.research.id, "T1", "结果 [E1]", ["E1"])
        with self.assertRaises(ValueError):
            database.save_task_checkpoint(self.checkpoint(evidence=self.evidence()))

    def test_legacy_checkpoint_survives_schema_upgrade(self):
        legacy = ResearchRun(question="旧问题")
        database.save_run(legacy)
        checkpoint = AgentCheckpoint(
            run_id=legacy.id, next_step=0, tool_count=0, model_name="old-model",
            messages=[{"role": "system", "content": "系统"},
                      {"role": "user", "content": legacy.question}], evidence=[],
        )
        database.save_checkpoint(checkpoint)
        self.execute_sql("DROP TABLE task_checkpoints")
        self.execute_sql("DROP TABLE research_evidence")
        database.init_db()
        self.assertEqual(database.load_checkpoint(legacy.id), checkpoint)
        with self.assertRaises(ValueError):
            database.save_checkpoint(self.checkpoint())

    def test_resume_agent_uses_saved_round_without_searching_again(self):
        checkpoint = self.after_search(self.checkpoint())
        database.save_task_checkpoint(checkpoint)
        loaded = database.load_task_checkpoint(self.research.id, "T1")
        client = client_with(response(text="结论 [E1]"))
        with patch.object(agent, "execute_tool") as tool:
            answer = asyncio.run(agent.run_agent(
                client, self.plan.tasks[0].question, EvidenceStore.from_records(loaded.evidence),
                lambda *args: None, checkpoint=loaded,
            ))
        tool.assert_not_called()
        self.assertIn("结论 [E1]", answer)
        arguments = client.chat.completions.create.call_args.kwargs
        self.assertEqual(arguments["messages"], loaded.messages)
        self.assertEqual(arguments["model"], loaded.model_name)

    def test_new_process_loads_snapshot_without_original_objects(self):
        checkpoint = self.after_search(self.checkpoint())
        database.save_task_checkpoint(checkpoint)
        script = (
            "import json, sys; from pathlib import Path; from dataclasses import asdict; "
            "import database; database.DB_PATH=Path(sys.argv[1]); "
            "print(json.dumps(asdict(database.load_task_checkpoint(sys.argv[2], 'T1'))))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script, str(database.DB_PATH), self.research.id],
            cwd=Path(database.__file__).parent, capture_output=True, text=True,
            check=True, timeout=20,
        )
        self.assertEqual(TaskCheckpoint.from_dict(json.loads(completed.stdout)), checkpoint)

    def test_same_task_id_different_research_has_separate_evidence(self):
        first = self.checkpoint(evidence=self.evidence())
        database.save_task_checkpoint(first)
        other = ResearchRun(question=self.research.question)
        database.save_run(other)
        database.save_plan(other.id, self.plan)
        database.initialize_task_executions(other.id)
        database.start_task_execution(other.id, "T1")
        second = replace(self.checkpoint(), run_id=other.id)
        database.save_task_checkpoint(second)
        self.assertEqual(database.load_research_evidence(other.id), [])
        self.assertEqual(database.load_research_evidence(self.research.id), self.evidence())
        self.assertEqual(database.load_task_checkpoint(self.research.id, "T1"), first)


if __name__ == "__main__":
    unittest.main()
