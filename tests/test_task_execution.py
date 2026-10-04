import unittest

from task_execution import TaskExecution, validate_task_execution


class TaskExecutionTests(unittest.TestCase):
    def test_valid_states(self):
        """四种状态都可以有合法的记录。"""
        cases = [
            TaskExecution(run_id="run001", task_id="T1"),
            TaskExecution(
                run_id="run001",
                task_id="T1",
                status="running",
                evidence_ids=["E1"],
            ),
            TaskExecution(
                run_id="run001",
                task_id="T1",
                status="completed",
                answer="研究结果见 [E1]",
                evidence_ids=["E1"],
            ),
            TaskExecution(
                run_id="run001",
                task_id="T1",
                status="failed",
                error="搜索请求超时",
                evidence_ids=["E1"],
            ),
        ]

        for task in cases:
            with self.subTest(status=task.status):
                validate_task_execution(task)

    def test_inconsistent_states_are_rejected(self):
        """字段类型正确，也可能在含义上互相矛盾。"""
        cases = [
            {"status": "completed"},
            {"status": "completed", "answer": "结果", "error": "失败"},
            {"status": "failed"},
            {"status": "failed", "error": "超时", "answer": "结果"},
            {"status": "pending", "answer": "结果"},
            {"status": "pending", "evidence_ids": ["E1"]},
            {"status": "running", "error": "超时"},
        ]

        for fields in cases:
            with self.subTest(fields=fields):
                task = TaskExecution(
                    run_id="run001", task_id="T1", **fields
                )
                with self.assertRaises(ValueError):
                    validate_task_execution(task)

    def test_invalid_fields_are_rejected(self):
        """检查错误的类型、空文本和编号格式。"""
        cases = [
            {"run_id": "   "},
            {"run_id": 123},
            {"task_id": "T0"},
            {"task_id": "T01"},
            {"task_id": None},
            {"status": "unknown"},
            {"answer": ""},
            {"answer": 123},
            {"error": "   "},
            {"error": 123},
        ]

        for fields in cases:
            with self.subTest(fields=fields):
                values = {"run_id": "run001", "task_id": "T1"}
                values.update(fields)
                task = TaskExecution(**values)

                with self.assertRaises(ValueError):
                    validate_task_execution(task)

    def test_invalid_evidence_ids_are_rejected(self):
        """证据编号必须是列表，格式正确，而且不能重复。"""
        cases = [
            "E1",
            None,
            ["E0"],
            ["E01"],
            ["T1"],
            [123],
            ["E1", "E1"],
        ]

        for evidence_ids in cases:
            with self.subTest(evidence_ids=evidence_ids):
                task = TaskExecution(
                    run_id="run001",
                    task_id="T1",
                    status="running",
                    evidence_ids=evidence_ids,
                )
                with self.assertRaises(ValueError):
                    validate_task_execution(task)

    def test_evidence_lists_are_independent(self):
        """修改 T1 的列表，不能影响 T2。"""
        first = TaskExecution(run_id="run001", task_id="T1")
        second = TaskExecution(run_id="run001", task_id="T2")

        first.evidence_ids.append("E1")

        self.assertEqual(second.evidence_ids, [])


if __name__ == "__main__":
    unittest.main()