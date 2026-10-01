"""计划解析的输入边界验证，不调用模型或数据库。"""
import json
import unittest
from dataclasses import asdict

from plan import ResearchPlan, parse_plan


class PlanTests(unittest.TestCase):
    def test_valid_plan_has_program_assigned_ids_and_serializes(self):
        data = {"tasks": [{"question": " 技术路线 "}, {"question": "成本因素"},
                          {"question": "环保影响"}]}
        plan = parse_plan(" 电池回收研究 ", data)
        self.assertIsInstance(plan, ResearchPlan)
        self.assertEqual(plan.question, "电池回收研究")
        self.assertEqual([task.id for task in plan.tasks], ["T1", "T2", "T3"])
        self.assertEqual(plan.tasks[0].question, "技术路线")
        self.assertEqual(data["tasks"][0]["question"], " 技术路线 ")
        stored = json.loads(json.dumps(asdict(plan), ensure_ascii=False))
        self.assertEqual(stored["tasks"][0], {"id": "T1", "question": "技术路线"})

    def test_single_task_and_300_character_boundary(self):
        plan = parse_plan("问题", {"tasks": [{"question": "字" * 300}]})
        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.tasks[0].id, "T1")
        with self.assertRaises(ValueError):
            parse_plan("问题", {"tasks": [{"question": "字" * 301}]})

    def test_invalid_root_and_task_count(self):
        for data in [None, [], "{}", {}, {"tasks": [], "extra": 1},
                     {"tasks": None}, {"tasks": {}}, {"tasks": []},
                     {"tasks": [{"question": str(i)} for i in range(4)]}]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_plan("问题", data)

    def test_invalid_task_fields_and_types(self):
        for item in [None, "问题", {}, {"question": None}, {"question": 1},
                     {"question": True}, {"question": " \n\t"},
                     {"question": "问题", "id": "T99"}]:
            with self.subTest(item=item), self.assertRaises(ValueError):
                parse_plan("问题", {"tasks": [item]})

    def test_duplicates_ignore_case_and_whitespace(self):
        for questions in [("技术路线", " 技术路线 "), ("Battery Cost", "battery\t cost")]:
            with self.subTest(questions=questions), self.assertRaises(ValueError):
                parse_plan("问题", {"tasks": [{"question": q} for q in questions]})

    def test_invalid_parent_question(self):
        for question in [None, 123, "", " \n "]:
            with self.subTest(question=question), self.assertRaises(ValueError):
                parse_plan(question, {"tasks": [{"question": "子问题"}]})


if __name__ == "__main__":
    unittest.main()
