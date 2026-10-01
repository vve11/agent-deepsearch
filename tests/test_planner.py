"""独立验证规划器；模型请求和等待均模拟，不使用密钥。"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from openai import APIStatusError

from planner import create_plan


def reply(text, finish_reason="stop", tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason,
        message=SimpleNamespace(content=text, tool_calls=tool_calls),
    )])


def service_error(status):
    response = httpx.Response(status, request=httpx.Request("POST", "https://example.com"))
    return APIStatusError("模拟服务错误", response=response, body=None)


class PlannerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.create = AsyncMock()
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        self.events = []
        self.valid_text = json.dumps({"tasks": [
            {"question": "电池回收有哪些主要技术？"},
            {"question": "不同电池回收技术的成本如何？"},
        ]}, ensure_ascii=False)

    def record(self, event_type, payload):
        self.events.append((event_type, payload))

    async def plan(self):
        return await create_plan(self.client, " 比较电池回收技术及成本 ", self.record)

    async def test_valid_plan_and_request_contract(self):
        self.create.return_value = reply(self.valid_text)
        with patch.dict("os.environ", {"DEEPSEEK_MODEL": "configured-model"}):
            plan = await self.plan()
        self.assertEqual(plan.question, "比较电池回收技术及成本")
        self.assertEqual([task.id for task in plan.tasks], ["T1", "T2"])
        self.assertEqual(plan.tasks[1].question, "不同电池回收技术的成本如何？")
        self.assertEqual(self.events, [("plan_started", {}), ("plan_generated", {"task_count": 2})])
        self.create.assert_awaited_once()
        request = self.create.await_args.kwargs
        self.assertEqual(request["model"], "configured-model")
        self.assertEqual(request["messages"][1], {"role": "user", "content": plan.question})
        self.assertNotIn("tools", request)

    async def test_invalid_input_never_calls_model(self):
        for question in [None, 123, "", " \n "]:
            with self.subTest(question=question), self.assertRaises(ValueError):
                await create_plan(self.client, question, self.record)
        self.create.assert_not_awaited()
        self.assertEqual(self.events, [])

    async def test_invalid_json_or_schema_fails_without_repair_request(self):
        for text in ["not json", '```json\n{"tasks": []}\n```', "null", "[]",
                     '{"tasks": []}', '{"tasks": [{"question": "A"}, {"question": " a "}]}',
                     '{"tasks": [{"id": "T9", "question": "A"}]}']:
            with self.subTest(text=text):
                self.events.clear()
                self.create.reset_mock()
                self.create.return_value = reply(text)
                with self.assertRaises(ValueError):
                    await self.plan()
                self.create.assert_awaited_once()
                self.assertEqual(self.events, [("plan_started", {}), ("plan_failed", {"error_type": "ValueError"})])

    async def test_incomplete_empty_and_tool_responses_rejected(self):
        for response in [reply(self.valid_text, "length"), reply(""), reply(None),
                         reply(self.valid_text, tool_calls=[object()]), SimpleNamespace(choices=[])]:
            with self.subTest(response=response):
                self.events.clear()
                self.create.return_value = response
                with self.assertRaises(ValueError):
                    await self.plan()
                self.assertEqual(self.events[-1][0], "plan_failed")

    async def test_transient_request_retries_in_planning_phase(self):
        self.create.side_effect = [service_error(503), reply(self.valid_text)]
        with patch("model_requests.asyncio.sleep", new_callable=AsyncMock) as sleep:
            plan = await self.plan()
        self.assertEqual(len(plan.tasks), 2)
        self.assertEqual(self.create.await_count, 2)
        sleep.assert_awaited_once_with(1)
        self.assertEqual([name for name, _ in self.events], ["plan_started", "model_retrying", "plan_generated"])
        self.assertEqual(self.events[1][1]["phase"], "planning")

    async def test_authentication_failure_preserves_exception(self):
        failure = service_error(401)
        self.create.side_effect = failure
        with self.assertRaises(APIStatusError) as caught:
            await self.plan()
        self.assertIs(caught.exception, failure)
        self.create.assert_awaited_once()
        self.assertEqual(self.events[-1], ("plan_failed", {"error_type": "APIStatusError"}))

    async def test_exhausted_retries_record_one_plan_failure(self):
        failure = service_error(503)
        self.create.side_effect = failure
        with patch("model_requests.asyncio.sleep", new_callable=AsyncMock):
            with self.assertRaises(APIStatusError) as caught:
                await self.plan()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.create.await_count, 3)
        self.assertEqual([name for name, _ in self.events],
                         ["plan_started", "model_retrying", "model_retrying", "plan_failed"])
