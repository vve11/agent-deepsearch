"""重试次数、错误分类和搜索不重放；等待与模型均模拟。"""
import asyncio
import ssl
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError

import agent
from evidence import EvidenceStore
from model_requests import request_model
from test_checkpoints import response


def status_error(code, body=None, headers=None):
    reply = httpx.Response(code, headers=headers, request=httpx.Request("POST", "https://example.com"))
    return APIStatusError("Simulated failure", response=reply, body=body)


class ModelRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.create = AsyncMock()
        self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.create)))
        self.events = []
        self.sleep = AsyncMock()
        patcher = patch("model_requests.asyncio.sleep", self.sleep)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def request(self):
        return await request_model(
            self.client, lambda name, payload: self.events.append((name, payload)),
            {"phase": "research", "step": 2}, model="test", messages=[],
        )

    async def test_transient_errors_then_success(self):
        self.create.side_effect = [APITimeoutError(httpx.Request("POST", "https://example.com")),
                                   status_error(503), "answer"]
        self.assertEqual(await self.request(), "answer")
        self.assertEqual(self.create.await_count, 3)
        self.assertEqual([c.args[0] for c in self.sleep.await_args_list], [1, 2])
        self.assertEqual([p["retry_number"] for _, p in self.events], [1, 2])
        self.assertTrue(all(p["step"] == 2 for _, p in self.events))
        self.assertEqual(self.create.await_args_list[0], self.create.await_args_list[2])

    async def test_exhaustion_preserves_last_exception(self):
        failure = status_error(503)
        self.create.side_effect = failure
        with self.assertRaises(APIStatusError) as caught:
            await self.request()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.create.await_count, 3)
        self.assertEqual(self.sleep.await_count, 2)

    async def test_permanent_and_programming_errors_do_not_retry(self):
        for failure in [status_error(400), status_error(401), status_error(402), status_error(403),
                        status_error(429, {"error": {"code": "insufficient_quota"}}),
                        ValueError("Invalid data")]:
            with self.subTest(failure=failure):
                self.create.reset_mock()
                self.create.side_effect = failure
                with self.assertRaises(type(failure)):
                    await self.request()
                self.assertEqual(self.create.await_count, 1)
        self.sleep.assert_not_awaited()

    async def test_retry_after_is_respected_or_stops_when_too_long(self):
        self.create.side_effect = [status_error(429, headers={"Retry-After": "5"}), "ok"]
        self.assertEqual(await self.request(), "ok")
        self.sleep.assert_awaited_once_with(5)
        self.sleep.reset_mock()
        self.create.reset_mock()
        self.create.side_effect = status_error(429, headers={"Retry-After": "120"})
        with self.assertRaises(APIStatusError):
            await self.request()
        self.sleep.assert_not_awaited()
        self.assertEqual(self.create.await_count, 1)

    async def test_certificate_failure_does_not_retry(self):
        failure = APIConnectionError(request=httpx.Request("POST", "https://example.com"))
        failure.__cause__ = ssl.SSLCertVerificationError("bad certificate")
        self.create.side_effect = failure
        with self.assertRaises(APIConnectionError):
            await self.request()
        self.sleep.assert_not_awaited()

    async def test_cancel_during_wait_stops_requests(self):
        self.create.side_effect = status_error(503)
        self.sleep.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.request()
        self.assertEqual(self.create.await_count, 1)

    async def test_model_retry_after_search_never_replays_tool(self):
        self.create.side_effect = [response(queries=["test"]), status_error(503), response("Answer [E1]")]
        with patch.object(agent, "execute_tool", return_value={"result": [
            {"title": "Title", "url": "https://example.com", "content": "Fact"},
        ]}) as tool:
            answer = await agent.run_agent(self.client, "Question", EvidenceStore(),
                                          lambda name, payload: self.events.append((name, payload)))
        self.assertIn("Answer [E1]", answer)
        tool.assert_awaited_once()
        self.assertEqual(self.create.await_count, 3)
        self.assertEqual(sum(name == "model_retrying" for name, _ in self.events), 1)

    async def test_citation_repair_request_also_retries(self):
        store = EvidenceStore()
        store.add({"title": "Title", "url": "https://example.com", "content": "Fact"})
        self.create.side_effect = [status_error(503), response("Fixed [E1]")]
        answer = await agent.validate_or_repair_answer(
            self.client, [], "Wrong [E99]", store,
            lambda name, payload: self.events.append((name, payload)),
        )
        self.assertEqual(answer, "Fixed [E1]")
        event = next(p for name, p in self.events if name == "model_retrying")
        self.assertEqual(event["phase"], "citation_repair")
        self.assertEqual(self.create.call_args.kwargs["tool_choice"], "none")
