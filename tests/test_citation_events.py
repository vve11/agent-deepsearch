"""离线验证引用检查与修复事件；不调用 API，不修改正式数据库。"""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import database
from agent import run_agent, validate_or_repair_answer
from evidence import EvidenceStore
from run import ResearchRun


def make_response(text, finish_reason="stop", tool_calls=None):
    message = SimpleNamespace(content=text, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice])


class CitationEventTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.store = EvidenceStore()
        self.store.add({
            "title": "Example", "url": "https://example.com", "content": "Evidence",
        })
        self.events = []
        self.create = AsyncMock()
        self.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=self.create))
        )

    def record(self, event_type, payload):
        self.events.append({"type": event_type, "payload": payload})

    def event_types(self):
        return [event["type"] for event in self.events]

    async def check_answer(self, text="Draft [E99]"):
        return await validate_or_repair_answer(
            self.client, [], text, self.store, self.record,
        )

    async def test_valid_answer_needs_no_repair(self):
        self.assertEqual(await self.check_answer("Valid [E1]"), "Valid [E1]")
        self.assertEqual(self.event_types(), ["citation_check_passed"])
        self.assertEqual(self.events[0]["payload"]["stage"], "initial")
        self.create.assert_not_awaited()

    async def test_answer_without_citations_passes_only_existence_check(self):
        await self.check_answer("Hello")
        self.assertEqual(self.event_types(), ["citation_check_passed"])
        self.create.assert_not_awaited()

    async def test_successful_repair_event_order(self):
        self.create.return_value = make_response("Fixed [E1]")
        self.assertEqual(await self.check_answer(), "Fixed [E1]")
        self.assertEqual(self.event_types(), [
            "citation_check_failed", "repair_started", "model_started",
            "model_completed", "citation_check_passed", "repair_completed",
        ])
        self.assertEqual(self.events[0]["payload"]["unknown_ids"], ["E99"])
        self.assertEqual(self.events[4]["payload"]["stage"], "after_repair")
        self.assertEqual(self.events[2]["payload"]["phase"], "citation_repair")
        self.assertEqual(self.create.await_count, 1)
        self.assertEqual(self.create.call_args.kwargs["tool_choice"], "none")

    async def test_unknown_citation_after_repair_stops(self):
        self.create.return_value = make_response("Still wrong [E99]")
        with self.assertRaises(ValueError):
            await self.check_answer()
        self.assertEqual(self.event_types(), [
            "citation_check_failed", "repair_started", "model_started",
            "model_completed", "citation_check_failed", "repair_failed",
        ])
        self.assertEqual(self.create.await_count, 1)

    async def test_request_error_is_preserved(self):
        failure = RuntimeError("Simulated network failure")
        self.create.side_effect = failure
        with self.assertRaises(RuntimeError) as caught:
            await self.check_answer()
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.event_types(), [
            "citation_check_failed", "repair_started", "model_started",
            "model_failed", "repair_failed",
        ])
        self.assertEqual(self.create.await_count, 1)

    async def test_invalid_repair_responses_are_not_success(self):
        for response in [
            make_response("   "),
            make_response("Partial [E1]", finish_reason="length"),
            make_response(None, finish_reason="tool_calls", tool_calls=[object()]),
        ]:
            with self.subTest(response=response):
                self.events.clear()
                self.create.reset_mock()
                self.create.return_value = response
                with self.assertRaises(ValueError):
                    await self.check_answer()
                self.assertEqual(self.event_types(), [
                    "citation_check_failed", "repair_started", "model_started",
                    "model_completed", "repair_failed",
                ])
                self.assertEqual(self.create.await_count, 1)

    async def test_real_callback_persists_events_in_temporary_database(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(database, "DB_PATH", Path(directory) / "test.db"):
                database.init_db()
                run = ResearchRun(question="Offline integration check")
                database.save_run(run)

                def record(event_type, payload):
                    database.append_event(run.id, event_type, payload)

                # 没有搜索证据：原稿的 E99 非法，修复稿诚实说明证据不足。
                self.create.side_effect = [
                    make_response("Unsupported [E99]"),
                    make_response("Insufficient evidence."),
                ]
                answer = await run_agent(self.client, run.question, EvidenceStore(), record)
                self.assertEqual(answer, "Insufficient evidence.")
                events = database.load_events(run.id)
                self.assertEqual([event["event_type"] for event in events], [
                    "model_started", "model_completed", "citation_check_failed",
                    "repair_started", "model_started", "model_completed",
                    "citation_check_passed", "repair_completed",
                ])
                self.assertEqual(self.create.await_count, 2)


if __name__ == "__main__":
    unittest.main()
