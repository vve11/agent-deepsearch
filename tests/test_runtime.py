"""用临时数据库验证生命周期和 CLI；不调用真实模型。"""
import contextlib
import io
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cli
import database
import runtime
from plan import parse_plan


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        db_patch = patch.object(database, "DB_PATH", Path(directory.name) / "test.db")
        db_patch.start()
        self.addCleanup(db_patch.stop)
        database.init_db()
        # These tests isolate research/recovery; planner has dedicated integration tests.
        async def planned(client, question, record):
            return parse_plan(question, {"tasks": [{"question": question}]})
        planner_patch = patch.object(runtime, "create_plan", side_effect=planned)
        planner_patch.start()
        self.addCleanup(planner_patch.stop)

    def saved_run(self):
        with contextlib.closing(sqlite3.connect(database.DB_PATH)) as connection:
            run_id = connection.execute("SELECT id FROM runs").fetchone()[0]
        return database.load_run(run_id)

    async def test_success_saves_answer_evidence_and_events(self):
        async def agent(client, question, store, record, **kwargs):
            self.assertEqual(question, "Research")
            store.add({"title": "Source", "url": "https://example.com", "content": "Fact"})
            record("model_started", {"step": 1})
            return "Answer [E1]"

        with patch.object(runtime, "run_agent", side_effect=agent):
            run = await runtime.execute_research(object(), "Research")
        loaded = database.load_run(run.id)
        self.assertEqual(loaded, run)
        self.assertEqual(run.status, "completed")
        self.assertEqual(run.answer, "Answer [E1]")
        self.assertEqual(run.evidence[0]["id"], "E1")
        self.assertEqual([e["event_type"] for e in database.load_events(run.id)],
                         ["run_started", "plan_saved", "model_started", "run_completed"])

    async def test_failure_preserves_evidence_and_original_exception(self):
        failure = ValueError("Research failed")

        async def agent(client, question, store, record, **kwargs):
            store.add({"title": "Partial", "url": "https://example.com", "content": "Fact"})
            raise failure

        with patch.object(runtime, "run_agent", side_effect=agent):
            with self.assertRaises(ValueError) as caught:
                await runtime.execute_research(object(), "Research")
        self.assertIs(caught.exception, failure)
        run = self.saved_run()
        self.assertEqual(run.status, "failed")
        self.assertIsNone(run.answer)
        self.assertEqual(len(run.evidence), 1)
        self.assertEqual(run.error, "ValueError: Research failed")
        self.assertEqual(database.load_events(run.id)[-1]["event_type"], "run_failed")

    async def test_startup_save_failure_does_not_run_agent(self):
        with patch.object(runtime, "save_run_with_event", side_effect=sqlite3.OperationalError("Unavailable")), \
             patch.object(runtime, "run_agent", new_callable=AsyncMock) as agent:
            with self.assertRaises(sqlite3.OperationalError):
                await runtime.execute_research(object(), "Research")
            agent.assert_not_awaited()

    async def test_final_save_failure_does_not_mask_research_error(self):
        failure = ValueError("Original error")
        calls = 0

        def save(run, event_type, payload, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise sqlite3.OperationalError("Final save failed")
            database.save_run_with_event(run, event_type, payload, **kwargs)

        with patch.object(runtime, "save_run_with_event", side_effect=save), \
             patch.object(runtime, "run_agent", side_effect=failure), \
             self.assertLogs("runtime", level="WARNING") as logs:
            with self.assertRaises(ValueError) as caught:
                await runtime.execute_research(object(), "Research")
        self.assertIs(caught.exception, failure)
        self.assertIn("Final save failed", logs.output[0])

    async def test_cli_queries_history_without_starting_research(self):
        with patch.object(runtime, "run_agent", return_value="Saved answer"):
            run = await runtime.execute_research(object(), "Research")
        output = io.StringIO()
        with patch("builtins.input", side_effect=[
            "/show " + run.id, "/events " + run.id, "/events", "/unknown", "/exit",
        ]), patch.object(cli, "execute_research", new_callable=AsyncMock) as execute, \
             contextlib.redirect_stdout(output):
            await cli.run_cli(object())
            execute.assert_not_awaited()
        self.assertIn("Saved answer", output.getvalue())
        self.assertIn("run_completed", output.getvalue())

    async def test_cli_research_error_allows_next_question(self):
        output = io.StringIO()
        with patch.object(runtime, "run_agent", side_effect=[ValueError("Bad answer"), "Good answer"]) as agent, \
             patch("builtins.input", side_effect=["First", "Second", "/exit"]), \
             contextlib.redirect_stdout(output):
            await cli.run_cli(object())
        self.assertEqual(agent.await_count, 2)
        self.assertIn("Bad answer", output.getvalue())
        self.assertIn("Good answer", output.getvalue())


if __name__ == "__main__":
    unittest.main()
