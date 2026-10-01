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
from checkpoint import AgentCheckpoint
from run import ResearchRun


class RecoveryMenuTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        patcher = patch.object(database, "DB_PATH", Path(directory.name) / "test.db")
        patcher.start()
        self.addCleanup(patcher.stop)
        database.init_db()

    def add_run(self, title, status="failed"):
        run = ResearchRun(question=title, status=status)
        database.save_run_with_event(run, "run_started", {})
        checkpoint = AgentCheckpoint(run.id, 0, 0, [
            {"role": "system", "content": "Assistant"}, {"role": "user", "content": title},
        ], [], "test-model")
        database.save_checkpoint(checkpoint)
        return run

    async def drive_cli(self, commands):
        output = io.StringIO()
        with patch("builtins.input", side_effect=commands), \
             patch.object(cli, "resume_research", new_callable=AsyncMock) as resume, \
             patch.object(cli, "execute_research", new_callable=AsyncMock) as execute, \
             contextlib.redirect_stdout(output):
            resume.return_value = ResearchRun(question="Recovered", answer="Done")
            await cli.run_cli(object())
        return output.getvalue(), resume, execute

    async def test_startup_and_picker_use_title_and_validate_selection(self):
        first = self.add_run("电池研究")
        self.add_run("光伏研究")
        output, resume, execute = await self.drive_cli(["/resume", "bad", "9", "2", "/exit"])
        self.assertIn("发现 2 个可恢复任务", output)
        self.assertIn("1. 光伏研究", output)
        self.assertIn("2. 电池研究", output)
        self.assertEqual(resume.await_args.args[1], first.id)
        execute.assert_not_awaited()

    async def test_continue_uses_most_recent_activity(self):
        first = self.add_run("Older task")
        self.add_run("Newer task")
        database.append_event(first.id, "run_failed", {})
        _, resume, _ = await self.drive_cli(["/continue", "/exit"])
        self.assertEqual(resume.await_args.args[1], first.id)

    async def test_cancel_empty_list_unknown_and_old_id_form(self):
        _, resume, execute = await self.drive_cli(["/resume", "/continue", "/continue extra", "/exit"])
        resume.assert_not_awaited()
        execute.assert_not_awaited()
        run = self.add_run("Task")
        _, resume, _ = await self.drive_cli(["/resume", "0", "/resume " + run.id, "/exit"])
        resume.assert_awaited_once()
        self.assertEqual(resume.await_args.args[1], run.id)

    def test_candidates_exclude_completed_corrupt_missing_and_active(self):
        good = self.add_run("Good")
        self.add_run("Completed", "completed")
        corrupt = self.add_run("Corrupt")
        old = ResearchRun(question="No checkpoint")
        database.save_run(old)
        active = self.add_run("Active", "running")
        with contextlib.closing(sqlite3.connect(database.DB_PATH)) as connection:
            with connection:
                connection.execute("UPDATE checkpoints SET state_json = ? WHERE run_id = ?", ("{}", corrupt.id))
        with patch.object(runtime, "_active_runs", {active.id}):
            self.assertEqual([r["id"] for r in runtime.get_resumable_runs()], [good.id])
