"""顺序研究的阶段、总预算和汇总稿。旧单 Agent 表保持兼容。"""
import sqlite3
from contextlib import closing

import database as db
from budget import BudgetExceeded
from evidence import EvidenceStore


SCHEMA = """CREATE TABLE IF NOT EXISTS research_pipeline (
    run_id TEXT PRIMARY KEY NOT NULL REFERENCES runs(id),
    version INTEGER NOT NULL DEFAULT 1 CHECK(version = 1),
    phase TEXT NOT NULL DEFAULT 'planning'
        CHECK(phase IN ('planning', 'tasks', 'synthesis', 'completed')),
    model_name TEXT NOT NULL,
    model_limit INTEGER NOT NULL DEFAULT 30 CHECK(model_limit > 0),
    tool_limit INTEGER NOT NULL DEFAULT 9 CHECK(tool_limit >= 0),
    model_used INTEGER NOT NULL DEFAULT 0 CHECK(model_used BETWEEN 0 AND model_limit),
    tool_used INTEGER NOT NULL DEFAULT 0 CHECK(tool_used BETWEEN 0 AND tool_limit),
    report TEXT
)"""


def load_pipeline(run_id):
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM research_pipeline WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        return None  # 没有标记的历史任务继续用旧流程。
    state = dict(row)
    if state['version'] != 1 or state['phase'] not in {'planning', 'tasks', 'synthesis', 'completed'}:
        raise ValueError("不支持或损坏的顺序研究状态")
    if not isinstance(state['model_name'], str) or not state['model_name'].strip():
        raise ValueError("研究模型配置损坏")
    for kind in ('model', 'tool'):
        used, limit = state[kind + '_used'], state[kind + '_limit']
        if type(used) is not int or type(limit) is not int or not 0 <= used <= limit:
            raise ValueError("研究预算数据损坏")
    if state['model_limit'] < 1:
        raise ValueError("研究模型预算无效")
    if state['report'] is not None and (
        not isinstance(state['report'], str) or not state['report'].strip()
        or state['phase'] not in {'synthesis', 'completed'}
    ):
        raise ValueError("汇总稿与阶段不一致")
    if state['phase'] == 'completed' and state['report'] is None:
        raise ValueError("已完成研究缺少汇总稿")
    return state


def reserve_budget(run_id, kind):
    if kind not in {'model', 'tool'}:
        raise ValueError("未知预算类型")
    # 列名来自白名单；条件更新使读取、检查和扣减不可分割。
    with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
        cursor = conn.execute(
            f"UPDATE research_pipeline SET {kind}_used={kind}_used+1 "
            f"WHERE run_id=? AND {kind}_used<{kind}_limit AND phase!='completed'",
            (run_id,),
        )
        if cursor.rowcount != 1:
            raise BudgetExceeded(f"整次研究的 {kind} 调用预算已用尽或状态无效")
        db._append_event(conn, run_id, "budget_reserved", {"kind": kind})


def begin_tasks(run_id):
    plan = db.load_plan(run_id)
    with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT phase FROM research_pipeline WHERE run_id=?", (run_id,)).fetchone()
        if row != ('planning',):
            raise ValueError("当前阶段不能初始化子任务")
        # 不修补丢失记录；初始化与阶段推进必须一起提交。
        for task in plan.tasks:
            conn.execute("INSERT INTO task_executions(run_id, task_id) VALUES (?, ?)", (run_id, task.id))
        conn.execute("INSERT INTO research_evidence VALUES (?, '[]')", (run_id,))
        conn.execute("UPDATE research_pipeline SET phase='tasks' WHERE run_id=?", (run_id,))
        conn.execute("UPDATE run_workflow SET phase='research' WHERE run_id=?", (run_id,))
        db._append_event(conn, run_id, "tasks_initialized", {"task_count": len(plan.tasks)})


def begin_synthesis(run_id):
    with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute("SELECT status FROM task_executions WHERE run_id=?", (run_id,)).fetchall()
        if not rows or any(row[0] != 'completed' for row in rows):
            raise ValueError("还有未完成子任务，不能汇总")
        cursor = conn.execute("UPDATE research_pipeline SET phase='synthesis' WHERE run_id=? AND phase='tasks'", (run_id,))
        if cursor.rowcount != 1:
            raise ValueError("当前阶段不能开始汇总")
        db._append_event(conn, run_id, "synthesis_started", {})


def save_report(run_id, answer):
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("汇总稿不能为空")
    store = EvidenceStore.from_records(db.load_research_evidence(run_id))
    if store.find_unknown_citations(answer):
        raise ValueError("汇总稿引用未知证据")
    with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
        cursor = conn.execute("UPDATE research_pipeline SET report=? WHERE run_id=? AND phase='synthesis' AND report IS NULL", (answer, run_id))
        if cursor.rowcount != 1:
            raise ValueError("不能覆盖已有汇总稿或当前阶段不允许保存")
        db._append_event(conn, run_id, "synthesis_saved", {"answer_chars": len(answer)})


def finish_run(run):
    """最终结果与终态原子提交；失败时已落盘的汇总稿可直接重用。"""
    with closing(sqlite3.connect(db.DB_PATH)) as conn, conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT report FROM research_pipeline WHERE run_id=? AND phase='synthesis'", (run.id,)).fetchone()
        if row is None or not row[0]:
            raise ValueError("缺少已保存汇总稿")
        evidence = db._read_research_evidence(conn, run.id)
        from quality_storage import final_material
        report, evidence = final_material(conn, run.id, row[0], evidence)
        if EvidenceStore.from_records(evidence).find_unknown_citations(report):
            raise ValueError("汇总稿引用未知证据")
        if evidence and '\n\n本次收集的证据：' not in report:
            report += '\n\n本次收集的证据：\n' + '\n'.join(
                f"[{e['id']}] {e['title']}\n    {e['url']}" for e in evidence)
        run.answer, run.evidence, run.status, run.error = report, evidence, 'completed', None
        db._save_run(conn, run)
        conn.execute("UPDATE research_pipeline SET phase='completed' WHERE run_id=?", (run.id,))
        db._append_event(conn, run.id, "run_completed", {"answer_chars": len(run.answer), "evidence_count": len(evidence)})
        from memory_jobs import enqueue
        enqueue(conn, run.id)
