import sqlite3
import json
from contextlib import closing
from pathlib import Path
from run import ResearchRun
from dataclasses import asdict
from checkpoint import AgentCheckpoint
from plan import ResearchPlan, restore_plan, plan_context
from task_execution import TaskExecution, validate_task_execution
DB_PATH = Path(__file__).resolve().parent / "data" / "research.db"


def init_db() -> None:
    """按需创建数据表，保留已有任务和事件。"""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(DB_PATH)) as connection:
        # 外键检查需要为每个连接单独启用
        connection.execute("PRAGMA foreign_keys = ON")

        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY NOT NULL,
                    question TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN (
                            'running',
                            'completed',
                            'failed'
                        )),
                    answer TEXT,
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    error TEXT
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (
                        strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    ),
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
                """
            )

            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS
                    idx_run_events_run_id_id
                ON run_events (run_id, id)
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id TEXT PRIMARY KEY NOT NULL,
                    state_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
                """
            )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS research_plans (
                    run_id TEXT PRIMARY KEY NOT NULL,
                    plan_json TEXT NOT NULL,
                    FOREIGN KEY (run_id) REFERENCES runs(id)
                )
                """
            )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS run_workflow (
                    run_id TEXT PRIMARY KEY NOT NULL REFERENCES runs(id),
                    phase TEXT NOT NULL CHECK (phase IN ('planning', 'planned', 'research'))
                )"""
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS task_executions (
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN (
                            'pending',
                            'running',
                            'completed',
                            'failed'
                        )),
                    answer TEXT,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    error TEXT,

                    PRIMARY KEY (run_id, task_id),
                    FOREIGN KEY (run_id)
                        REFERENCES research_plans(run_id)
                )
                """
            )

def _save_run(connection: sqlite3.Connection, run: ResearchRun) -> None:
    """只执行写入；连接和事务由调用方管理。"""
    evidence_json = json.dumps(run.evidence, ensure_ascii=False)

    connection.execute(
        """
        INSERT INTO runs (
            id,
            question,
            status,
            answer,
            evidence_json,
            error
        )
        VALUES (?, ?, ?, ?, ?, ?)

        ON CONFLICT(id) DO UPDATE SET
            question = excluded.question,
            status = excluded.status,
            answer = excluded.answer,
            evidence_json = excluded.evidence_json,
            error = excluded.error
        """,
        (
            run.id,
            run.question,
            run.status,
            run.answer,
            evidence_json,
            run.error,
        ),
    )


def save_run(run: ResearchRun) -> None:
    """单独保存任务；编号已存在时更新。"""
    with closing(sqlite3.connect(DB_PATH)) as connection:
        with connection:
            _save_run(connection, run)


def load_run(run_id: str) -> ResearchRun:
    """根据任务编号查询记录，重建 ResearchRun 对象。"""

    if (
        len(run_id) != 32
        or any(char not in "0123456789abcdef" for char in run_id)
    ):
        raise ValueError("任务编号格式不正确")

    with closing(sqlite3.connect(DB_PATH)) as connection:
        # 让查询结果支持通过字段名取值
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT id, question, status, answer, evidence_json, error
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

    if row is None:
        raise LookupError(f"没有找到任务：{run_id}")

    return ResearchRun(
        question=row["question"],
        id=row["id"],
        status=row["status"],
        answer=row["answer"],
        evidence=json.loads(row["evidence_json"]),
        error=row["error"],
    )

def _append_event(
    connection: sqlite3.Connection,
    run_id: str,
    event_type: str,
    payload: dict,
) -> int:
    """只执行插入；不创建连接，也不单独提交。"""
    event_type = event_type.strip()
    if not event_type:
        raise ValueError("事件类型不能为空")
    payload_json = json.dumps(payload, ensure_ascii=False)

    cursor = connection.execute(
        """
        INSERT INTO run_events (
            run_id,
            event_type,
            payload_json
        )
        VALUES (?, ?, ?)
        """,
        (
            run_id,
            event_type,
            payload_json,
        ),
    )
    return cursor.lastrowid


def append_event(run_id: str, event_type: str, payload: dict) -> int:
    """单独追加事件，返回事件编号。"""
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            event_id = _append_event(connection, run_id, event_type, payload)
    return event_id


def save_run_with_event(
    run: ResearchRun, event_type: str, payload: dict, *, start_planning: bool = False,
) -> int:
    """任务与事件一起提交；任一步失败，两次写入一起回滚。"""
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            # 同一连接、顺序执行，中间不提交。
            _save_run(connection, run)
            if start_planning:
                connection.execute(
                    "INSERT INTO run_workflow (run_id, phase) VALUES (?, 'planning')",
                    (run.id,),
                )
            event_id = _append_event(connection, run.id, event_type, payload)
    return event_id


def load_events(run_id: str) -> list[dict]:
    """按记录顺序，读取指定任务的全部事件。"""

    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT id, run_id, created_at, event_type, payload_json
            FROM run_events
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()

    events = []

    for row in rows:
        events.append(
            {
                "id": row["id"],
                "run_id": row["run_id"],
                "created_at": row["created_at"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
            }
        )

    return events


def save_checkpoint(checkpoint: AgentCheckpoint) -> None:
    """最新完整状态和 checkpoint_saved 事件在同一事务提交。"""
    checkpoint.validate()
    state_json = json.dumps(asdict(checkpoint), ensure_ascii=False, allow_nan=False)
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            workflow = connection.execute(
                "SELECT phase FROM run_workflow WHERE run_id = ?", (checkpoint.run_id,),
            ).fetchone()
            if workflow is not None:
                saved_plan = connection.execute(
                    "SELECT plan_json FROM research_plans WHERE run_id = ?", (checkpoint.run_id,),
                ).fetchone()
                if workflow[0] not in {"planned", "research"} or saved_plan is None:
                    raise ValueError("必须先保存计划才能进入研究阶段")
                plan = restore_plan(json.loads(saved_plan[0]))
                if (checkpoint.messages[1]["content"].strip() != plan.question
                        or not checkpoint.messages[0]["content"].endswith(plan_context(plan))):
                    raise ValueError("检查点与已保存计划不一致")
            connection.execute(
                """INSERT INTO checkpoints (run_id, state_json) VALUES (?, ?)
                ON CONFLICT(run_id) DO UPDATE SET state_json = excluded.state_json""",
                (checkpoint.run_id, state_json),
            )
            connection.execute(
                "UPDATE run_workflow SET phase = 'research' WHERE run_id = ?",
                (checkpoint.run_id,),
            )
            _append_event(connection, checkpoint.run_id, "checkpoint_saved", {
                "next_step": checkpoint.next_step,
                "tool_count": checkpoint.tool_count,
                "evidence_count": len(checkpoint.evidence),
            })


def load_checkpoint(run_id: str) -> AgentCheckpoint:
    with closing(sqlite3.connect(DB_PATH)) as connection:
        row = connection.execute(
            "SELECT state_json FROM checkpoints WHERE run_id = ?", (run_id,),
        ).fetchone()
    if row is None:
        raise LookupError("这个任务没有检查点；旧任务需要重新发起研究")
    checkpoint = AgentCheckpoint.from_dict(json.loads(row[0]))
    if checkpoint.run_id != run_id:
        raise ValueError("检查点所属任务不匹配")
    return checkpoint


def list_checkpoint_candidates() -> list[dict]:
    """读取检查点任务及新流程任务；Runtime 再验证能否安全恢复。"""
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT r.id, r.question, r.status, c.state_json
            FROM runs AS r LEFT JOIN checkpoints AS c ON c.run_id = r.id
            LEFT JOIN run_workflow AS w ON w.run_id = r.id
            WHERE r.status IN ('running', 'failed')
                AND (c.run_id IS NOT NULL OR w.run_id IS NOT NULL)
            ORDER BY (
                SELECT MAX(e.id) FROM run_events AS e WHERE e.run_id = r.id
            ) DESC, r.rowid DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


def save_plan(run_id: str, plan: ResearchPlan) -> None:
    """保存固定计划和 plan_saved 事件；相同计划重复保存不产生新事件。"""
    if not isinstance(plan, ResearchPlan):
        raise ValueError("计划必须是 ResearchPlan 对象")
    # dataclass 本身不做字段校验，写入前也必须验证。
    validated = restore_plan(asdict(plan))
    plan_json = json.dumps(asdict(validated), ensure_ascii=False, allow_nan=False)

    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with connection:
            parent = connection.execute(
                "SELECT question FROM runs WHERE id = ?", (run_id,),
            ).fetchone()
            if parent is None:
                raise LookupError("请先保存研究任务，再保存计划")
            if parent[0].strip() != validated.question:
                raise ValueError("计划问题与所属研究任务不一致")

            cursor = connection.execute(
                """INSERT INTO research_plans (run_id, plan_json) VALUES (?, ?)
                ON CONFLICT(run_id) DO NOTHING""",
                (run_id, plan_json),
            )
            if cursor.rowcount == 0:
                existing = connection.execute(
                    "SELECT plan_json FROM research_plans WHERE run_id = ?", (run_id,),
                ).fetchone()
                if restore_plan(json.loads(existing[0])) != validated:
                    raise ValueError("任务已有不同计划，不能覆盖；重新规划需后续版本机制")
                return

            # 事件失败会连同刚插入的计划一起回滚。
            connection.execute(
                "UPDATE run_workflow SET phase = 'planned' WHERE run_id = ? AND phase = 'planning'",
                (run_id,),
            )
            _append_event(connection, run_id, "plan_saved", {
                "task_count": len(validated.tasks),
                "task_ids": [task.id for task in validated.tasks],
            })


def load_plan(run_id: str) -> ResearchPlan:
    """读取固定计划，校验编号并还原 ResearchTask，而非返回裸字典。"""
    with closing(sqlite3.connect(DB_PATH)) as connection:
        row = connection.execute(
            """SELECT p.plan_json, r.question
            FROM research_plans AS p JOIN runs AS r ON r.id = p.run_id
            WHERE p.run_id = ?""",
            (run_id,),
        ).fetchone()
    if row is None:
        raise LookupError("这个任务尚未保存研究计划")
    plan = restore_plan(json.loads(row[0]))
    if plan.question != row[1].strip():
        raise ValueError("已保存计划的问题与任务不一致")
    return plan


def load_workflow_phase(run_id: str) -> str | None:
    """无阶段记录表示旧流程任务，不能据此推断它尚未开始研究。"""
    with closing(sqlite3.connect(DB_PATH)) as connection:
        row = connection.execute(
            "SELECT phase FROM run_workflow WHERE run_id = ?", (run_id,),
        ).fetchone()
    return row[0] if row else None


def list_runs(limit: int = 50, offset: int = 0) -> list[dict]:
    """网页列表只读摘要，避免把所有回答、检查点传给浏览器。"""
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """SELECT r.id, r.question, r.status, w.phase,
                (SELECT MAX(created_at) FROM run_events WHERE run_id = r.id) AS updated_at
            FROM runs AS r LEFT JOIN run_workflow AS w ON w.run_id = r.id
            ORDER BY (SELECT MAX(id) FROM run_events WHERE run_id = r.id) DESC, r.rowid DESC
            LIMIT ? OFFSET ?""", (limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]


def load_recent_events(run_id: str, limit: int = 100) -> list[dict]:
    """详情页显示最近事件；完整历史仍保存在数据库中。"""
    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM run_events WHERE run_id = ? ORDER BY id DESC LIMIT ?",
            (run_id, limit),
        ).fetchall()
    return [{**dict(row), "payload": json.loads(row["payload_json"])}
            for row in reversed(rows)]

def initialize_task_executions(run_id: str) -> None:
    """根据已保存的计划创建执行记录；已有记录保持不变。"""
    plan = load_plan(run_id)

    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.execute("PRAGMA foreign_keys = ON")

        with connection:
            for task in plan.tasks:
                execution = TaskExecution(
                    run_id=run_id,
                    task_id=task.id,
                )
                validate_task_execution(execution)

                connection.execute(
                    """
                    INSERT INTO task_executions (
                        run_id,
                        task_id,
                        status,
                        answer,
                        evidence_ids_json,
                        error
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id, task_id) DO NOTHING
                    """,
                    (
                        execution.run_id,
                        execution.task_id,
                        execution.status,
                        execution.answer,
                        json.dumps(
                            execution.evidence_ids,
                            ensure_ascii=False,
                        ),
                        execution.error,
                    ),
                )
def load_task_execution(run_id: str, task_id: str) -> TaskExecution:
    """读取一个子任务的执行记录，并校验数据。"""
    plan = load_plan(run_id)

    if not any(task.id == task_id for task in plan.tasks):
        raise ValueError("这个子任务不属于当前研究计划")

    with closing(sqlite3.connect(DB_PATH)) as connection:
        connection.row_factory = sqlite3.Row

        row = connection.execute(
            """
            SELECT run_id, task_id, status, answer,
                   evidence_ids_json, error
            FROM task_executions
            WHERE run_id = ? AND task_id = ?
            """,
            (run_id, task_id),
        ).fetchone()

    if row is None:
        raise LookupError("这个子任务尚未初始化执行记录")

    execution = TaskExecution(
        run_id=row["run_id"],
        task_id=row["task_id"],
        status=row["status"],
        answer=row["answer"],
        evidence_ids=json.loads(row["evidence_ids_json"]),
        error=row["error"],
    )

    validate_task_execution(execution)
    return execution