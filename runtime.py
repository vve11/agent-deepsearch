"""管理一次研究任务的生命周期；不读取终端输入。"""
import logging
import sqlite3
import asyncio

from openai import AsyncOpenAI

from agent import run_agent
from database import save_run, save_run_with_event, append_event, DB_PATH
from database import load_run, save_checkpoint, load_checkpoint
from database import list_checkpoint_candidates, load_plan, save_plan, load_workflow_phase
from planner import create_plan
from plan import plan_context
from checkpoint import AgentCheckpoint
from evidence import EvidenceStore
from run import ResearchRun

logger = logging.getLogger(__name__)
# 第一版只支持单进程；阻止本进程同时恢复同一任务。
_active_runs: set[str] = set()


def get_resumable_runs() -> list[dict]:
    """给界面提供可恢复任务摘要；隐藏活动中、损坏和预算已耗尽的记录。"""
    summaries = []
    for row in list_checkpoint_candidates():
        if row["id"] in _active_runs:
            continue
        try:
            checkpoint = _recovery_checkpoint(load_run(row["id"]))
        except (ValueError, TypeError, LookupError):
            logger.warning("[恢复列表] 跳过无效恢复状态：%s", row["id"])
            continue
        summaries.append({
            "id": row["id"], "question": row["question"], "status": row["status"],
            "next_step": checkpoint.next_step if checkpoint else 0,
            "evidence_count": len(checkpoint.evidence) if checkpoint else 0,
            "phase": "research" if checkpoint else load_workflow_phase(row["id"]),
        })
    return summaries


async def execute_research(client: AsyncOpenAI, question: str) -> ResearchRun:
    """成功返回任务；失败保存已有证据后，继续向上传递原异常。"""
    run = create_research_run(question)
    return await execute_created_research(client, run)


def create_research_run(question: str) -> ResearchRun:
    """先持久化任务，API 才能立即返回一个可查询、可恢复的编号。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("研究问题必须是非空字符串")
    run = ResearchRun(question=question.strip())
    save_run_with_event(run, "run_started", {"question": run.question}, start_planning=True)
    return run


async def execute_created_research(client: AsyncOpenAI, run: ResearchRun) -> ResearchRun:
    """仅供刚创建任务的调度入口使用；恢复已有任务使用 resume_research。"""
    return await _execute_run(client, run, startup_saved=True)


async def resume_research(client: AsyncOpenAI, run_id: str) -> ResearchRun:
    """读取最近完整检查点；沿用任务编号，不重复已保存的工具轮次。"""
    if run_id in _active_runs:
        raise ValueError("任务正在本进程执行，不能同时恢复")
    run = load_run(run_id)
    if run.status == "completed":
        raise ValueError("任务已经完成，请使用 /show 查看结果")
    if run.status not in {"running", "failed"}:
        raise ValueError("当前任务状态不允许恢复")
    checkpoint = _recovery_checkpoint(run)
    return await _execute_run(client, run, checkpoint, resuming=True)


def _recovery_checkpoint(run: ResearchRun) -> AgentCheckpoint | None:
    """缺少检查点只在明确的研究前阶段允许；损坏不能当成不存在。"""
    phase = load_workflow_phase(run.id)
    try:
        checkpoint = load_checkpoint(run.id)
    except LookupError:
        if phase not in {"planning", "planned"}:
            raise LookupError("任务缺少研究检查点，不能安全恢复；请重新发起研究")
        try:
            load_plan(run.id)
        except LookupError:
            if phase == "planned":
                raise ValueError("已保存计划丢失，不能重新规划")
        else:
            if phase == "planning":
                raise ValueError("计划与流程阶段不一致")
        return None
    if checkpoint.messages[1]["content"] != run.question:
        raise ValueError("检查点问题与任务不一致")
    if checkpoint.next_step >= checkpoint.max_tool_calls + 1:
        raise ValueError("模型轮次预算已用尽，不能恢复；请重新发起任务")
    if phase is not None:
        if phase != "research":
            raise ValueError("检查点与流程阶段不一致")
        plan = load_plan(run.id)
        if not checkpoint.messages[0]["content"].endswith(plan_context(plan)):
            raise ValueError("检查点中的计划与已保存计划不一致")
    # 旧任务没有阶段标记，沿用原检查点，不在恢复中途插入计划。
    return checkpoint


async def prepare_plan(client, run, record_event):
    phase = load_workflow_phase(run.id)
    if phase == "planning":
        logger.info("[研究计划] 正在生成计划")
        plan = await create_plan(client, run.question, record_event)
        save_plan(run.id, plan)  # 必须成功落盘，才允许开始研究。
    elif phase == "planned":
        plan = load_plan(run.id)
        record_event("plan_reused", {"task_count": len(plan.tasks)})
    else:
        raise ValueError("当前阶段不允许准备计划")
    for task in plan.tasks:
        logger.info("[研究计划] %s：%s", task.id, task.question)
    return plan


async def _execute_run(
    client: AsyncOpenAI, run: ResearchRun,
    checkpoint: AgentCheckpoint | None = None,
    *, resuming: bool = False, startup_saved: bool = False,
) -> ResearchRun:
    if run.id in _active_runs:
        raise ValueError("任务正在本进程执行，不能同时恢复")
    _active_runs.add(run.id)
    try:
        return await _drive_run(client, run, checkpoint, resuming=resuming, startup_saved=startup_saved)
    finally:
        _active_runs.remove(run.id)


async def _drive_run(
    client: AsyncOpenAI, run: ResearchRun,
    checkpoint: AgentCheckpoint | None,
    *, resuming: bool = False, startup_saved: bool = False,
) -> ResearchRun:
    evidence_store = (
        EvidenceStore.from_records(checkpoint.evidence)
        if checkpoint is not None else EvidenceStore()
    )
    run.status = "running"
    run.answer = None
    run.error = None
    run.evidence = evidence_store.all()
    logger.info("[任务编号] %s", run.id)
    logger.info("[任务状态] %s", run.status)

    # 启动记录与启动事件在同一事务中保存。
    if not resuming and not startup_saved:
        save_run_with_event(run, "run_started", {"question": run.question}, start_planning=True)
    elif resuming:
        save_run_with_event(run, "run_resumed", {
            "next_step": checkpoint.next_step if checkpoint else 0,
            "tool_count": checkpoint.tool_count if checkpoint else 0,
            "phase": "research" if checkpoint else load_workflow_phase(run.id),
        })

    def record_event(event_type: str, payload: dict) -> None:
        try:
            append_event(run.id, event_type, payload)
        except sqlite3.Error as error:
            # 保留现有策略：中间事件保存失败，告警但不中断研究。
            logger.warning("[事件记录失败] %s：%s", event_type, error)

    def record_checkpoint(state: dict) -> None:
        # 与普通事件不同：检查点保存失败必须中止，异常交给下方处理。
        save_checkpoint(AgentCheckpoint(run_id=run.id, **state))

    try:
        plan = await prepare_plan(client, run, record_event) if checkpoint is None else None
        run.answer = await run_agent(
            client, run.question, evidence_store, record_event,
            checkpoint=checkpoint, record_checkpoint=record_checkpoint, plan=plan,
        )
        run.status = "completed"
        return run
    except Exception as error:
        run.status = "failed"
        run.error = f"{type(error).__name__}: {error}"
        raise
    except asyncio.CancelledError:
        run.status = "failed"
        run.error = "CancelledError: 执行被取消，可使用 /resume 从已保存阶段继续"
        raise
    finally:
        # Agent 使用的就是这个证据库，因此失败前收集的资料也能保留。
        run.evidence = evidence_store.all()
        logger.info("[任务状态] %s", run.status)
        logger.info("[证据数量] %s", len(run.evidence))
        try:
            if run.status == "completed":
                save_run_with_event(run, "run_completed", {
                    "evidence_count": len(run.evidence),
                    "answer_chars": len(run.answer or ""),
                })
            elif run.status == "failed":
                save_run_with_event(run, "run_failed", {
                    "error": run.error,
                    "evidence_count": len(run.evidence),
                })
            else:
                # 保留原有非终态保存行为，取消与恢复将在后续实现。
                save_run(run)
            logger.info("[任务记录已保存] %s", run.id)
            logger.info("[数据库] %s", DB_PATH)
        except sqlite3.Error as error:
            # 保存失败不覆盖研究异常，也不丢弃已生成的回答。
            logger.warning("[任务或事件保存失败] %s", error)
