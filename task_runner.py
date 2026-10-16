"""单进程顺序执行：只跨越已持久化的边界，不重跑已完成子任务。"""
import asyncio
import logging
import re
import sqlite3
from contextlib import closing

import database as db
import pipeline_storage as storage
from agent import run_agent
from budget import BudgetExceeded, use_budget
from checkpoint import TaskCheckpoint
from evidence import EvidenceStore
from synthesizer import synthesize

logger = logging.getLogger(__name__)


def inspect_pipeline(run_id, *, for_resume=False):
    """恢复前完整校验；缺失/损坏记录不能伪装成一个待执行的新任务。"""
    state = storage.load_pipeline(run_id)
    if state is None:
        raise LookupError("没有顺序研究记录")
    phase = state['phase']
    with closing(sqlite3.connect(db.DB_PATH)) as conn:
        task_ids = {r[0] for r in conn.execute('SELECT task_id FROM task_executions WHERE run_id=?', (run_id,))}
        cp_ids = {r[0] for r in conn.execute('SELECT task_id FROM task_checkpoints WHERE run_id=?', (run_id,))}
    if phase == 'planning':
        if task_ids or cp_ids:
            raise ValueError("规划阶段存在子任务执行记录")
        workflow = db.load_workflow_phase(run_id)
        try:
            db.load_plan(run_id)
        except LookupError:
            if workflow != 'planning':
                raise ValueError("已保存计划丢失")
        else:
            if workflow != 'planned':
                raise ValueError("计划与规划阶段不一致")
        executions, checkpoints = [], {}
    else:
        plan = db.load_plan(run_id)
        if task_ids != {task.id for task in plan.tasks} or not cp_ids <= task_ids:
            raise ValueError("子任务执行记录缺失或出现额外任务")
        shared = db.load_research_evidence(run_id)
        if db.load_workflow_phase(run_id) != 'research':
            raise ValueError("子任务与工作流阶段不一致")
        executions, checkpoints = [], {}
        unfinished = False
        saved_ids = set()
        for event in db.load_events(run_id):
            if event['event_type'] == 'task_checkpoint_saved':
                payload = event['payload']
                if not isinstance(payload, dict) or not isinstance(payload.get('task_id'), str):
                    raise ValueError("检查点保存事件数据损坏")
                saved_ids.add(payload['task_id'])
        for task in plan.tasks:
            execution = db.load_task_execution(run_id, task.id)
            if unfinished and execution.status != 'pending':
                raise ValueError("子任务状态不符合顺序执行约束")
            unfinished = unfinished or execution.status != 'completed'
            if task.id in cp_ids:
                cp = db.load_task_checkpoint(run_id, task.id)
                if execution.status == 'pending' or cp.model_name != state['model_name']:
                    raise ValueError("检查点与子任务状态或模型不一致")
                if execution.status != 'completed' and cp.evidence != shared:
                    raise ValueError("活动子任务检查点与共享证据不一致")
                checkpoints[task.id] = cp
            elif execution.status == 'completed' or task.id in saved_ids:
                raise ValueError("已保存的子任务检查点丢失")
            if execution.status == 'completed':
                cited = set(re.findall(r'\[(E\d+)\]', execution.answer))
                if not cited <= set(execution.evidence_ids):
                    raise ValueError("子任务答案与已登记引用不一致")
            executions.append(execution)
        latest = next(reversed(checkpoints.values()), None)
        if (latest.evidence if latest else []) != shared:
            raise ValueError("共享证据缺少对应的最新子任务检查点")
        if phase in {'synthesis', 'completed'} and unfinished:
            raise ValueError("汇总阶段仍有未完成子任务")
        if state['report'] is not None and EvidenceStore.from_records(shared).find_unknown_citations(state['report']):
            raise ValueError("已保存汇总稿引用未知证据")
    if for_resume:
        import quality_storage
        quality = quality_storage.load(run_id)
        if quality and quality[1]['phase'] == 'exhausted':
            raise ValueError('质量检查未通过且补救已终止：' + str(quality[1]['reason']))
        if phase == 'completed':
            raise ValueError("研究已经完成")
        if state['report'] is None and state['model_used'] >= state['model_limit']:
            raise BudgetExceeded("整次研究模型预算已用尽，不能恢复")
        for execution in executions:
            cp = checkpoints.get(execution.task_id)
            if execution.status != 'completed' and cp is not None and cp.next_step >= cp.max_tool_calls + 1:
                raise BudgetExceeded("子任务模型轮次已用尽，不能恢复")
    return state, executions, checkpoints


def recovery_checkpoint(run_id):
    _, executions, checkpoints = inspect_pipeline(run_id, for_resume=True)
    return next((checkpoints.get(task.task_id) for task in executions if task.status != 'completed'), None)


async def execute_pipeline(client, run, prepare_plan, *, resuming=False):
    state, _, _ = inspect_pipeline(run.id, for_resume=True)
    run.status, run.answer, run.error = 'running', None, None
    if resuming:
        db.save_run_with_event(run, 'run_resumed', {'phase': state['phase']})

    def record(event_type, payload):
        # 进度事件可丢失；状态、检查点、预算的事务事件绝不能忽略。
        try:
            db.append_event(run.id, event_type, payload)
        except sqlite3.Error:
            logger.exception("进度事件保存失败：%s", event_type)

    try:
        with use_budget(run.id, state['model_name']):
            if state['phase'] == 'planning':
                await prepare_plan(client, run, record)
                storage.begin_tasks(run.id)
            state, executions, checkpoints = inspect_pipeline(run.id)
            plan = db.load_plan(run.id)
            if state['phase'] == 'tasks':
                for task, execution in zip(plan.tasks, executions):
                    if execution.status == 'completed':
                        continue
                    cp = checkpoints.get(task.id)
                    if execution.status == 'running':
                        db.fail_task_execution(run.id, task.id, '上次执行中断，从已保存检查点恢复')
                    db.start_task_execution(run.id, task.id)
                    store = EvidenceStore.from_records(cp.evidence if cp else db.load_research_evidence(run.id))

                    def task_event(event_type, payload):
                        record(event_type, {**payload, 'task_id': task.id})

                    def save_checkpoint(snapshot):
                        db.save_task_checkpoint(TaskCheckpoint(run_id=run.id, task_id=task.id, **snapshot))

                    try:
                        answer = await run_agent(
                            client, task.question, store, task_event,
                            checkpoint=cp, record_checkpoint=save_checkpoint, append_sources=False,
                        )
                        cited = set(re.findall(r'\[(E\d+)\]', answer))
                        ids = [item['id'] for item in store.all() if item['id'] in cited]
                        db.complete_task_execution(run.id, task.id, answer, ids)
                    except (Exception, asyncio.CancelledError) as error:
                        try:
                            db.fail_task_execution(run.id, task.id, f'{type(error).__name__}: {error}')
                        except Exception:
                            logger.exception("子任务失败状态保存失败，保留原异常")
                        raise
                storage.begin_synthesis(run.id)
            state, executions, _ = inspect_pipeline(run.id)
            if state['report'] is None:
                evidence = EvidenceStore.from_records(db.load_research_evidence(run.id))
                answer = await synthesize(client, plan, executions, evidence, record, state['model_name'])
                storage.save_report(run.id, answer)
            from quality_gate import ensure_quality
            await ensure_quality(client, run, state['model_name'], record)
            storage.finish_run(run)
            return run
    except (Exception, asyncio.CancelledError) as error:
        run.status, run.answer = 'failed', None
        run.error = f'{type(error).__name__}: {error}'
        try:
            state = storage.load_pipeline(run.id)
            run.evidence = db.load_research_evidence(run.id) if state['phase'] != 'planning' else []
            import quality_storage
            quality = quality_storage.load(run.id)
            if quality and quality[1]['phase'] not in {'pending','skipped'}:
                run.evidence = quality[1]['evidence']
            db.save_run_with_event(run, 'run_failed', {'error': run.error, 'evidence_count': len(run.evidence)})
        except Exception:
            logger.exception("研究失败记录保存失败，保留原异常")
        raise
