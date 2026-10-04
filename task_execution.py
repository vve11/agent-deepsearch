"""子任务的执行状态；暂时不调用模型，也不操作数据库。"""
import re
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class TaskExecution:
    run_id: str
    task_id: str

    status: Literal[
        "pending", "running", "completed", "failed"
    ] = "pending"

    answer: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    error: str | None = None


def validate_task_execution(task: TaskExecution) -> None:
    """检查单条执行记录是否合法；不修改数据。"""
    if not isinstance(task, TaskExecution):
        raise ValueError("必须传入 TaskExecution 对象")

    if not isinstance(task.run_id, str) or not task.run_id.strip():
        raise ValueError("研究编号不能为空")

    if not isinstance(task.task_id, str) or not re.fullmatch(
        r"T[1-9][0-9]*", task.task_id
    ):
        raise ValueError("子任务编号必须是 T1、T2 这样的格式")

    if task.status not in ("pending", "running", "completed", "failed"):
        raise ValueError("子任务状态不合法")

    if task.answer is not None:
        if not isinstance(task.answer, str) or not task.answer.strip():
            raise ValueError("回答必须是非空字符串或 None")

    if task.error is not None:
        if not isinstance(task.error, str) or not task.error.strip():
            raise ValueError("错误信息必须是非空字符串或 None")

    if not isinstance(task.evidence_ids, list):
        raise ValueError("证据编号必须是列表")

    for evidence_id in task.evidence_ids:
        if not isinstance(evidence_id, str) or not re.fullmatch(
            r"E[1-9][0-9]*", evidence_id
        ):
            raise ValueError("证据编号必须是 E1、E2 这样的格式")

    if len(task.evidence_ids) != len(set(task.evidence_ids)):
        raise ValueError("证据编号不能重复")

    if task.status == "completed":
        if task.answer is None or task.error is not None:
            raise ValueError("完成的子任务必须有回答，且不能有错误")
    elif task.status == "failed":
        if task.error is None or task.answer is not None:
            raise ValueError("失败的子任务必须有错误，且不能有最终回答")
    else:
        if task.answer is not None or task.error is not None:
            raise ValueError("待执行或执行中的子任务不能有最终回答或错误")

    if task.status == "pending" and task.evidence_ids:
        raise ValueError("尚未执行的子任务不能已有证据")