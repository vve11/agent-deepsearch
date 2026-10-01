"""第一版检查点：只恢复到消息完整的下一次模型请求。"""
import json
from dataclasses import dataclass, fields

from evidence import EvidenceStore


@dataclass
class AgentCheckpoint:
    run_id: str
    next_step: int
    tool_count: int
    messages: list[dict]
    evidence: list[dict]
    model_name: str
    max_tool_calls: int = 3
    version: int = 1
    next_action: str = "request_model"

    @classmethod
    def from_dict(cls, data: dict) -> "AgentCheckpoint":
        if not isinstance(data, dict) or set(data) != {f.name for f in fields(cls)}:
            raise ValueError("检查点字段不完整或格式不受支持")
        checkpoint = cls(**data)
        checkpoint.validate()
        return checkpoint

    def validate(self) -> None:
        if (not isinstance(self.run_id, str) or len(self.run_id) != 32
                or any(c not in "0123456789abcdef" for c in self.run_id)):
            raise ValueError("检查点任务编号不正确")
        if type(self.version) is not int or self.version != 1:
            raise ValueError("不支持的检查点版本")
        if self.next_action != "request_model":
            raise ValueError("不支持的恢复阶段")
        if (type(self.max_tool_calls) is not int or self.max_tool_calls != 3
                or type(self.next_step) is not int
                or not 0 <= self.next_step <= self.max_tool_calls + 1
                or type(self.tool_count) is not int
                or not 0 <= self.tool_count <= self.max_tool_calls):
            raise ValueError("检查点调用预算或轮次不正确")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("检查点模型名称不正确")
        EvidenceStore.from_records(self.evidence)

        # 首两条是系统提示和问题；之后只允许完整的 assistant/tools 轮次。
        if not isinstance(self.messages, list) or len(self.messages) < 2:
            raise ValueError("检查点消息历史不完整")
        for message, role in zip(self.messages[:2], ("system", "user")):
            if (not isinstance(message, dict) or message.get("role") != role
                    or not isinstance(message.get("content"), str)):
                raise ValueError("检查点缺少系统提示或用户问题")
        pending = set()
        seen = set()
        rounds = 0
        for message in self.messages[2:]:
            if not isinstance(message, dict):
                raise ValueError("检查点消息必须是字典")
            if message.get("role") == "assistant" and not pending:
                calls = message.get("tool_calls")
                if not isinstance(calls, list) or not calls:
                    raise ValueError("检查点中的助手消息必须包含工具请求")
                rounds += 1
                for call in calls:
                    if not isinstance(call, dict):
                        raise ValueError("工具请求格式不正确")
                    call_id = call.get("id")
                    function = call.get("function")
                    if (not isinstance(call_id, str) or not call_id or call_id in seen
                            or call.get("type") != "function"
                            or not isinstance(function, dict)
                            or not isinstance(function.get("name"), str)
                            or not isinstance(function.get("arguments"), str)):
                        raise ValueError("工具请求编号或参数格式不正确")
                    seen.add(call_id)
                    pending.add(call_id)
            elif message.get("role") == "tool":
                call_id = message.get("tool_call_id")
                if (not isinstance(call_id, str) or call_id not in pending
                        or not isinstance(message.get("content"), str)):
                    raise ValueError("工具结果与请求不匹配")
                pending.remove(call_id)
            else:
                raise ValueError("检查点包含不完整的工具轮次")
        if pending or rounds != self.next_step:
            raise ValueError("检查点轮次不匹配或缺少工具结果")
        if self.tool_count != min(len(seen), self.max_tool_calls):
            raise ValueError("工具调用次数与消息历史不匹配")
        try:
            json.dumps(self.messages, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("消息历史不能序列化") from error
