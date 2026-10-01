import json
from .search import search_web, validate_search_arguments

# 给模型看的工具说明
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "搜索互联网资料，返回网页标题、链接和相关内容。"
                "用于查找最新信息或获取支持回答的外部资料。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "具体、清晰的搜索关键词",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    }
]


# 给程序使用的工具名称与函数映射
AVAILABLE_TOOLS = {
    "search_web": {
        "function": search_web,
        "validate": validate_search_arguments,
    },
}

async def execute_tool(tool_call) -> dict:
    name = tool_call.function.name

    if name not in AVAILABLE_TOOLS:
        raise ValueError(f"未注册的工具：{name}")

    arguments = json.loads(tool_call.function.arguments)

    if not isinstance(arguments, dict):
        raise ValueError("工具参数必须是一个 JSON 对象")

    tool = AVAILABLE_TOOLS[name]

    # 使用这个工具自己的参数校验器
    validated_arguments = tool["validate"](arguments)

    print(
    "[调用参数] "
    + json.dumps(validated_arguments, ensure_ascii=False)
    )
    # 将校验后的参数交给这个工具
    result = await tool["function"](**validated_arguments)

    return {"result": result}