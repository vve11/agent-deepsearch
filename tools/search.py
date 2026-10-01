import os

import httpx

def validate_search_arguments(arguments: dict) -> dict:
    if set(arguments) != {"query"}:
        raise ValueError("搜索工具必须且只能提供 query 参数")

    query = arguments["query"]

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")

    return {"query": query.strip()}

async def search_web(query: str) -> list[dict[str, str]]:
    """搜索网页，返回标题、链接和相关内容。"""

    # 1. 检查输入
    query = query.strip()
    if not query:
        raise ValueError("搜索关键词不能为空")

    # 2. 获取搜索服务的密钥
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("请先在 .env 中填写 TAVILY_API_KEY")

    # 3. 发送搜索请求
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": 3,
                "include_answer": False,
                "include_raw_content": False,
            },
        )

        # 4. 检查响应状态，再解析返回数据
        response.raise_for_status()
        data = response.json()

    # 5. 整理成我们项目使用的结果格式
    results = []

    for item in data["results"]:
        results.append(
            {
                "title": item["title"],
                "url": item["url"],
                "content": item["content"],
            }
        )

    return results