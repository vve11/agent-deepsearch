"""浏览器联调专用：临时数据库和模拟模型，端口 8001，不读取 .env。

首次研究在搜索后模拟失败；点击继续研究后完成。Ctrl+C 退出并清理临时数据库。
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import uvicorn
import agent
import database
from api import create_app
from test_checkpoints import client_with, response


async def model(**kwargs):
    await asyncio.sleep(1)
    if "tools" not in kwargs:
        return response(json.dumps({"tasks": [{"question": "调查电池回收技术路线"}, {"question": "比较各路线的适用条件"}]}))
    if len(kwargs["messages"]) == 2:
        return response(queries=["电池回收技术比较"])
    if not model.failed:
        model.failed = True
        raise ValueError("浏览器测试：模拟一次中断，请点击继续研究")
    return response("## 模拟研究报告\n\n这是一条用于验证网页联动的测试回答，不是真实检索结果。[E1]\n\n| 项目 | 说明 |\n|---|---|\n| 恢复 | 沿用原检查点 |\n| 引用 | 来源已登记 |")


model.failed = False


async def search(call):
    await asyncio.sleep(1)
    return {"result": [{"title": "浏览器测试资料", "url": "https://example.com", "content": "模拟证据，仅用于测试网页。"}]}


if __name__ == "__main__":
    client = client_with()
    client.chat.completions.create.side_effect = model
    with tempfile.TemporaryDirectory() as folder, patch.object(agent, "execute_tool", side_effect=search):
        database.DB_PATH = Path(folder) / "browser-test.db"
        uvicorn.run(create_app(client_factory=lambda: client, load_environment=False), host="127.0.0.1", port=8001)
