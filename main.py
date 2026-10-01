import asyncio
import logging
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from cli import run_cli
from database import init_db


async def main():
    load_dotenv(Path(__file__).with_name(".env"))
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        init_db()
    except (sqlite3.Error, OSError) as error:
        print(f"数据库初始化失败：{error}")
        return

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请先在 .env 中填写 DEEPSEEK_API_KEY")

    async with AsyncOpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "DEEPSEEK_BASE_URL",
            "https://api.deepseek.com",
        ),
        timeout=60.0,
        max_retries=0,
    ) as client:
        await run_cli(client)


if __name__ == "__main__":
    asyncio.run(main())
