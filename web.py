"""本地网页启动入口：单进程，不启用会中断研究的开发自动重载。"""
import logging
import uvicorn

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    uvicorn.run("api:app", host="127.0.0.1", port=8000, workers=1)
