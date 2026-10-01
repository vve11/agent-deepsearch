"""非流式模型请求的有限重试；不重跑 Agent 或工具。"""
import asyncio
import logging
import math
import ssl
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from openai import APIConnectionError, APIStatusError

logger = logging.getLogger(__name__)
MAX_RETRIES = 2  # 首次请求之外最多再试两次
MAX_RETRY_DELAY = 30.0


def is_retryable(error: Exception) -> bool:
    if isinstance(error, APIConnectionError):
        # 超时继承 APIConnectionError；证书校验失败则应修正配置。
        cause = error
        seen = set()
        while cause is not None and id(cause) not in seen:
            seen.add(id(cause))
            if isinstance(cause, ssl.SSLCertVerificationError):
                return False
            cause = cause.__cause__ or cause.__context__
        return True
    if not isinstance(error, APIStatusError):
        return False
    if error.status_code == 429:
        body = error.body if isinstance(error.body, dict) else {}
        details = body.get("error", body)
        if not isinstance(details, dict):
            details = {}
        markers = {str(details.get(key, "")).lower() for key in ("code", "type")}
        if markers & {"insufficient_quota", "quota_exceeded", "insufficient_balance",
                      "billing_hard_limit_reached", "spend_limit_exceeded"}:
            return False
    return error.status_code in {408, 429, 500, 502, 503, 504, 529}


def retry_delay(error: Exception, retry_index: int) -> float | None:
    delay = float(2 ** retry_index)  # 默认等待 1 秒、2 秒
    if isinstance(error, APIStatusError):
        header = error.response.headers.get("retry-after")
        if header:
            try:
                requested = float(header)
            except ValueError:
                try:
                    date = parsedate_to_datetime(header)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    requested = (date - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    requested = 0
            if math.isfinite(requested):
                delay = max(delay, requested)
    # 不提前于服务端要求重发，也不让 CLI 无期限等待。
    return delay if delay <= MAX_RETRY_DELAY else None


async def request_model(client, record_event, context: dict, **request):
    """仅重试未返回的非流式请求；最终仍抛出原异常。"""
    if request.get("stream"):
        raise ValueError("当前重试器只支持非流式模型请求")
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await client.chat.completions.create(**request)
        except (APIConnectionError, APIStatusError) as error:
            if attempt == MAX_RETRIES or not is_retryable(error):
                raise
            delay = retry_delay(error, attempt)
            if delay is None:
                raise
            record_event("model_retrying", {
                **context,
                "retry_number": attempt + 1,
                "max_retries": MAX_RETRIES,
                "delay_seconds": delay,
                "error_type": type(error).__name__,
                "status_code": getattr(error, "status_code", None),
            })
            logger.warning("[模型重试] %s 秒后重试（%s/%s）", delay, attempt + 1, MAX_RETRIES)
            await asyncio.sleep(delay)
