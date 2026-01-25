"""
Callback Service

estimate_id 기반 callback URL로 결과 전송
"""

import logging
from typing import Any, Dict, Optional

import aiohttp

from api.config import (
    CALLBACK_URL_TEMPLATE,
    CALLBACK_TIMEOUT_SECONDS,
    CALLBACK_RETRY_COUNT,
)

logger = logging.getLogger(__name__)


async def send_callback(
    estimate_id: int,
    result_data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> bool:
    """
    Callback URL로 결과 전송.

    Args:
        estimate_id: 견적 ID
        result_data: 성공 시 결과 데이터 (TDD Response 형식)
        error: 실패 시 에러 메시지

    Returns:
        True if callback succeeded, False otherwise
    """
    # Build callback URL
    callback_url = CALLBACK_URL_TEMPLATE.replace("{estimateId}", str(estimate_id))

    # Build payload
    if error:
        payload = {"error": error}
    elif result_data:
        payload = result_data
    else:
        payload = {"error": "Unknown error"}

    logger.info(f"[Callback] Sending to {callback_url}")

    # Send with retry
    for attempt in range(CALLBACK_RETRY_COUNT + 1):
        try:
            timeout = aiohttp.ClientTimeout(total=CALLBACK_TIMEOUT_SECONDS)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    callback_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as response:
                    if response.status >= 200 and response.status < 300:
                        logger.info(
                            f"[Callback] Success for estimate_id={estimate_id}, status={response.status}"
                        )
                        return True
                    else:
                        response_text = await response.text()
                        logger.warning(
                            f"[Callback] Failed for estimate_id={estimate_id}, "
                            f"status={response.status}, response={response_text[:200]}"
                        )

        except aiohttp.ClientError as e:
            logger.warning(
                f"[Callback] Network error for estimate_id={estimate_id}, "
                f"attempt={attempt + 1}/{CALLBACK_RETRY_COUNT + 1}: {e}"
            )
        except Exception as e:
            logger.error(
                f"[Callback] Unexpected error for estimate_id={estimate_id}: {e}"
            )

        # Retry if not last attempt
        if attempt < CALLBACK_RETRY_COUNT:
            logger.info(f"[Callback] Retrying estimate_id={estimate_id}...")

    logger.error(
        f"[Callback] All retries failed for estimate_id={estimate_id}"
    )
    return False
