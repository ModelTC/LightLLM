"""Shared helpers for OpenAI-compatible API error responses."""

from http import HTTPStatus
from typing import Any, Mapping

from fastapi.responses import JSONResponse

from lightllm.utils.error_utils import ServerBusyError

_RATE_LIMIT_ERROR_TYPES = {"RateLimitError", "rate_limit_error"}


def is_rate_limit_error(error: Mapping[str, Any]) -> bool:
    return error.get("code") == HTTPStatus.TOO_MANY_REQUESTS or error.get("type") in _RATE_LIMIT_ERROR_TYPES


def create_error_response(
    status_code: HTTPStatus, message: str, err_type: str = None, param: str = None
) -> JSONResponse:
    if err_type is None:
        if status_code.value >= 500:
            err_type = "InternalServerError"
        elif status_code == HTTPStatus.NOT_FOUND:
            err_type = "NotFoundError"
        elif status_code == HTTPStatus.TOO_MANY_REQUESTS:
            err_type = "RateLimitError"
        else:
            err_type = "BadRequestError"

    from .api_http import g_objs

    g_objs.metric_client.counter_inc("lightllm_request_failure")
    return JSONResponse(
        {"error": {"message": message, "type": err_type, "param": param, "code": status_code.value}},
        status_code=status_code.value,
    )


def create_server_busy_response(exc: ServerBusyError) -> JSONResponse:
    return create_error_response(HTTPStatus(exc.status_code), str(exc))
