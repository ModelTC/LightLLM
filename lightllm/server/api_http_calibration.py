"""Internal FP8 KV calibration HTTP control route.

The router is included only by an export-calibration service during FastAPI
startup.  A per-job nonce prevents a client from operating an unrelated service.
"""
from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from .api_errors import create_error_response

router = APIRouter(prefix="/_calibration", include_in_schema=False)
_ALLOWED = {"status": "calibration_status", "begin": "calibration_begin", "snapshot": "calibration_snapshot"}


@router.post("/{operation}", include_in_schema=False)
async def calibration_operation(operation: str, request: Request):
    from .api_http import g_objs

    if not g_objs.args.export_fp8kv_calibration:
        return create_error_response(HTTPStatus.NOT_FOUND, "calibration route unavailable")
    if operation not in _ALLOWED:
        return create_error_response(HTTPStatus.NOT_FOUND, "unknown calibration operation")
    body = await request.json()
    if not isinstance(body, dict) or body.get("job_id") != g_objs.args.calibration_job_id:
        return create_error_response(HTTPStatus.FORBIDDEN, "calibration job_id does not match this service")
    http_pending = len(g_objs.httpserver_manager.req_id_to_out_inf)
    if operation != "status" and http_pending:
        return create_error_response(
            HTTPStatus.CONFLICT, f"calibration operation requires idle HTTP server, pending={http_pending}"
        )
    result = await g_objs.httpserver_manager.calibration_op(_ALLOWED[operation], body["job_id"])
    if not result.success:
        return create_error_response(HTTPStatus.BAD_REQUEST, result.msg)
    return JSONResponse(
        {"success": True, "operation": operation, "http_pending": http_pending, "ranks": result.op_result}
    )
