"""RL control-plane HTTP APIs.

需要启动参数 ``--enable_rl``。供 RL / 在线训推一体场景使用，不走普通 generate 路径。

调用链：
  HTTP → HttpServerManager → HttpRlController
       → (多数) RlOpReq → Router → Model RlBackendOps

路由在模块级 ``router`` 上注册，由 ``api_http`` ``include_router`` 挂载。
``g_objs`` / ``create_error_response`` 在 handler 内懒导入，避免与 api_http 循环依赖。
"""

from http import HTTPStatus

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from lightllm.server.io_struct import (
    AbortReq,
    DestroyWeightsUpdateGroupReq,
    FlushCacheReq,
    InitWeightsUpdateGroupReq,
    ReleaseMemoryReq,
    ResumeMemoryReq,
    RlOpRsp,
    UpdateWeightsFromDistributedReq,
    UpdateWeightsFromIPCReq,
    UpdateWeightsFromTensorReq,
)
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

router = APIRouter()


async def handle_request_common(request_obj, handler):
    from .api_http import create_error_response

    try:
        ret: RlOpRsp = await handler(request_obj)
        if ret.success:
            return JSONResponse({"success": ret.success, "message": ret.msg}, status_code=200)
        else:
            return create_error_response(HTTPStatus.BAD_REQUEST, ret.msg)
    except Exception as e:
        logger.error("handle_request_common (%s) error occurred: %s", str(request_obj), str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, f"error: {str(e)}")


@router.post("/abort_request")
async def abort_request(request: AbortReq, raw_request: Request):
    """Abort a request."""
    from .api_http import create_error_response, g_objs

    try:
        success, msg = await g_objs.httpserver_manager.abort_request(request)
        if not success:
            return create_error_response(HTTPStatus.REQUEST_TIMEOUT, msg, err_type="AbortRequestTimeout")
        return Response(status_code=200)
    except Exception as e:
        logger.error("abort_request error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, f"error: {str(e)}")


@router.post("/init_weights_update_group")
async def init_weights_update_group(request: InitWeightsUpdateGroupReq, raw_request: Request):
    """Init weights update group."""
    from .api_http import g_objs

    return await handle_request_common(request, g_objs.httpserver_manager.init_weights_update_group)


@router.post("/destroy_weights_update_group")
async def destroy_weights_update_group(request: DestroyWeightsUpdateGroupReq, raw_request: Request):
    """Destroy weights update group."""
    from .api_http import g_objs

    return await handle_request_common(request, g_objs.httpserver_manager.destroy_weights_update_group)


@router.post("/update_weights_from_distributed")
async def update_weights_from_distributed(request: UpdateWeightsFromDistributedReq, raw_request: Request):
    """Update model parameter from distributed online."""
    from .api_http import g_objs

    return await handle_request_common(request, g_objs.httpserver_manager.update_weights_from_distributed)


@router.post("/update_weights_from_tensor")
async def update_weights_from_tensor(request: UpdateWeightsFromTensorReq, raw_request: Request):
    """Update model parameter from distributed online."""
    from .api_http import g_objs

    return await handle_request_common(request, g_objs.httpserver_manager.update_weights_from_tensor)


@router.post("/update_weights_from_ipc")
async def update_weights_from_ipc(request: UpdateWeightsFromIPCReq, raw_request: Request):
    from .api_http import g_objs

    return await handle_request_common(request, g_objs.httpserver_manager.update_weights_from_ipc)


@router.post("/flush_cache")
@router.get("/flush_cache")
async def flush_cache():
    """Flush the radix cache."""
    from .api_http import g_objs

    return await handle_request_common(FlushCacheReq(), g_objs.httpserver_manager.flush_cache)


@router.post("/pause_generation")
async def pause_generation():
    from .api_http import g_objs

    await g_objs.httpserver_manager.pause_generation()
    return Response(content="Generation paused successfully.", status_code=200)


@router.post("/continue_generation")
async def continue_generation():
    from .api_http import g_objs

    await g_objs.httpserver_manager.continue_generation()
    return Response(content="Generation continued successfully.", status_code=200)


@router.get("/release_memory_occupation")
@router.post("/release_memory_occupation")
async def release_memory_occupation(request: ReleaseMemoryReq):
    from .api_http import g_objs

    return await handle_request_common(request, g_objs.httpserver_manager.release_memory_occupation)


@router.get("/resume_memory_occupation")
@router.post("/resume_memory_occupation")
async def resume_memory_occupation(request: ResumeMemoryReq):
    from .api_http import g_objs

    return await handle_request_common(request, g_objs.httpserver_manager.resume_memory_occupation)
