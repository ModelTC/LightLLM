# Adapted from vllm/entrypoints/api_server.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import collections
import time

import uvloop
import requests
import base64
import os
from io import BytesIO
import setproctitle

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import ujson as json
from http import HTTPStatus
import uuid
from PIL import Image
import multiprocessing as mp
from typing import AsyncGenerator, Union
from typing import Callable
from lightllm.server import TokenLoad
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response, JSONResponse
from lightllm.server.core.objs.sampling_params import SamplingParams
from lightllm.server.core.objs import StartArgs
from .multimodal_params import MultimodalParams
from .httpserver.manager import HttpServerManager
from .httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from .api_lightllm import lightllm_get_score
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.error_utils import ClientDisconnected, ServerBusyError
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.shm_port_args import get_shm_port_args
from dataclasses import asdict, dataclass, is_dataclass

from .api_errors import create_error_response, create_server_busy_response
from .api_openai import chat_completions_impl, completions_impl
from .visual_chat_proxy import (
    VisualChatProxyError,
    VisualProxyCapacityError,
    VisualProxyRuntime,
    VisualProxySettings,
    VisualProxyTimeoutError,
    VisualProxyUpstreamError,
    should_use_visual_proxy,
    visual_chat_completions_impl,
)
from .api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    CompletionResponse,
    ModelCard,
    ModelListResponse,
)
from .build_prompt import build_prompt, init_tokenizer

logger = init_logger(__name__)


@dataclass
class G_Objs:
    app: FastAPI = None
    metric_client: MetricClient = None
    args: StartArgs = None
    g_generate_func: Callable = None
    g_generate_stream_func: Callable = None
    httpserver_manager: Union[HttpServerManager, HttpServerManagerForPDMaster] = None
    shared_token_load: TokenLoad = None
    # OpenAI-compatible "created" timestamp for /v1/models.
    # Should be stable for the lifetime of this server process.
    model_created: int = None
    visual_proxy_runtime: VisualProxyRuntime = None

    def set_args(self, args: StartArgs):
        self.args = args
        from .api_lightllm import lightllm_generate, lightllm_generate_stream
        from .api_tgi import tgi_generate_impl, tgi_generate_stream_impl

        if args.use_tgi_api:
            self.g_generate_func = tgi_generate_impl
            self.g_generate_stream_func = tgi_generate_stream_impl
        else:
            self.g_generate_func = lightllm_generate
            self.g_generate_stream_func = lightllm_generate_stream

        setproctitle.setproctitle(f"lightllm::{get_unique_server_name()}::api_server")

        init_tokenizer(args)  # for openai api
        SamplingParams.load_generation_cfg(args.model_dir)
        CompletionRequest.load_generation_cfg(args.model_dir)
        ChatCompletionRequest.load_generation_cfg(args.model_dir)

        if self.model_created is None:
            self.model_created = int(time.time())

        if args.run_mode == "pd_master":
            self.metric_client = MetricClient(get_shm_port_args().metric_port)
            self.httpserver_manager = HttpServerManagerForPDMaster(
                args=args,
            )
        else:
            self.metric_client = MetricClient(get_shm_port_args().metric_port)
            self.httpserver_manager = HttpServerManager(args=args)
            dp_size_in_node = max(1, args.dp // args.nnodes)  # 兼容多机纯tp的运行模式，这时候 1 // 2 == 0, 需要兼容
            self.shared_token_load = TokenLoad(f"{get_unique_server_name()}_shared_token_load", dp_size_in_node)


g_objs = G_Objs()

app = FastAPI()
g_objs.app = app

_ACCESS_LOG_STATUS_COLORS = {2: "\033[32m", 3: "\033[36m", 4: "\033[33m", 5: "\033[31m"}
_ACCESS_LOG_RESET = "\033[0m"


class _AccessLogMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        status_holder = {"status": 0}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            if scope["type"] == "http":
                status = status_holder["status"]
                msg = f"{scope['method']} {scope['path']} {status}"
                color = _ACCESS_LOG_STATUS_COLORS.get(status // 100, "")
                if color:
                    msg = color + msg + _ACCESS_LOG_RESET
                logger.info(msg)


app.add_middleware(_AccessLogMiddleware)


@app.exception_handler(ServerBusyError)
async def server_busy_exception_handler(request: Request, exc: ServerBusyError) -> JSONResponse:
    logger.warning(str(exc))

    # Streaming responses can raise during their first body iteration, after
    # the route handler has already returned. Preserve the Anthropic error
    # envelope for that deferred failure path as well.
    if request.url.path == "/v1/messages":
        from .api_anthropic import _anthropic_error_response

        g_objs.metric_client.counter_inc("lightllm_request_failure")
        return _anthropic_error_response(HTTPStatus(exc.status_code), str(exc))

    return create_server_busy_response(exc)


@app.get("/liveness")
@app.post("/liveness")
def liveness():
    return {"status": "ok"}


@app.get("/readiness")
@app.post("/readiness")
def readiness():
    if g_objs.args.run_mode == "pd_master":
        pd_nodes_are_ready = g_objs.httpserver_manager.pd_manager.is_pd_nodes_ready()
        return JSONResponse(
            {"status": "ok" if pd_nodes_are_ready else "not ready"},
            status_code=200 if pd_nodes_are_ready else 503,
        )
    return {"status": "ok"}


@app.get("/get_model_name")
@app.post("/get_model_name")
def get_model_name():
    return {"model_name": g_objs.args.model_name}


@app.get("/get_server_info")
@app.post("/get_server_info")
def get_server_info():
    if is_dataclass(g_objs.args):
        return asdict(g_objs.args)

    # HTTP workers restore StartArgs from the environment as an EasyDict.
    return dict(g_objs.args)


@app.get("/get_weight_version")
@app.post("/get_weight_version")
def get_weight_version():
    return {"weight_version": g_objs.args.weight_version}


@app.get("/healthz", summary="Check server health")
@app.get("/health", summary="Check server health")
@app.head("/health", summary="Check server health")
async def healthcheck(request: Request):
    if os.environ.get("DEBUG_HEALTHCHECK_RETURN_FAIL") == "true":
        return JSONResponse({"message": "Error"}, status_code=503)

    if g_objs.args.run_mode == "pd_master":
        httpserver_manager = g_objs.httpserver_manager
        pd_manager = httpserver_manager.pd_manager
        if g_objs.args.pd_master_mode == "elastic":
            inference_is_healthy = httpserver_manager.is_healthy()
            pd_nodes_are_ready = pd_manager.is_pd_nodes_ready()
            is_healthy = inference_is_healthy and pd_nodes_are_ready
            health_info = {
                "inference_healthy": inference_is_healthy,
                "pd_nodes_ready": pd_nodes_are_ready,
            }
        else:
            inference_is_healthy = httpserver_manager.is_healthy()
            pd_nodes_are_ready = pd_manager.is_pd_nodes_ready()
            pd_nodes_are_healthy = (
                inference_is_healthy and pd_nodes_are_ready and await pd_manager.check_pd_nodes_health()
            )
            is_healthy = pd_nodes_are_healthy
            health_info = {
                "inference_healthy": inference_is_healthy,
                "pd_nodes_ready": pd_nodes_are_ready,
                "pd_nodes_healthy": pd_nodes_are_healthy,
            }

        health_info.update(
            {
                "message": "Ok" if is_healthy else "Error",
                "pd_master_mode": g_objs.args.pd_master_mode,
                "registered_prefill_nodes": len(pd_manager.prefill_nodes),
                "registered_decode_nodes": len(pd_manager.decode_nodes),
            }
        )
        return JSONResponse(health_info, status_code=200 if is_healthy else 503)

    from lightllm.utils.health_check import health_check

    is_healthy = health_check(g_objs.httpserver_manager.shm_req_manager)
    return JSONResponse(
        {"message": "Ok" if is_healthy else "Error"},
        status_code=200 if is_healthy else 503,
    )


@app.get("/token_load", summary="Get the current server's load of tokens")
async def token_load(request: Request):
    ans_dict = {
        # 当前使用 token 量，估计的负载
        "current_load": [
            float(g_objs.shared_token_load.get_current_load(dp_index)) for dp_index in range(g_objs.args.dp)
        ],
        # 朴素估计的负载，简单将当前请求的输入和输出长度想加得到,目前已未使用，其值与 dynamic_max_load 一样。
        "logical_max_load": [
            float(g_objs.shared_token_load.get_logical_max_load(dp_index)) for dp_index in range(g_objs.args.dp)
        ],
        # 动态估计的最大负载，考虑请求中途退出的情况的负载
        "dynamic_max_load": [
            float(g_objs.shared_token_load.get_dynamic_max_load(dp_index)) for dp_index in range(g_objs.args.dp)
        ],
    }

    if g_objs.args.dp == 1:
        ans_dict = {k: v[0] for k, v in ans_dict.items()}

    return JSONResponse(ans_dict, status_code=200)


@app.post("/generate")
async def generate(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        return await g_objs.g_generate_func(request, g_objs.httpserver_manager)
    except ServerBusyError as e:
        logger.warning(str(e))
        return create_server_busy_response(e)
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        return await g_objs.g_generate_stream_func(request, g_objs.httpserver_manager)
    except ServerBusyError as e:
        logger.warning(str(e))
        return create_server_busy_response(e)
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/get_score")
async def get_score(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        return await lightllm_get_score(request, g_objs.httpserver_manager)
    except ServerBusyError as e:
        logger.warning(str(e))
        return create_server_busy_response(e)
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/")
async def compat_generate(request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    request_dict = await request.json()
    stream = request_dict.pop("stream", False)
    if stream:
        return await generate_stream(request)
    else:
        return await generate(request)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, raw_request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        visual_remote_url = getattr(g_objs.args, "visual_remote_url", None)
        if should_use_visual_proxy(visual_remote_url, request):
            logger.info(
                "[visual-chat-proxy][external_request] method=POST " "path=/v1/chat/completions model=%s stream=%s",
                request.model,
                request.stream,
            )
            runtime = g_objs.visual_proxy_runtime
            if runtime is None:
                raise VisualChatProxyError("Visual proxy runtime is not initialized")
            resp = await visual_chat_completions_impl(
                request=request,
                raw_request=raw_request,
                runtime=runtime,
                main_chat_handler=chat_completions_impl,
            )
            if isinstance(resp, ChatCompletionResponse):
                resp = JSONResponse(resp.model_dump(mode="json", exclude_none=True))
        else:
            resp = await chat_completions_impl(request, raw_request)
    except VisualProxyCapacityError as e:
        logger.warning("%s", str(e))
        return create_error_response(HTTPStatus.TOO_MANY_REQUESTS, str(e), err_type="RateLimitError")
    except VisualProxyTimeoutError as e:
        logger.warning("%s", str(e))
        return create_error_response(HTTPStatus.GATEWAY_TIMEOUT, str(e), err_type="UpstreamTimeoutError")
    except VisualProxyUpstreamError as e:
        logger.warning("%s", str(e))
        return create_error_response(HTTPStatus.BAD_GATEWAY, str(e), err_type="UpstreamError")
    except VisualChatProxyError as e:
        logger.error("%s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.BAD_GATEWAY, str(e), err_type="UpstreamError")
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ServerBusyError as e:
        logger.warning(str(e))
        return create_server_busy_response(e)
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    return resp


@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest, raw_request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )

    try:
        resp = await completions_impl(request, raw_request)
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ServerBusyError as e:
        logger.warning(str(e))
        return create_server_busy_response(e)
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    return resp


@app.post("/v1/messages")
async def anthropic_messages(raw_request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )
    from .api_anthropic import _anthropic_error_response, anthropic_messages_impl

    try:
        return await anthropic_messages_impl(raw_request)
    except ServerBusyError as e:
        logger.warning(str(e))
        g_objs.metric_client.counter_inc("lightllm_request_failure")
        return _anthropic_error_response(HTTPStatus(e.status_code), str(e))
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)


@app.post("/v1/messages/count_tokens")
async def anthropic_count_tokens(raw_request: Request) -> Response:
    from .api_anthropic import _anthropic_error_response, anthropic_count_tokens_impl

    try:
        return await anthropic_count_tokens_impl(raw_request)
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return _anthropic_error_response(HTTPStatus.EXPECTATION_FAILED, f"error: {str(e)}")


@app.post("/v1/responses")
async def openai_responses(raw_request: Request) -> Response:
    if get_env_start_args().run_mode in ["prefill", "decode"]:
        return create_error_response(
            HTTPStatus.EXPECTATION_FAILED, "service in pd mode dont recv reqs from http interface"
        )
    from .api_responses import responses_impl

    try:
        return await responses_impl(raw_request)
    except ServerBusyError as e:
        logger.warning(str(e))
        return create_server_busy_response(e)
    except ValueError as e:
        return create_error_response(HTTPStatus.BAD_REQUEST, str(e))
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.get("/v1/models", response_model=ModelListResponse)
async def get_models(raw_request: Request):
    model_name = g_objs.args.model_name
    max_model_len = g_objs.httpserver_manager.get_real_supported_max_req_total_len()

    if model_name == "default_model_name" and g_objs.args.model_dir:
        model_name = os.path.basename(g_objs.args.model_dir.rstrip("/"))

    return ModelListResponse(
        data=[
            ModelCard(
                id=model_name,
                created=g_objs.model_created,
                max_model_len=max_model_len,
                owned_by=g_objs.args.model_owner or "lightllm",
            )
        ]
    )


@app.get("/tokens")
@app.post("/tokens")
async def tokens(request: Request):
    try:
        request_dict = await request.json()
        prompt = request_dict.pop("text")
        sample_params_dict = request_dict.pop("parameters", {})

        sampling_params = SamplingParams()
        sampling_params.init(tokenizer=g_objs.httpserver_manager.tokenizer, **sample_params_dict)
        sampling_params.verify()

        multimodal_params_dict = request_dict.get("multimodal_params", {})
        multimodal_params = MultimodalParams(**multimodal_params_dict)
        await multimodal_params.verify_and_preload(request)
        return JSONResponse(
            {
                "ntokens": g_objs.httpserver_manager.tokens(
                    prompt, multimodal_params, sampling_params, sample_params_dict
                )
            },
            status_code=200,
        )
    except ClientDisconnected as e:
        logger.warning(str(e))
        return Response(status_code=499)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, f"error: {str(e)}")


@app.get("/metrics")
async def metrics() -> Response:
    data = await g_objs.metric_client.generate_latest()
    response = Response(data)
    response.mimetype = "text/plain"
    return response


# RL 控制面接口（abort / pause / flush / memory / weight update），见 api_http_rl.py
from .api_http_rl import router as rl_router

app.include_router(rl_router)

# PD 分离控制面接口（P/D 注册与 KV 状态上报），见 api_http_pd.py
from .api_http_pd import router as pd_router

app.include_router(pd_router)


@app.get("/profiler_start")
async def profiler_start() -> Response:
    if g_objs.args.enable_profiling:
        await g_objs.httpserver_manager.profiler_cmd("start")
        return JSONResponse({"status": "ok"})
    else:
        return JSONResponse({"message": "Profiling support not enabled"}, status_code=400)


@app.get("/profiler_stop")
async def profiler_stop() -> Response:
    if g_objs.args.enable_profiling:
        await g_objs.httpserver_manager.profiler_cmd("stop")
        return JSONResponse({"status": "ok"})
    else:
        return JSONResponse({"message": "Profiling support not enabled"}, status_code=400)


@app.on_event("shutdown")
async def shutdown():
    logger.info("Received signal to shutdown. Performing graceful shutdown...")
    if g_objs.visual_proxy_runtime is not None:
        await g_objs.visual_proxy_runtime.close()
        g_objs.visual_proxy_runtime = None
    await asyncio.sleep(3)

    # 杀掉所有子进程
    import psutil
    import signal

    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        os.kill(child.pid, signal.SIGKILL)
    logger.info("Graceful shutdown completed.")
    return


@app.on_event("startup")
async def startup_event():
    logger.info("server start up")
    loop = asyncio.get_event_loop()
    g_objs.set_args(get_env_start_args())
    if getattr(g_objs.args, "visual_remote_url", None):
        g_objs.visual_proxy_runtime = VisualProxyRuntime(VisualProxySettings.from_args(g_objs.args))
    loop.create_task(g_objs.httpserver_manager.handle_loop())
    logger.info(f"server start up ok, loop use is {asyncio.get_event_loop()}")
    return
