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
import os
import pickle
from .build_prompt import build_prompt, init_tokenizer

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import ujson as json
from http import HTTPStatus
import uuid
import multiprocessing as mp
from typing import AsyncGenerator, Union
from typing import Callable
from lightllm.server import TokenLoad
from fastapi import BackgroundTasks, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse, JSONResponse
from lightllm.server.core.objs.sampling_params import SamplingParams
from .multimodal_params import MultimodalParams
from .httpserver.manager import HttpServerManager
from .httpserver_for_pd_master.manager import HttpServerManagerForPDMaster
from .api_lightllm import lightllm_get_score, lightllm_pd_generate_stream
from lightllm.utils.envs_utils import get_env_start_args

from .api_models import (
    ChatCompletionRequest,
    UsageInfo,
    ChatMessage,
    ChatCompletionResponseChoice,
    ChatCompletionResponse,
    DeltaMessage,
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
)

from lightllm.utils.log_utils import init_logger
from lightllm.server.metrics.manager import MetricClient
from lightllm.utils.envs_utils import get_unique_server_name
from dataclasses import dataclass

logger = init_logger(__name__)


@dataclass
class G_Objs:
    app: FastAPI = None
    metric_client: MetricClient = None
    args: object = None
    g_generate_func: Callable = None
    g_generate_stream_func: Callable = None
    httpserver_manager: Union[HttpServerManager, HttpServerManagerForPDMaster] = None
    shared_token_load: TokenLoad = None

    def set_args(self, args):
        self.args = args
        from .api_lightllm import lightllm_generate, lightllm_generate_stream
        from .api_tgi import tgi_generate_impl, tgi_generate_stream_impl

        if args.use_tgi_api:
            self.g_generate_func = tgi_generate_impl
            self.g_generate_stream_func = tgi_generate_stream_impl
        else:
            self.g_generate_func = lightllm_generate
            self.g_generate_stream_func = lightllm_generate_stream

        if args.run_mode == "pd_master":
            self.metric_client = MetricClient(args.metric_port)
            self.httpserver_manager = HttpServerManagerForPDMaster(
                args,
                metric_port=args.metric_port,
            )
        else:
            init_tokenizer(args)  # for openai api
            SamplingParams.load_generation_cfg(args.model_dir)
            self.metric_client = MetricClient(args.metric_port)
            self.httpserver_manager = HttpServerManager(
                args,
                router_port=args.router_port,
                cache_port=args.cache_port,
                detokenization_pub_port=args.detokenization_pub_port,
                visual_port=args.visual_port,
                enable_multimodal=args.enable_multimodal,
                metric_port=args.metric_port,
            )
            dp_size_in_node = max(1, args.dp // args.nnodes)  # 兼容多机纯tp的运行模式，这时候 1 // 2 == 0, 需要兼容
            self.shared_token_load = TokenLoad(f"{get_unique_server_name()}_shared_token_load", dp_size_in_node)


g_objs = G_Objs()

app = FastAPI()
g_objs.app = app


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    g_objs.metric_client.counter_inc("lightllm_request_failure")
    return JSONResponse({"message": message}, status_code=status_code.value)


@app.get("/liveness")
@app.post("/liveness")
def liveness():
    return {"status": "ok"}


@app.get("/readiness")
@app.post("/readiness")
def readiness():
    return {"status": "ok"}


@app.get("/get_model_name")
@app.post("/get_model_name")
def get_model_name():
    return {"model_name": g_objs.args.model_name}


@app.get("/healthz", summary="Check server health")
@app.get("/health", summary="Check server health")
@app.head("/health", summary="Check server health")
async def healthcheck(request: Request):
    if os.environ.get("DEBUG_HEALTHCHECK_RETURN_FAIL") == "true":
        return JSONResponse({"message": "Error"}, status_code=503)
    from lightllm.utils.health_check import health_check, health_obj

    asyncio.create_task(health_check(g_objs.args, g_objs.httpserver_manager, None))
    if health_obj.is_health():
        return JSONResponse({"message": "Ok"}, status_code=200)
    else:
        return JSONResponse({"message": "Error"}, status_code=503)


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
    try:
        return await g_objs.g_generate_func(request, g_objs.httpserver_manager)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    try:
        return await g_objs.g_generate_stream_func(request, g_objs.httpserver_manager)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/pd_generate_stream")
async def pd_generate_stream(request: Request) -> Response:
    try:
        return await lightllm_pd_generate_stream(request, g_objs.httpserver_manager)
    except Exception as e:
        logger.error("An error occurred: %s", str(e), exc_info=True)
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/get_score")
async def get_score(request: Request) -> Response:
    try:
        return await lightllm_get_score(request, g_objs.httpserver_manager)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@app.post("/")
async def compat_generate(request: Request) -> Response:
    request_dict = await request.json()
    stream = request_dict.pop("stream", False)
    if stream:
        return await generate_stream(request)
    else:
        return await generate(request)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, raw_request: Request) -> Response:

    if request.logit_bias is not None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The logit_bias parameter is not currently supported",
        )

    if request.function_call != "none":
        return create_error_response(HTTPStatus.BAD_REQUEST, "The function call feature is not supported")

    created_time = int(time.time())
    prompt = await build_prompt(request)
    sampling_params_dict = {
        "do_sample": request.do_sample,
        "presence_penalty": request.presence_penalty,
        "frequency_penalty": request.frequency_penalty,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "ignore_eos": request.ignore_eos,
        "max_new_tokens": request.max_tokens,
        "stop_sequences": request.stop,
        "n": request.n,
        "best_of": request.n,
        "add_special_tokens": False,
    }
    sampling_params = SamplingParams()
    sampling_params.init(tokenizer=g_objs.httpserver_manager.tokenizer, **sampling_params_dict)

    sampling_params.verify()
    multimodal_params = MultimodalParams(images=[])

    results_generator = g_objs.httpserver_manager.generate(
        prompt, sampling_params, multimodal_params, request=raw_request
    )

    # Non-streaming case
    if not request.stream:
        final_output_dict = collections.defaultdict(list)
        count_output_tokens_dict = collections.defaultdict(lambda: 0)
        finish_reason_dict = {}
        prompt_tokens_dict = {}
        completion_tokens = 0
        async for sub_req_id, request_output, metadata, finish_status in results_generator:
            from .req_id_generator import convert_sub_id_to_group_id

            group_request_id = convert_sub_id_to_group_id(sub_req_id)
            count_output_tokens_dict[sub_req_id] += 1
            final_output_dict[sub_req_id].append(request_output)
            if finish_status.is_finished():
                finish_reason_dict[sub_req_id] = finish_status.get_finish_reason()
                prompt_tokens_dict[sub_req_id] = metadata["prompt_tokens"]
        choices = []
        sub_ids = list(final_output_dict.keys())[: request.n]
        for i in range(request.n):
            sub_req_id = sub_ids[i]
            prompt_tokens = prompt_tokens_dict[sub_req_id]
            completion_tokens = count_output_tokens_dict[sub_req_id]
            usage = UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )
            chat_message = ChatMessage(role="assistant", content="".join(final_output_dict[sub_req_id]))
            choice = ChatCompletionResponseChoice(
                index=i, message=chat_message, finish_reason=finish_reason_dict[sub_req_id]
            )
            choices.append(choice)
        resp = ChatCompletionResponse(
            id=group_request_id, created=created_time, model=request.model, choices=choices, usage=usage
        )
        return resp

    if sampling_params.n != 1:
        raise Exception("stream api only support n = 1")

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        finish_reason = None
        from .req_id_generator import convert_sub_id_to_group_id

        async for sub_req_id, request_output, metadata, finish_status in results_generator:
            group_request_id = convert_sub_id_to_group_id(sub_req_id)

            delta_message = DeltaMessage(role="assistant", content=request_output)
            if finish_status.is_finished():
                finish_reason = finish_status.get_finish_reason()
            stream_choice = ChatCompletionStreamResponseChoice(
                index=0, delta=delta_message, finish_reason=finish_reason
            )
            stream_resp = ChatCompletionStreamResponse(
                id=group_request_id,
                created=created_time,
                model=request.model,
                choices=[stream_choice],
            )
            yield ("data: " + json.dumps(stream_resp.dict(), ensure_ascii=False) + "\n\n").encode("utf-8")

    background_tasks = BackgroundTasks()
    return StreamingResponse(stream_results(), media_type="text/event-stream", background=background_tasks)


@app.get("/tokens")
@app.post("/tokens")
async def tokens(request: Request):
    try:
        request_dict = await request.json()
        prompt = request_dict.pop("text")
        parameters = request_dict.pop("parameters", {})
        return JSONResponse({"ntokens": g_objs.httpserver_manager.tokens(prompt, parameters)}, status_code=200)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, f"error: {str(e)}")


@app.get("/metrics")
async def metrics() -> Response:
    data = await g_objs.metric_client.generate_latest()
    response = Response(data)
    response.mimetype = "text/plain"
    return response


@app.websocket("/pd_register")
async def register_and_keep_alive(websocket: WebSocket):
    await websocket.accept()
    client_ip, client_port = websocket.client
    logger.info(f"Client connected from IP: {client_ip}, Port: {client_port}")
    regist_json = json.loads(await websocket.receive_text())
    logger.info(f"recieved regist_json {regist_json}")
    await g_objs.httpserver_manager.register_pd(regist_json, websocket)

    try:
        while True:
            # 等待接收消息，设置超时为10秒
            data = await websocket.receive_bytes()
            obj = pickle.loads(data)
            await g_objs.httpserver_manager.put_to_handle_queue(obj)

    except (WebSocketDisconnect, Exception, RuntimeError) as e:
        logger.error(f"client {regist_json} has error {str(e)}")
        logger.exception(str(e))
    finally:
        logger.error(f"client {regist_json} removed")
        await g_objs.httpserver_manager.remove_pd(regist_json)
    return


@app.websocket("/kv_move_status")
async def kv_move_status(websocket: WebSocket):
    await websocket.accept()
    client_ip, client_port = websocket.client
    logger.info(f"kv_move_status Client connected from IP: {client_ip}, Port: {client_port}")
    try:
        while True:
            # 等待接收消息，设置超时为10秒
            data = await websocket.receive_text()
            json_data = json.loads(data)
            from .pd_io_struct import UpKVStatus

            upkv_status = UpKVStatus(**json_data)
            await g_objs.httpserver_manager.update_req_status(upkv_status)
    except (WebSocketDisconnect, Exception, RuntimeError) as e:
        logger.error(f"kv_move_status client {(client_ip, client_port)} has error {str(e)}")
        logger.exception(str(e))
    return


@app.on_event("shutdown")
async def shutdown():
    logger.info("Received signal to shutdown. Performing graceful shutdown...")
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
    loop.create_task(g_objs.httpserver_manager.handle_loop())
    logger.info(f"server start up ok, loop use is {asyncio.get_event_loop()}")
    return
