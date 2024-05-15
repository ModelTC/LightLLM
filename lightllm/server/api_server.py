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
import torch
import uvloop
import sys
import os

from .build_prompt import build_prompt

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import argparse
import json
from http import HTTPStatus
import uuid
import multiprocessing as mp
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response, StreamingResponse, JSONResponse
import uvicorn
from .sampling_params import SamplingParams
from .multimodal_params import MultimodalParams
from .httpserver.manager import HttpServerManager
from .detokenization.manager import start_detokenization_process
from .router.manager import start_router_process
from .embed_cache.manager import start_cache_manager
from .visualserver.manager import start_visual_process
from .req_id_generator import ReqIDGenerator
from .api_tgi import tgi_generate_impl, tgi_generate_stream_impl
from .api_lightllm import lightllm_generate, lightllm_generate_stream

from lightllm.utils.net_utils import alloc_can_use_network_port
from lightllm.utils.start_utils import start_submodule_processes

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

from .metrics import monitor
from prometheus_client import generate_latest

logger = init_logger(__name__)

TIMEOUT_KEEP_ALIVE = 5  # seconds.

g_id_gen = ReqIDGenerator()
app = FastAPI()
server = uvicorn.Server(uvicorn.Config(app))

isFirst = True


def first_set_handle_loop():
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False
    return


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse({"message": message}, status_code=status_code.value)


@app.get("/liveness")
@app.post("/liveness")
def liveness():
    return {"status": "ok"}


@app.get("/readiness")
@app.post("/readiness")
def readiness():
    return {"status": "ok"}


@app.get("/healthz")
@app.get("/health")
@app.head("/health")
async def healthcheck(request: Request):
    first_set_handle_loop()
    if os.environ.get("DEBUG_HEALTHCHECK_RETURN_FAIL") == "true":
        return JSONResponse({"message": "Error"}, status_code=404)

    from lightllm.utils.health_check import health_check

    if health_check(httpserver_manager, g_id_gen, request):
        return JSONResponse({"message": "Ok"}, status_code=200)
    else:
        return JSONResponse({"message": "Error"}, status_code=404)


@app.get("/token_load")
async def token_load(request: Request):
    return JSONResponse(
        {
            # 当前使用token量，估计的负载
            "current_load": float(shared_token_load.get_current_load()),
            # 朴素估计的负载，简单将当前请求的输入和输出长度想加得到,目前已未使用，其值与dynamic_max_load一样。
            "logical_max_load": float(shared_token_load.get_logical_max_load()),
            # 动态估计的最大负载，考虑请求中途退出的情况的负载
            "dynamic_max_load": float(shared_token_load.get_dynamic_max_load()),
        },
        status_code=200,
    )


@monitor.histogram_timer("lightllm_request_duration")
@app.post("/generate")
async def generate(request: Request) -> Response:
    first_set_handle_loop()
    try:
        return await g_generate_func(request, g_id_gen, httpserver_manager)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@monitor.histogram_timer("lightllm_request_duration")
@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    first_set_handle_loop()
    try:
        return await g_generate_stream_func(request, g_id_gen, httpserver_manager)
    except Exception as e:
        return create_error_response(HTTPStatus.EXPECTATION_FAILED, str(e))


@monitor.histogram_timer("lightllm_request_duration")
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, raw_request: Request) -> Response:
    monitor.counter_inc("lightllm_request_count")
    first_set_handle_loop()

    if request.logit_bias is not None:
        return create_error_response(
            HTTPStatus.BAD_REQUEST,
            "The logit_bias parameter is not currently supported",
        )

    if request.n > 1:
        return create_error_response(HTTPStatus.BAD_REQUEST, "The n parameter currently only supports 1")

    if request.function_call != "none":
        return create_error_response(HTTPStatus.BAD_REQUEST, "The function call feature is not supported")

    created_time = int(time.time())
    prompt = await build_prompt(request)
    sampling_params = SamplingParams(
        do_sample=request.do_sample,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        ignore_eos=request.ignore_eos,
        max_new_tokens=request.max_tokens,
        stop_sequences=request.stop,
    )
    sampling_params.verify()
    multimodal_params = MultimodalParams(images=[])

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    results_generator = httpserver_manager.generate(
        prompt, sampling_params, request_id, multimodal_params, request=raw_request
    )

    # Non-streaming case
    if not request.stream:
        final_output = []
        prompt_tokens = -1
        completion_tokens = 0
        async for sub_req_id, request_output, metadata, _ in results_generator:
            completion_tokens += 1
            if prompt_tokens == -1:
                prompt_tokens = metadata["prompt_tokens"]
            final_output.append(request_output)

        usage = UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        chat_message = ChatMessage(role="assistant", content="".join(final_output))
        choice = ChatCompletionResponseChoice(index=0, message=chat_message)
        resp = ChatCompletionResponse(
            id=request_id, created=created_time, model=request.model, choices=[choice], usage=usage
        )
        return resp

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for sub_req_id, request_output, metadata, _ in results_generator:
            delta_message = DeltaMessage(role="assistant", content=request_output)

            stream_choice = ChatCompletionStreamResponseChoice(index=0, delta=delta_message)

            stream_resp = ChatCompletionStreamResponse(
                id=request_id,
                created=created_time,
                model=request.model,
                choices=[stream_choice],
            )
            yield ("data: " + stream_resp.json(ensure_ascii=False) + "\n\n").encode("utf-8")

    async def abort_request() -> None:
        await httpserver_manager.abort(request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)
    monitor.counter_inc("lightllm_request_success")
    return StreamingResponse(stream_results(), media_type="text/event-stream", background=background_tasks)


@app.get("/metrics")
async def metrics() -> Response:
    metrics_data = generate_latest()
    response = Response(metrics_data)
    response.mimetype = "text/plain"
    return response


@app.on_event("shutdown")
async def shutdown():
    logger.info("Received signal to shutdown. Performing graceful shutdown...")
    await asyncio.sleep(3)
    logger.info("Graceful shutdown completed.")

    # 杀掉所有子进程
    import psutil
    import signal

    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        os.kill(child.pid, signal.SIGKILL)

    server.should_exit = True
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)

    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="the model weight dir path, the app will load config, weights and tokenizer from this dir",
    )
    parser.add_argument(
        "--tokenizer_mode",
        type=str,
        default="slow",
        help="""tokenizer load mode, can be slow or auto, slow mode load fast but run slow, slow mode is
          good for debug and test, when you want to get best performance, try auto mode""",
    )
    parser.add_argument(
        "--load_way",
        type=str,
        default="HF",
        help="""the way of loading model weights, the default is HF(Huggingface format), llama also supports
          DS(Deepspeed)""",
    )
    parser.add_argument(
        "--max_total_token_num",
        type=int,
        default=6000,
        help="the total token nums the gpu and model can support, equals = max_batch * (input_len + output_len)",
    )
    parser.add_argument(
        "--batch_max_tokens",
        type=int,
        default=None,
        help="max tokens num for new cat batch, it control prefill batch size to Preventing OOM",
    )
    parser.add_argument("--eos_id", nargs="+", type=int, default=[2], help="eos stop token id")
    parser.add_argument(
        "--running_max_req_size", type=int, default=1000, help="the max size for forward requests in the same time"
    )
    parser.add_argument("--tp", type=int, default=1, help="model tp parral size, the default is 1")
    parser.add_argument("--max_req_input_len", type=int, default=2048, help="the max value for req input tokens num")
    parser.add_argument(
        "--max_req_total_len", type=int, default=2048 + 1024, help="the max value for req_input_len + req_output_len"
    )
    parser.add_argument(
        "--nccl_port", type=int, default=28765, help="the nccl_port to build a distributed environment for PyTorch"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=[],
        nargs="+",
        help="""Model mode: [triton_int8kv | ppl_int8kv | ppl_fp16 | triton_flashdecoding
                        | triton_gqa_attention | triton_gqa_flashdecoding]
                        [triton_w4a16 | triton_w8a16 | lmdeploy_w4a16 | ppl_w4a16 | ppl_w8a8],
                        triton_flashdecoding mode is for long context, current support llama llama2 qwen;
                        triton_gqa_attention and triton_gqa_flashdecoding is fast kernel for model which use GQA;
                        triton_int8kv mode use int8 to store kv cache, can increase token capacity, use triton kernel;
                        ppl_int8kv mode use int8 to store kv cache, and use ppl fast kernel;
                        ppl_fp16 mode use ppl fast fp16 decode attention kernel;
                        triton_int8weight and triton_int4weight and lmdeploy_int4weight or ppl_int4weight mode
                        use int8 and int4 to store weights;
                        you need to read source code to make sure the supported detail mode for all models""",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
    )
    parser.add_argument("--disable_log_stats", action="store_true", help="disable logging throughput stats.")
    parser.add_argument("--log_stats_interval", type=int, default=10, help="log stats interval in second.")

    parser.add_argument("--router_token_ratio", type=float, default=0.0, help="token ratio to control router dispatch")
    parser.add_argument(
        "--router_max_new_token_len", type=int, default=1024, help="the request max new token len for router"
    )

    parser.add_argument(
        "--router_max_wait_tokens",
        type=int,
        default=10,
        help="schedule new requests after every router_max_wait_tokens decode steps.",
    )

    parser.add_argument("--use_dynamic_prompt_cache", action="store_true", help="use_dynamic_prompt_cache test")

    parser.add_argument("--splitfuse_block_size", type=int, default=256, help="splitfuse block size")

    parser.add_argument("--splitfuse_mode", action="store_true", help="use splitfuse mode")
    parser.add_argument("--beam_mode", action="store_true", help="use beamsearch mode")
    parser.add_argument("--diverse_mode", action="store_true", help="diversity generation mode")
    parser.add_argument("--token_healing_mode", action="store_true", help="code model infer mode")

    parser.add_argument(
        "--enable_multimodal", action="store_true", help="Whether or not to allow to load additional multimodal models."
    )
    parser.add_argument(
        "--cache_capacity", type=int, default=200, help="cache server capacity for multimodal resources"
    )
    parser.add_argument(
        "--cache_reserved_ratio", type=float, default=0.5, help="cache server reserved capacity ratio after clear"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["fp16", "float16", "bf16", "bfloat16", "fp32", "float32"],
        default="float16",
        help="the data type of the model weight",
    )
    parser.add_argument("--return_all_prompt_logprobs", action="store_true", help="return all prompt tokens logprobs")

    parser.add_argument(
        "--long_truncation_mode",
        type=str,
        choices=[None, "head", "center"],
        default=None,
        help="""use to select the handle way when input token len > max_req_input_len.
                        None : raise Exception
                        head : remove some head tokens to make input token len <= max_req_input_len
                        center : remove some tokens in center loc to make input token len <= max_req_input_len""",
    )
    parser.add_argument("--use_tgi_api", action="store_true", help="use tgi input and ouput format")
    parser.add_argument(
        "--health_monitor", action="store_true", help="check the health of service and restart when error"
    )

    args = parser.parse_args()

    global g_generate_func
    global g_generate_stream_func
    if args.use_tgi_api:
        g_generate_func = tgi_generate_impl
        g_generate_stream_func = tgi_generate_stream_impl
    else:
        g_generate_func = lightllm_generate
        g_generate_stream_func = lightllm_generate_stream

    logger.info(f"use tgi api: {args.use_tgi_api}")

    assert args.max_req_input_len < args.max_req_total_len
    assert args.max_req_total_len <= args.max_total_token_num
    monitor.init_api_server_monitor(args)

    # 这些模式不能同时设置。
    assert [args.splitfuse_mode, args.beam_mode, args.diverse_mode, args.token_healing_mode].count(True) <= 1
    # 部分模式目前还无法与dynamic_prompt_cache一起跑，to do。
    if args.use_dynamic_prompt_cache:
        assert args.beam_mode is False
        assert args.token_healing_mode is False

    # 部分模式还不能支持与高级动态调度算法协同，to do.
    if args.beam_mode or args.diverse_mode:
        assert args.router_token_ratio == 0.0

    if not args.splitfuse_mode:
        # 普通模式下
        if args.batch_max_tokens is None:
            batch_max_tokens = int(1 / 6 * args.max_total_token_num)
            batch_max_tokens = max(batch_max_tokens, args.max_req_total_len)
            args.batch_max_tokens = batch_max_tokens
        else:
            assert args.batch_max_tokens >= args.max_req_total_len, "batch_max_tokens must >= max_req_total_len"
    else:
        # splitfuse 模式下
        # assert args.batch_max_tokens is not None, "need to set by yourself"
        if args.batch_max_tokens is None:
            batch_max_tokens = int(1 / 6 * args.max_total_token_num)
            batch_max_tokens = max(batch_max_tokens, args.splitfuse_block_size)
            args.batch_max_tokens = batch_max_tokens

    can_use_ports = alloc_can_use_network_port(num=5 + args.tp, used_nccl_port=args.nccl_port)
    router_port, detokenization_port, httpserver_port, visual_port, cache_port = can_use_ports[0:5]
    model_rpc_ports = can_use_ports[5:]

    if args.enable_multimodal:
        start_submodule_processes(
            start_funcs=[
                start_cache_manager,
            ],
            start_args=[(cache_port, args)],
        )

    # help to manage data stored on Ceph
    if "s3://" in args.model_dir:
        from lightllm.utils.petrel_helper import s3_model_prepare

        s3_model_prepare(args.model_dir)

    global httpserver_manager
    httpserver_manager = HttpServerManager(
        args,
        router_port=router_port,
        cache_port=cache_port,
        visual_port=visual_port,
        httpserver_port=httpserver_port,
        enable_multimodal=args.enable_multimodal,
    )

    start_submodule_processes(
        start_funcs=[start_router_process, start_detokenization_process],
        start_args=[
            (args, router_port, detokenization_port, model_rpc_ports),
            (args, detokenization_port, httpserver_port),
        ],
    )
    if args.enable_multimodal:
        start_submodule_processes(
            start_funcs=[
                start_visual_process,
            ],
            start_args=[
                (args, router_port, visual_port, cache_port),
            ],
        )

    if "s3://" in args.model_dir:
        from lightllm.utils.petrel_helper import s3_model_clear

        s3_model_clear(args.model_dir)

    if args.health_monitor:
        from lightllm.server.health_monitor.manager import start_health_check_process

        start_submodule_processes(start_funcs=[start_health_check_process], start_args=[(args,)])

    # 共享变量，用于获取router端调度分析得到的机器负载信息
    from lightllm.server import TokenLoad

    global shared_token_load
    shared_token_load = TokenLoad(f"{str(args.nccl_port)}_shared_token_load")

    server.install_signal_handlers()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
        loop="uvloop",
    )


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn"),  # this code will not be ok for settings to fork to subprocess
    main()
