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
import time
import torch
import uvloop
import sys

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

logger = init_logger(__name__)

TIMEOUT_KEEP_ALIVE = 5  # seconds.

g_id_gen = ReqIDGenerator()
app = FastAPI()

isFirst = True


def create_error_response(status_code: HTTPStatus, message: str) -> JSONResponse:
    return JSONResponse({"message": message}, status_code=status_code.value)


@app.get("/healthz")
@app.get("/health")
def healthcheck():
    return "OK"


@app.post("/generate")
async def generate(request: Request) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    request_dict = await request.json()
    prompt = request_dict.pop("inputs")
    sample_params_dict = request_dict["parameters"]
    return_details = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()
    multimodal_params_dict = request_dict.get("multimodal_params", {})
    multimodal_params = MultimodalParams(**multimodal_params_dict)

    request_id = g_id_gen.generate_id()
    results_generator = httpserver_manager.generate(prompt, sampling_params, request_id, multimodal_params)

    # Non-streaming case
    final_output = []
    count_output_tokens = 0
    tokens = []
    prompt_logprobs = None
    prompt_token_ids = None
    is_first_metadata = True
    async for request_output, metadata, finish_status in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await httpserver_manager.abort(request_id)
            return Response(status_code=499)

        # when set "--return_all_prompt_logprobs", the first token metadata will contains
        # prompt_logprobs and prompt_token_ids
        if is_first_metadata:
            prompt_logprobs = metadata.get("prompt_logprobs", None)
            prompt_token_ids = metadata.get("prompt_token_ids", None)
            if prompt_logprobs is not None:
                del metadata["prompt_logprobs"]
            if prompt_token_ids is not None:
                del metadata["prompt_token_ids"]
            is_first_metadata = False

        count_output_tokens += 1
        final_output.append(request_output)
        if return_details:
            metadata["text"] = request_output
            tokens.append(metadata)

    assert final_output is not None
    ret = {
        "generated_text": ["".join(final_output)],
        "count_output_tokens": count_output_tokens,
        "finish_reason": finish_status.get_finish_reason(),
    }
    if return_details:
        ret["tokens"] = tokens
    if prompt_token_ids is not None:
        ret["prompt_token_ids"] = prompt_token_ids
    if prompt_logprobs is not None:
        ret["prompt_logprobs"] = prompt_logprobs
    return Response(content=json.dumps(ret, ensure_ascii=False).encode("utf-8"))


@app.post("/generate_stream")
async def generate_stream(request: Request) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

    request_dict = await request.json()
    prompt = request_dict.pop("inputs")
    sample_params_dict = request_dict["parameters"]
    _ = sample_params_dict.pop("return_details", False)
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()
    multimodal_params_dict = request_dict.get("multimodal_params", {})
    multimodal_params = MultimodalParams(**multimodal_params_dict)

    request_id = g_id_gen.generate_id()
    results_generator = httpserver_manager.generate(prompt, sampling_params, request_id, multimodal_params)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output, metadata, finish_status in results_generator:
            ret = {
                "token": {
                    "id": metadata.get("id", None),
                    "text": request_output,
                    "logprob": metadata.get("logprob", None),
                    "special": metadata.get("special", False),
                    "count_output_tokens": metadata.get("count_output_tokens", 0),
                },
                "generated_text": None,
                "finished": finish_status.is_finished(),
                "finish_reason": finish_status.get_finish_reason(),
                "details": None,
            }

            yield ("data:" + json.dumps(ret, ensure_ascii=False) + "\n\n").encode("utf-8")

    async def abort_request() -> None:
        await httpserver_manager.abort(request_id)

    background_tasks = BackgroundTasks()
    # Abort the request if the client disconnects.
    background_tasks.add_task(abort_request)

    return StreamingResponse(stream_results(), media_type="text/event-stream", background=background_tasks)


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, raw_request: Request) -> Response:
    global isFirst
    if isFirst:
        loop = asyncio.get_event_loop()
        loop.create_task(httpserver_manager.handle_loop())
        isFirst = False

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
    results_generator = httpserver_manager.generate(prompt, sampling_params, request_id, multimodal_params)

    # Non-streaming case
    if not request.stream:
        final_output = []
        prompt_tokens = -1
        completion_tokens = 0
        async for request_output, metadata, _ in results_generator:
            if await raw_request.is_disconnected():
                # Abort the request if the client disconnects.
                await httpserver_manager.abort(request_id)
                return Response(status_code=499)
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
        async for request_output, metadata, _ in results_generator:
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

    return StreamingResponse(stream_results(), media_type="text/event-stream", background=background_tasks)


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
    parser.add_argument("--eos_id", nargs='+', type=int, default=[2], help="eos stop token id")
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

    parser.add_argument("--use_dynamic_prompt_cache", action="store_true", help="use_dynamic_prompt_cache test")

    parser.add_argument("--splitfuse_mode", action="store_true", help="use splitfuse mode")
    
    parser.add_argument("--splitfuse_block_size", type=int, default=256, help="splitfuse block size")
    parser.add_argument(
        "--enable_multimodal", action="store_true", help="Whether or not to allow to load additional multimodal models."
    )
    parser.add_argument(
        "--cache_capacity", type=int, default=200, help="cache server capacity for multimodal resources"
    )
    parser.add_argument(
        "--cache_reserved_ratio", type=float, default=0.5, help="cache server reserved capacity ratio after clear"
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

    args = parser.parse_args()

    assert args.max_req_input_len < args.max_req_total_len
    assert args.max_req_total_len <= args.max_total_token_num

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
