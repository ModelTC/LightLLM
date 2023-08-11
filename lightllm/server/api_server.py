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
import uvloop
import sys
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import argparse
import json
import uuid
import multiprocessing as mp
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn
from .sampling_params import SamplingParams
from .httpserver.manager import HttpServerManager
from .detokenization.manager import start_detokenization_process
from .router.manager import start_router_process

from lightllm.utils.net_utils import alloc_can_use_network_port
from lightllm.common.configs.config import setting

TIMEOUT_KEEP_ALIVE = 5  # seconds.

app = FastAPI()

isFirst = True


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
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()
    
    request_id = uuid.uuid4().hex
    results_generator = httpserver_manager.generate(prompt, sampling_params, request_id)

    # Non-streaming case
    final_output = []
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await httpserver_manager.abort(request_id)
            return Response(status_code=499)
        final_output.append(request_output)

    assert final_output is not None
    ret = {"generated_text": ["".join(final_output)]}
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
    sampling_params = SamplingParams(**sample_params_dict)
    sampling_params.verify()
    
    request_id = uuid.uuid4().hex
    results_generator = httpserver_manager.generate(prompt, sampling_params, request_id)

    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            ret = {"token":{
                         "id": None, 
                         "text":request_output,
                         "logprob":None,
                         "special":False},
                         "generated_text":None,
                         "details":None}
            yield ("data:"+ json.dumps(ret, ensure_ascii=False) + f"\n\n").encode("utf-8")

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
    parser.add_argument("--model_dir", type=str, default=None,
                        help="the model weight dir path, the app will load config, weights and tokenizer from this dir")
    parser.add_argument("--tokenizer_mode", type=str, default="slow",
                        help="""tokenizer load mode, can be slow or auto, slow mode load fast but run slow, slow mode is good for debug and test, 
                        when you want to get best performance, try auto mode""")
    parser.add_argument("--max_total_token_num", type=int, default=6000,
                        help="the total token nums the gpu and model can support, equals = max_batch * (input_len + output_len)")
    parser.add_argument("--batch_max_tokens", type=int, default=None,
                        help="max tokens num for new cat batch, it control prefill batch size to Preventing OOM")
    parser.add_argument("--eos_id", type=int, default=2,
                        help="eos stop token id")
    parser.add_argument("--running_max_req_size", type=int, default=1000,
                        help="the max size for forward requests in the same time")
    parser.add_argument("--tp", type=int, default=1,
                        help="model tp parral size, the default is 1")
    parser.add_argument("--max_req_input_len", type=int, default=2048,
                        help="the max value for req input tokens num")
    parser.add_argument("--max_req_total_len", type=int, default=2048 + 1024,
                        help="the max value for req_input_len + req_output_len")
    parser.add_argument("--nccl_port", type=int, default=28765,
                        help="the nccl_port to build a distributed environment for PyTorch")
    parser.add_argument("--mode", type=str, default="",
                        help="model inference mode, now only support 'int8kv' or '' default")
    parser.add_argument("--trust_remote_code", action='store_true',
                        help="Whether or not to allow for custom models defined on the Hub in their own modeling files.")
    parser.add_argument("--disable_log_stats", action='store_true',
                        help="disable logging throughput stats.")
    parser.add_argument("--log_stats_interval", type=int, default=10,
                        help="log stats interval in second.")
    args = parser.parse_args()

    assert args.max_req_input_len < args.max_req_total_len
    setting['max_req_total_len'] = args.max_req_total_len
    setting['nccl_port'] = args.nccl_port

    if args.batch_max_tokens is None:
        batch_max_tokens = int(1 / 6 * args.max_total_token_num)
        batch_max_tokens = max(batch_max_tokens, args.max_req_total_len)
        args.batch_max_tokens = batch_max_tokens
    else:
        assert args.batch_max_tokens >= args.max_req_total_len, "batch_max_tokens must >= max_req_total_len"

    can_use_ports = alloc_can_use_network_port(
        num=3 + args.tp, used_nccl_port=args.nccl_port)
    router_port, detokenization_port, httpserver_port = can_use_ports[0:3]
    model_rpc_ports = can_use_ports[3:]

    global httpserver_manager
    httpserver_manager = HttpServerManager(args.model_dir,
                                           args.tokenizer_mode,
                                           router_port=router_port,
                                           httpserver_port=httpserver_port,
                                           total_token_num=args.max_total_token_num,
                                           max_req_input_len=args.max_req_input_len,
                                           max_req_total_len=args.max_req_total_len,
                                           trust_remote_code=args.trust_remote_code)
    pipe_router_reader, pipe_router_writer = mp.Pipe(duplex=False)
    pipe_detoken_reader, pipe_detoken_writer = mp.Pipe(duplex=False)
    proc_router = mp.Process(target=start_router_process, args=(
        args, router_port, detokenization_port, model_rpc_ports, args.mode, pipe_router_writer))
    proc_router.start()
    proc_detoken = mp.Process(target=start_detokenization_process, args=(
        args, detokenization_port, httpserver_port, pipe_detoken_writer, args.trust_remote_code))
    proc_detoken.start()
    
    # wait load model ready
    router_init_state = pipe_router_reader.recv()
    detoken_init_state = pipe_detoken_reader.recv()

    if router_init_state != "init ok" or detoken_init_state != "init ok":
        proc_router.kill()
        proc_detoken.kill()
        print("router init state:", router_init_state, "detoken init state:", detoken_init_state)
        sys.exit(1)

    assert proc_router.is_alive() and proc_detoken.is_alive()

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE, loop="uvloop")


if __name__ == "__main__":
    main()
