# Adapted from benchmarks/benchmark_serving.py
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
import os
import socket
import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple, Union

import aiohttp
import numpy as np
from transformers import PreTrainedTokenizerBase
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)

QUESTION = {}
def get_tokenizer(
    tokenizer_name: str,
    tokenizer_mode: str = "slow",
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError(
                "Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = True
    if "llama" in tokenizer_name.lower() and kwargs.get("use_fast", True):
        pass
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, *args,
                                                  **kwargs)
    except TypeError as e:
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA-based "
            f"model, use '{_FAST_LLAMA_TOKENIZER}' instead of the original "
            "tokenizer.")
        raise RuntimeError(err_msg) from e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        pass
    return tokenizer

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def sample_requests(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    data = []
    with open(dataset_path, "r") as f:
        questions = f.readlines()
    gts = {}
    for question in questions:
        question = json.loads(question.strip())
        file_name = question["file_name"].split(".")[0]
        data.append((file_name, question['question_id'], question['instruction'], question['answer']))
        if file_name not in QUESTION:
            QUESTION[file_name] = {}
        QUESTION[file_name][question["question_id"]] = [question["answer"]]

    print("read data set finish")
    return data


async def get_request(
    input_requests: List[Tuple[str, str, str, str]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request
        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


async def send_request(
    request: str,
    output_len: int,
    port: int,
) -> None:
    request_start_time = time.time()
    headers = {'Content-Type': 'application/json'}
    headers = {"User-Agent": "Benchmark Client"}
    file_name, question_id, inputs, answer = request 
    prompt = f"<系统> <对话历史> <知识> <最新问题> 用户：给出以下问题的答案:\n{inputs} SenseChat："
    print(prompt)
    # prompt=  "[Round {}]\n\n问：{}\n\n答：".format(1, inputs)
    url = f'http://localhost:{port}/generate'
    data = {
        'inputs': prompt,
        'parameters': {
            'do_sample': False,
            'ignore_eos': True,
            'max_new_tokens': output_len,
            # 'do_sample':True, 
            # 'top_p':0.8,
            # 'temperature':0.8
             # 'temperature': 0.1,
        }
    }
    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(url, headers=headers, json=data) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)
            QUESTION[file_name][question_id].append(output["generated_text"][0])
            if "error" not in output:
                break

async def benchmark(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
    port: int,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        task = asyncio.create_task(send_request(request, 1, port))
        tasks.append(task)
    await asyncio.gather(*tasks)


def IsOpen(ip, port):
    s = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    index=1
    try:
        s.connect((ip,int(port)))
        s.shutdown(2)

        print('successfully launch model')
        return True
    except:
        time.sleep(10)
        return False

def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = get_tokenizer(args.tokenizer, "slow")
    input_requests = sample_requests(args.dataset, tokenizer)

    benchmark_start_time = time.time()
    asyncio.run(benchmark(input_requests, args.request_rate, args.port))
    rights, alls = 0, 0
    for file_name in QUESTION:
        for idx in QUESTION[file_name]:
            alls += 1
            if QUESTION[file_name][idx][0] == QUESTION[file_name][idx][1]:
                rights += 1
    print(QUESTION)
    score = rights / alls
    print("score: {}".format(score))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Name or path of the tokenizer.")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Number of requests per second. If this is inf, "
                             "then all the requests are sent at time 0. "
                             "Otherwise, we use Poisson process to synthesize "
                             "the request arrival times.")
    parser.add_argument("--port", type=int, default=8000,
                    help="port number")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
