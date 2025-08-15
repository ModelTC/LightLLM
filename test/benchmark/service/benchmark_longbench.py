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

import argparse
import asyncio
import json
import random
import time
from typing import AsyncGenerator, List, Tuple, Union

import aiohttp
import numpy as np
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast


def get_tokenizer(
    tokenizer_name: str,
    tokenizer_mode: str = "auto",
    *args,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False

    if "llama" in tokenizer_name.lower() and kwargs.get("use_fast", True):
        pass
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, *args, **kwargs)
    except TypeError as e:
        err_msg = "Failed to load the tokenizer. {e}"
        raise RuntimeError(err_msg) from e

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        pass
    return tokenizer


# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
    max_total_tokens: int = 16384,
) -> List[Tuple[List[dict], str, int, int]]:
    # Load the dataset (jsonl)
    dataset = []
    with open(dataset_path) as f:
        for line in f.readlines():
            if not line.strip():
                continue
            dataset.append(json.loads(line))
    print("read data set finish")

    def render_with_template(messages: List[dict]) -> str:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            parts = []
            for m in messages:
                parts.append(f"{m['role']}: {m['content']}")
            parts.append("assistant:")
            return "\n".join(parts)

    built_examples: List[Tuple[List[dict], str, int, int]] = []

    for data in dataset:
        context = data.get("context") or ""
        question = data.get("input") or "Summarizing government work reports"
        answers = data.get("answers")
        if not isinstance(context, str) or not isinstance(question, str):
            continue

        # Build messages: system + user with context and question
        system_prompt = "You are a helpful assistant. Read the context and answer the question concisely."
        user_content = f"Context:\n{context}\nInput:\n{question}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        rendered_prompt = render_with_template(messages)
        prompt_len = len(tokenizer(rendered_prompt).input_ids)

        # Estimate output length from reference answer if available
        target_text = ""
        if isinstance(answers, list) and len(answers) > 0:
            first_ans = answers[0]
            if isinstance(first_ans, str):
                target_text = first_ans
            else:
                target_text = str(first_ans)
        elif isinstance(answers, str):
            target_text = answers

        estimated_out = len(tokenizer(target_text).input_ids) if target_text else 128

        # Fit within max_total_tokens
        available_out = max_total_tokens - 1 - prompt_len
        if available_out < 4:
            # Skip samples that are too long
            continue
        output_len = min(estimated_out, available_out)

        built_examples.append((messages, rendered_prompt, prompt_len, output_len))

    # Take the first N valid samples
    sampled_requests = built_examples[:num_requests]
    sum_len = 0
    for _, _, prompt_len, output_len in sampled_requests:
        sum_len += prompt_len + output_len
    print("total tokens:", sum_len)
    return sampled_requests


async def get_request(
    input_requests: List[Tuple[List[dict], str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[List[dict], str, int, int], None]:
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
    messages: List[dict], rendered_prompt: str, prompt_len: int, output_len: int, use_openai_api: bool
) -> None:
    if use_openai_api:
        # Use OpenAI API to send the request.
        # Use local server to send the request.
        request_start_time = time.time()
        headers = {"Content-Type": "application/json", "User-Agent": "Benchmark Client"}
        url = "http://localhost:8000/v1/chat/completions"

        data = {
            "model": "DeepSeek-R1",
            "messages": messages,
            "top_k": 1,
            "top_p": 1.0,
            "temperature": 0,
            "stream": True,
            "ignore_eos": True,
            "max_tokens": output_len,
        }
        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        receive_n = 1

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=data) as response:
                chunks = []
                text = ""
                start_time = time.time()
                is_first = True
                async for chunk, _ in response.content.iter_chunks():
                    now_time = time.time()
                    delta_time = now_time - start_time
                    if is_first:
                        is_first = False
                        ttft = delta_time
                    text += json.loads(chunk.decode("utf-8")[6:])["choices"][0]["delta"].get("content", "")
                    if delta_time < 0.005:
                        receive_n += 1
                    chunks.append(delta_time)
                    start_time = now_time

    else:
        # Use local server to send the request.
        request_start_time = time.time()
        headers = {"Content-Type": "application/json", "User-Agent": "Benchmark Client"}
        url = "http://localhost:8000/generate_stream"

        data = {
            "inputs": rendered_prompt,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": output_len,
            },
        }

        timeout = aiohttp.ClientTimeout(total=3 * 3600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            receive_n = 0
            text = ""
            async with session.post(url, headers=headers, json=data) as response:
                chunks = []
                start_time = time.time()
                is_first = True
                async for chunk, _ in response.content.iter_chunks():
                    now_time = time.time()
                    delta_time = now_time - start_time
                    if is_first:
                        is_first = False
                        ttft = delta_time
                    if delta_time < 0.005:
                        receive_n += 1
                    chunks.append(chunk)
                    text += json.loads(chunk.decode("utf-8")[5:])["token"]["text"]
                    start_time = now_time

    request_end_time = time.time()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency, ttft))


async def benchmark(
    input_requests: List[Tuple[List[dict], str, int, int]],
    request_rate: float,
    use_openai_api: bool = False,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        messages, rendered_prompt, prompt_len, output_len = request
        task = asyncio.create_task(send_request(messages, rendered_prompt, prompt_len, output_len, use_openai_api))
        tasks.append(task)
    await asyncio.gather(*tasks)


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = get_tokenizer(args.tokenizer, "slow")
    input_requests = sample_requests(args.dataset, args.num_prompts, tokenizer, args.max_total_tokens)

    benchmark_start_time = time.time()
    asyncio.run(benchmark(input_requests, args.request_rate, args.use_openai_api))
    benchmark_end_time = time.time()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {args.num_prompts / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency, _ in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_time_to_first_token = np.mean([ttft for _, _, _, ttft in REQUEST_LATENCY])
    print("Average time to first token: " f"{avg_time_to_first_token:.2f} s")
    avg_per_token_latency = (
        np.mean([latency / (prompt_len + output_len) for prompt_len, output_len, latency, _ in REQUEST_LATENCY]) * 1000
    )
    print(f"Average latency per token: {avg_per_token_latency:.1f} ms")
    # avg_per_output_token_latency = np.mean([latency / output_len for _, output_len, latency, _ in REQUEST_LATENCY])
    # print("Average latency per output token: " f"{avg_per_output_token_latency:.2f} s")
    avg_inter_token_latency = (
        np.mean(
            [(latency - ttft) / (output_len - 1) for _, output_len, latency, ttft in REQUEST_LATENCY if output_len > 1]
        )
        * 1000
    )
    print(f"Average inter-token latency: {avg_inter_token_latency:.1f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark the online serving throughput.")
    parser.add_argument("--use_openai_api", default=False, action="store_true", help="Use OpenAI API for requests.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--tokenizer", type=str, required=True, help="Name or path of the tokenizer.")
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--num-prompts", type=int, default=1, help="Number of prompts to process.")
    parser.add_argument("--max-total-tokens", type=int, default=16384, help="Max total tokens (input + output).")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
