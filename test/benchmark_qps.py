import os
import argparse
import yaml
import requests
import json
import time
import random
import numpy as np
from tqdm import tqdm
from typing import Union, List, Tuple
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
import aiohttp
import asyncio


def seed_all(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def get_tokenizer(
    tokenizer_name: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return tokenizer


def get_output_length(reqs_num: int, output_len: int) -> List[int]:
    min_len, max_len = 2, output_len * 2
    mean = (min_len + max_len) * 0.5
    std = mean
    output_lens = []
    for _ in range(reqs_num):
        cur_len = random.gauss(mean, std)
        cur_len = round(cur_len)
        if cur_len < min_len:
            cur_len = min_len
        elif cur_len > max_len:
            cur_len = max_len
        output_lens.append(cur_len)
    return output_lens


def gen_random_input_text(tokenizer) -> str:
    random_ids = [random.randint(512, 8192) for _ in range(1024)]
    random_text = tokenizer.decode(random_ids)
    return random_text


def gen_random_data(
    input_len: int, output_len: int, reqs_num: int, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Tuple[List[str], List[int], List[int]]:
    prompts = []
    input_lens = []
    output_lens = get_output_length(reqs_num, output_len)
    for i in range(reqs_num):
        input_text = gen_random_input_text(tokenizer)
        prompts.append(input_text)
        input_lens.append(input_len)
    print("Generate random data finish.")
    return prompts, input_lens, output_lens


async def async_post_stream_lightllm(url, text_input, max_new_tokens, session) -> List[float]:
    try:
        data = {
            "inputs": text_input,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": max_new_tokens,
            },
        }
        headers = {"Content-Type": "application/json"}
        used_time = []
        start_time = time.time()
        last_time = start_time

        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                return []

            async for line in response.content:
                if line and line.startswith(b"data:"):
                    # print(line)
                    current_time = time.time()
                    elapsed_time = current_time - last_time
                    used_time.append(elapsed_time)
                    last_time = current_time
        return used_time
    except Exception:
        pass


async def continuous_sender(
    session, pending_tasks, async_task, url, prompts, max_new_tokens, request_queue, stop_send, sent_count, input_qps
):
    prompt_index = 0
    while not stop_send.is_set():
        prompt = prompts[prompt_index % len(prompts)]
        max_tokens = max_new_tokens[prompt_index % len(max_new_tokens)]

        task = asyncio.create_task(async_task(url, prompt, max_tokens, session))
        pending_tasks.append(task)
        await request_queue.put(task)

        prompt_index += 1
        sent_count[0] += 1
        # 控制发送速率
        await asyncio.sleep(1.0 / input_qps)


async def response_collector(
    request_queue,
    results,
    reqs_num,
    stop_event,
    stop_send,
    counter,
    end_time,
    sent_count,
    force_terminate,
    pending_tasks,
):
    try:
        while True:
            try:
                task = await asyncio.wait_for(request_queue.get(), timeout=1.0)
                result = await task
                request_queue.task_done()
                assert result is not None
                if len(result) > 1 and not stop_send.is_set():
                    results.append(result)
                current_count = counter[0] + 1
                counter[0] = current_count
                print(f"\rfinished_reqs:{current_count} / target_reqs:{reqs_num} / sent_reqs:{sent_count[0]}", end="")

                if len(results) >= reqs_num and not stop_send.is_set():
                    end_time[0] = time.time()
                    print("\nReached target number of responses")
                    stop_send.set()
                    if force_terminate and not stop_event.is_set():
                        stop_event.set()
                    else:
                        print("\nWaiting remining responses to finish...")

                if current_count >= sent_count[0] and not stop_event.is_set():
                    stop_event.set()

                if stop_event.is_set() and (force_terminate or request_queue.empty()):
                    return

            except asyncio.TimeoutError:
                if stop_event.is_set() and (force_terminate or request_queue.empty()):
                    return
                continue
            except Exception as e:
                print(f"\nError collecting response: {e}")
    finally:
        if force_terminate:
            for task in pending_tasks:
                if not task.done():
                    task.cancel()


async def run_continuous_benchmark(
    async_task, url, prompts, max_new_tokens, reqs_num, num_clients, input_qps, force_terminate
):
    request_queue = asyncio.Queue()
    stop_event = asyncio.Event()
    stop_send = asyncio.Event()
    results_data = []
    counter = [0]
    sent_count = [0]
    end_time = [0.0]
    pending_tasks = []

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=10 * reqs_num)) as session:
        sender_task = asyncio.create_task(
            continuous_sender(
                session,
                pending_tasks,
                async_task,
                url,
                prompts,
                max_new_tokens,
                request_queue,
                stop_send,
                sent_count,
                input_qps,
            )
        )

        collector_task = [
            asyncio.create_task(
                response_collector(
                    request_queue,
                    results_data,
                    reqs_num,
                    stop_event,
                    stop_send,
                    counter,
                    end_time,
                    sent_count,
                    force_terminate,
                    pending_tasks,
                )
            )
            for _ in range(num_clients)
        ]
        await asyncio.wait(collector_task)

        if not sender_task.done():
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass

    return results_data, sent_count[0], end_time[0]


model_name = []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/generate_stream")
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--input_num", type=int, default=2000)
    parser.add_argument("--input_qps", type=float, default=30.0)
    parser.add_argument("--input_len", type=int, default=1024)
    parser.add_argument("--output_len", type=int, default=128)
    parser.add_argument("--server_api", type=str, default="lightllm")
    parser.add_argument("--dump_file", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force_terminate",
        type=int,
        default=0,
        help="0: waiting all reqs return; 1: only waiting input_num reqs return",
    )

    args = parser.parse_args()
    if args.dump_file and os.path.exists(args.dump_file):
        # 读取并输出 JSON 内容
        with open(args.dump_file, "r") as json_file:
            content = json.load(json_file)
            print(json.dumps(content, indent=4))
        return

    assert args.tokenizer_path is not None
    model_name.append(args.tokenizer_path)
    seed_all(args.seed)
    url = args.url
    tokenizer = get_tokenizer(args.tokenizer_path)
    # qps发送模式发送请求的数量不固定，这里暂定为reqs_num的10倍
    prompts, input_lens, max_new_tokens = gen_random_data(
        args.input_len, args.output_len, 10 * args.input_num, tokenizer
    )

    percentiles = [25, 50, 75, 90, 95, 99, 100]
    if args.server_api == "lightllm":
        async_post_stream = async_post_stream_lightllm
    else:
        raise Exception(f"Not support {args.server_api} server_api.")

    dump_dict = {}
    dump_dict["backend"] = args.server_api
    dump_dict["clients"] = args.num_clients

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    start_time = time.time()
    results, sent_reqs, end_time = loop.run_until_complete(
        run_continuous_benchmark(
            async_post_stream,
            url,
            prompts,
            max_new_tokens,
            args.input_num,
            args.num_clients,
            args.input_qps,
            args.force_terminate,
        )
    )
    loop.close()

    first_token_time = []
    decode_token_time = []
    request_time = []
    final_output_lens = []
    valid_num = 0
    for result in results:
        if len(result) > 1:  # 统计至少decode出两个token的数据
            first_token_time.append(result[0])
            decode_token_time.append(sum(result[1:]) / len(result[1:]))
            request_time.append(sum(result))
            final_output_lens.append(len(result))
            valid_num += 1

    print(
        f"\n\nvalid num = {valid_num}; all data num = {len(results)}; valid ratio = {valid_num * 1.0 / len(results)}\n"
    )
    print(f"Total QPS: {valid_num / (end_time - start_time)}")
    print(f"Sender QPS: {sent_reqs / (end_time - start_time)}")
    print(f"Avg Input Length: {sum(input_lens) / len(input_lens)}")
    print(f"Avg Output Length: {sum(final_output_lens) / len(final_output_lens)}")
    print(f"Total Throughput: {(sum(input_lens) + sum(final_output_lens)) / (end_time - start_time)} token/s")
    print(f"Input Throughput: {sum(input_lens) / (end_time - start_time)} token/s")
    print(f"Output Throughput: {sum(final_output_lens) / (end_time - start_time)} token/s")
    print("-" * 10)
    dump_dict["request_num"] = valid_num
    dump_dict["Total QPS"] = valid_num / (end_time - start_time)
    dump_dict["Sender QPS"] = sent_reqs / (end_time - start_time)
    dump_dict["Avg Input Length"] = sum(input_lens) / len(input_lens)
    dump_dict["Avg Output Length"] = sum(final_output_lens) / len(final_output_lens)
    dump_dict["Total Throughput"] = (sum(input_lens) + sum(final_output_lens)) / (end_time - start_time)
    dump_dict["Input Throughput"] = sum(input_lens) / (end_time - start_time)
    dump_dict["Output Throughput"] = sum(final_output_lens) / (end_time - start_time)

    values = np.percentile(request_time, percentiles)
    request_time_dict = {}
    for percentile, value in zip(percentiles, values):
        print(f"request_time P{percentile}: {value:.6f}s")
        request_time_dict[f"P{percentile}"] = value
    dump_dict["request_time"] = request_time_dict
    print("-" * 10)

    first_token_time_dict = {}
    values = np.percentile(first_token_time, percentiles)
    for percentile, value in zip(percentiles, values):
        print(f"first_token_time  P{percentile}: {value:.6f}s")
        first_token_time_dict[f"P{percentile}"] = value
    dump_dict["first_token_time_dict"] = first_token_time_dict
    print("-" * 10)

    decode_token_time_dict = {}
    values = np.percentile(decode_token_time, percentiles)
    for percentile, value in zip(percentiles, values):
        print(f"decode_token_time  P{percentile}: {value * 1000:.6f}ms")
        decode_token_time_dict[f"P{percentile}"] = value * 1000
    dump_dict["decode_token_time_dict"] = decode_token_time_dict
    print(dump_dict)

    if args.dump_file:
        with open(args.dump_file, "w") as json_file:
            json.dump(dump_dict, json_file, indent=4)
        print(f"Results have been written to {args.dump_file}")


if __name__ == "__main__":
    main()
