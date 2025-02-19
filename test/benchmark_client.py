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


def get_output_length(input_num: int, output_len: int) -> List[int]:
    min_len, max_len = 2, output_len * 2
    mean = (min_len + max_len) * 0.5
    std = mean
    output_lens = []
    for _ in range(input_num):
        cur_len = random.gauss(mean, std)
        cur_len = round(cur_len)
        if cur_len < min_len:
            cur_len = min_len
        elif cur_len > max_len:
            cur_len = max_len
        output_lens.append(cur_len)
    return output_lens


def gen_random_input_text(input_len, tokenizer) -> str:
    random_ids = [random.randint(512, 8192) for _ in range(1024)]
    random_text = tokenizer.decode(random_ids)
    return random_text


def gen_random_data(
    input_len: int, output_len: int, input_num: int, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Tuple[List[str], List[int], List[int]]:
    prompts = []
    input_lens = []
    output_lens = get_output_length(input_num, output_len)
    for i in range(input_num):
        input_text = gen_random_input_text(input_len, tokenizer)
        prompts.append(input_text)
        input_lens.append(input_len)
    print("Generate random data finish.")
    return prompts, input_lens, output_lens


def post_stream_lightllm(url: str, text_input: str, max_new_tokens: int) -> List[float]:
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
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
    if response.status_code != 200:
        print(response.json())
    assert response.status_code == 200
    for line in response.iter_lines():
        if line:
            current_time = time.time()
            elapsed_time = current_time - last_time
            used_time.append(elapsed_time)
            # print(line.decode("utf-8"))
            last_time = current_time
    return used_time


model_name = []


def post_stream_openai(url: str, text_input: str, max_new_tokens: int) -> List[float]:
    data = {
        "model": model_name[0],
        "prompt": text_input,
        "n": 1,
        "ignore_eos": True,
        "max_tokens": max_new_tokens,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    used_time = []
    start_time = time.time()
    last_time = start_time
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
    assert response.status_code == 200
    for line in response.iter_content(chunk_size=8192):
        line = line.strip()
        if line:
            line = line.decode("utf-8")[6:]  # remove "data: "
            if line == "[DONE]":
                continue
            data = json.loads(line)
            if not data["choices"][0]["text"]:
                continue
            current_time = time.time()
            elapsed_time = current_time - last_time
            used_time.append(elapsed_time)
            last_time = current_time
    return used_time


def post_stream_triton(url: str, text_input: str, max_new_tokens: int) -> List[float]:
    data = {"text_input": text_input, "max_tokens": max_new_tokens, "stream": True}
    headers = {"Content-Type": "application/json"}
    used_time = []
    start_time = time.time()
    last_time = start_time
    response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
    assert response.status_code == 200
    for line in response.iter_lines():
        if line:
            current_time = time.time()
            elapsed_time = current_time - last_time
            used_time.append(elapsed_time)
            last_time = current_time
    return used_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/generate_stream")
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--input_num", type=int, default=2000)
    parser.add_argument("--input_len", type=int, default=1024)
    parser.add_argument("--output_len", type=int, default=128)
    parser.add_argument("--server_api", type=str, default="lightllm")
    parser.add_argument("--dump_file", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)

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
    prompts, input_lens, max_new_tokens = gen_random_data(args.input_len, args.output_len, args.input_num, tokenizer)

    percentiles = [25, 50, 75, 90, 95, 99, 100]
    if args.server_api == "lightllm":
        post_stream = post_stream_lightllm
    elif args.server_api == "openai":
        post_stream = post_stream_openai
    elif args.server_api == "triton":
        post_stream = post_stream_triton
    else:
        raise Exception(f"Not support {args.server_api} server_api.")

    dump_dict = {}
    dump_dict["backend"] = args.server_api
    dump_dict["clients"] = args.num_clients
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.num_clients) as executor:
        results = list(
            tqdm(
                executor.map(lambda p: post_stream(url, p[0], p[1]), zip(prompts, max_new_tokens)),
                total=len(prompts),
                desc="Running tests",
            )
        )
    end_time = time.time()
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
    print(f"Avg Input Length: {sum(input_lens) / len(input_lens)}")
    print(f"Avg Output Length: {sum(final_output_lens) / len(final_output_lens)}")
    print(f"Total Throughput: {(sum(input_lens) + sum(final_output_lens)) / (end_time - start_time)} token/s")
    print(f"Input Throughput: {sum(input_lens) / (end_time - start_time)} token/s")
    print(f"Output Throughput: {sum(final_output_lens) / (end_time - start_time)} token/s")
    print("-" * 10)
    dump_dict["request_num"] = valid_num
    dump_dict["Total QPS"] = valid_num / (end_time - start_time)
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
