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
from PIL import Image
import io
import base64


def generate_random_image_and_encode_to_base64(width=448, height=448):
    # Step 1: Generate a random image (RGB)
    random_image = np.random.randint(100, 256, (height, width, 3), dtype=np.uint8)

    # Step 2: Convert NumPy array to PIL Image
    image = Image.fromarray(random_image)

    # Step 3: Save the image to a BytesIO buffer
    buffered = io.BytesIO()
    image.save(buffered, format="bmp")

    # Step 4: Encode the image bytes to Base64
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return base64_image


def seed_all(seed):
    if not seed:
        seed = int(time.time())
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def get_tokenizer(
    tokenizer_name: str,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return tokenizer


def get_random_length(reqs_num: int, length: int, range_ratio: float) -> List[int]:
    lens = []
    lens = np.random.randint(
        max(int(length * range_ratio), 1),
        length + 1,
        size=reqs_num,
    )
    return lens.tolist()


def gen_random_input_text(tokenizer, input_len) -> str:
    random_ids = [random.randint(1, tokenizer.vocab_size) for _ in range(input_len)]
    random_text = tokenizer.decode(random_ids)
    return random_text


def build_image_placeholders(tokenizer_path: str, num_images: int) -> str:
    if num_images <= 0:
        return ""

    tokenizer_name = tokenizer_path.lower()
    if "internvl" in tokenizer_name:
        return "".join("<image>\n" for _ in range(num_images))

    return "".join("<img></img>" for _ in range(num_images))


def gen_random_data(
    input_len: int,
    output_len: int,
    reqs_num: int,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    range_ratio: float,
    num_images: int,
) -> Tuple[List[str], List[int], List[int]]:
    prompts = []
    output_lens = get_random_length(reqs_num, output_len, range_ratio)
    input_lens = get_random_length(reqs_num, input_len, range_ratio)
    for i in range(reqs_num):
        input_text = gen_random_input_text(tokenizer, input_lens[i])
        images = []
        for _ in range(num_images):
            images.append({"type": "base64", "data": generate_random_image_and_encode_to_base64()})
            # input_text += "<|vision_start|><|image_pad|><|vision_end|>"
        prompts.append((input_text, input_lens[i], images))
    print("Generate random data finish.")
    return prompts, output_lens


def get_custom_input_data(data_path, output_len, tokenizer, range_ratio):
    prompts = []
    with open(data_path, "r") as f:
        for line in f.readlines():
            data_line = json.loads(line)
            input_data = tokenizer.apply_chat_template(
                data_line["messages"], add_generation_prompt=True, tokenize=False
            )
            input_len = len(tokenizer.encode(input_data))
            prompts.append([input_data, input_len])
    output_lens = get_random_length(len(prompts), output_len, range_ratio)
    print("Load random data finish.")
    return prompts, output_lens


model_name = [""]


async def async_post_stream_openai(url, prompt, max_new_tokens, session):
    input_len = 0
    try:
        text_input, input_len, images = prompt
        if images:
            text_input = build_image_placeholders(model_name[-1], len(images)) + text_input
        text_input = "a" + text_input + "<|im_start|>assistant\n"
        content = [{"type": "text", "text": text_input}]
        for img in images:
            b64 = img["data"]
            mime = "image/png"
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        messages = [{"role": "user", "content": content}]
        # print(messages)
        data = {
            # "model": "../InternVL2_5-26B",
            "model": "/mnt/mtc/niushengxiao/251024_math_ocr_15b_v1.6.5",
            "messages": messages,
            "max_tokens": max_new_tokens,
            "ignore_eos": True,
            "stream": True,
            "temperature": 0.0,
            # "best_of": 1,
        }
        headers = {"Content-Type": "application/json"}
        used_time = []
        start_time = time.time()
        last_time = start_time
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                return [], input_len

            while True:
                line = await response.content.readline()
                if not line:
                    break
                line = line.strip()
                if line:
                    current_time = time.time()
                    elapsed_time = current_time - last_time
                    used_time.append(elapsed_time)
                    last_time = current_time
            return used_time, input_len
    except Exception as e:
        print(f"openai request failed: {repr(e)}")
        return [], input_len


async def async_post_stream_lightllm(url, prompt, max_new_tokens, session):
    input_len = 0
    try:
        text_input, input_len, images = prompt
        if images:
            text_input = build_image_placeholders(model_name[-1], len(images)) + text_input
        data = {
            "inputs": text_input,
            "parameters": {
                "do_sample": False,
                "ignore_eos": True,
                "max_new_tokens": max_new_tokens,
                "add_special_tokens": False,
                "return_details": True,
                # "image_max_patch_num": 1
            },
            "multimodal_params": {
                "images": images,
            },
        }
        headers = {"Content-Type": "application/json"}
        used_time = []
        start_time = time.time()
        last_time = start_time
        async with session.post(url, headers=headers, json=data) as response:
            if response.status != 200:
                return [], input_len

            while True:
                line = await response.content.readline()
                if not line:
                    break
                if line and line.startswith(b"data:"):
                    current_time = time.time()
                    elapsed_time = current_time - last_time
                    used_time.append(elapsed_time)
                    last_time = current_time
                    line = json.loads(line[5:].strip())
                    input_len = int(line["token"]["prompt_tokens"])
        return used_time, input_len
    except Exception as e:
        print(f"lightllm request failed: {repr(e)}")
        return [], input_len


async def continuous_sender(
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
    max_count,
    continuous_send,
):
    prompt_index = 0
    while not stop_send.is_set():
        if not continuous_send and sent_count[0] >= max_count:
            break
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
                task_result = await task
                request_queue.task_done()
                if task_result is None:
                    raise ValueError("task returned None")

                result, input_len = task_result
                if result is None:
                    result = []

                if len(result) >= 1 and not stop_send.is_set():
                    results.append((result, input_len))
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
    async_task, url, prompts, max_new_tokens, reqs_num, num_clients, input_qps, force_terminate, continuous_send
):
    request_queue = asyncio.Queue()
    stop_event = asyncio.Event()
    stop_send = asyncio.Event()
    results_data = []
    counter = [0]
    sent_count = [0]
    end_time = [0.0]
    pending_tasks = []

    timeout = aiohttp.ClientTimeout(
        total=3600,  # 总超时时间1小时
        connect=300,  # 连接超时5分钟
        sock_connect=300,
        sock_read=3600,
    )

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=10 * reqs_num),
        timeout=timeout,
    ) as session:
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
                reqs_num,
                continuous_send,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:18009/generate_stream",
        help="lightllm:http://127.0.0.1:18007/generate_stream, openai:http://127.0.0.1:18007/v1/completions",
    )
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/data_vqa/wangruohui/train_internvl_qwen3/RUN/qwen3_32B_vit_300m_mlp_vit_10k_sft/5000_hf",
    )
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--input_num", type=int, default=200)
    parser.add_argument("--input_qps", type=float, default=30.0)
    parser.add_argument("--input_len", type=int, default=4096)
    parser.add_argument("--output_len", type=int, default=1)
    parser.add_argument("--server_api", type=str, default="lightllm")
    parser.add_argument("--dump_file", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--range_ratio", type=float, default=1.0)
    parser.add_argument(
        "--force_terminate",
        type=int,
        default=0,
        help="0: waiting all reqs return; 1: only waiting input_num reqs return",
    )
    parser.add_argument(
        "--continuous_send",
        type=int,
        default=0,
        help="0: only send input_num reqs; 1: send continuously until receiving input_num reqs",
    )
    parser.add_argument("--num_images", type=int, default=13)
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
    if args.data_path is not None:
        prompts, max_new_tokens = get_custom_input_data(args.data_path, args.output_len, tokenizer, args.range_ratio)
        args.input_num = len(prompts)
    else:
        # qps发送模式发送请求的数量不固定，这里暂定为input_num的10倍
        prompts, max_new_tokens = gen_random_data(
            args.input_len,
            args.output_len,
            args.input_num if not args.continuous_send else 10 * args.input_num,
            tokenizer,
            args.range_ratio,
            num_images=args.num_images,
        )

    percentiles = [25, 50, 75, 90, 95, 99, 100]
    if args.server_api == "lightllm":
        async_post_stream = async_post_stream_lightllm
    elif args.server_api == "openai":
        async_post_stream = async_post_stream_openai
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
            args.continuous_send,
        )
    )
    loop.close()
    print(len(results))
    first_token_time = []
    decode_token_time = []
    request_time = []
    final_output_lens = []
    valid_num = 0
    input_lens = []
    for result, input_len in results:
        if len(result) > 1:  # 统计至少decode出两个token的数据
            first_token_time.append(result[0])
            decode_token_time.append(sum(result[1:]) / len(result[1:]))
            request_time.append(sum(result))
            final_output_lens.append(len(result))
            input_lens.append(input_len)
            valid_num += 1
        else:
            first_token_time.append(result[0])
            decode_token_time.append(0)  # no decode
            request_time.append(sum(result))
            final_output_lens.append(len(result))
            input_lens.append(input_len)
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

# import os
# import argparse
# import json
# import time
# import random
# import numpy as np
# from typing import Union, List, Tuple
# from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast
# import aiohttp
# import asyncio
# from PIL import Image
# import io
# import base64


# # -------------------- 工具函数 --------------------

# def seed_all(seed):
#     if not seed:
#         seed = int(time.time())
#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)


# def get_tokenizer(tokenizer_name: str) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
#     return AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


# def generate_random_image_and_encode_to_base64(width=448, height=448):
#     arr = np.random.randint(1, 256, (height, width, 3), dtype=np.uint8)
#     img = Image.fromarray(arr)
#     buf = io.BytesIO()
#     img.save(buf, format="bmp")
#     return base64.b64encode(buf.getvalue()).decode("utf-8")


# def encode_image_to_base64(image_path):
#     with open(image_path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")


# def get_random_length(reqs_num, length, range_ratio):
#     return np.random.randint(max(int(length * range_ratio), 1), length + 1, size=reqs_num).tolist()


# # -------------------- 数据生成 --------------------

# def gen_random_data(input_len, output_len, reqs_num, tokenizer, range_ratio, num_images):
#     prompts = []
#     output_lens = get_random_length(reqs_num, output_len, range_ratio)
#     input_lens = get_random_length(reqs_num, input_len, range_ratio)
#     for i in range(reqs_num):
#         random_ids = [random.randint(1, tokenizer.vocab_size - 1) for _ in range(input_lens[i])]
#         text = tokenizer.decode(random_ids)
#         images = [{"type": "base64", "data": generate_random_image_and_encode_to_base64()} for _ in range(num_images)]
#         prompts.append((text, input_lens[i], images))
#     print("Generated random data.")
#     return prompts, output_lens


# def get_prompts_from_json(json_path, tokenizer, output_len, range_ratio):
#     with open(json_path, "r") as f:
#         data = json.load(f)
#     prompts = []
#     for item in data:
#         input_data = tokenizer.apply_chat_template(item["messages"], add_generation_prompt=True, tokenize=False)
#         input_len = len(tokenizer.encode(input_data))
#         images = []
#         for img_path in item.get("images", []):
#             if os.path.exists(img_path):
#                 img_b64 = encode_image_to_base64(img_path)
#                 images.append({"type": "base64", "data": img_b64})
#             else:
#                 print(f"[Warning] Image not found: {img_path}")
#         prompts.append((input_data, input_len, images))
#     output_lens = get_random_length(len(prompts), output_len, range_ratio)
#     print(f"Loaded {len(prompts)} prompts from JSON file.")
#     return prompts, output_lens


# # -------------------- 请求函数 --------------------

# async def async_post_stream_lightllm(url, prompt, max_new_tokens, session):
#     try:
#         text_input, input_len, images = prompt
#         text_input += "<|vision_start|><|image_pad|><|vision_end|>\n<|im_start|>assistant\n"
#         print(f"text_input is {text_input}")
#         data = {
#             "inputs": text_input,
#             "parameters": {
#                 "do_sample": False,
#                 "ignore_eos": True,
#                 "max_new_tokens": max_new_tokens,
#                 "add_special_tokens": False,
#                 "return_details": True,
#             },
#             "multimodal_params": {"images": images},
#         }
#         headers = {"Content-Type": "application/json"}
#         used_time = []
#         start = time.time()
#         last = start
#         async with session.post(url, headers=headers, json=data) as resp:
#             if resp.status != 200:
#                 return [], input_len
#             async for line in resp.content:
#                 if line and line.startswith(b"data:"):
#                     now = time.time()
#                     used_time.append(now - last)
#                     last = now
#                     try:
#                         line = json.loads(line[5:])
#                         input_len = int(line["token"]["prompt_tokens"])
#                     except Exception:
#                         pass
#         return used_time, input_len
#     except Exception as e:
#         print(f"[Error] {e}")
#         return [], 0


# async def async_post_stream_openai(url, prompt, max_new_tokens, session):
#     try:
#         text_input, input_len, images = prompt
#         text_input = "a" + text_input + "<|im_start|>assistant\n"
#         content = [{"type": "text", "text": text_input}]
#         for img in images:
#             mime = "image/png"
#             content.append({
#                 "type": "image_url",
#                 "image_url": {"url": f"data:{mime};base64,{img['data']}"}
#             })
#         messages = [{"role": "user", "content": content}]
#         data = {
#             "model": "test_model",
#             "messages": messages,
#             "max_tokens": max_new_tokens,
#             "ignore_eos": True,
#             "stream": True,
#             "temperature": 0.0,
#         }
#         headers = {"Content-Type": "application/json"}
#         used_time = []
#         start = time.time()
#         last = start
#         async with session.post(url, headers=headers, json=data) as resp:
#             if resp.status != 200:
#                 return [], input_len
#             async for line in resp.content:
#                 line = line.strip()
#                 if line:
#                     now = time.time()
#                     used_time.append(now - last)
#                     last = now
#         return used_time, input_len
#     except Exception as e:
#         print(f"[Error] {e}")
#         return [], 0


# # -------------------- 并发控制 --------------------

# async def worker(semaphore, session, async_task, url, prompt, max_new_tokens, results):
#     async with semaphore:
#         res = await async_task(url, prompt, max_new_tokens, session)
#         if res and len(res[0]) > 0:
#             results.append(res)


# async def run_fixed_concurrency_benchmark(async_task, url, prompts, max_new_tokens, num_concurrent):
#     timeout = aiohttp.ClientTimeout(total=3600)
#     semaphore = asyncio.Semaphore(num_concurrent)
#     results = []
#     start = time.time()
#     async with aiohttp.ClientSession(timeout=timeout) as session:
#         tasks = [
#             asyncio.create_task(worker(semaphore, session, async_task, url, p, max_new_tokens[i], results))
#             for i, p in enumerate(prompts)
#         ]
#         await asyncio.gather(*tasks)
#     end = time.time()
#     return results, start, end


# # -------------------- 主程序 --------------------

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--url", type=str, default="http://localhost:18009/generate_stream")
#     parser.add_argument("--num_concurrent", type=int, default=5, help="并发数量")
#     parser.add_argument("--tokenizer_path", type=str, required=True)
#     parser.add_argument("--data_path", type=str, default=None)
#     parser.add_argument("--input_num", type=int, default=100)
#     parser.add_argument("--input_len", type=int, default=4096)
#     parser.add_argument("--output_len", type=int, default=1)
#     parser.add_argument("--server_api", type=str, default="lightllm")
#     parser.add_argument("--range_ratio", type=float, default=1.0)
#     parser.add_argument("--num_images", type=int, default=2)
#     parser.add_argument("--dump_file", type=str, default="")
#     parser.add_argument("--seed", type=int, default=0)
#     args = parser.parse_args()

#     seed_all(args.seed)
#     tokenizer = get_tokenizer(args.tokenizer_path)

#     if args.data_path:
#         prompts, max_new_tokens = get_prompts_from_json(args.data_path, tokenizer, args.output_len, args.range_ratio)
#     else:
#         prompts, max_new_tokens = gen_random_data(
#             args.input_len, args.output_len, args.input_num, tokenizer, args.range_ratio, args.num_images
#         )

#     if args.server_api == "lightllm":
#         async_task = async_post_stream_lightllm
#     else:
#         async_task = async_post_stream_openai

#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     results, start_time, end_time = loop.run_until_complete(
#         run_fixed_concurrency_benchmark(async_task, args.url, prompts, max_new_tokens, args.num_concurrent)
#     )
#     loop.close()

#     # ---------------- 统计部分（与原始保持一致） ----------------
#     percentiles = [25, 50, 75, 90, 95, 99, 100]
#     first_token_time = []
#     decode_token_time = []
#     request_time = []
#     final_output_lens = []
#     input_lens = []

#     for result, input_len in results:
#         if len(result) > 1:
#             first_token_time.append(result[0])
#             decode_token_time.append(sum(result[1:]) / len(result[1:]))
#         else:
#             first_token_time.append(result[0])
#             decode_token_time.append(0)
#         request_time.append(sum(result))
#         final_output_lens.append(len(result))
#         input_lens.append(input_len)

#     valid_num = len(results)
#     print(f"\nvalid num = {valid_num}; all data num = {len(prompts)}; valid ratio = {valid_num / len(prompts):.4f}")
#     print(f"Total QPS: {valid_num / (end_time - start_time)}")
#     print(f"Avg Input Length: {np.mean(input_lens):.2f}")
#     print(f"Avg Output Length: {np.mean(final_output_lens):.2f}")
#     print(f"Total Throughput: {(sum(input_lens) + sum(final_output_lens)) / (end_time - start_time):.2f} token/s")
#     print(f"Input Throughput: {sum(input_lens) / (end_time - start_time):.2f} token/s")
#     print(f"Output Throughput: {sum(final_output_lens) / (end_time - start_time):.2f} token/s")
#     print("-" * 10)

#     dump_dict = {
#         "backend": args.server_api,
#         "clients": args.num_concurrent,
#         "request_num": valid_num,
#         "Total QPS": valid_num / (end_time - start_time),
#         "Avg Input Length": np.mean(input_lens),
#         "Avg Output Length": np.mean(final_output_lens),
#         "Total Throughput": (sum(input_lens) + sum(final_output_lens)) / (end_time - start_time),
#         "Input Throughput": sum(input_lens) / (end_time - start_time),
#         "Output Throughput": sum(final_output_lens) / (end_time - start_time),
#     }

#     # 各项延迟分位数
#     for name, arr, scale, unit in [
#         ("request_time", request_time, 1, "s"),
#         ("first_token_time", first_token_time, 1, "s"),
#         ("decode_token_time", decode_token_time, 1000, "ms"),
#     ]:
#         vals = np.percentile(arr, percentiles)
#         d = {}
#         for p, v in zip(percentiles, vals):
#             print(f"{name} P{p}: {v * scale:.6f}{unit}")
#             d[f"P{p}"] = v * scale
#         dump_dict[name] = d
#         print("-" * 10)

#     if args.dump_file:
#         with open(args.dump_file, "w") as f:
#             json.dump(dump_dict, f, indent=4)
#         print(f"Results have been written to {args.dump_file}")


# if __name__ == "__main__":
#     main()
