# Adapted from https://github.com/sgl-project/sglang/blob/main/benchmark/gsm8k/bench_other.py
import argparse
import ast
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import requests
from tqdm import tqdm

INVALID = -9999999


def read_jsonl(filename: str):
    """Read a JSONL file."""
    with open(filename) as fin:
        for line in fin:
            if line.startswith("#"):
                continue
            yield json.loads(line)


def dump_state_text(filename: str, states: list, mode: str = "w"):
    """Dump program state in a text file."""
    with open(filename, mode) as fout:
        for i, s in enumerate(states):
            if isinstance(s, str):
                fout.write(f"==== {i} ====\n{s}\n")
            else:
                fout.write(f"==== {i} ====\n{str(s)}\n")


def download_and_cache_file(url: str, filename: Optional[str] = None):
    """Read and cache a file from a url."""
    if filename is None:
        filename = os.path.join("/tmp", url.split("/")[-1])

    # Check if the cache file already exists
    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    # Stream the response to show the progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for request errors

    # Total size of the file in bytes
    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024  # Download in chunks of 1KB

    # Use tqdm to display the progress bar
    with open(filename, "wb") as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            size = file.write(chunk)
            bar.update(size)

    return filename


def call_generate_lightllm(prompt, temperature, max_tokens, stop=None, url=None):
    """Call LightLLM API for text generation."""
    assert url is not None

    data = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "stop_sequences": stop,
        },
    }
    res = requests.post(url, json=data)
    assert res.status_code == 200, f"API request failed with status code {res.status_code}: {res.text}"

    response_json = res.json()
    if "generated_text" not in response_json:
        raise ValueError(f"Invalid API response format. Expected 'generated_text' key, got: {response_json.keys()}")
    if not isinstance(response_json["generated_text"], list) or len(response_json["generated_text"]) == 0:
        raise ValueError(
            "Invalid API response format. 'generated_text' should be a non-empty list, "
            f"got: {response_json['generated_text']}"
        )

    pred = response_json["generated_text"][0]
    return pred


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def get_answer_value(answer_str):
    answer_str = answer_str.replace(",", "")
    # First try to find the answer after "####" marker (GSM8K format)
    match = re.search(r"####\s*(-?\d+)", answer_str)
    if match:
        try:
            return ast.literal_eval(match.group(1))
        except SyntaxError:
            pass
    # Fallback: find all numbers and take the last one
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=256)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--num-questions", type=int, default=200)
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    parser.add_argument("--data-path", type=str, default="test.jsonl")
    return parser.parse_args()


def main(args):
    # LightLLM API URL
    url = f"{args.host}:{args.port}/generate"

    # Read data
    url_data = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    filename = download_and_cache_file(url_data)
    lines = list(read_jsonl(filename))

    # Construct prompts
    num_questions = args.num_questions
    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(lines, num_shots)

    # Ensure we have enough samples and avoid data leakage
    # Test questions should start after few-shot examples
    max_available = len(lines) - num_shots
    if num_questions > max_available:
        print(
            "Warning: Requested {} questions, but only {} available after reserving {} for few-shot. "
            "Using {} questions.".format(num_questions, max_available, num_shots, max_available)
        )
        num_questions = max_available

    questions = []
    labels = []
    for i in range(num_shots, num_shots + num_questions):
        questions.append(get_one_example(lines, i, False))
        labels.append(get_answer_value(lines[i]["answer"]))
    assert all(label != INVALID for label in labels)

    states = [None] * len(labels)

    # Run requests using thread pool
    def get_one_answer(i):
        answer = call_generate_lightllm(
            prompt=few_shot_examples + questions[i],
            temperature=0,
            max_tokens=1024,
            stop=["Question", "Assistant:", "<|separator|>", "Human:", "\n\nQuestion"],
            url=url,
        )
        states[i] = answer

    tic = time.perf_counter()
    if args.parallel == 1:
        for i in tqdm(range(len(questions))):
            get_one_answer(i)
    else:
        with ThreadPoolExecutor(args.parallel) as executor:
            list(
                tqdm(
                    executor.map(get_one_answer, list(range(len(questions)))),
                    total=len(questions),
                )
            )

    latency = time.perf_counter() - tic

    preds = []
    for i in range(len(states)):
        preds.append(get_answer_value(states[i]))

    # Compute accuracy
    acc = np.mean(np.array(preds) == np.array(labels))
    invalid = np.mean(np.array(preds) == INVALID)

    # Print results
    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")

    # Dump results
    dump_state_text("tmp_output_lightllm.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "gsm8k",
            "backend": "lightllm",
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "num_requests": args.num_questions,
            "other": {
                "num_questions": args.num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
