# ARC-Challenge accuracy test, following the same pattern as test_gsmk.py
import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import requests
from tqdm import tqdm

INVALID = "INVALID"


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

    if os.path.exists(filename):
        return filename

    print(f"Downloading from {url} to {filename}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    chunk_size = 1024

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


def load_arc_data(data_path=None):
    """Load ARC-Challenge dataset via HuggingFace datasets library."""
    if data_path and os.path.exists(data_path):
        with open(data_path) as f:
            return json.load(f)

    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        lines = []
        for item in ds:
            choices = item["choices"]
            lines.append(
                {
                    "question": item["question"],
                    "choices": {
                        "label": choices["label"],
                        "text": choices["text"],
                    },
                    "answerKey": item["answerKey"],
                }
            )
        return lines
    except ImportError:
        raise RuntimeError("Please install the datasets library: pip install datasets")


def load_arc_train_data():
    """Load ARC-Challenge train split for few-shot examples."""
    try:
        from datasets import load_dataset

        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
        lines = []
        for item in ds:
            choices = item["choices"]
            lines.append(
                {
                    "question": item["question"],
                    "choices": {
                        "label": choices["label"],
                        "text": choices["text"],
                    },
                    "answerKey": item["answerKey"],
                }
            )
        return lines
    except ImportError:
        raise RuntimeError("Please install the datasets library: pip install datasets")


def format_choices(item):
    """Format choices as A. xxx  B. xxx  C. xxx  D. xxx"""
    choices = item["choices"]
    parts = []
    for label, text in zip(choices["label"], choices["text"]):
        parts.append(f"{label}. {text}")
    return "\n".join(parts)


def get_one_example(lines, i, include_answer):
    ret = "Question: " + lines[i]["question"] + "\n" + format_choices(lines[i]) + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answerKey"]
    return ret


def get_few_shot_examples(lines, k):
    ret = ""
    for i in range(k):
        ret += get_one_example(lines, i, True) + "\n\n"
    return ret


def call_generate_lightllm(prompt, temperature, max_tokens, stop=None, url=None):
    """Call LightLLM API for text generation."""
    assert url is not None

    data = {
        "inputs": prompt,
        "parameters": {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "stop_sequences": stop,
            "repetition_penalty": 1.0,
            "top_p": 1.0,
            "top_k": 1,
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


def parse_answer(answer_str):
    """Extract the answer letter (A/B/C/D) from model output."""
    if not answer_str:
        return INVALID
    answer_str = answer_str.strip()

    # Check if the response starts with a single letter
    if len(answer_str) >= 1 and answer_str[0] in "ABCDE":
        # Make sure it's not part of a longer word
        if len(answer_str) == 1 or not answer_str[1].isalpha():
            return answer_str[0]

    # Look for patterns like "The answer is A" or "Answer: B"
    match = re.search(r"(?:answer\s+is|answer:)\s*([A-E])", answer_str, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Look for standalone letter with period or parenthesis: "A." or "(A)" or "A)"
    match = re.search(r"(?:^|\s|\()([A-E])(?:\.|(?:\)))", answer_str)
    if match:
        return match.group(1).upper()

    # Last resort: find the first standalone A-E letter
    match = re.search(r"\b([A-E])\b", answer_str)
    if match:
        return match.group(1).upper()

    return INVALID


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", type=int, default=256)
    parser.add_argument("--host", type=str, default="http://127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--num-questions", type=int, default=None, help="Number of questions (default: all)")
    parser.add_argument("--result-file", type=str, default="result.jsonl")
    parser.add_argument("--data-path", type=str, default=None)
    return parser.parse_args()


def main(args):
    url = f"{args.host}:{args.port}/generate"

    # Load data
    print("Loading ARC-Challenge test data...")
    test_lines = load_arc_data(args.data_path)
    print(f"Loaded {len(test_lines)} test questions")

    print("Loading ARC-Challenge train data for few-shot examples...")
    train_lines = load_arc_train_data()
    print(f"Loaded {len(train_lines)} train examples")

    num_shots = args.num_shots
    few_shot_examples = get_few_shot_examples(train_lines, num_shots)

    num_questions = args.num_questions if args.num_questions else len(test_lines)
    if num_questions > len(test_lines):
        print(f"Warning: Requested {num_questions} questions, but only {len(test_lines)} available. Using all.")
        num_questions = len(test_lines)

    questions = []
    labels = []
    for i in range(num_questions):
        questions.append(get_one_example(test_lines, i, False))
        labels.append(test_lines[i]["answerKey"])

    states = [None] * len(labels)

    def get_one_answer(i):
        answer = call_generate_lightllm(
            prompt=few_shot_examples + questions[i],
            temperature=0,
            max_tokens=64,
            stop=["Question", "\n\nQuestion"],
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
        preds.append(parse_answer(states[i]))

    # Compute accuracy
    preds_arr = np.array(preds)
    labels_arr = np.array(labels)
    acc = np.mean(preds_arr == labels_arr)
    invalid = np.mean(preds_arr == INVALID)

    print(f"Accuracy: {acc:.3f}")
    print(f"Invalid: {invalid:.3f}")
    print(f"Latency: {latency:.3f} s")
    print(f"Total questions: {num_questions}")

    dump_state_text("tmp_output_arc_lightllm.txt", states)

    with open(args.result_file, "a") as fout:
        value = {
            "task": "arc_challenge",
            "backend": "lightllm",
            "num_gpus": 1,
            "latency": round(latency, 3),
            "accuracy": round(acc, 3),
            "num_requests": num_questions,
            "other": {
                "num_questions": num_questions,
                "parallel": args.parallel,
            },
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
