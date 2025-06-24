"""
benchmark_multi_server.py

This script is used for automated benchmarking of multiple model services (e.g., llama-7b, llama-13b),
evaluating their performance under different input lengths, output lengths, number of turns, concurrent users,
and worker threads.

Main features:
- Supports automated testing for multiple models and parameter combinations.
- Collects and outputs various performance metrics, including throughput, QPS, and latency.
- Saves results as a Markdown table for easy analysis.

Parameter description:
- models: Model names and their service URLs to be tested.
- first_input_lens: List of token lengths for the first input.
- subsequent_input_lens: List of token lengths for subsequent inputs.
- output_lens: List of output token lengths.
- num_turns: List of dialogue turns.
- num_workers: List of concurrent worker counts.
- num_users: List of concurrent user counts.
- result_dir: Directory to save results.

Example:
    python benchmark_multi_server.py
"""
import os
import itertools
from easydict import EasyDict
from datetime import datetime
from benchmark_prompt_cache import run


# settings
models = {"llama-7b": "http://localhost:8080", "llama-13b": "http://localhost:8081"}
warm_up = True
first_input_lens = [16000, 64000]
subsequent_input_lens = [128]
output_lens = [128]
num_turns = [5, 10]
num_workers = [1, 8, 16, 24, 32]
num_users = [60]
result_dir = "./llama"
result_path = os.path.join(result_dir, f"summary_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.md")
heads = [
    "model_name",
    "first_input_len",
    "subsequent_input_len",
    "output_len",
    "num_turns",
    "num_workers",
    "num_users",
    "prefill_throughput(tokens/s)",
    "decode_throughput(tokens/s)",
    "total_throughput(tokens/s)",
    "qps",
    "首字延迟25%(ms)",
    "首字延迟50%(ms)",
    "首字延迟75%(ms)",
    "首字延迟99%(ms)",
    "首字延迟100%(ms)",
    "包间延迟25%(ms)",
    "包间延迟50%(ms)",
    "包间延迟75%(ms)",
    "包间延迟99%(ms)",
    "包间延迟100%(ms)",
]

results = []
for first_input_len, subsequent_input_len, output_len, num_turn, num_worker, num_user in itertools.product(
    first_input_lens, subsequent_input_lens, output_lens, num_turns, num_workers, num_users
):
    for model_name, model_url in models.items():
        args = EasyDict(
            {
                "model_name": model_name,
                "model_url": model_url,
                "first_input_len": first_input_len,
                "subsequent_input_len": subsequent_input_len,
                "output_len": output_len,
                "num_turns": num_turn,
                "num_workers": num_worker,
                "num_users": num_user,
                "result_dir": result_dir,
                "print": False,
                "cache": True,
                "use_cache": True,
            }
        )
        if warm_up:
            warm_up = False
            args["cache"] = False
            result = run(args)
            args["cache"] = True
        result = run(args)
        results.append(args | result)

with open(result_path, "w") as md_file:
    md_file.write("|")
    for head in heads:
        md_file.write(head + "|")
    md_file.write("\r\n")
    md_file.write("|")
    for _ in range(len(heads)):
        md_file.write("------|")
    md_file.write("\r\n")
    for result in results:
        md_file.write("|")
        for head in heads:
            md_file.write(str(result[head]) + "|")
        md_file.write("\r\n")
