import argparse
import glob
import json
import os
import random
import subprocess
import sys
import unittest
from types import SimpleNamespace

"""
git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install -e lmms-eval/
"""

# VLM models for testing
MODELS = [
    SimpleNamespace(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        mmmu_accuracy=0.4,
    ),
]
os.environ["OPENAI_API_KEY"] = "lightllm123"
os.environ["OPENAI_API_BASE"] = "http://localhost:8000/v1"


def run_mmmu_eval(
    model_version: str,
    output_path: str,
):
    """
    Evaluate a VLM on the MMMU validation set with lmms‑eval.
    Only `model_version` (checkpoint) and `chat_template` vary;
    We are focusing only on the validation set due to resource constraints.
    """
    # -------- fixed settings --------
    model = "openai_compatible"
    tp = 1
    tasks = "mmmu_val"
    batch_size = 16
    log_suffix = "openai_compatible"
    os.makedirs(output_path, exist_ok=True)

    # -------- compose --model_args --------
    model_args = f"model_version={model_version}," f"tp={tp}"
    print(model_args)

    # -------- build command list --------
    cmd = [
        "python3",
        "-m",
        "lmms_eval",
        "--model",
        model,
        "--model_args",
        model_args,
        "--tasks",
        tasks,
        "--batch_size",
        str(batch_size),
        "--log_samples",
        "--log_samples_suffix",
        log_suffix,
        "--output_path",
        str(output_path),
    ]

    subprocess.run(
        cmd,
        check=True,
        timeout=3600,
    )


run_mmmu_eval("Qwen/Qwen2.5-VL-7B-Instruct", "./logs")
