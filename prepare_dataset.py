#!/usr/bin/env python3
"""
Download and prepare benchmark datasets for benchmark_sharegpt.py
Extended with: PubMedQA (medical), LegalBench, FinQA, MATH
"""

import argparse
import json
import os
from typing import Dict, List, Any
from datasets import load_dataset


# ========== Common Helper ==========

def build_prompt(question: str, context: str = "", long_answer: bool = False) -> str:
    prompt = question.strip()
    if context:
        prompt += f"\n\nContext:\n{context.strip()}"
    if long_answer:
        prompt += "\n\nPlease provide a detailed, step-by-step explanation."
    return prompt


# ========== Existing Datasets ==========

def prepare_gsm8k(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("gsm8k", "main", split="test")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        prompt = build_prompt(item["question"], long_answer=long_answer)

        results.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": item["answer"]}
            ]
        })

    json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def prepare_humaneval(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("openai_humaneval", split="test")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        instruction = f"Complete the following Python function:\n\n{item['prompt']}"

        results.append({
            "conversations": [
                {"from": "human", "value": instruction},
                {"from": "assistant", "value": item["canonical_solution"]}
            ]
        })

    json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def prepare_mmlu(output_path: str, num_samples: int = None, long_answer: bool = False):
    subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics"]
    results = []

    for subject in subjects:
        dataset = load_dataset("cais/mmlu", subject, split="test")

        for item in dataset:
            if num_samples and len(results) >= num_samples:
                break

            choices = item["choices"]
            answer = choices[item["answer"]]

            choices_text = "\n".join([f"{chr(65+j)}. {c}" for j, c in enumerate(choices)])
            prompt = build_prompt(
                f"{item['question']}\n\nChoices:\n{choices_text}\n\nAnswer with the correct choice.",
                long_answer=long_answer
            )

            results.append({
                "conversations": [
                    {"from": "human", "value": prompt},
                    {"from": "assistant", "value": answer}
                ]
            })

    json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def prepare_truthfulqa(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("truthful_qa", "generation", split="validation")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        prompt = build_prompt(item["question"], long_answer=long_answer)

        results.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": item["best_answer"]}
            ]
        })

    json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def prepare_sharegpt(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        convs = item.get("conversations", [])
        if len(convs) >= 2:
            results.append({"conversations": convs})

    json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def prepare_alpaca(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        prompt = item["instruction"]
        if item.get("input"):
            prompt += f"\n\n{item['input']}"

        if long_answer:
            prompt += "\n\nProvide a detailed answer."

        results.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": item["output"]}
            ]
        })

    json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def prepare_pubmedqa(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        context = " ".join(item["context"]["contexts"])
        prompt = build_prompt(item["question"], context, long_answer)

        results.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": item["long_answer"]}
            ]
        })

    json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def prepare_legalbench(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("lex_glue", "ecthr_a", split="test")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        labels = ", ".join(map(str, item["labels"]))
        prompt = build_prompt(
            "Read the following legal case and identify relevant labels:\n\n" + item["text"],
            long_answer=long_answer
        )

        results.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": labels}
            ]
        })

    json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def prepare_finqa(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("finqa", split="train")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        context = " ".join(item["context"])
        prompt = build_prompt(item["question"], context, long_answer)

        results.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": str(item["answer"])}
            ]
        })

    json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def prepare_math(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("hendrycks/competition_math", split="test")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        prompt = build_prompt(item["problem"], long_answer=long_answer)

        results.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": item["solution"]}
            ]
        })

    json.dump(results, open(output_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ========== Registry ==========

DATASET_HANDLERS = {
    "gsm8k": prepare_gsm8k,
    "humaneval": prepare_humaneval,
    "mmlu": prepare_mmlu,
    "truthfulqa": prepare_truthfulqa,
    "sharegpt": prepare_sharegpt,
    "alpaca": prepare_alpaca,
    "pubmedqa": prepare_pubmedqa,
    "legalbench": prepare_legalbench,
    "finqa": prepare_finqa,
    "math": prepare_math,
}


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=list(DATASET_HANDLERS.keys()) + ["all"])
    parser.add_argument("--output-dir", default="./datasets")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--long-answer", action="store_true",
                        help="Force detailed answers (recommended for MTP testing)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "all":
        for name, fn in DATASET_HANDLERS.items():
            path = os.path.join(args.output_dir, f"{name}.json")
            print(f"Preparing {name}...")
            fn(path, args.num_samples, args.long_answer)
    else:
        fn = DATASET_HANDLERS[args.dataset]
        path = os.path.join(args.output_dir, f"{args.dataset}.json")
        fn(path, args.num_samples, args.long_answer)


if __name__ == "__main__":
    main()