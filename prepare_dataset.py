#!/usr/bin/env python3
"""
Download and prepare benchmark datasets for benchmark_sharegpt.py
Supports: gsm8k, humaneval, mmlu, truthfulqa, sharegpt
"""

import argparse
import json
import os
from typing import Dict, List, Any
from datasets import load_dataset


def prepare_gsm8k(output_path: str, num_samples: int = None) -> None:
    """
    Download and prepare GSM8K dataset.
    GSM8K contains grade school math problems.
    """
    print("Loading GSM8K dataset...")
    dataset = load_dataset("gsm8k", "main", split="test")

    results = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Format: question -> answer
        conversations = [
            {"from": "human", "value": item["question"]},
            {"from": "assistant", "value": item["answer"]}
        ]
        results.append({"conversations": conversations})

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"GSM8K dataset saved to {output_path}, total samples: {len(results)}")


def prepare_humaneval(output_path: str, num_samples: int = None) -> None:
    """
    Download and prepare HumanEval dataset.
    HumanEval contains code generation problems.
    """
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")

    results = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Format: prompt (includes function signature and docstring) -> canonical_solution
        prompt = item["prompt"]
        canonical_solution = item["canonical_solution"]

        # Create a more natural conversation format
        instruction = f"Complete the following Python function:\n\n{prompt}"

        conversations = [
            {"from": "human", "value": instruction},
            {"from": "assistant", "value": canonical_solution}
        ]
        results.append({"conversations": conversations})

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"HumanEval dataset saved to {output_path}, total samples: {len(results)}")


def prepare_mmlu(output_path: str, num_samples: int = None) -> None:
    """
    Download and prepare MMLU dataset.
    MMLU contains multiple choice questions across various subjects.
    """
    print("Loading MMLU dataset...")
    # Load a subset of MMLU (all subjects can be huge)
    subjects = ["abstract_algebra", "anatomy", "astronomy", "business_ethics"]

    results = []
    for subject in subjects:
        try:
            dataset = load_dataset("cais/mmlu", subject, split="test")
            for i, item in enumerate(dataset):
                if num_samples and len(results) >= num_samples:
                    break

                # Format: question with choices -> answer
                question = item["question"]
                choices = item["choices"]
                answer_idx = item["answer"]
                answer = choices[answer_idx]

                # Format choices
                choices_text = "\n".join([f"{chr(65+j)}. {choice}" for j, choice in enumerate(choices)])
                full_question = f"{question}\n\nChoices:\n{choices_text}\n\nAnswer with the correct choice."

                conversations = [
                    {"from": "human", "value": full_question},
                    {"from": "assistant", "value": answer}
                ]
                results.append({"conversations": conversations})
        except Exception as e:
            print(f"Warning: Could not load subject {subject}: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"MMLU dataset saved to {output_path}, total samples: {len(results)}")


def prepare_truthfulqa(output_path: str, num_samples: int = None) -> None:
    """
    Download and prepare TruthfulQA dataset.
    TruthfulQA contains questions that test model truthfulness.
    """
    print("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthful_qa", "generation", split="validation")

    results = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Format: question -> best answer
        question = item["question"]
        best_answer = item["best_answer"]

        conversations = [
            {"from": "human", "value": question},
            {"from": "assistant", "value": best_answer}
        ]
        results.append({"conversations": conversations})

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"TruthfulQA dataset saved to {output_path}, total samples: {len(results)}")


def prepare_sharegpt(output_path: str, num_samples: int = None) -> None:
    """
    Download and prepare ShareGPT dataset.
    ShareGPT contains real user conversations with ChatGPT.
    """
    print("Loading ShareGPT dataset...")
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")

    results = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # ShareGPT format has conversations array
        conversations = item.get("conversations", [])
        if not conversations:
            continue

        # Convert to the expected format
        formatted_convs = []
        for turn in conversations:
            from_val = turn.get("from", "")
            value = turn.get("value", "")
            if from_val and value:
                formatted_convs.append({"from": from_val, "value": value})

        if len(formatted_convs) >= 2:  # Need at least 2 turns
            results.append({"conversations": formatted_convs})

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"ShareGPT dataset saved to {output_path}, total samples: {len(results)}")


def prepare_alpaca(output_path: str, num_samples: int = None) -> None:
    """
    Download and prepare Alpaca dataset.
    Alpaca contains instruction-following examples.
    """
    print("Loading Alpaca dataset...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    results = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        # Format: instruction + input -> output
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]

        # Combine instruction and input
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction

        conversations = [
            {"from": "human", "value": prompt},
            {"from": "assistant", "value": output}
        ]
        results.append({"conversations": conversations})

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Alpaca dataset saved to {output_path}, total samples: {len(results)}")


def prepare_wikidata_qa(output_path: str, num_samples: int = None) -> None:
    """
    Download and prepare WikiData QA dataset.
    Simple question-answer pairs.
    """
    print("Loading WikiData QA dataset...")
    dataset = load_dataset("lmms-lab/WikiDataQA", split="train")

    results = []
    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        question = item.get("question", "")
        answer = item.get("answer", "")

        if question and answer:
            conversations = [
                {"from": "human", "value": question},
                {"from": "assistant", "value": answer}
            ]
            results.append({"conversations": conversations})

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"WikiData QA dataset saved to {output_path}, total samples: {len(results)}")


DATASET_HANDLERS = {
    "gsm8k": prepare_gsm8k,
    "humaneval": prepare_humaneval,
    "mmlu": prepare_mmlu,
    "truthfulqa": prepare_truthfulqa,
    "truthful_qa": prepare_truthfulqa,
    "sharegpt": prepare_sharegpt,
    "alpaca": prepare_alpaca,
    "wikidata_qa": prepare_wikidata_qa,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare benchmark datasets for benchmark_sharegpt.py"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_HANDLERS.keys()) + ["all"],
        help="Dataset to prepare. Use 'all' to prepare all datasets."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets",
        help="Directory to save the prepared datasets"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Maximum number of samples to include (default: all)"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "all":
        # Prepare all datasets
        for name, handler in DATASET_HANDLERS.items():
            output_path = os.path.join(args.output_dir, f"{name}.json")
            try:
                handler(output_path, args.num_samples)
            except Exception as e:
                print(f"Error preparing {name}: {e}")
    else:
        # Prepare specific dataset
        handler = DATASET_HANDLERS[args.dataset]
        output_path = os.path.join(args.output_dir, f"{args.dataset}.json")
        handler(output_path, args.num_samples)


if __name__ == "__main__":
    main()
