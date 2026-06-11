#!/usr/bin/env python3
"""
Download and prepare benchmark datasets for benchmark_sharegpt.py
Extended with: PubMedQA, LegalBench, FinQA, MATH and several colder / rarer QA datasets
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


def save_results(output_path: str, results: List[Dict[str, Any]]):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def get_choice_answer(choices: Dict[str, List[str]], answer_key: Any) -> str:
    labels = choices["label"]
    texts = choices["text"]
    if isinstance(answer_key, int):
        return texts[answer_key]
    for label, text in zip(labels, texts):
        if label == answer_key:
            return text
    raise ValueError(f"Cannot find answer {answer_key} in choices {labels}")


def build_choices_text(choices: Dict[str, List[str]]) -> str:
    return "\n".join([f"{label}. {text}" for label, text in zip(choices["label"], choices["text"])])


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

    save_results(output_path, results)


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

    save_results(output_path, results)


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

    save_results(output_path, results)


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

    save_results(output_path, results)


def prepare_sharegpt(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        convs = item.get("conversations", [])
        if len(convs) >= 2:
            results.append({"conversations": convs})

    save_results(output_path, results)


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

    save_results(output_path, results)


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

    save_results(output_path, results)


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

    save_results(output_path, results)


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

    save_results(output_path, results)


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

    save_results(output_path, results)


def prepare_drop(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("ucinlp/drop", split="validation")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        spans = item.get("answers_spans", {}).get("spans", [])
        if not spans:
            continue

        prompt = build_prompt(item["question"], item["passage"], long_answer)
        results.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": spans[0]}
            ]
        })

    save_results(output_path, results)


def prepare_qasc(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("allenai/qasc", split="validation")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        choices = item["choices"]
        answer = get_choice_answer(choices, item["answerKey"])
        facts = []
        for key in ["fact1", "fact2", "combinedfact"]:
            value = item.get(key)
            if value:
                facts.append(value)

        prompt = build_prompt(
            f"{item['question']}\n\nChoices:\n{build_choices_text(choices)}\n\nAnswer with the correct choice.",
            context="\n".join(facts),
            long_answer=long_answer,
        )

        results.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": answer}
            ]
        })

    save_results(output_path, results)


def prepare_quartz(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("allenai/quartz", split="validation")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        choices = item["choices"]
        answer = get_choice_answer(choices, item["answerKey"])
        prompt = build_prompt(
            f"{item['question']}\n\nChoices:\n{build_choices_text(choices)}\n\nAnswer with the correct choice.",
            context=item.get("para", ""),
            long_answer=long_answer,
        )

        results.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": answer}
            ]
        })

    save_results(output_path, results)


def prepare_sciq(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("allenai/sciq", split="test")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        choices = {
            "label": ["A", "B", "C", "D"],
            "text": [
                item["correct_answer"],
                item["distractor1"],
                item["distractor2"],
                item["distractor3"],
            ],
        }
        prompt = build_prompt(
            f"{item['question']}\n\nChoices:\n{build_choices_text(choices)}\n\nAnswer with the correct choice.",
            context=item.get("support", ""),
            long_answer=long_answer,
        )

        results.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": item["correct_answer"]}
            ]
        })

    save_results(output_path, results)


def prepare_openbookqa(output_path: str, num_samples: int = None, long_answer: bool = False):
    dataset = load_dataset("allenai/openbookqa", "additional", split="test")
    results = []

    for i, item in enumerate(dataset):
        if num_samples and i >= num_samples:
            break

        choices = item["choices"]
        answer = get_choice_answer(choices, item["answerKey"])
        prompt = build_prompt(
            f"{item['question_stem']}\n\nChoices:\n{build_choices_text(choices)}\n\nAnswer with the correct choice.",
            context=item.get("fact1", ""),
            long_answer=long_answer,
        )

        results.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "assistant", "value": answer}
            ]
        })

    save_results(output_path, results)


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
    "drop": prepare_drop,
    "qasc": prepare_qasc,
    "quartz": prepare_quartz,
    "sciq": prepare_sciq,
    "openbookqa": prepare_openbookqa,
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
