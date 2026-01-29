#!/usr/bin/env python3
"""
BFCL (Berkeley Function Calling Leaderboard) Evaluation Script for LightLLM

This script evaluates function/tool calling capabilities on the BFCL benchmark.

Usage:
    # Start LightLLM server first:
    python -m lightllm.server.api_server --model_dir /path/to/GLM-4.7-Flash --tp 1

    # Run evaluation:
    python eval_bfcl.py \
        --model_name GLM-4.7-Flash \
        --base_url http://localhost:8000/v1 \
        --test_category simple

Test Categories:
    - simple: Single function calls (400 examples)
    - multiple: Select one function from multiple options (200 examples)
    - parallel: Multiple function calls in parallel (200 examples)
    - parallel_multiple: Combination of parallel and multiple (200 examples)
    - java: Java function calls (100 examples)
    - javascript: JavaScript function calls (70 examples)
    - irrelevance: Detect when no function should be called
    - all: Run all categories

Requirements:
    pip install openai tqdm huggingface_hub
"""

import argparse
import json
import os
import re
import ast
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    exit(1)


# BFCL Dataset on HuggingFace
BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"

# Test category mappings to filenames
TEST_CATEGORIES = {
    "simple": "BFCL_v3_simple.json",
    "multiple": "BFCL_v3_multiple.json",
    "parallel": "BFCL_v3_parallel.json",
    "parallel_multiple": "BFCL_v3_parallel_multiple.json",
    "java": "BFCL_v3_java.json",
    "javascript": "BFCL_v3_javascript.json",
    "irrelevance": "BFCL_v3_irrelevance.json",
    "live_simple": "BFCL_v3_live_simple.json",
    "live_multiple": "BFCL_v3_live_multiple.json",
    "live_parallel": "BFCL_v3_live_parallel.json",
    "live_parallel_multiple": "BFCL_v3_live_parallel_multiple.json",
    "rest": "BFCL_v3_rest.json",
    "sql": "BFCL_v3_sql.json",
}

# Possible answer files for ground truth
ANSWER_FILES = {
    "simple": "possible_answer/BFCL_v3_simple.json",
    "multiple": "possible_answer/BFCL_v3_multiple.json",
    "parallel": "possible_answer/BFCL_v3_parallel.json",
    "parallel_multiple": "possible_answer/BFCL_v3_parallel_multiple.json",
    "java": "possible_answer/BFCL_v3_java.json",
    "javascript": "possible_answer/BFCL_v3_javascript.json",
    "live_simple": "possible_answer/BFCL_v3_live_simple.json",
    "live_multiple": "possible_answer/BFCL_v3_live_multiple.json",
    "live_parallel": "possible_answer/BFCL_v3_live_parallel.json",
    "live_parallel_multiple": "possible_answer/BFCL_v3_live_parallel_multiple.json",
    "sql": "possible_answer/BFCL_v3_sql.json",
}


@dataclass
class EvalResult:
    """Result of a single evaluation."""

    task_id: str
    category: str
    passed: bool
    model_output: str
    expected: Any
    error: Optional[str] = None


def download_bfcl_file(filename: str) -> str:
    """Download a BFCL file from HuggingFace Hub."""
    try:
        local_path = hf_hub_download(
            repo_id=BFCL_REPO,
            filename=filename,
            repo_type="dataset",
        )
        return local_path
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None


def load_jsonl_or_json(filepath: str) -> List[Dict[str, Any]]:
    """Load data from JSON or JSONL file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
        # Try as JSON array first
        try:
            data = json.loads(content)
            if isinstance(data, dict):
                data = [data]
        except json.JSONDecodeError:
            # Try as JSONL
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return data


def load_bfcl_data(category: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load BFCL dataset for a specific category."""
    filename = TEST_CATEGORIES.get(category)
    if not filename:
        print(f"Unknown category: {category}")
        return []

    print(f"Downloading {filename} from HuggingFace...")
    filepath = download_bfcl_file(filename)
    if not filepath:
        return []

    print(f"Loading data from {filepath}")
    data = load_jsonl_or_json(filepath)

    # Also load ground truth answers if available
    answer_file = ANSWER_FILES.get(category)
    if answer_file:
        print(f"Downloading answer file {answer_file}...")
        answer_path = download_bfcl_file(answer_file)
        if answer_path:
            answers = load_jsonl_or_json(answer_path)
            # Create a mapping from id to answer
            answer_map = {}
            for ans in answers:
                ans_id = ans.get("id", "")
                answer_map[ans_id] = ans.get("ground_truth", ans.get("result", []))

            # Merge answers into data
            for item in data:
                item_id = item.get("id", "")
                if item_id in answer_map:
                    item["ground_truth"] = answer_map[item_id]

    if limit:
        data = data[:limit]

    print(f"Loaded {len(data)} examples for category: {category}")
    return data


def fix_schema_types(schema: Any) -> Any:
    """
    Fix Python type names to JSON Schema types.
    BFCL uses Python type names like 'dict', 'list' but JSON Schema needs 'object', 'array'.
    """
    if isinstance(schema, dict):
        result = {}
        for key, value in schema.items():
            if key == "type" and isinstance(value, str):
                # Map Python types to JSON Schema types
                type_mapping = {
                    "dict": "object",
                    "list": "array",
                    "str": "string",
                    "int": "integer",
                    "float": "number",
                    "bool": "boolean",
                    "NoneType": "null",
                    "tuple": "array",
                }
                result[key] = type_mapping.get(value, value)
            else:
                result[key] = fix_schema_types(value)
        return result
    elif isinstance(schema, list):
        return [fix_schema_types(item) for item in schema]
    else:
        return schema


def convert_to_openai_tools(functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert BFCL function format to OpenAI tools format."""
    tools = []
    for func in functions:
        if isinstance(func, str):
            func = json.loads(func)

        # Fix the parameters schema to use valid JSON Schema types
        parameters = fix_schema_types(func.get("parameters", {}))

        tool = {
            "type": "function",
            "function": {
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": parameters,
            },
        }
        tools.append(tool)
    return tools


def parse_function_call(response: str) -> List[Dict[str, Any]]:
    """Parse function calls from model response."""
    calls = []

    # Try to parse as JSON array
    try:
        parsed = json.loads(response)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, dict):
            return [parsed]
    except json.JSONDecodeError:
        pass

    # Try to find function call patterns
    # Pattern 1: function_name(args)
    func_pattern = r"(\w+)\s*\((.*?)\)"
    matches = re.findall(func_pattern, response, re.DOTALL)
    for name, args_str in matches:
        try:
            # Try to parse args as Python dict/kwargs
            args_str = args_str.strip()
            if args_str:
                # Convert to dict format
                args = eval(f"dict({args_str})")
            else:
                args = {}
            calls.append({"name": name, "arguments": args})
        except:
            pass

    # Pattern 2: JSON-like tool_calls
    tool_call_pattern = r'\{"name":\s*"([^"]+)",\s*"arguments":\s*(\{[^}]+\})\}'
    matches = re.findall(tool_call_pattern, response)
    for name, args_str in matches:
        try:
            args = json.loads(args_str)
            calls.append({"name": name, "arguments": args})
        except:
            pass

    return calls


def extract_tool_calls_from_response(response) -> List[Dict[str, Any]]:
    """Extract tool calls from OpenAI API response."""
    calls = []

    if hasattr(response, "choices") and response.choices:
        choice = response.choices[0]
        message = choice.message

        # Check for tool_calls in response
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                func = tool_call.function
                try:
                    args = json.loads(func.arguments) if func.arguments else {}
                except json.JSONDecodeError:
                    args = {}
                calls.append({"name": func.name, "arguments": args})

        # Also check content for function calls (some models output in content)
        if hasattr(message, "content") and message.content:
            content_calls = parse_function_call(message.content)
            if content_calls and not calls:
                calls = content_calls

    return calls


def normalize_value(value: Any) -> Any:
    """Normalize values for comparison."""
    if isinstance(value, str):
        # Try to parse as number
        try:
            return float(value)
        except ValueError:
            return value.lower().strip()
    elif isinstance(value, bool):
        return value
    elif isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, list):
        return [normalize_value(v) for v in value]
    elif isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    return value


def value_matches_expected(predicted_value: Any, expected_values: Any) -> bool:
    """
    Check if predicted value matches expected value(s).
    BFCL format: expected values can be a list of acceptable values.
    """
    # Normalize predicted value
    pred_normalized = normalize_value(predicted_value)

    # If expected is a list, check if predicted matches any item
    if isinstance(expected_values, list):
        for exp_val in expected_values:
            exp_normalized = normalize_value(exp_val)
            if pred_normalized == exp_normalized:
                return True
            # Also try string comparison for edge cases
            if str(pred_normalized) == str(exp_normalized):
                return True
        return False
    else:
        exp_normalized = normalize_value(expected_values)
        return pred_normalized == exp_normalized or str(pred_normalized) == str(exp_normalized)


def compare_function_calls(
    predicted: List[Dict[str, Any]], expected: List[Dict[str, Any]], strict: bool = False
) -> Tuple[bool, str]:
    """Compare predicted function calls with expected ones."""
    if not predicted and not expected:
        return True, ""

    if len(predicted) != len(expected):
        return False, f"Count mismatch: predicted {len(predicted)}, expected {len(expected)}"

    # Sort by function name for comparison
    pred_sorted = sorted(predicted, key=lambda x: x.get("name", ""))
    exp_sorted = sorted(expected, key=lambda x: x.get("name", ""))

    for pred, exp in zip(pred_sorted, exp_sorted):
        pred_name = pred.get("name", "")
        exp_name = exp.get("name", "")

        if pred_name != exp_name:
            return False, f"Function name mismatch: {pred_name} vs {exp_name}"

        pred_args = pred.get("arguments", {})
        exp_args = exp.get("arguments", {})

        # Check required arguments match (BFCL format: values are lists of acceptable values)
        for key, expected_values in exp_args.items():
            if key not in pred_args:
                return False, f"Missing argument {key} in {pred_name}"
            if not value_matches_expected(pred_args[key], expected_values):
                return False, f"Argument {key} mismatch in {pred_name}"

    return True, ""


def parse_expected_output(ground_truth: Any) -> List[Dict[str, Any]]:
    """
    Parse expected output from BFCL ground truth.

    BFCL format: [{"func_name": {"arg1": [val1, val2], "arg2": [val3]}}]
    Convert to: [{"name": "func_name", "arguments": {"arg1": [val1, val2], "arg2": [val3]}}]
    """
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            # Try parsing as Python literal
            try:
                ground_truth = ast.literal_eval(ground_truth)
            except:
                return []

    if not ground_truth:
        return []

    # Ensure it's a list
    if isinstance(ground_truth, dict):
        ground_truth = [ground_truth]

    result = []
    for item in ground_truth:
        if isinstance(item, dict):
            # Check if it's already in standard format {"name": ..., "arguments": ...}
            if "name" in item and "arguments" in item:
                result.append(item)
            else:
                # BFCL format: {"func_name": {"arg1": [v1], "arg2": [v2]}}
                for func_name, args in item.items():
                    if isinstance(args, dict):
                        result.append({"name": func_name, "arguments": args})
                    else:
                        # Handle edge case where args might not be a dict
                        result.append({"name": func_name, "arguments": {}})

    return result


class BFCLEvaluator:
    """BFCL Benchmark Evaluator using OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        api_key: str = "EMPTY",
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate_response(
        self, prompt: str, tools: List[Dict[str, Any]], system_prompt: Optional[str] = None
    ) -> Tuple[Any, List[Dict[str, Any]]]:
        """Generate response from the model with tool calling."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            tool_calls = extract_tool_calls_from_response(response)
            return response, tool_calls
        except Exception as e:
            print(f"API Error: {e}")
            return None, []

    def evaluate_single(self, item: Dict[str, Any], category: str) -> EvalResult:
        """Evaluate a single BFCL example."""
        task_id = item.get("id", "unknown")

        # Extract question and functions
        question = item.get("question", [[{"role": "user", "content": ""}]])
        if isinstance(question, str):
            prompt = question
        elif isinstance(question, list) and question:
            if isinstance(question[0], dict):
                prompt = question[0].get("content", "")
            elif isinstance(question[0], list) and question[0]:
                prompt = question[0][0].get("content", "")
            else:
                prompt = str(question[0])
        else:
            prompt = str(question)

        # Get functions
        functions = item.get("function", [])
        if isinstance(functions, str):
            try:
                functions = json.loads(functions)
            except:
                functions = []

        if not isinstance(functions, list):
            functions = [functions]

        # Convert to OpenAI tools format
        tools = convert_to_openai_tools(functions)

        # Get expected output
        ground_truth = item.get("ground_truth", item.get("answer", []))
        expected = parse_expected_output(ground_truth)

        # Generate response
        system_prompt = (
            "You are a helpful assistant that can use tools/functions to help answer questions. "
            "When you need to call a function, use the provided tools."
        )

        response, predicted_calls = self.generate_response(prompt, tools, system_prompt)

        if response is None:
            return EvalResult(
                task_id=task_id,
                category=category,
                passed=False,
                model_output="",
                expected=expected,
                error="API call failed",
            )

        # For irrelevance category, model should NOT call any function
        if "irrelevance" in category.lower():
            passed = len(predicted_calls) == 0
            error = "Model called function when it shouldn't" if not passed else None
        else:
            # Compare function calls
            passed, error = compare_function_calls(predicted_calls, expected)

        model_output = json.dumps(predicted_calls, indent=2) if predicted_calls else str(response)

        return EvalResult(
            task_id=task_id, category=category, passed=passed, model_output=model_output, expected=expected, error=error
        )

    def evaluate_category(self, category: str, limit: Optional[int] = None, num_workers: int = 4) -> Dict[str, Any]:
        """Evaluate all examples in a category."""
        print(f"\nLoading BFCL dataset for category: {category}")
        data = load_bfcl_data(category, limit)

        if not data:
            print(f"No data found for category: {category}")
            return {"category": category, "total": 0, "passed": 0, "accuracy": 0.0}

        print(f"Loaded {len(data)} examples")

        results = []

        # Use ThreadPoolExecutor for concurrent evaluation
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.evaluate_single, item, category): item for item in data}

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {category}"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error evaluating: {e}")

        # Calculate metrics
        total = len(results)
        passed = sum(1 for r in results if r.passed)
        accuracy = passed / total * 100 if total > 0 else 0.0

        # Collect errors for analysis
        errors = defaultdict(int)
        for r in results:
            if not r.passed and r.error:
                errors[r.error[:50]] += 1

        return {
            "category": category,
            "total": total,
            "passed": passed,
            "accuracy": accuracy,
            "results": results,
            "error_summary": dict(errors),
        }


def main():
    parser = argparse.ArgumentParser(description="BFCL Evaluation for LightLLM")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--base_url", type=str, default="http://localhost:8000/v1", help="OpenAI-compatible API base URL"
    )
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key (use EMPTY for local)")
    parser.add_argument(
        "--test_category",
        type=str,
        default="simple",
        choices=list(TEST_CATEGORIES.keys()) + ["all"],
        help="Test category to evaluate",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples (for testing)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output file for detailed results")

    args = parser.parse_args()

    print("=" * 60)
    print("BFCL (Berkeley Function Calling Leaderboard) Evaluation")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"API URL: {args.base_url}")
    print(f"Test Category: {args.test_category}")
    print()

    evaluator = BFCLEvaluator(
        base_url=args.base_url,
        model_name=args.model_name,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Determine categories to evaluate
    if args.test_category == "all":
        categories = list(TEST_CATEGORIES.keys())
    else:
        categories = [args.test_category]

    all_results = {}

    for category in categories:
        result = evaluator.evaluate_category(category, limit=args.limit, num_workers=args.num_workers)
        all_results[category] = result

        print(f"\n{category.upper()} Results:")
        print(f"  Total: {result['total']}")
        print(f"  Passed: {result['passed']}")
        print(f"  Accuracy: {result['accuracy']:.2f}%")

        if result.get("error_summary"):
            print("  Common errors:")
            for error, count in sorted(result["error_summary"].items(), key=lambda x: -x[1])[:5]:
                print(f"    - {error}: {count}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Category':<25} {'Total':>8} {'Passed':>8} {'Accuracy':>10}")
    print("-" * 60)

    total_all = 0
    passed_all = 0

    for category, result in all_results.items():
        print(f"{category:<25} {result['total']:>8} {result['passed']:>8} {result['accuracy']:>9.2f}%")
        total_all += result["total"]
        passed_all += result["passed"]

    if len(all_results) > 1:
        print("-" * 60)
        overall_acc = passed_all / total_all * 100 if total_all > 0 else 0
        print(f"{'OVERALL':<25} {total_all:>8} {passed_all:>8} {overall_acc:>9.2f}%")

    print("=" * 60)

    # Save detailed results
    if args.output:
        output_data = {
            "model": args.model_name,
            "config": {
                "base_url": args.base_url,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
            },
            "results": {
                cat: {
                    "total": r["total"],
                    "passed": r["passed"],
                    "accuracy": r["accuracy"],
                    "error_summary": r.get("error_summary", {}),
                }
                for cat, r in all_results.items()
            },
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
