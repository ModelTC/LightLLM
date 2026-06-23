import re
import sys
from collections import defaultdict

def detailed_stats(file_path: str):
    patterns = {
        "mtp_avg_token_per_step": r"mtp_avg_token_per_step:\s*(\d+\.?\d*)",
        "mtp_avg_verify_tokens_per_step": r"mtp_avg_verify_tokens_per_step:\s*(\d+\.?\d*)",
        "gpu_cache_hit": r"gpu cache hit:\s*(True|False)",
        "gpu_prompt_cache_ratio": r"gpu_prompt_cache_ratio:\s*(\d+\.?\d*)",
        "pure_decode_time_ms": r"pure_decode_time_ms:\s*(\d+\.?\d*)",
        "pure_decode_token_num": r"pure_decode_token_num:\s*(\d+)",
        "pure_decode_time_per_token_ms": r"pure_decode_time_per_token_ms:\s*(\d+\.?\d*)",
        "pure_decode_throughput": r"pure_decode_throughput:\s*(\d+\.?\d*)",
    }

    metric_values = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            for metric_name, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    raw_value = match.group(1)
                    if metric_name == "gpu_cache_hit":
                        value = 1.0 if raw_value == "True" else 0.0
                    else:
                        value = float(raw_value)
                    metric_values[metric_name].append(value)

    if not metric_values:
        print("No matching data found")
        return

    def print_stats(name, values):
        if not values:
            return

        avg = sum(values) / len(values)

        print(f"{name}.avg = {avg:.4f}")
        print(f"{name}.min = {min(values):.4f}")
        print(f"{name}.max = {max(values):.4f}")
        print(f"{name}.count = {len(values)}")

    for name in [
        "mtp_avg_token_per_step",
        "mtp_avg_verify_tokens_per_step",
        "gpu_cache_hit",
        "gpu_prompt_cache_ratio",
        "pure_decode_time_ms",
        "pure_decode_token_num",
        "pure_decode_time_per_token_ms",
        "pure_decode_throughput",
    ]:
        values = metric_values.get(name, [])
        print_stats(name, values)

    total_pure_decode_time_ms = sum(metric_values.get("pure_decode_time_ms", []))
    total_pure_decode_token_num = sum(metric_values.get("pure_decode_token_num", []))
    if total_pure_decode_time_ms > 0.0 and total_pure_decode_token_num > 0.0:
        total_pure_decode_time_per_token_ms = total_pure_decode_time_ms / total_pure_decode_token_num
        total_pure_decode_throughput = total_pure_decode_token_num * 1000.0 / total_pure_decode_time_ms
        print(f"pure_decode_time_ms.sum = {total_pure_decode_time_ms:.4f}")
        print(f"pure_decode_token_num.sum = {total_pure_decode_token_num:.4f}")
        print(f"pure_decode_time_per_token_ms.global = {total_pure_decode_time_per_token_ms:.4f}")
        print(f"pure_decode_throughput.global = {total_pure_decode_throughput:.4f}")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "output.log"
    detailed_stats(file_path)
