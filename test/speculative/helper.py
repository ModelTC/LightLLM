import re
import sys
from collections import defaultdict

def detailed_stats(file_path: str):
    # 1. Define patterns to match metrics and their regex
    patterns = {
        'mtp_avg_token_per_step': r'mtp_avg_token_per_step:\s*(\d+\.?\d*)',
        'mtp_avg_verify_tokens_per_step': r'mtp_avg_verify_tokens_per_step:\s*(\d+\.?\d*)'
    }
    
    # 2. Store extracted values, structured as {metric_name: [list of values]}
    metric_values = defaultdict(list)
    
    # (Optional) If you want to record by Step and overwrite old values, enable the line below
    # step_values = defaultdict(float) 
    # Note: If you want to record by Step as a list, use: step_values = defaultdict(lambda: defaultdict(list))

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Iterate through all defined patterns to match
            for metric_name, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    value = float(match.group(1))
                    metric_values[metric_name].append(value)
                    
                    # --- (Optional) Step-based recording logic ---
                    # step_match = re.search(r'Step\s+(\d+)', line)
                    # if step_match:
                    #     step = int(step_match.group(1))
                    #     step_values[step] = value # This will keep the last value for each step

    if not metric_values:
        print("No matching data found")
        return

    # 3. Define helper function to print statistics
    def print_stats(name, values):
        if not values:
            return

        avg = sum(values) / len(values)

        # Output in parseable format: metric_name.avg = value
        print(f"{name}.avg = {avg:.4f}")
        print(f"{name}.min = {min(values):.4f}")
        print(f"{name}.max = {max(values):.4f}")
        print(f"{name}.count = {len(values)}")

    # 4. Print metrics in parseable format
    for name, values in metric_values.items():
        print_stats(name, values)

if __name__ == "__main__":
    # If no file path is provided, default to output.log
    file_path = sys.argv[1] if len(sys.argv) > 1 else "output.log"
    detailed_stats(file_path)