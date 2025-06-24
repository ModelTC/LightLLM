"""
This script starts the inference server, sends a set of prompts,
collects the outputs, and supports comparing the results between
the current commit and a specified historical commit for accuracy testing.

The command is:
python compare_with_previous_commit..py --tp 2 --model_dir /xx/xx --compare_commit_id xxxx

"""
import difflib
import argparse
import subprocess
import time
import os
import requests
import sys
import json
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True, help="Number of GPUs to use.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the model.")
    parser.add_argument("--compare_commit_id", type=str, default=None, help="The commit id of the baseline.")
    return parser.parse_args()


def start_server(tp, model_dir):
    cmd = [
        "python",
        "-m",
        "lightllm.server.api_server",
        "--tp",
        str(tp),
        "--model_dir",
        model_dir,
        "--data_type",
        "fp16",
        "--mode",
        "triton_gqa_flashdecoding",
        "--trust_remote_code",
        "--tokenizer_mode",
        "fast",
        "--host",
        "0.0.0.0",
        "--port",
        "8080",
    ]
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return process


def check_health():
    health_url = "http://localhost:8080/health"
    try:
        r = requests.get(health_url, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def send_prompts(prompts, output_file):
    for prompt in prompts:
        while not check_health():
            time.sleep(1)

        request_data = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1024, "frequency_penalty": 1, "do_sample": False},
            "multimodal_params": {},
        }

        try:
            r = requests.post("http://localhost:8080/generate", json=request_data, timeout=10)
            response_json = json.loads(r.text)
            generated_text = (
                response_json["generated_text"][0] if "generated_text" in response_json else "No generated_text."
            )
        except Exception as e:
            generated_text = f"ERROR: {str(e)}"

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"===== prompt: {prompt} =====\n")
            f.write(f"{generated_text}\n\n")

        print(f"===================Ouput saved in {output_file}===========================")


def compare_files(file1, file2, diff_output_file="diff.txt"):
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

    diff = difflib.unified_diff(lines1, lines2, fromfile=file1, tofile=file2, lineterm="")

    diff_list = list(diff)
    if not diff_list:
        print("两个文件内容完全一致。")
        return

    # 打印到终端
    for line in diff_list:
        if line.startswith("+") and not line.startswith("+++"):
            print("\033[32m" + line + "\033[0m", end="")  # 绿色
        elif line.startswith("-") and not line.startswith("---"):
            print("\033[31m" + line + "\033[0m", end="")  # 红色
        else:
            print(line, end="")

    # 保存到文件
    with open(diff_output_file, "w", encoding="utf-8") as f:
        for line in diff_list:
            f.write(line + "\n")
    print(f"\n差异已保存到 {diff_output_file}")


def run_and_save(tp, model_dir, output_file, prompts):
    """
    Start the server, send prompts, and save the results to output_file.
    """
    # Remove the old result file if it exists
    if os.path.exists(output_file):
        os.remove(output_file)

    process = None
    try:
        # Start the inference server
        process = start_server(tp, model_dir)
        # Send prompts and save results
        send_prompts(prompts, output_file)
    finally:
        # Shutdown the server
        if process is not None:
            process.terminate()
            process.wait()


def main():
    # Parse arguments
    args = parse_args()
    tp = args.tp
    model_dir = args.model_dir
    compare_commit_id = args.compare_commit_id

    # Prompts to test
    prompts = [
        "What is the machine learning?",
        "1+1等于几",
        "What role does attention play in transformer architectures?",
        "西红柿炒鸡蛋怎么做？",
        "Describe the concept of overfitting and underfitting.",
        "CPU和GPU的区别是什么？",
        "What is the role of a loss function in machine learning?",
    ]

    # Run and save results for the current commit
    current_output_file = "test_results_current.txt"
    run_and_save(tp, model_dir, current_output_file, prompts)

    # If compare_commit_id is provided, run and save results for the baseline commit
    if compare_commit_id:
        # Get the absolute path of the current script
        script_path = os.path.abspath(__file__)
        script_name = os.path.basename(script_path)
        tmp_script = f"/tmp/{script_name}"
        # Copy the current script to /tmp to ensure it exists in the baseline commit
        shutil.copy(script_path, tmp_script)
        # Save current commit id
        current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        # Save current branch name (if any)
        current_branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
        # Stash any local changes
        subprocess.run(["git", "stash"])
        # Checkout the baseline commit
        subprocess.run(["git", "checkout", compare_commit_id])
        # Copy the script back to the original location in case it does not exist in the baseline commit
        shutil.copy(tmp_script, script_path)
        try:
            compare_output_file = "test_results_compare.txt"
            run_and_save(tp, model_dir, compare_output_file, prompts)
        finally:
            # Checkout back to the original branch or commit
            if current_branch != "HEAD":
                subprocess.run(["git", "checkout", current_branch])
            else:
                subprocess.run(["git", "checkout", current_commit])
            # Pop the stashed changes
            subprocess.run(["git", "stash", "pop"])
            # Remove the temporary script file
            if os.path.exists(tmp_script):
                os.remove(tmp_script)
        # Compare the results
        compare_files(current_output_file, compare_output_file)


if __name__ == "__main__":
    main()
