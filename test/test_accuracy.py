import argparse
import subprocess
import time
import os
import requests
import sys
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, required=True, help="Number of GPUs to use.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the model.")
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


def main():
    # args
    args = parse_args()
    tp = args.tp
    model_dir = args.model_dir

    # output_file
    output_file = "test_results.txt"

    if os.path.exists(output_file):
        os.remove(output_file)

    # start server
    process = start_server(tp, model_dir)

    # prompts
    prompts = [
        "What is the machine learning?",
        "1+1等于几",
        "What role does attention play in transformer architectures?",
        "西红柿炒鸡蛋怎么做？",
        "Describe the concept of overfitting and underfitting.",
        "CPU和GPU的区别是什么？",
        "What is the role of a loss function in machine learning?",
    ]

    send_prompts(prompts, output_file)

    # shutdown server
    process.terminate()
    process.wait()


if __name__ == "__main__":
    main()

# python test_accuracy.py --tp 2 --model_dir /xx/xx
