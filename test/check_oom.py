"""Manual CLI wrapper for the OOM/stability self-check.

Usage:
    python test/check_oom.py --model_dir /path/to/model --port 18888 \\
        --running_max_req_size 128 --max_req_total_len 262144

Runs the same probe the server triggers on LIGHTLLM_CHECK_OOM=1.
"""
import argparse

from lightllm.server.oom_check.runner import run_oom_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--running_max_req_size", type=int, required=True)
    parser.add_argument("--max_req_total_len", type=int, required=True)
    args = parser.parse_args()

    run_oom_check(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
        trust_remote_code=args.trust_remote_code,
        running_max_req_size=args.running_max_req_size,
        max_req_total_len=args.max_req_total_len,
    )


if __name__ == "__main__":
    main()
