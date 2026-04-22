"""Manual CLI wrapper for the OOM/stability self-check.

Usage:
    python test/check_oom.py --model_dir /path/to/model --port 18888 \\
        --running_max_req_size 128 --max_req_total_len 262144

Runs the same burst probe the server triggers on LIGHTLLM_CHECK_OOM=1: a single
wave of ``2 * running_max_req_size`` /generate requests fires at t=0 so ViT
encode and LLM decode peak together. The probe is hard-stopped at
``--deadline_s`` (default 60 s); any still-in-flight requests are cancelled and
the aiohttp session is closed, so the server observes client disconnects and
aborts the underlying requests.

Any default is overridable by flag or the matching env var
(LIGHTLLM_OOM_CHECK_IMAGE_SIZE / TOTAL / CONCURRENCY / MAX_NEW_TOKENS /
DEADLINE_S).
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
    parser.add_argument("--image_size", type=int, default=None, help="side length of the square test image (px)")
    parser.add_argument(
        "--total", type=int, default=None, help="total requests (default: 2 * running_max_req_size)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="concurrent in-flight requests (default: 2 * running_max_req_size — burst)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=None, help="decode length per request (default 256)")
    parser.add_argument("--deadline_s", type=float, default=None, help="hard wall clock in seconds (default 60)")
    args = parser.parse_args()

    run_oom_check(
        host=args.host,
        port=args.port,
        model_dir=args.model_dir,
        trust_remote_code=args.trust_remote_code,
        running_max_req_size=args.running_max_req_size,
        max_req_total_len=args.max_req_total_len,
        image_size=args.image_size,
        total=args.total,
        concurrency=args.concurrency,
        max_new_tokens=args.max_new_tokens,
        deadline_s=args.deadline_s,
    )


if __name__ == "__main__":
    main()
