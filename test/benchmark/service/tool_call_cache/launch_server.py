"""Launch an existing image with an immutable source checkout; never build images.

Run under exp on the GPU host. The experiment receives the exact launch manifest
and a copy of this launcher. Model files and the source snapshot are read-only.
"""

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time


def wait_for_port(port, timeout=180):
    """Let a retired service release its TCP endpoint before strict startup checks."""
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        with socket.socket() as probe:
            try:
                probe.bind(("", port))
                return
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"port {port} is still occupied; no process was stopped") from exc
        if not announced:
            print(f"Waiting for port {port} to become bindable before service startup (untimed)", flush=True)
            announced = True
        time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-dir", default="/mtc/models/Qwen3-4B")
    parser.add_argument("--model-name", default="tool-cache-qwen4b")
    parser.add_argument("--tool-call-parser", default="qwen25")
    parser.add_argument("--image", default="ghcr.io/modeltc/lightllm:main")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--cache-tokens", type=int, default=32768)
    parser.add_argument("--max-req-tokens", type=int, default=16384)
    parser.add_argument("--batch-tokens", type=int, default=8192)
    parser.add_argument("--max-requests", type=int, default=32)
    parser.add_argument("--retention", type=float, default=0)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--disable-cudagraph", action="store_true")
    parser.add_argument("--original-base", action="store_true", help="Source predates retention flags; omit them")
    parser.add_argument("--kernel-cache", type=Path, help="Reuse compiled kernels across service restarts")
    parser.add_argument("--tilelang-cache", type=Path, help="Reuse TileLang kernels across service restarts")
    parser.add_argument("--linear-att-page-block-num", type=int)
    parser.add_argument("--prefill-queue-strategy", choices=["auto", "default", "promote_shortest", "hrrn"])
    parser.add_argument("--output-constraint-mode", choices=["none", "xgrammar"])
    parser.add_argument("--python-deps", type=Path, help="Read-only, preinstalled runtime dependency overlay")
    parser.add_argument("--log-level", choices=["info", "debug"], default="info")
    args = parser.parse_args()
    if "EXP_RUN_DIR" not in os.environ or len(args.revision) != 40:
        parser.error("run under exp and supply a full source revision")
    record = Path(os.environ["EXP_RUN_DIR"])
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        args.name,
        "--gpus",
        f'"device={args.gpu}"',
        "--network",
        "host",
        "--shm-size=8g",
        "-v",
        f"{args.source}:/work:ro",
        "-v",
        f"{args.model_dir}:/model:ro",
        "-w",
        "/work",
        "-e",
        "PYTHONPATH=/work",
        "-e",
        f"LIGHTLLM_LOG_LEVEL={args.log_level}",
        args.image,
        "python",
        "-m",
        "lightllm.server.api_server",
        "--model_dir",
        "/model",
        "--model_name",
        args.model_name,
        "--tp",
        str(args.tp),
        "--port",
        str(args.port),
        "--max_total_token_num",
        str(args.cache_tokens),
        "--max_req_total_len",
        str(args.max_req_tokens),
        "--batch_max_tokens",
        str(args.batch_tokens),
        "--running_max_req_size",
        str(args.max_requests),
        "--tool_call_parser",
        args.tool_call_parser,
        "--tool_call_cache_retention_seconds",
        str(args.retention),
        "--tool_call_cache_max_fraction",
        str(args.fraction),
        "--llm_prefill_att_backend",
        "triton",
        "--disable_vision",
        "--disable_audio",
        "--graph_max_batch_size",
        str(args.max_requests),
        "--graph_max_len_in_batch",
        str(args.max_req_tokens),
    ]
    if args.disable_cudagraph:
        command.append("--disable_cudagraph")
    if args.prefill_queue_strategy is not None:
        command.extend(["--prefill_queue_strategy", args.prefill_queue_strategy])
    if args.output_constraint_mode is not None:
        command.extend(["--output_constraint_mode", args.output_constraint_mode])
    dependency_hashes = {}
    if args.python_deps is not None:
        dependency_hashes = {
            str(path.relative_to(args.python_deps)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(args.python_deps.rglob("*"))
            if path.is_file()
        }
        if not dependency_hashes:
            parser.error("runtime dependency overlay is empty")
        offset = command.index(args.image)
        command[offset:offset] = ["-v", f"{args.python_deps}:/deps:ro", "-e", "PYTHONPATH=/work:/deps"]
    if args.linear_att_page_block_num is not None:
        command.extend(["--linear_att_page_block_num", str(args.linear_att_page_block_num)])
    if args.original_base:
        if args.retention:
            parser.error("original base cannot enable retention")
        offset = command.index("--tool_call_cache_retention_seconds")
        del command[offset : offset + 4]
    if args.kernel_cache is not None:
        (args.kernel_cache / "flashinfer").mkdir(parents=True, exist_ok=True)
        (args.kernel_cache / "triton").mkdir(parents=True, exist_ok=True)
        offset = command.index(args.image)
        command[offset:offset] = [
            "-v",
            f"{args.kernel_cache}:/kernel-cache",
            "-v",
            f"{args.kernel_cache / 'flashinfer'}:/root/.cache/flashinfer",
            "-e",
            "TRITON_CACHE_DIR=/kernel-cache/triton",
        ]
    if args.tilelang_cache is not None:
        args.tilelang_cache.mkdir(parents=True, exist_ok=True)
        offset = command.index(args.image)
        command[offset:offset] = ["-v", f"{args.tilelang_cache}:/root/.tilelang"]
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image], text=True))[0]
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version", "--format=csv"], text=True
    )
    manifest = {
        "source_revision": args.revision,
        "image_id": image["Id"],
        "image_digests": image["RepoDigests"],
        "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_config_sha256": hashlib.sha256((Path(args.model_dir) / "config.json").read_bytes()).hexdigest(),
        "python_dependency_sha256": dependency_hashes,
        "hardware": hardware,
        "command": command,
    }
    (record / "service-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(__file__, record / "launch_server.py")
    print(json.dumps(manifest, indent=2), flush=True)
    wait_for_port(args.port)
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
