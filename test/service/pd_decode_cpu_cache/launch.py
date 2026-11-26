"""Launch real P/D and CPU-cache-off reference services for a manual regression run.

Example:
    python test/service/pd_decode_cpu_cache/launch.py hybrid_normal \
        --model-dir /path/to/Qwen3.5-0.8B --output-dir /tmp/pd-cache-run
Keep this process alive while running run_cases.py; Ctrl-C cleans up only its children.
"""
import argparse
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "phase",
    choices=[
        "hybrid_normal",
        "hybrid_pf",
        "dense_normal",
        "dense_pf",
        "hybrid_dp2_graph",
        "hybrid_tp2_graph",
        "hybrid_pcpu_normal",
        "hybrid_pcpu_pf",
    ],
)
parser.add_argument("--model-dir", required=True)
parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "pd_decode_cpu_cache_service")
parser.add_argument("--host", default=socket.gethostbyname(socket.gethostname()))
parser.add_argument("--port", type=int, default=None)
parser.add_argument("--prefill-gpus")
parser.add_argument("--decode-gpus")
parser.add_argument("--baseline-gpus")
options = parser.parse_args()
OUT = options.output_dir.resolve()
phase, model, host = options.phase, options.model_dir, options.host
base = options.port or random.randrange(24000, 28000, 10)
if host in ("127.0.0.1", "localhost", "0.0.0.0"):
    parser.error("PD requires an actual interface IP; specify --host")
common = [
    "--model_dir",
    model,
    "--model_name",
    "cpu-cache-service-test",
    "--host",
    host,
    "--max_req_total_len",
    "3072",
    "--running_max_req_size",
    "8",
    "--batch_max_tokens",
    "1024",
    "--chunked_prefill_size",
    "512",
    "--max_total_token_num",
    "8192",
    "--graph_max_batch_size",
    "8",
    "--disable_vision",
    "--disable_audio",
    "--output_constraint_mode",
    "none",
    "--httpserver_workers",
    "1",
    "--page_size",
    "4",
    "--linear_att_hash_page_size",
    "128",
    "--linear_att_page_block_num",
    "4",
    "--linear_att_cache_size",
    "16",
    "--disable_cudagraph",
    "--att_backend",
    "gqa.prefill=triton",
    "gdn.prefill=triton",
    "gqa.decode=triton",
    "gdn.decode=triton",
]
master_args = [
    "--model_dir",
    model,
    "--model_name",
    "cpu-cache-service-test",
    "--host",
    host,
    "--max_req_total_len",
    "3072",
    "--running_max_req_size",
    "8",
    "--disable_vision",
    "--disable_audio",
    "--output_constraint_mode",
    "none",
    "--httpserver_workers",
    "1",
    "--run_mode",
    "pd_master",
    "--port",
    str(base),
    "--pd_master_mode",
    "1p1d",
    "--select_p_d_node_strategy",
    "round_robin",
]
if phase.endswith("_pf"):
    master_args += ["--pd_prefill_first"]
specs = [
    ("master", "", master_args),
    (
        "prefill",
        "0",
        common
        + [
            "--run_mode",
            "prefill",
            "--port",
            str(base + 1),
            "--nccl_port",
            str(base + 1001),
            "--pd_trans_mode",
            "nccl",
            "--pd_kv_page_size",
            "512",
            "--pd_master_ip",
            host,
            "--pd_master_port",
            str(base),
        ],
    ),
    (
        "decode",
        "1",
        common
        + [
            "--run_mode",
            "decode",
            "--port",
            str(base + 2),
            "--nccl_port",
            str(base + 1002),
            "--pd_trans_mode",
            "nccl",
            "--pd_kv_page_size",
            "512",
            "--pd_master_ip",
            host,
            "--pd_master_port",
            str(base),
            "--enable_cpu_cache",
            "--cpu_cache_storage_size",
            ("0.5" if phase.startswith("dense") else "0.125"),
            "--cpu_cache_token_page_size",
            "256",
            "--cache_placement_strategy",
            "legacy",
        ],
    ),
    ("baseline", "2", common + ["--run_mode", "normal", "--port", str(base + 3), "--nccl_port", str(base + 1003)]),
]
if phase.startswith("dense") and "gpu" not in phase:
    specs[2] = (specs[2][0], specs[2][1], specs[2][2] + ["--disable_dynamic_prompt_cache"])
if "pcpu" in phase:
    specs[1] = (
        specs[1][0],
        specs[1][1],
        specs[1][2] + ["--enable_cpu_cache", "--cpu_cache_storage_size", "1", "--cache_placement_strategy", "legacy"],
    )
if "dp2" in phase:
    specs[2] = (specs[2][0], "1,3", specs[2][2] + ["--tp", "2", "--dp", "2"])
if "tp2" in phase and "dp2" not in phase:
    specs[1] = (specs[1][0], "0,4", specs[1][2] + ["--tp", "2"])
    specs[2] = (specs[2][0], "1,3", specs[2][2] + ["--tp", "2"])
    specs[3] = (specs[3][0], "2,5", specs[3][2] + ["--tp", "2"])
if "graph" in phase:
    specs[2] = (specs[2][0], specs[2][1], [a for a in specs[2][2] if a != "--disable_cudagraph"])
for index, override in ((1, options.prefill_gpus), (2, options.decode_gpus), (3, options.baseline_gpus)):
    if override is not None:
        specs[index] = (specs[index][0], override, specs[index][2])
run_dir = OUT / phase
if (run_dir / "processes.json").exists():
    raise RuntimeError("Choose a fresh --output-dir to preserve the previous run's evidence")
run_dir.mkdir(parents=True, exist_ok=True)
procs = []
manifest = []
for port in [base, base + 1, base + 2, base + 3, base + 1001, base + 1002, base + 1003]:
    with socket.socket() as s:
        s.bind(("", port))
try:
    for name, gpus, args in specs:
        env = os.environ.copy()
        env.update(
            CUDA_VISIBLE_DEVICES=gpus,
            PYTHONUNBUFFERED="1",
            OMP_NUM_THREADS="2",
            LOADWORKER="4",
            LIGHTLLM_LOG_LEVEL="debug",
            NO_PROXY=f"{host},127.0.0.1,localhost",
            no_proxy=f"{host},127.0.0.1,localhost",
            PYTHONPATH=str(ROOT),
        )
        cmd = [sys.executable, "-m", "lightllm.server.api_server", *args]
        log = open(run_dir / f"{name}.log", "w")
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, log))
        manifest.append(dict(name=name, pid=p.pid, gpus=gpus, command=cmd, log=str(run_dir / f"{name}.log")))
    (run_dir / "processes.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(phase=phase, processes=manifest)), flush=True)
    client = httpx.Client(trust_env=False, timeout=2)
    deadline = time.time() + 600
    pending = {name: base + i for i, (name, _, _) in enumerate(specs)}
    while pending and time.time() < deadline:
        for name, p, _ in procs:
            if p.poll() is not None:
                raise RuntimeError(f"{name} exited: {p.returncode}; see {run_dir/name}.log")
        for name, port in list(pending.items()):
            try:
                r = client.get(f"http://{host}:{port}/readiness")
                if r.status_code == 200:
                    pending.pop(name)
                    print(f"READY {name} {port}", flush=True)
            except Exception:
                pass
        if pending:
            time.sleep(2)
    if pending:
        raise RuntimeError(f"startup timed out: {pending}")
    (run_dir / "ready.json").write_text(
        json.dumps(
            dict(
                phase=phase,
                time=time.time(),
                ports=dict(master=base, prefill=base + 1, decode=base + 2, baseline=base + 3),
            )
        )
    )
    print("ALL_READY", flush=True)
    while True:
        for name, p, _ in procs:
            if p.poll() is not None:
                raise RuntimeError(f"{name} exited: {p.returncode}")
        time.sleep(2)
except KeyboardInterrupt:
    pass
finally:
    for name, p, _ in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    end = time.time() + 20
    for _, p, log in procs:
        try:
            p.wait(timeout=max(0.1, end - time.time()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait(timeout=5)
        # A failed parent may exit while children still ignore SIGTERM during initialization.
        # Each service owns its process group; reclaim surviving children even when p.wait() succeeded.
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        log.close()
    print("STOPPED", flush=True)
