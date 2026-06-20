"""
Test release_memory_occupation / resume_memory_occupation / update_weights_from_tensor
against a running lightllm server.

Sequence:
  1. baseline generate (sanity)
  2. release_memory_occupation  -> GPU memory should drop sharply
  3. resume_memory_occupation   -> GPU memory should grow back
                                   (without --enable_weight_cpu_backup the weight
                                    memory is allocated empty, so generation right
                                    after resume is expected to be garbage)
  4. update_weights_from_tensor (per-batch CUDA-IPC handoff) for every parameter
     found on disk -> repopulate weights
  5. final generate -> should produce a sensible answer again

The "trainer" runs in this same process: it holds tensors on a free GPU, serialises
them via lightllm.utils.serializer.MultiprocessingSerializer (CUDA IPC handles, not
data), then asks the server to clone them into its weight buffers. No NCCL group
is required, so this is safe to interrupt without leaving the server hung.

Usage:
  python test/test_api/test_rl_endpoints.py \
      --url http://127.0.0.1:8000 \
      --model_dir /nvme/models/Qwen3.5-35B-A3B \
      --tp 4 \
      --server_devices 0,1,2,3 \
      --client_device 4

Notes:
  - This script must run on the same machine as the server (CUDA IPC).
  - --server_devices are nvidia-smi GPU indices for the TP workers. If omitted,
    the script infers them from the top --tp memory consumers before release.
  - --client_device picks a free CUDA device for the in-process trainer; it is
    independent from --server_devices and should not overlap the TP workers.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from glob import glob
from typing import Dict, List, Tuple

# Make the repo importable when this script is invoked by path rather than -m.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import requests
import torch
from safetensors import safe_open

from lightllm.utils.patch_torch import monkey_patch_torch_reductions
from lightllm.utils.serializer import MultiprocessingSerializer


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def banner(msg: str):
    print(f"\n{YELLOW}=== {msg} ==={RESET}", flush=True)


def ok(msg: str):
    print(f"  {GREEN}OK{RESET}  {msg}", flush=True)


def fail(msg: str):
    print(f"  {RED}FAIL{RESET}  {msg}", flush=True)


def gpu_mem_used_mib() -> List[int]:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode()
    return [int(x.strip()) for x in out.strip().splitlines()]


def _select_gpu_mem(mem: List[int], devices: List[int]) -> List[int]:
    return [mem[i] for i in devices]


def _resolve_server_devices(server_devices: str, tp: int, mem: List[int]) -> List[int]:
    if tp <= 0:
        raise ValueError(f"--tp must be positive, got {tp}")
    if tp > len(mem):
        raise ValueError(f"--tp={tp} but nvidia-smi only returned {len(mem)} GPUs")

    value = server_devices.strip()
    if value.lower() == "auto":
        return sorted(range(len(mem)), key=lambda i: mem[i], reverse=True)[:tp]

    devices = [int(x.strip()) for x in value.split(",") if x.strip()]
    if len(devices) != tp:
        raise ValueError(f"--server_devices must contain exactly --tp entries; got {devices} for tp={tp}")
    if len(set(devices)) != len(devices):
        raise ValueError(f"--server_devices contains duplicates: {devices}")

    bad = [i for i in devices if i < 0 or i >= len(mem)]
    if bad:
        raise ValueError(f"--server_devices contains invalid GPU indices {bad}; nvidia-smi returned {len(mem)} GPUs")
    return devices


def post(url: str, path: str, payload=None, timeout=600):
    r = requests.post(url + path, json=payload or {}, timeout=timeout)
    try:
        body = r.json()
    except Exception:
        body = r.text
    return r.status_code, body


def generate(url: str, prompt: str, max_new_tokens: int = 16) -> str:
    r = requests.post(
        url + "/generate",
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": max_new_tokens, "do_sample": False},
        },
        timeout=120,
    )
    r.raise_for_status()
    data = r.json()
    if isinstance(data.get("generated_text"), list):
        return data["generated_text"][0]
    return data.get("generated_text", json.dumps(data))


def looks_garbage(text: str) -> bool:
    """Heuristic: post-resume text is usually a single repeated character (e.g. '!!!!')."""
    s = text.strip()
    if not s:
        return True
    return len(set(s)) == 1


# ---------------- weight-update helpers (update_weights_from_tensor) ----------------


def _list_safetensor_shards(model_dir: str) -> List[str]:
    shards = sorted(glob(os.path.join(model_dir, "*.safetensors")))
    if not shards:
        raise RuntimeError(f"no .safetensors found under {model_dir}")
    return shards


def _send_update_from_tensor(
    url: str,
    serialized_per_rank: List[str],
    flush_cache: bool = False,
):
    code, body = post(
        url,
        "/update_weights_from_tensor",
        {
            "serialized_named_tensors": serialized_per_rank,
            "load_format": None,
            "flush_cache": flush_cache,
            "abort_all_requests": False,
        },
        timeout=600,
    )
    return code, body


def update_weights_from_disk_via_tensor_api(
    url: str,
    model_dir: str,
    tp: int,
    client_device: int,
    batch_per_request: int = 8,
    flush_cache_at_end: bool = True,
):
    """
    Acts as an in-process "trainer": loads every safetensor shard onto
    cuda:client_device, then ships each batch of (name, tensor) to the server
    via /update_weights_from_tensor. The server worker on each TP rank receives
    a CUDA IPC handle, copies into its weight buffer.
    """
    banner("update_weights_from_tensor (CUDA IPC)")
    # Server side patches its own copy; we patch ours so reductions can serialise
    # CUDA tensors with UUID-based device addressing.
    monkey_patch_torch_reductions()
    torch.cuda.set_device(client_device)
    device = f"cuda:{client_device}"

    shards = _list_safetensor_shards(model_dir)
    print(f"  found {len(shards)} safetensor shards, batch_per_request={batch_per_request}", flush=True)

    total_params = 0
    total_bytes = 0
    t0 = time.time()
    for shard_idx, shard in enumerate(shards):
        shard_t0 = time.time()
        with safe_open(shard, framework="pt") as f:
            keys = list(f.keys())
            for i in range(0, len(keys), batch_per_request):
                batch_keys = keys[i : i + batch_per_request]
                # Load batch onto the client GPU. .contiguous() guarantees a
                # whole-tensor allocation (safetensors slices are already
                # contiguous, but this is cheap insurance).
                tensors = [f.get_tensor(k).to(device).contiguous() for k in batch_keys]
                named: List[Tuple[str, torch.Tensor]] = list(zip(batch_keys, tensors))

                # Same payload to every TP rank — the server clones full
                # tensors per rank and lets model.load_weights handle the TP
                # sharding internally (matching how update_weights_from_*
                # paths are written).
                blob = MultiprocessingSerializer.serialize(named, output_str=True)
                serialized_per_rank = [blob] * tp

                # Last batch flushes the prefix cache so old KV from the
                # previous weight version cannot poison subsequent gens.
                is_last = (shard_idx == len(shards) - 1) and (i + batch_per_request >= len(keys))
                code, body = _send_update_from_tensor(
                    url,
                    serialized_per_rank,
                    flush_cache=(flush_cache_at_end and is_last),
                )
                if code != 200:
                    fail(f"update batch failed: {code} {body}")
                    raise RuntimeError(f"update batch failed: {code} {body}")
                total_params += len(batch_keys)
                total_bytes += sum(t.numel() * t.element_size() for t in tensors)
                # Free client-side memory before next batch — the worker has
                # already cloned the data by the time post() returned.
                for t in tensors:
                    del t
                del tensors, named
                torch.cuda.empty_cache()

        print(
            f"  shard {shard_idx+1}/{len(shards)} done "
            f"(+{len(keys)} tensors, {time.time()-shard_t0:.1f}s, "
            f"running total {total_params} params, {total_bytes/1e9:.1f} GB)",
            flush=True,
        )

    dt = time.time() - t0
    ok(f"streamed {total_params} params, {total_bytes/1e9:.1f} GB in {dt:.1f}s")


# ---------------- main flow ----------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument(
        "--server_devices",
        default="auto",
        help="comma-separated nvidia-smi GPU indices used by the server, or 'auto' to infer from memory usage",
    )
    ap.add_argument(
        "--client_device",
        type=int,
        default=2,
        help="GPU index for the in-process trainer; must differ from TP worker GPUs",
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--batch_per_request", type=int, default=8)
    ap.add_argument("--skip_update", action="store_true", help="run only release/resume, skip the update_weights phase")
    args = ap.parse_args()

    # ---------------- stage 1: baseline ----------------
    banner("baseline generate")
    base_text = generate(args.url, args.prompt, args.max_new_tokens)
    print(f"  prompt   : {args.prompt!r}")
    print(f"  generated: {base_text!r}")
    ok("baseline generated")

    # ---------------- stage 2: release ----------------
    banner("release_memory_occupation")
    before = gpu_mem_used_mib()
    try:
        server_devices = _resolve_server_devices(args.server_devices, args.tp, before)
    except ValueError as e:
        fail(str(e))
        sys.exit(1)
    print(f"  server GPUs   : {server_devices}")
    print(f"  GPU mem before: {_select_gpu_mem(before, server_devices)}")
    code, body = post(args.url, "/release_memory_occupation", {})
    print(f"  resp: {code} {body}")
    if code != 200:
        fail("release failed")
        sys.exit(1)
    time.sleep(2)
    after = gpu_mem_used_mib()
    print(f"  GPU mem after : {_select_gpu_mem(after, server_devices)}")
    drop = sum(_select_gpu_mem(before, server_devices)) - sum(_select_gpu_mem(after, server_devices))
    if drop < 10_000:
        fail(f"release did not free much memory (delta={drop} MiB)")
        sys.exit(1)
    ok(f"release freed ~{drop} MiB on TP GPUs")

    # ---------------- stage 3: resume ----------------
    banner("resume_memory_occupation")
    code, body = post(args.url, "/resume_memory_occupation", {})
    print(f"  resp: {code} {body}")
    if code != 200:
        fail("resume failed")
        sys.exit(1)
    time.sleep(2)
    print(f"  GPU mem after : {_select_gpu_mem(gpu_mem_used_mib(), server_devices)}")
    ok("resume returned success")

    banner("post-resume generate (likely garbage without weight cpu backup)")
    text_after_resume = generate(args.url, args.prompt, args.max_new_tokens)
    print(f"  generated: {text_after_resume!r}  garbage_heuristic={looks_garbage(text_after_resume)}")

    if args.skip_update:
        ok("done (skipped update_weights stage)")
        return

    # ---------------- stage 4: update_weights_from_tensor ----------------
    update_weights_from_disk_via_tensor_api(
        url=args.url,
        model_dir=args.model_dir,
        tp=args.tp,
        client_device=args.client_device,
        batch_per_request=args.batch_per_request,
        flush_cache_at_end=True,
    )

    # ---------------- stage 5: final generate ----------------
    banner("final generate (after weight reload)")
    final_text = generate(args.url, args.prompt, args.max_new_tokens)
    print(f"  prompt   : {args.prompt!r}")
    print(f"  generated: {final_text!r}")
    if looks_garbage(final_text):
        fail("final generation still looks like garbage; weight update did not stick")
        sys.exit(1)
    if final_text.strip() == base_text.strip():
        ok("final output matches baseline exactly")
    else:
        ok("final output is sensible (differs from baseline but not garbage)")

    print(f"\n{GREEN}ALL STAGES PASSED{RESET}")


if __name__ == "__main__":
    main()
