"""Sequential, isolated eight-GPU A/B with a strict full-vocabulary precision gate.

Precision and performance use separate snapshots/runs; capture I/O never enters
timing. Only processes started with this run's unique ownership token are stopped.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import socket
import subprocess
import time
import urllib.request
import uuid

import torch
from transformers import AutoTokenizer

from prepare_runtime import prepare, verify_baseline, package_hashes, REPO


HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
PYTHON = os.environ.get("PYTHON_BIN", "/opt/conda/bin/python")


def check_resources(gpus, port):
    if len(gpus.split(",")) != 8:
        raise ValueError("This baseline comparison preserves the supplied eight-GPU 2P1D topology")
    used = subprocess.check_output(
        [
            "nvidia-smi",
            f"--id={gpus}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    if any(int(line) > 512 for line in used.splitlines()):
        raise RuntimeError(f"An idle eight-GPU allocation is required; current memory usage MiB: {used.strip()}")
    # Keep fixed service ports below Linux's ephemeral range (32768+).
    for number in (port, 24061, 24062, 24063, 25061, 25062, 25063):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", number))


def owned_pids(token):
    marker = b"BSH_NSA_RUN_ID=" + token.encode()
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            try:
                if marker in (path / "environ").read_bytes().split(b"\0"):
                    yield int(path.name)
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                pass


def wait_resources(gpus, port, timeout=60):
    # Child CUDA contexts may outlive their parent briefly during teardown.
    # Wait for release; never stop processes from another allocation.
    deadline = time.monotonic() + timeout
    while True:
        try:
            check_resources(gpus, port)
            return
        except (RuntimeError, OSError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.5)


def stop(process, token):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in list(owned_pids(token)):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
            time.sleep(2)
    process.wait(timeout=15)


def prompt_tokens(tokenizer, length, seed):
    rng = random.Random(seed)
    text = [f"{rng.getrandbits(120):030x}: analyze the following measurements.\n"]
    text.extend(
        f"Row {i}: {rng.randrange(100000)} {rng.choice(['red', 'blue', 'green'])} value {rng.randrange(100000)}.\n"
        for i in range(length // 8 + 64)
    )
    ids = tokenizer.encode("".join(text), add_special_tokens=False)[:length]
    if len(ids) != length:
        raise RuntimeError("Synthetic prompt construction did not reach the requested token length")
    prompt = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if tokenizer.encode(prompt, add_special_tokens=False) != ids:
        raise RuntimeError("Tokenizer round trip changed benchmark inputs")
    return prompt


def complete(port, prompt, output_tokens):
    payload = {
        "model": "DSA_STEP519_BSH_NSA",
        "prompt": prompt,
        "temperature": 0.0,
        "do_sample": False,
        "max_tokens": output_tokens,
        "ignore_eos": output_tokens > 1,
        "stream": True,
    }
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    start, first, text, usage = time.monotonic(), None, [], None
    with HTTP.open(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            item = json.loads(data)
            if item.get("usage"):
                usage = item["usage"]
            for choice in item.get("choices") or []:
                if choice.get("text"):
                    first = first or time.monotonic()
                    text.append(choice["text"])
    end = time.monotonic()
    if usage is None or (first is None and output_tokens > 1):
        raise RuntimeError("Streaming response omitted text or token usage")
    first = first or end
    return {
        "output": "".join(text),
        "usage": usage,
        "ttft_s": first - start,
        "duration_s": end - start,
        "decode_tok_s": (usage["completion_tokens"] - 1) / max(end - first, 1e-9),
    }


def precision_cases(args, tokenizer):
    cases = []
    for length in (128, 1970, 2049, 8192, 32768, 65536, 131072, 524032):
        cases.append(
            (f"tokens_{length}", length, prompt_tokens(tokenizer, length, 519 + length), 128 if length == 1970 else 1)
        )
    tasks = json.loads(args.quality_set.read_text())["tasks"]
    names = list(tasks)
    random.Random(20261107).shuffle(names)
    short = []
    for name in names:
        prompt = (
            tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": "Choose the correct option. Return the letter only.\n" + tasks[name]["question"],
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "The correct answer is ("
        )
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) <= 1950:
            short.append((name, ids))
        if len(short) == 64:
            break
    if len(short) != 64:
        raise RuntimeError("Precision dataset needs 64 short cases under the baseline tokenizer")
    filler = tokenizer.encode(
        "Ignore the following unrelated log rows.\n"
        + "".join(f"Reference row {j}: alpha beta gamma observation {j * 17}.\n" for j in range(4000)),
        add_special_tokens=False,
    )[:16384]
    for group, chosen in (("short", short), ("long", short[:8])):
        for name, ids in chosen:
            if group == "long":
                ids = filler + ids
            prompt = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if tokenizer.encode(prompt, add_special_tokens=False) != ids:
                raise RuntimeError("Quality prompt token round trip changed inputs")
            cases.append((f"probe_{group}_{hashlib.sha256(name.encode()).hexdigest()[:16]}", len(ids), prompt, 1))
    return cases


def collect(args, tokenizer, capture, precision):
    records = []
    if args.phase == "precision":
        for case, length, prompt, output_tokens in precision:
            print(
                json.dumps({"event": "precision_case", "role": capture.name, "case": case, "tokens": length}),
                flush=True,
            )
            (capture / "armed.json").write_text(json.dumps({"id": case}))
            try:
                response = complete(args.port, prompt, output_tokens)
                if response["usage"]["prompt_tokens"] != length:
                    raise RuntimeError("Server token count differs from fixed precision input")
                records.append({"case": case, **response})
            finally:
                (capture / "armed.json").unlink(missing_ok=True)
    else:
        for length in args.lengths:
            for repeat in range(-1, args.repeats):
                prompt = prompt_tokens(tokenizer, length, length * 10 + repeat)
                response = complete(args.port, prompt, 256)
                if response["usage"]["prompt_tokens"] != length:
                    raise RuntimeError("Server token count differs from fixed timing input")
                records.append({"length": length, "repeat": repeat, **response})
                print(
                    json.dumps(
                        {
                            "event": "timing",
                            "role": capture.name,
                            "tokens": length,
                            "repeat": repeat,
                            "ttft_s": response["ttft_s"],
                            "decode_tok_s": response["decode_tok_s"],
                        }
                    ),
                    flush=True,
                )
        for concurrency in (8, 32):
            prompts = [prompt_tokens(tokenizer, 4096, 519000 + i) for i in range(concurrency)]
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                responses = list(pool.map(lambda prompt: complete(args.port, prompt, 128), prompts))
            elapsed = time.monotonic() - start
            records.append(
                {
                    "concurrency": concurrency,
                    "duration_s": elapsed,
                    "responses": responses,
                    "aggregate_output_tok_s": sum(r["usage"]["completion_tokens"] for r in responses) / elapsed,
                }
            )
    return records


def final_prefill(capture, case):
    snapshots = [torch.load(p, map_location="cpu", weights_only=True) for p in capture.glob(case + "_*.pt")]
    candidates = [s for s in snapshots if s["is_prefill"] and s["logits"].numel()]
    if not candidates:
        raise RuntimeError(f"Missing full-vocabulary capture for {case}")
    return max(candidates, key=lambda item: item["kv_length"])


def precision_result(root, results):
    comparisons = []
    for left, right in zip(results["baseline"], results["candidate"], strict=True):
        if left["case"] != right["case"]:
            raise RuntimeError("Precision cases were reordered")
        case = left["case"]
        before = final_prefill(root / "capture/baseline", case)
        after = final_prefill(root / "capture/candidate", case)
        a = before["logits"].float().reshape(-1, before["logits"].shape[-1])[-1]
        b = after["logits"].float().reshape(-1, after["logits"].shape[-1])[-1]
        same_inputs = torch.equal(before["input_ids"], after["input_ids"])
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        comparisons.append(
            {
                "case": case,
                "inputs_exact": same_inputs,
                "logits_exact": torch.equal(a, b),
                "generation_exact": left["output"] == right["output"],
                "finite": finite,
                "max_abs": float((a - b).abs().max()),
                "mean_abs": float((a - b).abs().mean()),
            }
        )
    return {
        "cases": comparisons,
        "passed": all(
            c["inputs_exact"] and c["logits_exact"] and c["generation_exact"] and c["finite"] for c in comparisons
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("precision", "performance"), default="precision")
    parser.add_argument("--precision-result", type=Path)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--port", type=int, default=18150)
    parser.add_argument("--candidate-topk", choices=("packed", "torch", "radix", "vllm", "auto"), default="auto")
    parser.add_argument("--query-tile", type=int, choices=(4, 8, 16), default=8)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1728, 8192, 32768, 65536, 131072, 262144, 524032])
    parser.add_argument(
        "--quality-set", type=Path, default=Path("/mnt/afs-toolcall/yankuihao/code/dsa_fa3_dense_20261107/gpqa.json")
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = verify_baseline()
    wait_resources(args.gpu_ids, args.port)
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if args.phase != "precision" or args.candidate_topk != "auto" or args.query_tile != 8:
            raise RuntimeError("Precision resume requires the frozen auto/tile8 configuration")
        runtime = json.loads((root / "runtimes/runtime_manifest.json").read_text())
        if not runtime["precision_hooks"] or runtime["baseline"] != manifest:
            raise RuntimeError("The saved precision baseline does not match")
        if package_hashes(REPO) != runtime["candidate_source_files"]:
            raise RuntimeError("Candidate engine source changed since the saved precision run")
        for role, path in runtime["runtimes"].items():
            expected = {k: v for k, v in runtime["files"][role].items() if "__pycache__" not in Path(k).parts}
            if package_hashes(Path(path)) != expected:
                raise RuntimeError("A frozen precision package changed: " + role)
    else:
        runtime = prepare(root / "runtimes", precision=args.phase == "precision")
    if args.phase == "performance":
        if args.precision_result is None:
            raise RuntimeError("Performance measurement requires --precision-result from a passing precision run")
        gate = json.loads(args.precision_result.read_text())
        if not gate.get("passed") or gate.get("candidate_source_files") != runtime["candidate_source_files"]:
            raise RuntimeError("Precision gate is missing, failed, or belongs to a different working tree")
        if gate.get("candidate_topk") != args.candidate_topk or gate.get("query_tile") != args.query_tile:
            raise RuntimeError("Precision gate used different selector/query-tile settings")
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_dir"], trust_remote_code=True)
    precision = precision_cases(args, tokenizer) if args.phase == "precision" else None
    results = {}
    for role in ("baseline", "candidate"):
        saved = root / f"{role}.json"
        if args.resume and saved.exists():
            records = json.loads(saved.read_text())
            if [r["case"] for r in records] != [c[0] for c in precision]:
                raise RuntimeError("Saved precision cases do not match the fixed inputs")
            results[role] = records
            print(json.dumps({"event": "reused_precision_cases", "role": role, "count": len(records)}), flush=True)
            continue
        wait_resources(args.gpu_ids, args.port)
        capture = root / "capture" / role
        if capture.exists():
            capture.rename(capture.with_name(role + f".attempt-{time.time_ns()}"))
        capture.mkdir(parents=True)
        token = uuid.uuid4().hex
        # Keep the original code and arguments, with independent mutable tuning
        # directories for the two P replicas. Concurrent file truncation can
        # otherwise leave one TP rank at a different collective during warmup.
        from run_matrix import node_roots, serve_script

        launch_folder = root / "launch" / role
        launch_folder.mkdir(parents=True, exist_ok=args.resume)
        node_code = node_roots(launch_folder, runtime["runtimes"][role])
        serve = serve_script(launch_folder, mtp=3, profile=False)
        env = os.environ.copy()
        env.update(
            {
                "MODE": role,
                "LIGHTLLM_ROOT": runtime["runtimes"][role],
                "SERVE_SCRIPT": str(serve),
                "BSH_NODE_CODE_ROOT": str(node_code),
                "GPU_IDS": args.gpu_ids,
                "PORT": str(args.port),
                "BSH_NSA_RUN_ID": token,
                "P_PORT": "24061",
                "P2_PORT": "24062",
                "D_PORT": "24063",
                "P_NCCL_PORT": "25061",
                "P2_NCCL_PORT": "25062",
                "D_NCCL_PORT": "25063",
                "DSA_TOPK_IMPL": args.candidate_topk if role == "candidate" else "packed",
                "DSA_SCORE_QUERY_TILE": str(args.query_tile if role == "candidate" else 4),
                "PYTHON_BIN": PYTHON,
            }
        )
        env.pop("DSA_PRECISION_DIR", None)
        if args.phase == "precision":
            env["DSA_PRECISION_DIR"] = str(capture)
        with (root / f"{role}.log").open("w") as log:
            print(json.dumps({"event": "starting", "role": role}), flush=True)
            process = subprocess.Popen(
                ["bash", str(REPO / "tools/sensenova_loop_nsa/launch.sh")],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 1800
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"{role} startup exited: see {root / (role + '.log')}")
                    try:
                        with HTTP.open(f"http://127.0.0.1:{args.port}/health", timeout=3) as response:
                            if response.status == 200:
                                break
                    except Exception:
                        pass
                    if time.monotonic() > deadline:
                        raise TimeoutError(role + " startup timed out")
                    time.sleep(3)
                print(json.dumps({"event": "ready", "role": role}), flush=True)
                results[role] = collect(args, tokenizer, capture, precision)
                (root / f"{role}.json").write_text(json.dumps(results[role], indent=2) + "\n")
            finally:
                stop(process, token)
        deadline = time.monotonic() + 60
        while list(owned_pids(token)) and time.monotonic() < deadline:
            time.sleep(1)
    result = {
        "phase": args.phase,
        "complete": True,
        "candidate_topk": args.candidate_topk,
        "query_tile": args.query_tile,
        "candidate_source_files": runtime["candidate_source_files"],
    }
    if args.phase == "precision":
        result.update(precision_result(root, results))
        result["quality_set_sha256"] = hashlib.sha256(args.quality_set.read_bytes()).hexdigest()
    else:
        result["results"] = results
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "candidate_source_files"}, indent=2))
    if args.phase == "precision" and not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
