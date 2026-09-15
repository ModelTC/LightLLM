#!/usr/bin/env python3
"""Run a loopback-only, self-owned FP8 KV calibration job.

This is deliberately not a server CLI feature.  The tool starts an isolated
normal-mode child process with an internal StartArgs flag, sends text/token-ID
samples, obtains rank-local snapshots, merges them on CPU, publishes a JSON and
report with atomic replacements, then terminates only its own process group.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Direct execution sets sys.path[0] to tools/.  Prefer this checkout over any
# separately installed LightLLM package so StartArgs and launch_server match.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

QMAX = 448.0


def _finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _validate_rank_snapshot(snapshot: dict[str, Any], expected: dict[str, Any] | None = None) -> None:
    required = {
        "rank",
        "layer_num",
        "per_head",
        "shape",
        "head_num",
        "counts",
        "observed_token_rows",
        "abs_max",
        "qmin",
        "qmax",
        "architecture",
        "num_target_layers",
        "num_draft_layers",
    }
    missing = required - snapshot.keys()
    if missing:
        raise ValueError(f"rank snapshot missing fields: {sorted(missing)}")
    for key, lower in {"rank": 0, "layer_num": 1, "head_num": 1, "num_target_layers": 1, "num_draft_layers": 0}.items():
        if type(snapshot[key]) is not int or snapshot[key] < lower:
            raise ValueError(f"rank snapshot has invalid {key}")
    if type(snapshot["per_head"]) is not bool:
        raise ValueError("rank snapshot per_head must be boolean")
    layer_num = snapshot["layer_num"]
    if snapshot["num_target_layers"] + snapshot["num_draft_layers"] != layer_num:
        raise ValueError("rank snapshot target/draft layer counts do not cover all layers")
    if any(
        not isinstance(snapshot[key], list) or len(snapshot[key]) != layer_num
        for key in ("abs_max", "counts", "observed_token_rows")
    ):
        raise ValueError("rank snapshot layer data length is inconsistent")
    width = 2 * snapshot["head_num"] if snapshot["per_head"] else 2
    if snapshot["shape"] != [layer_num, width] or snapshot["qmin"] != -QMAX or snapshot["qmax"] != QMAX:
        raise ValueError("rank snapshot shape or q-range is invalid")
    for counts in (snapshot["counts"], snapshot["observed_token_rows"]):
        if any(type(value) is not int or value <= 0 for value in counts):
            raise ValueError("rank snapshot has an unobserved calibration layer")
    for row in snapshot["abs_max"]:
        if (
            not isinstance(row, list)
            or len(row) != width
            or not all(_finite_number(value) and value >= 0 for value in row)
        ):
            raise ValueError("rank snapshot contains invalid maxima")
    if expected:
        for key in (
            "layer_num",
            "per_head",
            "shape",
            "head_num",
            "qmin",
            "qmax",
            "architecture",
            "num_target_layers",
            "num_draft_layers",
        ):
            if snapshot[key] != expected[key]:
                raise ValueError(f"rank snapshots disagree on {key}")


def merge_rank_snapshots(snapshots: list[dict[str, Any]], *, expected_ranks: int | None = None) -> dict[str, Any]:
    """Merge complete rank-local CPU snapshots without any GPU collective."""
    if not snapshots:
        raise ValueError("no rank snapshots received")
    for row in snapshots:
        _validate_rank_snapshot(row)
    ranks = sorted(snapshots, key=lambda row: row["rank"])
    if expected_ranks is not None and len(ranks) != expected_ranks:
        raise ValueError(f"expected {expected_ranks} rank snapshots, got {len(ranks)}")
    if [row["rank"] for row in ranks] != list(range(len(ranks))):
        raise ValueError("rank snapshots must be contiguous starting at zero")
    first = ranks[0]
    for row in ranks[1:]:
        _validate_rank_snapshot(row, first)
    layer_num, per_head = first["layer_num"], first["per_head"]
    if per_head:
        merged = []
        for layer in range(layer_num):
            keys, values = [], []
            for row in ranks:
                width = row["head_num"]
                keys.extend(row["abs_max"][layer][:width])
                values.extend(row["abs_max"][layer][width:])
            merged.append(keys + values)
    else:
        merged = [
            [
                max(float(row["abs_max"][layer][0]) for row in ranks),
                max(float(row["abs_max"][layer][1]) for row in ranks),
            ]
            for layer in range(layer_num)
        ]
    scales = [[float(value) / QMAX if float(value) > 0 else 1.0 for value in row] for row in merged]
    return {
        "quant_type": "per_head" if per_head else "per_tensor",
        "architectures": first["architecture"],
        "num_layers": layer_num,
        "num_target_layers": first["num_target_layers"],
        "num_draft_layers": first["num_draft_layers"],
        "num_head": sum(int(row["head_num"]) for row in ranks),
        "qmin": first["qmin"],
        "qmax": first["qmax"],
        "scales_shape": [layer_num, len(scales[0])],
        "scales": scales,
        "version": "1.0",
    }


def _post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = Request(url, data=json.dumps(payload).encode(), headers={"content-type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} {url}: {exc.read().decode(errors='replace')[:500]}") from exc
    except URLError as exc:
        raise RuntimeError(f"request {url} failed: {exc}") from exc


def _get(url: str, timeout: float) -> dict[str, Any]:
    with urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _pick_port(port: int = 0) -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
        return int(sock.getsockname()[1])


def _load_tokenizer(args):
    # Use the same helper as the server so trust_remote_code and an explicit
    # chat_template file are applied exactly as they are at request time.
    import lightllm.server.build_prompt as build_prompt
    from lightllm.utils.config_utils import get_model_architectures, get_vocab_size

    build_prompt.init_tokenizer(args)
    tokenizer = build_prompt.tokenizer
    model_vocab = int(get_vocab_size(args.model_dir))
    if model_vocab <= 0:
        raise ValueError("model config has no positive vocab_size")
    return tokenizer, model_vocab, get_model_architectures(args.model_dir)


def random_token_samples(
    tokenizer, *, num_samples: int, max_input_tokens: int, seed: int, model_vocab: int
) -> list[list[int]]:
    if num_samples <= 0 or max_input_tokens <= 0:
        raise ValueError("num_samples and max_input_tokens must be positive")
    specials = {int(token_id) for token_id in (getattr(tokenizer, "all_special_ids", []) or [])}
    vocab = tokenizer.get_vocab()
    candidates = sorted(
        {
            int(token_id)
            for token_id in vocab.values()
            if 0 <= int(token_id) < int(model_vocab) and int(token_id) not in specials
        }
    )
    if not candidates:
        raise ValueError("tokenizer has no non-special token IDs inside model vocab_size")
    rng = random.Random(seed)
    return [[rng.choice(candidates) for _ in range(rng.randint(1, max_input_tokens))] for _ in range(num_samples)]


def jsonl_token_samples(path: Path, tokenizer, *, max_input_tokens: int, required: int) -> list[list[int]]:
    samples: list[list[int]] = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no} must be an object")
            if "messages" in item:
                if not isinstance(item["messages"], list) or not item["messages"]:
                    raise ValueError(f"{path}:{line_no} messages must be a non-empty list")
                text = tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True)
            elif isinstance(item.get("prompt"), str) and item["prompt"].strip():
                text = item["prompt"]
            else:
                raise ValueError(f"{path}:{line_no} requires prompt or messages")
            token_ids = tokenizer.encode(text, add_special_tokens=False)[:max_input_tokens]
            if not token_ids:
                raise ValueError(f"{path}:{line_no} encoded to no tokens")
            samples.append(token_ids)
            if len(samples) == required:
                return samples
    raise ValueError(f"dataset has {len(samples)} usable rows, requires {required}; rows are not repeated")


def _parse() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--num_samples", type=int, default=128)
    parser.add_argument("--max_input_tokens", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--startup_timeout", type=float, default=600)
    parser.add_argument("--request_timeout", type=float, default=600)
    parser.add_argument("--drain_timeout", type=float, default=120)
    own, rest = parser.parse_known_args()
    if rest[:1] == ["--"]:
        rest = rest[1:]
    if "--export_fp8kv_calibration" in rest:
        raise ValueError("--export_fp8kv_calibration is internal and unavailable from the service CLI")
    return own, rest


def _service_args(service_argv: list[str], job_id: str):
    from lightllm.server.api_cli import add_cli_args
    from lightllm.server.core.objs import StartArgs

    parser = argparse.ArgumentParser(add_help=False)
    add_cli_args(parser)
    values = vars(parser.parse_args(service_argv))
    if values.get("nnodes", 1) != 1 or values.get("dp", 1) != 1 or values.get("run_mode", "normal") != "normal":
        raise ValueError("initial calibration supports only single-node dp=1 run_mode=normal")
    backends = values.get("llm_prefill_att_backend", [])
    backends = [backends] if isinstance(backends, str) else backends
    if any(backend == "auto" for backend in backends):
        raise ValueError("calibration requires explicit FA3 (per-head) or FlashInfer (per-tensor) prefill backend")
    per_head, per_tensor = "fa3" in backends, "flashinfer" in backends
    if per_head == per_tensor:
        raise ValueError("calibration requires exactly one supported prefill backend: FA3 or FlashInfer")
    expected = "fp8kv_sph" if per_head else "fp8kv_spt"
    requested = values.get("llm_kv_type", "None")
    if requested not in (None, "None", expected):
        raise ValueError(f"requested --llm_kv_type={requested} conflicts with {backends}; expected {expected}")
    values.update(
        {
            "export_fp8kv_calibration": True,
            "calibration_job_id": job_id,
            "enable_rl": False,
            "llm_kv_type": "None",  # collect unquantized KV after checking user granularity intent
            "disable_cudagraph": True,
            "enable_prefill_cudagraph": False,
            "disable_dynamic_prompt_cache": True,
            "enable_cpu_cache": False,
            "enable_disk_cache": False,
            "health_monitor": False,
            "httpserver_workers": 1,
            "enable_prefill_microbatch_overlap": False,
            "enable_decode_microbatch_overlap": False,
            "disable_vision": True,
            "disable_audio": True,
            "host": "127.0.0.1",
        }
    )
    explicit_port = any(token == "--port" or token.startswith("--port=") for token in service_argv)
    try:
        values["port"] = _pick_port(int(values["port"]) if explicit_port else 0)
    except OSError as exc:
        raise ValueError(f"requested --port {values['port']} is unavailable: {exc}") from exc
    if any(token == "--nccl_port" or token.startswith("--nccl_port=") for token in service_argv):
        if int(values["nccl_port"]) == int(values["port"]):
            raise ValueError("--nccl_port must differ from --port")
        try:
            _pick_port(int(values["nccl_port"]))
        except OSError as exc:
            raise ValueError(f"requested --nccl_port {values['nccl_port']} is unavailable: {exc}") from exc
    return StartArgs(**values)


def _spawn_service(args, state_file: Path, log_file: Path) -> subprocess.Popen:
    state_file.write_text(json.dumps(vars(args), default=str, sort_keys=True))
    code = (
        "import json; from lightllm.server.core.objs import StartArgs; "
        "from lightllm.server.api_server import launch_server; "
        f"launch_server(StartArgs(**json.load(open({str(state_file)!r}))))"
    )
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(_REPO_ROOT) if not existing_pythonpath else os.pathsep.join((str(_REPO_ROOT), existing_pythonpath))
    )
    with log_file.open("ab", buffering=0) as handle:
        return subprocess.Popen(
            [sys.executable, "-c", code], start_new_session=True, stdout=handle, stderr=subprocess.STDOUT, env=env
        )


def _terminate(proc: subprocess.Popen | None, timeout: float = 20) -> None:
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        pass
    # Always escalate for the owned PGID: the launcher may ignore TERM or
    # may have exited while a descendant remains alive.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # A reaped launcher says nothing about an orphan child; wait briefly
    # for the owned PGID to disappear after SIGKILL.
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)


def _wait_ready(proc: subprocess.Popen, base: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"calibration service exited during startup with code {proc.returncode}")
        try:
            _get(base + "/health", 2)
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError("calibration service startup timed out")


def _wait_idle(base: str, job_id: str, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        last = _post(base + "/_calibration/status", {"job_id": job_id}, min(10, timeout))
        ranks = last.get("ranks", [])
        if ranks and not last.get("http_pending", 0) and all(rank.get("idle") for rank in ranks):
            return last
        time.sleep(0.1)
    raise TimeoutError(f"calibration service did not drain before deadline: {last}")


def _generate_one(base: str, token_ids: list[int], own: argparse.Namespace) -> int:
    response = _post(
        base + "/generate",
        {
            "inputs": token_ids,
            "parameters": {
                "max_new_tokens": own.max_new_tokens,
                "do_sample": False,
                "ignore_eos": True,
                "return_details": False,
            },
        },
        own.request_timeout,
    )
    output_tokens = response.get("count_output_tokens")
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int) or output_tokens <= 0:
        raise RuntimeError("generation did not return a positive native count_output_tokens")
    if response.get("error") or response.get("finish_reason") in {"abort", "error"}:
        raise RuntimeError("generation returned an error/abort finish reason")
    return output_tokens


def _atomic_publish(
    output: Path, calibration: dict[str, Any], report: dict[str, Any], *, overwrite: bool, job_id: str
) -> None:
    report_path = output.with_name(output.name + ".report.json")
    if (output.exists() or report_path.exists()) and not overwrite:
        raise FileExistsError(f"output or report exists: {output}; pass --overwrite to replace both")
    output.parent.mkdir(parents=True, exist_ok=True)
    json_tmp = output.with_name(output.name + f".tmp-{job_id}")
    report_tmp = report_path.with_name(report_path.name + f".tmp-{job_id}")
    try:
        json_tmp.write_text(json.dumps(calibration, indent=2, sort_keys=True))
        report_tmp.write_text(json.dumps(report, indent=2, sort_keys=True))
        # Each file is atomically replaced; publishing JSON last avoids a success
        # JSON when report publication fails.
        os.replace(report_tmp, report_path)
        os.replace(json_tmp, output)
    finally:
        json_tmp.unlink(missing_ok=True)
        report_tmp.unlink(missing_ok=True)


def main() -> int:
    own, service_argv = _parse()
    for name in ("num_samples", "max_input_tokens", "max_new_tokens", "concurrency"):
        if isinstance(getattr(own, name), bool) or getattr(own, name) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in ("startup_timeout", "request_timeout", "drain_timeout"):
        if not _finite_number(getattr(own, name)) or getattr(own, name) <= 0:
            raise ValueError(f"{name} must be a finite positive timeout")
    job_id = uuid.uuid4().hex
    args = _service_args(service_argv, job_id)
    grain = "per_head" if "fa3" in args.llm_prefill_att_backend else "per_tensor"
    output = own.output or Path(f"kv_cache_calib_{grain}{'_with_draft' if args.mtp_mode else ''}.json")
    report_path = output.with_name(output.name + ".report.json")
    if (output.exists() or report_path.exists()) and not own.overwrite:
        raise FileExistsError(f"output or report exists: {output}; pass --overwrite to replace both")
    tokenizer, model_vocab, architectures = _load_tokenizer(args)
    samples = (
        jsonl_token_samples(own.dataset, tokenizer, max_input_tokens=own.max_input_tokens, required=own.num_samples)
        if own.dataset
        else random_token_samples(
            tokenizer,
            num_samples=own.num_samples,
            max_input_tokens=own.max_input_tokens,
            seed=own.seed,
            model_vocab=model_vocab,
        )
    )
    run_dir = output.parent / f".fp8kv-calibration-{job_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    state, log_file = run_dir / "start_args.json", run_dir / "service.log"
    print(f"calibration log={log_file}", flush=True)
    started, proc = time.time(), None
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def _exit_on_sigterm(signum, _frame):
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _exit_on_sigterm)
    try:
        proc = _spawn_service(args, state, log_file)
        base = f"http://127.0.0.1:{args.port}"
        _wait_ready(proc, base, own.startup_timeout)
        _wait_idle(base, job_id, own.drain_timeout)
        _post(base + "/_calibration/begin", {"job_id": job_id}, own.drain_timeout)
        output_tokens = 0
        pool = ThreadPoolExecutor(max_workers=own.concurrency, thread_name_prefix="fp8kv-calibration")
        futures = {pool.submit(_generate_one, base, ids, own): len(ids) for ids in samples}
        try:
            for index, future in enumerate(as_completed(futures), 1):
                prompt_tokens = futures[future]
                output_tokens += future.result()
                print(f"completed {index}/{len(samples)} prompt_tokens={prompt_tokens}", flush=True)
        except BaseException:
            # Do not let an executor context wait through every queued sample on
            # Ctrl-C or the first failure.  The outer finally tears down our PGID.
            for future in futures:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
        _wait_idle(base, job_id, own.drain_timeout)
        snapshot = _post(base + "/_calibration/snapshot", {"job_id": job_id}, own.drain_timeout)
        merged = merge_rank_snapshots(snapshot["ranks"], expected_ranks=args.tp)
        if merged["quant_type"] != grain:
            raise ValueError(f"snapshot grain {merged['quant_type']} differs from requested {grain}")
        if merged["architectures"] != architectures:
            raise ValueError("rank snapshot architecture disagrees with local model config")
        report = {
            "random_input": own.dataset is None,
            "seed": own.seed if own.dataset is None else None,
            "dataset": str(own.dataset) if own.dataset else None,
            "samples": len(samples),
            "input_tokens": sum(map(len, samples)),
            "output_tokens": output_tokens,
            "elapsed_s": time.time() - started,
            "service_args": {
                "tp": args.tp,
                "dp": args.dp,
                "mtp_mode": args.mtp_mode,
                "mtp_draft_model_dir": getattr(args, "mtp_draft_model_dir", None),
                "model_dir": args.model_dir,
                "max_new_tokens": own.max_new_tokens,
                "concurrency": own.concurrency,
            },
            "input_token_ids_sha256": hashlib.sha256(json.dumps(samples, separators=(",", ":")).encode()).hexdigest(),
            "job_id": job_id,
            "run_dir": str(run_dir),
            "rank_observations": [
                {"rank": row["rank"], "counts": row["counts"], "observed_token_rows": row["observed_token_rows"]}
                for row in snapshot["ranks"]
            ],
        }
        _atomic_publish(output, merged, report, overwrite=own.overwrite, job_id=job_id)
        print(f"exported {output} report={report_path} elapsed_s={report['elapsed_s']:.2f}", flush=True)
    finally:
        _terminate(proc)
        signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
