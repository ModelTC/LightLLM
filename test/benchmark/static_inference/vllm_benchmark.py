import json
import os
import re
import shutil
import socket
import time
from pathlib import Path

import numpy as np
import torch

from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig


batch_size = int(os.environ["BS"])
input_len = int(os.environ["ILEN"])
output_len = int(os.environ["OLEN"])
warmup_output_len = int(os.environ["WARMUP_OLEN"])
tp = int(os.environ["TP"])
bench_mode = os.environ["BENCH_MODE"]
cache_hit_len = int(os.environ["CACHE_HIT_LEN"])
model_dir = os.environ["MODEL_DIR"]
profile_dir = os.environ["PROFILE_DIR"]
profile_stage = os.environ.get("PROFILE_STAGE", "full")

assert bench_mode in {"static_forward", "cache_hit", "both"}, bench_mode
assert 0 <= cache_hit_len <= input_len, f"CACHE_HIT_LEN must be in [0, {input_len}], got {cache_hit_len}"
assert profile_stage in {"prefill", "decode", "full"}, profile_stage


def build_profiler_config() -> ProfilerConfig:
    # The profile window is driven explicitly by start_profile / stop_profile
    # (wrapped with torch.cuda.synchronize() to mirror lightllm's
    # `torch.cuda.synchronize(); with profile(...): fn(); synchronize()`
    # pattern). Do NOT set max_iterations here: vLLM's internal iteration
    # accounting can cut the window off mid-step (observed on cache-hit prefill:
    # ~140ms -> ~12ms), giving misleading kernel totals.
    return ProfilerConfig(
        profiler="torch",
        torch_profiler_dir=profile_dir,
        delay_iterations=0,
        max_iterations=0,
        ignore_frontend=True,
    )


def build_llm() -> LLM:
    # async_scheduling=False is critical: the V1 AsyncScheduler overlaps the
    # next iteration's forward with the current step's output handling, which
    # causes `engine.step()` to capture two forward passes (e.g. 188 attention
    # kernels instead of 94 for a 94-layer model). Forcing the sync scheduler
    # aligns the decode test cadence with lightllm, which profiles exactly one
    # isolated decode forward.
    #
    # enable_chunked_prefill=False is equally critical for prefill alignment:
    # with chunked prefill enabled, V1 can split a cold batch of
    # batch_size * input_len tokens across multiple scheduler steps (and often
    # only admits a single waiting request per step). Our profile window only
    # covers step_index == 0, so a chunked prefill would measure a tiny slice
    # of the true cold-prefill work — making static_forward prefill look as
    # cheap as cache_hit prefill. Disabling chunked prefill forces V1 to run
    # the entire `batch_size * input_len` token batch in one engine.step(),
    # which is the whole prefill (matches lightllm's single-shot prefill).
    # Safe because max_num_batched_tokens is sized to exactly fit the batch.
    return LLM(
        model=model_dir,
        skip_tokenizer_init=True,
        tensor_parallel_size=tp,
        dtype="bfloat16",
        max_model_len=32768,
        max_num_seqs=batch_size,
        max_num_batched_tokens=batch_size * input_len,
        enable_prefix_caching=True,
        enable_chunked_prefill=False,
        enforce_eager=False,
        disable_log_stats=False,
        async_scheduling=False,
        profiler_config=build_profiler_config(),
    )


def make_token_ids(req_index: int, total_len: int, suffix_seed: int = 0) -> list[int]:
    base = 1024 + req_index * (total_len + 17) + suffix_seed * 7919
    return [128 + ((base + i) % 20000) for i in range(total_len)]


def build_prompt_batches() -> tuple[list[dict], list[dict], list[dict]]:
    warmup_prompts = []
    cold_prompts = []
    hot_prompts = []

    for req_index in range(batch_size):
        cold_ids = make_token_ids(req_index, input_len)
        warmup_ids = make_token_ids(req_index, input_len, suffix_seed=999)
        if cache_hit_len == input_len:
            hot_ids = list(cold_ids)
        else:
            hot_suffix = make_token_ids(req_index, input_len - cache_hit_len, suffix_seed=31337)
            hot_ids = cold_ids[:cache_hit_len] + hot_suffix

        warmup_prompts.append({"prompt_token_ids": warmup_ids})
        cold_prompts.append({"prompt_token_ids": cold_ids})
        hot_prompts.append({"prompt_token_ids": hot_ids})

    return warmup_prompts, cold_prompts, hot_prompts


def percentile_dict(values: list[float]) -> dict[str, float]:
    percentiles = [50, 90, 99]
    return {f"P{p}": round(float(np.percentile(values, p)), 3) for p in percentiles}


def extract_latency_metrics(output) -> tuple[float, float, float, int, int, float | None]:
    metrics = output.metrics
    if metrics is None:
        raise RuntimeError("vLLM output metrics are missing")

    num_generation_tokens = int(getattr(metrics, "num_generation_tokens", 0))
    num_cached_tokens = int(
        getattr(output, "num_cached_tokens", None)
        if getattr(output, "num_cached_tokens", None) is not None
        else getattr(metrics, "num_cached_tokens", 0)
    )

    if hasattr(metrics, "prefill_time") and hasattr(metrics, "decode_time"):
        prefill_time = float(metrics.prefill_time)
        decode_time = float(metrics.decode_time)
        e2e_latency = float(getattr(metrics, "e2e_latency", prefill_time + decode_time))
        tpot = (
            float(metrics.mean_time_per_output_token)
            if num_generation_tokens > 1 and hasattr(metrics, "mean_time_per_output_token")
            else None
        )
        return prefill_time, decode_time, e2e_latency, num_generation_tokens, num_cached_tokens, tpot

    required_fields = ["scheduled_ts", "first_token_ts", "last_token_ts", "first_token_latency"]
    missing_fields = [field for field in required_fields if getattr(metrics, field, None) is None]
    if missing_fields:
        raise RuntimeError("vLLM output metrics do not contain enough timing fields: " + ", ".join(missing_fields))

    prefill_time = max(0.0, float(metrics.first_token_ts) - float(metrics.scheduled_ts))
    decode_time = max(0.0, float(metrics.last_token_ts) - float(metrics.first_token_ts))
    e2e_latency = max(0.0, float(metrics.first_token_latency) + decode_time)
    tpot = decode_time / (num_generation_tokens - 1) if num_generation_tokens > 1 else None
    return prefill_time, decode_time, e2e_latency, num_generation_tokens, num_cached_tokens, tpot


def summarize_outputs(label: str, outputs) -> dict:
    prefill_ms = []
    decode_ms = []
    tpot_ms = []
    e2e_ms = []
    cached_tokens = []
    generation_tokens = []

    for output in outputs:
        (
            prefill_time,
            decode_time,
            e2e_latency,
            num_generation_tokens,
            num_cached_tokens,
            tpot,
        ) = extract_latency_metrics(output)

        prefill_ms.append(prefill_time * 1000)
        decode_ms.append(decode_time * 1000)
        e2e_ms.append(e2e_latency * 1000)
        generation_tokens.append(num_generation_tokens)
        cached_tokens.append(num_cached_tokens)

        if tpot is not None:
            tpot_ms.append(tpot * 1000)

    summary = {
        "request_count": len(outputs),
        "input_len": input_len,
        "output_len": output_len,
        "requested_cache_hit_len": cache_hit_len,
        "avg_generation_tokens": round(float(np.mean(generation_tokens)), 3),
        "avg_cached_tokens": round(float(np.mean(cached_tokens)), 3),
        "min_cached_tokens": int(min(cached_tokens)),
        "max_cached_tokens": int(max(cached_tokens)),
        "prefill_latency_ms": {
            "mean": round(float(np.mean(prefill_ms)), 3),
            **percentile_dict(prefill_ms),
        },
        "decode_latency_ms": {
            "mean": round(float(np.mean(decode_ms)), 3),
            **percentile_dict(decode_ms),
        },
        "e2e_latency_ms": {
            "mean": round(float(np.mean(e2e_ms)), 3),
            **percentile_dict(e2e_ms),
        },
        "decode_tpot_ms": {
            "mean": round(float(np.mean(tpot_ms)), 3) if tpot_ms else 0.0,
            **(percentile_dict(tpot_ms) if tpot_ms else {}),
        },
    }
    print(json.dumps({label: summary}, indent=2, ensure_ascii=False))
    return summary


def snapshot_profile_files() -> dict[str, tuple[int, int]]:
    snapshots = {}
    root = Path(profile_dir)
    for path in root.iterdir():
        if path.is_dir():
            continue
        stat = path.stat()
        snapshots[path.name] = (stat.st_size, stat.st_mtime_ns)
    return snapshots


def get_profile_rank(file_name: str) -> int | None:
    # vLLM V1 trace filenames look like
    # `{prefix}_dp0_pp0_tp<rank>_dcp0_...pt.trace.json`. Match tp<rank> first
    # so we land on the tensor-parallel rank, which is what lightllm's layout
    # (`forward_{stage}_<rank>/`) keys off of.
    tp_match = re.search(r"tp(\d+)", file_name)
    if tp_match is not None:
        return int(tp_match.group(1))

    rank_match = re.search(r"rank(\d+)", file_name)
    if rank_match is not None:
        return int(rank_match.group(1))

    profiler_match = re.search(r"profiler_out_(\d+)", file_name)
    if profiler_match is not None:
        return int(profiler_match.group(1))

    return None


def _normalized_hostname() -> str:
    # Mirror torch.profiler.tensorboard_trace_handler's default worker_name,
    # which uses socket.gethostname() with dots replaced by dashes (since '.'
    # is the field separator in `<worker>.<ts>.pt.trace.json`).
    return socket.gethostname().replace(".", "-")


def _normalized_trace_name(rank: int | None, original_name: str) -> str:
    # Produce `{hostname}_{rank}.{ns}.pt.trace.json` to match lightllm's
    # `{hostname}_{pid}.{ns}.pt.trace.json`. We don't have vLLM worker PIDs,
    # so the TP rank is used as the per-worker identifier instead. This keeps
    # the tensorboard "Workers" dropdown values consistent with lightllm's
    # style (hostname-prefixed) rather than showing vLLM's internal profile
    # prefix (e.g. `cache_hit_prefill_dp0_pp0_tp0_dcp`).
    worker_id = "misc" if rank is None else str(rank)
    ts_ns = time.monotonic_ns()
    host = _normalized_hostname()
    suffix = ".pt.trace.json"
    if not original_name.endswith(suffix):
        # Fall back to the original extension if it's not a standard trace
        # (e.g. memory timeline html). Still apply the worker prefix so
        # downstream tooling keys off the same worker id.
        stem = Path(original_name).name
        return f"{host}_{worker_id}.{ts_ns}_{stem}"
    return f"{host}_{worker_id}.{ts_ns}{suffix}"


def organize_profile_artifacts(stage_name: str, before_snapshot: dict[str, tuple[int, int]]) -> list[str]:
    stage_dirs: set[str] = set()

    moved_count = 0
    root = Path(profile_dir)
    for path in root.iterdir():
        if path.is_dir():
            continue

        stat = path.stat()
        current_snapshot = (stat.st_size, stat.st_mtime_ns)
        if before_snapshot.get(path.name) == current_snapshot:
            continue

        rank = get_profile_rank(path.name)
        rank_suffix = "misc" if rank is None else str(rank)
        stage_dir = os.path.join(profile_dir, f"forward_{stage_name}_{rank_suffix}")
        os.makedirs(stage_dir, exist_ok=True)
        target_name = _normalized_trace_name(rank, path.name)
        shutil.move(str(path), os.path.join(stage_dir, target_name))
        stage_dirs.add(stage_dir)
        moved_count += 1

    if moved_count == 0:
        raise RuntimeError(f"No profiling artifacts were produced for stage: {stage_name}")

    return sorted(stage_dirs)


def supports_manual_step_profiling(llm: LLM) -> bool:
    return all(
        hasattr(llm, attr) for attr in ("_params_to_seq", "_preprocess_cmpl", "_render_and_add_requests", "llm_engine")
    ) and all(hasattr(llm.llm_engine, attr) for attr in ("step", "has_unfinished_requests"))


def generate_with_split_profile(
    llm: LLM,
    prompts: list[dict],
    sampling_params: SamplingParams,
    *,
    prefill_profile_prefix: str,
    decode_profile_prefix: str,
) -> list:
    if not supports_manual_step_profiling(llm):
        raise RuntimeError(
            "Installed vLLM does not expose the internal step API required for split prefill/decode profiling "
            "from a single model load."
        )

    engine_inputs = llm._preprocess_cmpl(prompts)
    params_seq = llm._params_to_seq(sampling_params, len(engine_inputs))
    llm._render_and_add_requests(prompts=engine_inputs, params=params_seq)
    finished_outputs = []

    profiled_stage: str | None = None
    profile_snapshot: dict[str, tuple[int, int]] | None = None
    decode_profile_step = output_len - 1 if output_len > 1 else None
    step_index = 0

    while llm.llm_engine.has_unfinished_requests():
        if step_index == 0:
            profiled_stage = "prefill"
            profile_snapshot = snapshot_profile_files()
            print(f"Profile Prefill -> {os.path.join(profile_dir, 'forward_prefill_*')}")
            # Sync before starting the profiler so prior-step kernels do not
            # leak into the profile window (matches lightllm's
            # torch.cuda.synchronize() before `with profile(...)`).
            torch.cuda.synchronize()
            llm.start_profile(profile_prefix=prefill_profile_prefix)
        elif decode_profile_step is not None and step_index == decode_profile_step:
            profiled_stage = "decode"
            profile_snapshot = snapshot_profile_files()
            print(f"Profile Decode -> {os.path.join(profile_dir, 'forward_decode_*')}")
            torch.cuda.synchronize()
            llm.start_profile(profile_prefix=decode_profile_prefix)

        step_outputs = llm.llm_engine.step()

        if profiled_stage is not None:
            # Ensure the profile window contains exactly this step's GPU
            # kernels before stopping, mirroring lightllm's
            # torch.cuda.synchronize(); prof.step() inside the context.
            torch.cuda.synchronize()
            llm.stop_profile()
            if profiled_stage == "prefill":
                organize_profile_artifacts("prefill", profile_snapshot or {})
            else:
                organize_profile_artifacts("decode", profile_snapshot or {})
            profiled_stage = None
            profile_snapshot = None

        for output in step_outputs:
            if getattr(output, "finished", False):
                finished_outputs.append(output)

        step_index += 1

    if len(finished_outputs) != len(engine_inputs):
        raise RuntimeError(f"Expected {len(engine_inputs)} finished outputs, got {len(finished_outputs)}")

    return finished_outputs


def generate_once(
    llm: LLM,
    prompts: list[dict],
    sampling_params: SamplingParams,
    *,
    do_profile: bool,
    profile_prefix: str,
):
    if do_profile:
        torch.cuda.synchronize()
        llm.start_profile(profile_prefix=profile_prefix)
    try:
        return llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    finally:
        if do_profile:
            torch.cuda.synchronize()
            llm.stop_profile()


def main():
    os.makedirs(profile_dir, exist_ok=True)
    llm = build_llm()

    warmup_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=warmup_output_len,
        detokenize=False,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=output_len,
        detokenize=False,
    )

    warmup_prompts, cold_prompts, hot_prompts = build_prompt_batches()

    print("Warm up request path with a separate prompt batch...")
    generate_once(llm, warmup_prompts, warmup_sampling_params, do_profile=False, profile_prefix="warmup_forward")

    reset_ok = llm.reset_prefix_cache(reset_running_requests=True)
    print(f"Prefix cache reset before measurement: {reset_ok}")
    if not reset_ok:
        raise RuntimeError("Failed to reset vLLM prefix cache before measurement")

    results = {}

    if bench_mode in {"static_forward", "both"}:
        print(f"Profiling cold prefill+decode into {profile_dir} ...")
        # Use the same profile prefixes as cache_hit so that intermediate file
        # names and rank parsing are identical across bench modes; outputs are
        # then renamed to `{hostname}_{rank}.{ns}.pt.trace.json` to match
        # lightllm's layout exactly.
        cold_outputs = generate_with_split_profile(
            llm,
            cold_prompts,
            sampling_params,
            prefill_profile_prefix="forward_prefill",
            decode_profile_prefix="forward_decode",
        )
        results["cold_forward"] = summarize_outputs("cold_forward", cold_outputs)

    if bench_mode == "cache_hit":
        print("Seed prefix cache with a first-time request batch before hot run...")
        generate_once(llm, cold_prompts, sampling_params, do_profile=False, profile_prefix="seed_forward")

    if bench_mode in {"cache_hit", "both"}:
        print(f"Profiling cache-hit prefill+decode into {profile_dir} ...")
        hot_outputs = generate_with_split_profile(
            llm,
            hot_prompts,
            sampling_params,
            prefill_profile_prefix="forward_prefill",
            decode_profile_prefix="forward_decode",
        )
        results["cache_hit_forward"] = summarize_outputs("cache_hit_forward", hot_outputs)

    print(json.dumps({"profile_dir": profile_dir, "results": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
