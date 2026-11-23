"""Time the expanded Prefill routed expert path on one idle Hopper GPU.

From the candidate checkout, run with CUDA_VISIBLE_DEVICES set to one GPU:
    PYTHONPATH=. python test/benchmark/kernel/benchmark_glm53_prefill_expert_sms.py --model-dir /path/to/checkpoint
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
from statistics import median
from unittest.mock import patch


EXPERTS, HIDDEN, INTERMEDIATE, BLOCK = 18, 4096, 2048, 128
CASES = [(4096, 6144), (12540, 16896)]


def load_weights(torch, checkpoint):
    from safetensors import safe_open

    mapping = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    suffix = "layers.3.mlp.experts.0.gate_proj.weight"
    (prefix,) = [name[: -len(suffix)] for name in mapping if name.endswith(suffix)]
    w13 = torch.empty((EXPERTS, 2 * INTERMEDIATE, HIDDEN), dtype=torch.float8_e4m3fn, device="cuda")
    w2 = torch.empty((EXPERTS, HIDDEN, INTERMEDIATE), dtype=torch.float8_e4m3fn, device="cuda")
    s13 = torch.empty((EXPERTS, 2 * INTERMEDIATE // BLOCK, HIDDEN // BLOCK), dtype=torch.float32, device="cuda")
    s2 = torch.empty((EXPERTS, HIDDEN // BLOCK, INTERMEDIATE // BLOCK), dtype=torch.float32, device="cuda")
    slots = []
    for expert in range(EXPERTS):
        name = f"{prefix}layers.3.mlp.experts.{expert}"
        gate, up = w13[expert].chunk(2)
        gate_scale, up_scale = s13[expert].chunk(2)
        for proj, weight, scale in (
            ("gate_proj", gate, gate_scale),
            ("up_proj", up, up_scale),
            ("down_proj", w2[expert], s2[expert]),
        ):
            slots.extend(((f"{name}.{proj}.weight", weight), (f"{name}.{proj}.weight_scale_inv", scale)))
    with ExitStack() as stack:
        files = {
            shard: stack.enter_context(safe_open(checkpoint / shard, framework="pt", device="cpu"))
            for shard in {mapping[key] for key, _ in slots}
        }
        for key, dest in slots:
            value = files[mapping[key]].get_tensor(key)
            assert value.shape == dest.shape
            assert value.dtype == dest.dtype or (
                dest.dtype == torch.float32 and value.dtype == getattr(torch, "float8_e8m0fnu", None)
            )
            dest.copy_(value.to(dest.dtype))
    return w13, s13, w2, s2


def fixture(torch, dense, expanded):
    rng = random.Random(368000 + dense)
    blocks = [i % EXPERTS for i in range(expanded // BLOCK)]
    blocks[: len(blocks) // 7] = [0] * (len(blocks) // 7)
    sizes = [blocks.count(expert) * BLOCK for expert in range(EXPERTS)]
    counts = [size - (expert * 13 + 17) % 74 if size else 0 for expert, size in enumerate(sizes)]
    slots = []
    start = 0
    for expert, (size, count) in enumerate(zip(sizes, counts, strict=True)):
        slots.extend((expert, index) for index in range(start, start + count))
        start += size
    rng.shuffle(slots)
    rows = [[route] for route in slots[:dense]]
    row = rng.randrange(dense)
    for expert, index in slots[dense:]:
        while len(rows[row]) == 8 or any(item[0] == expert for item in rows[row]):
            row = (row + 1) % dense
        rows[row].append((expert, index))
        row = (row + 1) % dense
    metadata = [
        [row % 16, row // 16] + [item[1] for item in routes] + [-1] * (8 - len(routes))
        for row, routes in enumerate(rows)
    ]
    return (
        sizes,
        torch.tensor(counts, dtype=torch.int32, device="cuda"),
        torch.tensor(metadata, dtype=torch.int32, device="cuda"),
    )


def benchmark_case(torch, kernels, weights, dense, expanded, repeats, samples, use_ue8m0):
    sizes, counts, original = fixture(torch, dense, expanded)
    metadata = original.clone()
    x = torch.randn(
        (expanded, HIDDEN),
        dtype=torch.bfloat16,
        device="cuda",
        generator=torch.Generator(device="cuda").manual_seed(36817 + dense),
    )
    recv_x = kernels.per_token_group_quant_fp8(
        x, BLOCK, column_major_scales=True, scale_tma_aligned=True, use_ue8m0_scales=use_ue8m0
    )
    del x
    route_weights = (torch.arange(expanded, dtype=torch.float32, device="cuda") % 29 + 1) / 232

    def forward(enable):
        metadata.copy_(original)
        return kernels.chunked_expanded_moe_forward(
            sizes,
            counts,
            recv_x,
            route_weights,
            metadata,
            *weights,
            BLOCK,
            torch.bfloat16,
            alpha=1.0,
            limit=10.0,
            clamp_up_add_one=False,
            allow_full_sms=enable,
        )

    baseline = forward(False)
    expected_metadata = metadata.clone()
    real_gemm = kernels.deepgemm_grouped_fp8_nt_contiguous
    sm_trace = []

    def trace(*args, **kwargs):
        sm_trace.append(kernels.deep_gemm.get_num_sms())
        return real_gemm(*args, **kwargs)

    with patch.object(kernels, "deepgemm_grouped_fp8_nt_contiguous", trace):
        candidate = forward(True)
    assert sm_trace == [132, 132]
    assert kernels.deep_gemm.get_num_sms() == 120
    assert torch.isfinite(baseline).all().item()
    assert torch.equal(baseline.view(torch.int16), candidate.view(torch.int16))
    assert torch.equal(metadata, expected_metadata)
    del baseline, candidate

    def sample(enable):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            forward(enable)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / repeats

    for _ in range(4):
        forward(False)
        forward(True)
    torch.cuda.synchronize()
    pairs = []
    for i in range(samples):
        if i % 2:
            candidate_ms = sample(True)
            baseline_ms = sample(False)
        else:
            baseline_ms = sample(False)
            candidate_ms = sample(True)
        pairs.append((baseline_ms, candidate_ms))
    assert kernels.deep_gemm.get_num_sms() == 120
    return dict(
        dense_rows=dense,
        expanded_rows=expanded,
        baseline_ms=median(a for a, _ in pairs),
        candidate_ms=median(b for _, b in pairs),
        median_speedup=median(a / b for a, b in pairs),
        pairs_ms=pairs,
        gemm_sm_trace=sm_trace,
        bitwise_parity=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()
    assert args.repeats > 0 and args.samples > 0
    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(dict(model_dir=str(args.model_dir), run_mode="prefill"))

    import torch
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import deepgemm as kernels

    assert torch.cuda.get_device_properties(0).multi_processor_count == 132
    use_ue8m0 = kernels.get_env_use_ue8m0_scales()
    dg = kernels.deep_gemm
    previous_sms = dg.get_num_sms()
    previous_alignment = kernels.get_mk_alignment_for_contiguous_layout()
    kernels.set_mk_alignment_for_contiguous_layout(BLOCK)
    dg.set_num_sms(120)
    try:
        with torch.inference_mode():
            weights = load_weights(torch, args.model_dir)
            for dense, expanded in CASES:
                result = benchmark_case(torch, kernels, weights, dense, expanded, args.repeats, args.samples, use_ue8m0)
                result["ue8m0_scales"] = use_ue8m0
                result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() // (1 << 20)
                print(json.dumps(result), flush=True)
    finally:
        dg.set_num_sms(previous_sms)
        kernels.set_mk_alignment_for_contiguous_layout(previous_alignment)


if __name__ == "__main__":
    main()
