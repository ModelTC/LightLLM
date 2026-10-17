"""Measure a full KDA state cycle: all layers, acceptance, folds and export.

This synthetic kernel benchmark excludes convolutions, model projections and
HTTP scheduling. Use a matched service workload for end-to-end conclusions.
"""

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
import triton

from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.kda_decode import fused_recurrent_kda
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache
from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_compact import CompactSSMCache
from lightllm.common.basemodel.triton_kernel.linear_att.ssm_autotune import get_configs


def workload(args, batch, width, dtype, mode, acceptance):
    layers, heads, dim = args.layers, args.heads, 128
    shape = (1, batch * width, heads, dim) if width > 1 else (batch, 1, heads, dim)
    packed = torch.randn(batch * width, heads * dim * 3, dtype=torch.bfloat16, device="cuda")
    q, k, v = [x.view(shape) for x in packed.split(heads * dim, -1)]
    gate = torch.randn(batch * width, heads * dim, dtype=q.dtype, device="cuda") - 3
    beta = torch.randn(batch * width, heads, dtype=q.dtype, device="cuda")
    alog = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads * dim, device="cuda")
    reqs = torch.arange(batch, dtype=torch.int32, device="cuda")
    cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * width if width > 1 else None
    counts = reqs % width + 1 if acceptance == "partial" else torch.full_like(reqs, width)
    accepted = torch.cat((counts - 1, counts.new_zeros(1)))
    snapshot = torch.empty(layers, heads, dim, dim, dtype=dtype, device="cuda")
    slots = (batch + 1) * (width if mode == "native" else 1)
    state = torch.randn(layers, slots, heads, dim, dim, dtype=dtype, device="cuda") * 0.01
    cache = None
    if mode == "compact":
        cache = CompactSSMCache(state, width, q.dtype, kda=True)
    elif mode != "native":
        cache = ReplaySSMCache(state, 8, width, q.dtype, kda=True, projection_mode=mode)
    if cache is not None:
        cache._config_is_fixed = True
    indexes = reqs[:, None] * width + torch.arange(width, device="cuda", dtype=torch.int32)
    if width == 1:
        indexes = reqs
    bytes_used = (
        sum(x.numel() * x.element_size() for x in vars(cache).values() if isinstance(x, torch.Tensor))
        if cache
        else state.numel() * state.element_size()
    )

    native_index = 0 if acceptance == "partial" else width - 1

    def cycle():
        for _ in range(args.rounds):
            if cache is not None:
                cache.prepare_decode(reqs, cu)
            for layer in range(layers):
                if cache is None:
                    fused_recurrent_kda(
                        q,
                        k,
                        v,
                        gate.view(*shape[:2], -1),
                        beta.view(*shape[:2], heads),
                        alog,
                        bias,
                        state[layer],
                        indexes,
                        cu_seqlens=cu,
                        num_accepted_tokens=counts if cu is not None else None,
                    )
                else:
                    cache.forward(layer, q, k, v, gate, beta, alog, bias, reqs, cu)
            if cache is not None:
                cache.accept_updates(reqs, accepted)
        if isinstance(cache, ReplaySSMCache):
            cache._materialize_accepted_state(reqs[:1], snapshot, snapshot=True)
            cache.merge_accepted_updates(reqs)
        else:
            snapshot.copy_(state[:, native_index] if cache is None else state[:, 0])

    return cache, cycle, bytes_used


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, nargs="+", default=[16, 64, 192])
    p.add_argument("--width", type=int, nargs="+", default=[1, 3])
    p.add_argument("--dtype", nargs="+", choices=["float32", "bfloat16"], default=["float32", "bfloat16"])
    p.add_argument("--layers", type=int, default=34)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--tune", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(53)
    rows = []
    report = dict(
        settings={**vars(args), "output": str(args.output)},
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        triton=triton.__version__,
        results=rows,
    )
    for dtype_name in args.dtype:
        for batch in args.batch:
            for width in args.width:
                for acceptance in ["full", "partial"] if width > 1 else ["full"]:
                    for mode in ["native", "compact", "inline", "precompute"] if width > 1 else ["native", "inline"]:
                        cache, cycle, size = workload(args, batch, width, getattr(torch, dtype_name), mode, acceptance)
                        candidates = []
                        if cache is not None:
                            for config in get_configs() if args.tune else [{"BV": 32, "num_warps": 2, "num_stages": 1}]:
                                cache.run_config = config
                                cycle()
                                ms = triton.testing.do_bench_cudagraph(cycle, rep=50)
                                candidates.append(dict(config=config, ms=ms))
                            cache.run_config = min(candidates, key=lambda r: r["ms"])["config"]
                        cycle()
                        samples = [
                            triton.testing.do_bench_cudagraph(cycle, rep=100) / args.rounds for _ in range(args.repeats)
                        ]
                        row = dict(
                            batch=batch,
                            width=width,
                            dtype=dtype_name,
                            acceptance=acceptance,
                            mode=mode,
                            state_history_bytes=size,
                            ms_per_round=statistics.median(samples),
                            samples_ms=samples,
                            config=cache.run_config if cache else None,
                            candidates=candidates,
                        )
                        rows.append(row)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps(row), flush=True)
                        del cache, cycle
                        gc.collect()
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
