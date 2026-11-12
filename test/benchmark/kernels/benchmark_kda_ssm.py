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
from lightllm.common.basemodel.attention.linear.replay_kda_decode import get_configs, replay
from unit_tests.common.basemodel.triton_kernel.linear_att.replay_history import (
    accept_replay_window,
    attach_replay_history as ReplaySSMExecutor,
)


def workload(args, batch, width, dtype, mode, acceptance):
    layers, heads, dim = args.layers, args.heads, 128
    packed = torch.randn(batch * width, heads * dim * 3, dtype=torch.bfloat16, device="cuda")
    q, k, v = [x.view(batch * width, heads, dim) for x in packed.split(heads * dim, -1)]
    native_shape = (1, batch * width) if width > 1 else (batch, 1)
    gate = torch.randn(batch * width, heads * dim, dtype=q.dtype, device="cuda") - 3
    beta = torch.randn(batch * width, heads, dtype=q.dtype, device="cuda")
    alog = torch.zeros(heads, device="cuda")
    bias = torch.zeros(heads * dim, device="cuda")
    reqs = torch.arange(batch, dtype=torch.int32, device="cuda")
    # 原生 decode 在宽度为 1 时不收前缀和。replay 始终要，一行一个 token 时就是 0, 1, 2, ...。
    replay_cu = torch.arange(batch + 1, dtype=torch.int32, device="cuda") * width
    cu = replay_cu if width > 1 else None
    counts = reqs % width + 1 if acceptance == "partial" else torch.full_like(reqs, width)
    accepted = torch.cat((counts - 1, counts.new_zeros(1)))
    snapshot = torch.empty(layers, heads, dim, dim, dtype=dtype, device="cuda")
    slots = (batch + 1) * (width if mode == "native" else 1)
    state = torch.randn(layers, slots, heads, dim, dim, dtype=dtype, device="cuda") * 0.01
    ssm_backend = None
    if mode != "native":
        capacity = max(8, triton.next_power_of_2(width))
        ssm_backend = ReplaySSMExecutor(state, capacity, width, q.dtype, kda=True, compute_mode=mode)
    indexes = reqs[:, None] * width + torch.arange(width, device="cuda", dtype=torch.int32)
    if width == 1:
        indexes = reqs
    bytes_used = (
        sum(x.numel() * x.element_size() for x in vars(ssm_backend).values() if isinstance(x, torch.Tensor))
        if ssm_backend
        else state.numel() * state.element_size()
    )

    native_index = 0 if acceptance == "partial" else width - 1

    def cycle():
        for _ in range(args.rounds):
            for layer in range(layers):
                if ssm_backend is None:
                    fused_recurrent_kda(
                        q.view(*native_shape, heads, dim),
                        k.view(*native_shape, heads, dim),
                        v.view(*native_shape, heads, dim),
                        gate.view(*native_shape, -1),
                        beta.view(*native_shape, heads),
                        alog,
                        bias,
                        state[layer],
                        indexes,
                        cu_seqlens=cu,
                        num_accepted_tokens=counts if cu is not None else None,
                    )
                else:
                    replay(
                        ssm_backend,
                        layer,
                        q,
                        k,
                        v,
                        gate,
                        beta,
                        alog,
                        bias,
                        reqs,
                        replay_cu,
                        run_config=ssm_backend.run_config,
                    )
            if ssm_backend is not None:
                accept_replay_window(ssm_backend, reqs, accepted, replay_cu)
        if isinstance(ssm_backend, ReplaySSMExecutor):
            ssm_backend._materialize_accepted_state(reqs[:1], snapshot, snapshot=True)
            ssm_backend.fold_accepted_prefix(reqs)
        else:
            snapshot.copy_(state[:, native_index] if ssm_backend is None else state[:, 0])

    return ssm_backend, cycle, bytes_used


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
                    for mode in ["native", "inline", "precompute"] if width > 1 else ["native", "inline"]:
                        ssm_backend, cycle, size = workload(
                            args, batch, width, getattr(torch, dtype_name), mode, acceptance
                        )
                        candidates = []
                        if ssm_backend is not None:
                            for config in get_configs() if args.tune else [{"BV": 32, "num_warps": 2, "num_stages": 1}]:
                                if mode == "precompute" and config["BV"] < 16:
                                    continue
                                ssm_backend.run_config = config
                                cycle()
                                ms = triton.testing.do_bench_cudagraph(cycle, rep=50)
                                candidates.append(dict(config=config, ms=ms))
                            ssm_backend.run_config = min(candidates, key=lambda r: r["ms"])["config"]
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
                            cache_len=getattr(ssm_backend, "capacity", None),
                            state_history_bytes=size,
                            ms_per_round=statistics.median(samples),
                            samples_ms=samples,
                            config=dict(ssm_backend.run_config) if ssm_backend and ssm_backend.run_config else None,
                            candidates=candidates,
                        )
                        rows.append(row)
                        args.output.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps(row), flush=True)
                        del ssm_backend, cycle
                        gc.collect()
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
