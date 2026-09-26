"""Run with PYTHONPATH=. exp -m 'ReplaySSM kernel comparison' python test/benchmark/kernels/benchmark_replayssm.py."""
import argparse
import json

import torch
import triton

from lightllm.common.basemodel.triton_kernel.linear_att.replayssm import ReplaySSMCache
from lightllm.common.basemodel.triton_kernel.linear_att.mtp_fused_recurrent import (
    mtp_fused_recurrent_gated_delta_rule,
)
from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops.fused_recurrent import (
    fused_recurrent_gated_delta_rule,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64, 256])
    parser.add_argument("--widths", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--state-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--bv", type=int, default=64)
    parser.add_argument("--warps", type=int, default=2)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--accepted-tokens", type=int)
    args = parser.parse_args()
    torch.manual_seed(1)
    h, hv, kd, vd = args.heads, args.heads * 2 if args.value_heads is None else args.value_heads, 128, 128
    assert h > 0 and hv > 0 and hv % h == 0 and args.layers > 0
    for width in args.widths:
        accepted_tokens = width if args.accepted_tokens is None else args.accepted_tokens
        assert 1 <= accepted_tokens <= width
        for batch in args.batches:
            q = torch.randn(1, batch * width, h, kd, dtype=torch.bfloat16, device="cuda")
            k = torch.randn_like(q)
            v = torch.randn(1, batch * width, hv, vd, dtype=torch.bfloat16, device="cuda")
            a = torch.randn(batch * width, hv, device="cuda", dtype=torch.bfloat16) - 3
            b = torch.randn_like(a)
            alog = torch.zeros(hv, device="cuda")
            bias = torch.zeros_like(alog)
            state = torch.zeros(
                args.layers, batch + 1, hv, kd, vd, device="cuda", dtype=getattr(torch, args.state_dtype)
            )
            if args.compact:
                from lightllm.common.basemodel.triton_kernel.linear_att.replayssm_compact import CompactSSMCache

                replay = CompactSSMCache(state, width, torch.bfloat16)
            else:
                replay = ReplaySSMCache(state, args.capacity, width)
            reqs = torch.arange(batch, device="cuda", dtype=torch.int32)
            cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * width
            accepted = torch.full((batch + 1,), accepted_tokens - 1, device="cuda", dtype=torch.int32)
            baseline_states = torch.zeros(
                args.layers, (batch + 1) * width, hv, kd, vd, device="cuda", dtype=state.dtype
            )
            indexes = torch.arange(batch * width, device="cuda", dtype=torch.int32).view(batch, width)
            counts = torch.full((batch,), accepted_tokens, device="cuda", dtype=torch.int32)

            def run_replay():
                positions = replay.positions(reqs)
                kwargs = {} if args.compact else {"run_config": {"BV": args.bv, "num_warps": args.warps}}
                for layer in range(args.layers):
                    replay.forward(layer, q, k, v, a, b, alog, bias, reqs, positions, cu, **kwargs)
                replay.commit(reqs, accepted)

            def run_baseline_layer(baseline_state):
                if width > 1:
                    mtp_fused_recurrent_gated_delta_rule(
                        q,
                        k,
                        v,
                        baseline_state,
                        cu,
                        indexes,
                        indexes,
                        counts,
                        alog,
                        bias,
                        a,
                        b,
                        run_config={"BV": 8, "num_warps": 1, "num_stages": 1},
                    )
                else:
                    fused_recurrent_gated_delta_rule(
                        q=q.view(batch, 1, h, kd),
                        k=k.view(batch, 1, h, kd),
                        v=v.view(batch, 1, hv, vd),
                        initial_state=baseline_state,
                        inplace_final_state=True,
                        ssm_state_indices=reqs,
                        use_qk_l2norm_in_kernel=True,
                        A_log=alog,
                        dt_bias=bias,
                        a_raw=a,
                        b_raw=b,
                    )

            def run_baseline():
                for layer in range(args.layers):
                    run_baseline_layer(baseline_states[layer])

            # Warm every ring phase before timing. Timed graphs advance real
            # cursors too, including folds and metadata/acceptance launches.
            for _ in range(2 * args.capacity):
                run_replay()
                run_baseline()
            base_ms = triton.testing.do_bench_cudagraph(run_baseline, rep=300)
            replay_ms = triton.testing.do_bench_cudagraph(run_replay, rep=300)
            replay_bytes = sum(
                x.numel() * x.element_size() for x in vars(replay).values() if isinstance(x, torch.Tensor)
            )
            print(
                json.dumps(
                    dict(
                        batch=batch,
                        width=width,
                        heads=hv,
                        layers=args.layers,
                        accepted_tokens=accepted_tokens,
                        compact=args.compact,
                        state_dtype=args.state_dtype,
                        capacity=args.capacity,
                        bv=8 if args.compact else args.bv,
                        warps=1 if args.compact else args.warps,
                        baseline_ms=base_ms,
                        replay_ms=replay_ms,
                        speedup=base_ms / replay_ms,
                        baseline_ssm_bytes=baseline_states.numel() * baseline_states.element_size(),
                        replay_bytes=replay_bytes,
                    )
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
