"""Compare DS4 compressor kernels on synthetic MTP/prefill inputs and packed caches.

Extract the pre-change compressor with git show, pass it as --baseline, and run
with the checkout on PYTHONPATH. No model weights or distributed group are needed.
"""

import argparse
import hashlib
import importlib.util
import json
import statistics
import subprocess
from pathlib import Path
from types import SimpleNamespace

import torch
import triton

from lightllm.models.deepseek_v4.layer_infer import compressor


def make_case(requests, pattern, ratio=128, indexer=False, prefill_tokens=0):
    head = 128 if indexer else 512
    width = head * (2 if ratio == 4 else 1)
    ring = 11 if ratio == 4 else 132
    page_size = 1 if indexer else 256 // ratio
    if prefill_tokens:
        requests = 1
        rows = prefill_tokens
        positions = torch.arange(rows, device="cuda", dtype=torch.int32) + 17020
        req_idx = torch.zeros(1, device="cuda", dtype=torch.int32)
        mtp = torch.zeros(rows, device="cuda", dtype=torch.int32)
        seq_len = torch.full_like(req_idx, 17020 + rows)
        ready = torch.full_like(req_idx, 17020)
        q_start = torch.zeros_like(req_idx)
        token_to_batch = torch.zeros(rows, device="cuda", dtype=torch.int32)
        mem_index = torch.arange(rows, device="cuda", dtype=torch.int32)
    else:
        rows = requests * 4
        starts = torch.full((requests,), 17020, dtype=torch.int32, device="cuda")
        if pattern == "one_boundary":
            starts[1:] = 17004
        elif pattern == "no_boundary":
            starts[:] = 17004
        elif pattern == "padding":
            starts[-1] = 0
        mtp = torch.arange(4, dtype=torch.int32, device="cuda").repeat(requests)
        positions = starts.repeat_interleave(4) + mtp
        req_idx = torch.arange(requests, dtype=torch.int32, device="cuda").repeat_interleave(4)
        if pattern == "padding":
            req_idx[-4:] = requests
        seq_len = (positions + 1).int()
        ready = q_start = None
        token_to_batch = None
        mem_index = torch.arange(requests, dtype=torch.int32, device="cuda").repeat_interleave(4) * 256 + mtp
    pages = max(requests + 1, (rows + 255) // 256 + 1)
    max_pos = int(positions.max().item()) + 1
    swa_table = torch.full((requests + 1, (max_pos + 127) // 128 + 1), -1, dtype=torch.int32, device="cuda")
    for req in range(requests + 1):
        swa_table[req, -3:] = torch.arange(3, dtype=torch.int32, device="cuda") + req * 3
    if pattern == "padding":
        swa_table[requests, :2] = torch.arange(2, dtype=torch.int32, device="cuda") + requests * 3
    state_rows = (requests + 1) * (3 if ratio == 4 else 1) * ring
    state = torch.randn((state_rows, width * 2), dtype=torch.float32, device="cuda") * 0.4
    data_bytes = head - 64 + 64 * 2
    scales = (head - 64) // 64 + 1
    out = (
        torch.full((rows, head), 17, dtype=torch.bfloat16, device="cuda")
        if indexer
        else torch.full((pages, page_size * (data_bytes + scales)), 85, dtype=torch.uint8, device="cuda")
    )
    mm = SimpleNamespace(
        c128_state_ring=ring,
        c4_state_ring=ring,
        indexer_head_dim=head,
        c128_pool=SimpleNamespace(page_size=page_size),
        c4_pool=SimpleNamespace(page_size=page_size),
        get_c128_state_buffer=lambda _: state,
        get_c4_state_buffer=lambda _: state,
        get_c4_indexer_state_buffer=lambda _: state,
        get_compressed_kv_buffer=lambda _: out,
    )
    if ratio == 4:
        token_reqs = req_idx if not prefill_tokens else torch.zeros(rows, dtype=torch.int32, device="cuda")
        swa_slots = swa_table[token_reqs.long(), positions // 128] * 128 + positions % 128
    else:
        swa_slots = torch.zeros(rows, dtype=torch.int32, device="cuda")
    infer = SimpleNamespace(
        mem_manager=mm,
        mem_index=mem_index,
        position_ids=positions,
        b_req_idx=req_idx,
        b_mtp_index=mtp,
        b_seq_len=seq_len,
        b_ready_cache_len=ready,
        b_q_start_loc=q_start,
        _dsv4_token_to_batch_idx=token_to_batch,
        is_prefill=bool(prefill_tokens),
        req_manager=SimpleNamespace(req_to_swa_pages=swa_table),
        dsv4_swa_write_slots=swa_slots,
        microbatch_index=0,
        dsv4_workspace=SimpleNamespace(
            c128_compress_scratch=torch.empty((1, rows, head), dtype=torch.float32, device="cuda")
        ),
    )
    angles = (
        torch.arange(max_pos, dtype=torch.float32, device="cuda")[:, None]
        * torch.linspace(0.0001, 0.01, 32, device="cuda")[None, :]
    )
    args = dict(
        kv_score=torch.randn((rows, width * 2), device="cuda", dtype=torch.float32),
        infer_state=infer,
        layer_idx=0,
        norm_weight=torch.ones(head, dtype=torch.bfloat16, device="cuda"),
        eps=1e-6,
        head_dim=head,
        qk_rope_head_dim=64,
        compress_ratio=ratio,
        cos_table=angles.cos(),
        sin_table=angles.sin(),
        is_in_indexer=indexer,
        out_buffer=out if indexer else None,
    )
    return args, state, out


def decoded_cache(out, args):
    head, ratio = args["head_dim"], args["compress_ratio"]
    page_size = 256 // ratio
    data_bytes = head - 64 + 128
    bytes_per_scale = (head - 64) // 64 + 1
    data = out[:, : page_size * data_bytes].reshape(-1, data_bytes).contiguous()
    scales = out[:, page_size * data_bytes :].reshape(-1, bytes_per_scale)
    fp8 = data[:, : head - 64].contiguous().view(torch.float8_e4m3fn).float()
    scale = torch.exp2(scales[:, :-1].float() - 127).repeat_interleave(64, dim=1)
    rope = data[:, head - 64 :].contiguous().view(torch.bfloat16).float()
    return fp8 * scale, rope, scales


def validate(args, state, out, baseline):
    initial = state.clone()
    sentinel = out.clone()
    baseline.fused_compress(**args)
    old_out, old_state = out.clone(), state.clone()
    state.copy_(initial)
    out.copy_(sentinel)
    compressor.fused_compress(**args)
    torch.cuda.synchronize()
    torch.testing.assert_close(state, old_state, rtol=0, atol=0)
    different = int((out != old_out).sum().item())
    if args["compress_ratio"] == 4:
        torch.testing.assert_close(out, old_out, rtol=0, atol=0)
    else:
        # FP8/BF16 output can differ at a rounding boundary when the FP32
        # reduction is tiled. Bound the changed fraction and to one output ULP.
        assert different / out.numel() < 0.001, (different, out.numel())
        new_kv, new_rope, new_scales = decoded_cache(out, args)
        old_kv, old_rope, old_scales = decoded_cache(old_out, args)
        torch.testing.assert_close(new_scales, old_scales, rtol=0, atol=0)
        torch.testing.assert_close(new_kv, old_kv, rtol=0.126, atol=2e-5)
        torch.testing.assert_close(new_rope, old_rope, rtol=0.008, atol=1e-6)
        infer = args["infer_state"]
        boundary = ((infer.position_ids + 1) % 128 == 0).nonzero().flatten().tolist()
        scratch = infer.dsv4_workspace.c128_compress_scratch[0]
        errors = []
        for token in boundary:
            p = int(infer.position_ids[token].item())
            batch = int(infer._dsv4_token_to_batch_idx[token].item()) if infer.is_prefill else token
            req = int(infer.b_req_idx[batch].item())
            ready = (
                int(infer.b_ready_cache_len[batch].item())
                if infer.is_prefill
                else p - int(infer.b_mtp_index[token].item())
            )
            start = (
                int(infer.b_q_start_loc[batch].item())
                if infer.is_prefill
                else token - int(infer.b_mtp_index[token].item())
            )
            positions = torch.arange(p - 127, p + 1, device="cuda")
            current = positions >= ready
            states = initial[req * 132 + positions % 132].clone()
            states[current] = args["kv_score"][start + positions[current] - ready]
            kv, score = states.chunk(2, dim=1)
            expected = (kv * torch.softmax(score, dim=0)).sum(dim=0)
            torch.testing.assert_close(scratch[token], expected, rtol=3e-5, atol=2e-6)
            errors.append(float((scratch[token] - expected).abs().max().item()))
        return {"different_output_bytes": different, "fp32_max_abs_error": max(errors, default=0)}
    return {"different_output_bytes": different}


def observe(module, collected):
    original = module._fused_compress_norm_rope_insert_kernel.run

    def run(*args, **kwargs):
        kernel = original(*args, **kwargs)
        key = (
            kwargs.get("IS_PREFILL"),
            kwargs.get("IS_IN_INDEXER"),
            kwargs.get("COMPRESS_RATIO"),
            kwargs.get("BLOCK"),
            kwargs.get("STORE_COMPRESSED", False),
            kwargs.get("LOAD_COMPRESSED", False),
        )
        collected[key] = kernel
        return kernel

    module._fused_compress_norm_rope_insert_kernel.run = run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, nargs="+", default=[16, 24, 32, 64])
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("baseline_compressor", args.baseline)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    compiled = {"baseline": {}, "optimized": {}}
    observe(baseline, compiled["baseline"])
    observe(compressor, compiled["optimized"])
    cases = [
        (r, pattern, 128, False, 0)
        for r in args.requests
        for pattern in ["dense", "one_boundary", "no_boundary", "padding"]
    ]
    cases += [(16, "dense", 4, False, 0), (16, "dense", 4, True, 0)]
    cases += [(1, "prefill", 128, False, n) for n in [129, 512, 8192]]
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "optimized_sha256": hashlib.sha256(Path(compressor.__file__).read_bytes()).hexdigest(),
        "cases": [],
    }
    for requests, pattern, ratio, indexer, prefill in cases:
        torch.manual_seed(1012)
        kw, state, out = make_case(requests, pattern, ratio, indexer, prefill)
        parity = validate(kw, state, out, baseline)
        times = {"baseline": [], "optimized": []}
        for _ in range(args.rounds):
            for name, module in [("baseline", baseline), ("optimized", compressor)]:
                times[name].append(
                    triton.testing.do_bench_cudagraph(lambda kw=kw, module=module: module.fused_compress(**kw), rep=40)
                )
        old, new = (statistics.median(times[name]) * 1000 for name in ["baseline", "optimized"])
        row = {
            "requests_per_rank": requests,
            "query_rows": kw["kv_score"].shape[0],
            "pattern": pattern,
            "compress_ratio": ratio,
            "indexer": indexer,
            "prefill": bool(prefill),
            **parity,
            "baseline_us": old,
            "optimized_us": new,
            "speedup": old / new,
            "rounds_ms": times,
        }
        report["cases"].append(row)
        (args.out_dir / "results.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)
        del kw, state, out
    resources = []
    for version, kernels in compiled.items():
        for i, (key, kernel) in enumerate(kernels.items()):
            path = args.out_dir / f"{version}-{i}.cubin"
            path.write_bytes(kernel.asm["cubin"])
            resource = subprocess.check_output(
                ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", str(path)], text=True
            )
            (args.out_dir / f"{version}-{i}-resources.txt").write_text(resource)
            resources.append(
                {
                    "version": version,
                    "specialization": key,
                    "registers": kernel.n_regs,
                    "shared": kernel.metadata.shared,
                    "resource_usage": resource,
                    "cubin": str(path),
                }
            )
    (args.out_dir / "compiler-resources.json").write_text(json.dumps(resources, indent=2))
    print("COMPRESSOR_BENCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
