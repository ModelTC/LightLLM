"""Numeric check for the FlashInfer fused all_reduce + residual add + RMSNorm op.

Run on 2 GPUs:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
        unit_tests/distributed/test_fused_allreduce_rmsnorm.py

Asserts the fused op matches plain all_reduce + residual add + RMSNorm (the
fallback path) to within bf16 tolerance, and that the residual is updated
in place. Skips cleanly if FlashInfer disables itself (unsupported GPU/world).
"""
import os
import torch
import torch.distributed as dist


def rmsnorm_ref(x, weight, eps):
    # Matches lightllm RMSNormWeight._native_forward: fp32 accumulate, cast at end.
    x_fp32 = x.to(torch.float32)
    var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
    return ((x_fp32 * torch.rsqrt(var + eps)) * weight).to(x.dtype)


def main():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    from lightllm.distributed.flashinfer_all_reduce import FlashInferAllReduce

    cpu_group = dist.new_group(list(range(world_size)), backend="gloo")
    fi = FlashInferAllReduce(cpu_group, local_rank)
    if fi.disabled:
        if rank == 0:
            print("SKIP: FlashInferAllReduce disabled on this GPU/world_size")
        dist.destroy_process_group()
        return

    torch.manual_seed(1234 + rank)  # distinct per-rank partials
    hidden = 4096
    eps = 1e-6
    dtype = torch.bfloat16
    weight = torch.randn(hidden, dtype=dtype, device="cuda") * 0.1 + 1.0
    # broadcast weight so all ranks share the same gamma
    dist.broadcast(weight, src=0)

    max_diffs = []
    for tokens in [1, 8, 32]:
        partial = torch.randn(tokens, hidden, dtype=dtype, device="cuda") * 0.1
        residual = torch.randn(tokens, hidden, dtype=dtype, device="cuda") * 0.1
        dist.broadcast(residual, src=0)  # residual is shared (post-embedding) across ranks

        if not fi.should_use(partial):
            if rank == 0:
                print(f"tokens={tokens}: should_use False (unexpected for this size)")
            continue

        # reference: plain all_reduce, then add + rmsnorm
        summed = partial.clone()
        dist.all_reduce(summed, group=None)
        ref_residual = residual + summed
        ref_norm = rmsnorm_ref(ref_residual, weight, eps)

        # fused
        res_fused = residual.clone()
        norm_out = torch.empty_like(partial)
        fi.all_reduce_fused_add_rmsnorm(partial.clone(), res_fused, weight, eps, norm_out)

        d_res = (res_fused.float() - ref_residual.float()).abs().max().item()
        d_norm = (norm_out.float() - ref_norm.float()).abs().max().item()
        max_diffs.append((tokens, d_res, d_norm))
        # residual update must be near-exact (just an add over the reduced sum)
        assert d_res < 1e-2, f"tokens={tokens} residual mismatch {d_res}"
        assert d_norm < 3e-2, f"tokens={tokens} norm mismatch {d_norm}"

    if not max_diffs:
        dist.destroy_process_group()
        raise RuntimeError("FlashInfer initialized, but no fused all_reduce+add+rmsnorm case executed")

    dist.barrier()
    if rank == 0:
        for tokens, d_res, d_norm in max_diffs:
            print(f"tokens={tokens}: max|d residual|={d_res:.2e}  max|d norm|={d_norm:.2e}")
        print("PASS: fused all_reduce+add+rmsnorm matches reference")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
