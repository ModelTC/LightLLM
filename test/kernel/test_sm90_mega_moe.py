"""Distributed GLM FP8 MegaMoE correctness check; run with torchrun on idle SM90 GPUs."""

import argparse
import json
import os
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.expert_parallel_state import ExpertParallelState
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl import FuseMoeDeepGEMM
from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import prepare_ep_moe_weights
from lightllm.common.basemodel.triton_kernel.fused_moe.sm90_fp8_mega_moe import get_sm90_mega_moe_module
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.distributed import dist_group_manager


def quant_dequant(x, group):
    groups = x.float().reshape(x.shape[0], x.shape[1] // group, group)
    scale = groups.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448.0
    return ((groups / scale).to(torch.float8_e4m3fn).float() * scale).reshape_as(x)


def dequant_weight(weight, scale):
    n, k = weight.shape
    return (weight.float().view(n // 128, 128, k // 128, 128) * scale[:, None, :, None]).reshape(n, k)


def reference(x, ids, weights, w13, w2, rank, world, limit):
    """Independent FP32 MoE with the kernel's per-128 input/per-64 intermediate FP8 rounding."""
    local = (x.cpu(), ids.cpu(), weights.cpu())
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    sizes = [item[0].shape[0] for item in gathered]
    all_x, all_ids, all_weights = [torch.cat([item[i] for item in gathered]).cuda() for i in range(3)]
    all_x = quant_dequant(all_x, 128)
    contributions = torch.zeros(*all_ids.shape, x.shape[1], device="cuda", dtype=torch.float32)
    local_experts = w13.weight.shape[0]
    for expert in range(local_experts):
        rows, slots = torch.where(all_ids == rank * local_experts + expert)
        if rows.numel() == 0:
            continue
        up_gate = all_x[rows] @ dequant_weight(w13.weight[expert], w13.weight_scale[expert]).T
        gate, up = up_gate.chunk(2, dim=-1)
        activated = torch.nn.functional.silu(gate.clamp(max=limit)) * up.clamp(-limit, limit)
        activated *= all_weights[rows, slots, None]
        activated = quant_dequant(activated, 64)
        projected = activated @ dequant_weight(w2.weight[expert], w2.weight_scale[expert]).T
        contributions[rows, slots] = projected.bfloat16().float()
    dist.all_reduce(contributions)
    start = sum(sizes[:rank])
    return contributions.sum(1).bfloat16()[start : start + sizes[rank]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", type=int, default=288)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    os.environ["LIGHTLLM_CURRENT_DEVICE_ID"] = str(rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300), device_id=torch.device("cuda", rank))
    world = dist.get_world_size()
    assert args.experts % world == 0
    torch.manual_seed(5300 + rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    module = get_sm90_mega_moe_module()
    module.set_num_sms(torch.cuda.get_device_properties(rank).multi_processor_count // 2 * 2)
    dist_group_manager.ep_mega_moe_buffer = module.get_symm_buffer_for_sm90_mega_moe(
        dist.group.WORLD, args.experts, 256, args.topk, args.hidden, args.intermediate
    )
    dist_group_manager.ep_triton_moe_quant_method = None
    dist_group_manager.ep_mega_moe_mma_type = "fp8xfp8"
    dist_group_manager.ep_mega_moe_quant_method = "fp8w8a8-b128-deepgemm"
    quant = SimpleNamespace(method_name="fp8w8a8-b128-deepgemm", block_size=128, mega_moe_mma_type=None)
    backend = FuseMoeDeepGEMM(
        n_routed_experts=args.experts,
        num_fused_shared_experts=0,
        routed_scaling_factor=2.5,
        quant_method=quant,
        expert_parallel_state=ExpertParallelState(args.experts, world),
    )

    def pack(n, k):
        shape = (args.experts // world, n, k)
        weight = torch.randint(-16, 17, shape, device="cuda", dtype=torch.int8).to(torch.float8_e4m3fn)
        scale = torch.rand(shape[0], n // 128, k // 128, device="cuda") * 0.015 + 0.002
        return WeightPack(weight, scale)

    w13, w2 = pack(2 * args.intermediate, args.hidden), pack(args.hidden, args.intermediate)
    original = WeightPack(w13.weight.clone(), w13.weight_scale)
    prepare_ep_moe_weights(w13, w2, quant)
    results = []
    # Large -> small -> empty exposes stale padding routes. Rank zero can be
    # empty while other ranks still need its experts.
    for tokens in (129, 17, 1, 0):
        count = tokens + (1 if rank and tokens != 1 else 0)
        x = torch.randn(count, args.hidden, device="cuda", dtype=torch.bfloat16) * 2
        logits = torch.randn(count, args.experts, device="cuda", dtype=torch.float32)
        # Check the real routing path, including correction bias and scaling.
        bias = torch.linspace(-0.05, 0.05, args.experts, device="cuda")
        topk_weights, ids, _ = backend._select_experts(
            x, logits, bias, args.topk, True, True, 1, 1, "sigmoid", is_prefill=tokens > 1
        )
        scores = logits.sigmoid()
        expected_ids = (scores + bias).topk(args.topk, dim=-1).indices
        assert torch.equal(ids.long(), expected_ids.long())
        expected_weights = scores.gather(1, expected_ids)
        expected_weights = expected_weights / expected_weights.sum(-1, keepdim=True) * 2.5
        torch.testing.assert_close(topk_weights, expected_weights, rtol=1e-5, atol=1e-6)

        def forward():
            return backend._fused_experts(
                x, w13, w2, topk_weights, ids, is_prefill=tokens > 1, alpha=1.0, limit=10.0, clamp_up_add_one=False
            )

        actual = forward()
        torch.cuda.synchronize()
        expected = reference(x, ids, topk_weights, original, w2, rank, world, 10.0)
        assert actual.shape == expected.shape and torch.isfinite(actual).all()
        denominator = expected.float().square().sum().clamp_min(1e-10)
        relative_rmse = ((actual.float() - expected.float()).square().sum() / denominator).sqrt().item()
        assert relative_rmse < 0.04, relative_rmse
        record = dict(rank=rank, tokens=count, relative_rmse=relative_rmse)
        if tokens == 17:
            # Warmed state and weights must be reused inside CUDA graph capture.
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = forward()
            for _ in range(2):
                x.normal_()
                graph.replay()
                torch.cuda.synchronize()
                eager = forward()
                torch.testing.assert_close(captured, eager, rtol=0, atol=0)
            record["graph_replays"] = 2
        results.append(record)
        print(json.dumps(record), flush=True)
    dist.barrier()
    dist_group_manager.ep_mega_moe_buffer.destroy()
    dist.destroy_process_group()
    print(json.dumps(dict(status="passed", config=vars(args), results=results)), flush=True)


if __name__ == "__main__":
    main()
