import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or int(os.environ.get("WORLD_SIZE", "1")) < 2,
    reason="Run with torchrun on multiple CUDA GPUs and DeepEP/DeepGEMM installed",
)


def test_bf16_ep():
    import deep_ep
    import deep_gemm
    from lightllm.distributed import dist_group_manager
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils import envs_utils
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep.bf16 import fused_experts_bf16
    from lightllm.common.triton_utils.autotuner import Autotuner

    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    for name, value in {
        "LIGHTLLM_GLOBAL_RANK": rank,
        "LIGHTLLM_GLOBAL_WORLD_SIZE": world,
        "LIGHTLLM_CURRENT_RANK_IN_NODE": rank,
        "LIGHTLLM_CURRENT_DEVICE_ID": rank,
        "LIGHTLLM_DP_WORLD_SIZE": world,
        "LIGHTLLM_CURRENT_RANK_IN_DP": rank,
    }.items():
        os.environ[name] = str(value)
    args = StartArgs(
        batch_max_tokens=8192, running_max_req_size=256, mtp_step=3, tp=world, dp=1, enable_tpsp_mix_mode=True
    )
    os.environ["LIGHTLLM_START_ARGS"] = json.dumps(vars(args))
    os.environ.pop("NUM_MAX_DISPATCH_TOKENS_PER_RANK_PREFILL", None)
    os.environ.pop("NUM_MAX_DISPATCH_TOKENS_PER_RANK_DECODE", None)
    envs_utils.get_deepep_num_max_dispatch_tokens_per_rank_prefill.cache_clear()
    envs_utils.get_deepep_num_max_dispatch_tokens_per_rank_decode.cache_clear()
    envs_utils.get_deepep_num_max_dispatch_tokens_per_rank.cache_clear()
    envs_utils.get_env_start_args.cache_clear()
    deep_gemm.set_mk_alignment_for_contiguous_layout(128)
    dist.barrier(device_ids=[rank])
    buffer = deep_ep.ElasticBuffer(
        dist.group.WORLD,
        num_max_tokens_per_rank=envs_utils.get_deepep_num_max_dispatch_tokens_per_rank(),
        hidden=256,
        num_topk=2,
        use_fp8_dispatch=False,
        allow_multiple_reduction=True,
    )
    dist_group_manager.ep_buffer = buffer
    autotune_group = dist.new_group(backend="gloo")
    from unittest.mock import patch

    torch.manual_seed(1042)
    experts = world * 2
    up = torch.randn(experts, 256, 256, device="cuda", dtype=torch.bfloat16) * 0.05
    down = torch.randn(experts, 256, 128, device="cuda", dtype=torch.bfloat16) * 0.05
    local_up = up[rank * 2 : (rank + 1) * 2].contiguous()
    local_down = down[rank * 2 : (rank + 1) * 2].contiguous()

    def reference(x, ids, weights):
        total = torch.zeros_like(x, dtype=torch.float32)
        for e in range(experts):
            rows, routes = torch.where(ids == e)
            projected = F.linear(x[rows], up[e])
            gate, value = projected.chunk(2, dim=-1)
            activated = (F.silu(gate.float()) * value.float()).to(x.dtype)
            output = F.linear(activated, down[e])
            total.index_add_(0, rows, output.float() * weights[rows, routes, None])
        return total.to(x.dtype)

    try:
        for tokens, concentrated in ((3, False), (7, True), (17, False), (128, True)):
            dummy_x = torch.zeros(256, 256, device="cuda", dtype=torch.bfloat16)
            dummy_y = torch.empty_like(dummy_x)
            dummy_up = torch.empty_like(dummy_x)
            dummy_mid = torch.zeros(256, 128, device="cuda", dtype=torch.bfloat16)
            dummy_layout = torch.tensor([128, 256], device="cuda", dtype=torch.int32)
            expected_m = tokens
            for a, b, out in ((dummy_x, local_up, dummy_up), (dummy_mid, local_down, dummy_y)):
                deep_gemm.m_grouped_bf16_gemm_nt_contiguous(
                    a,
                    b,
                    out,
                    dummy_layout,
                    use_psum_layout=True,
                    expected_m_for_psum_layout=expected_m,
                )
            torch.cuda.synchronize()
            dist.barrier(device_ids=[rank])
            torch.manual_seed(1200 + rank)
            x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16) * 0.2
            saved = x.clone()
            ids = torch.stack(
                [
                    torch.arange(tokens, device="cuda") % (2 if concentrated else experts),
                    (torch.arange(tokens, device="cuda") + 1) % (2 if concentrated else experts),
                ],
                dim=-1,
            ).long()
            weights = torch.softmax(torch.randn(tokens, 2, device="cuda"), dim=-1)
            expected = reference(x, ids, weights)

            shared_func = None

            def run(prefill):
                return fused_experts_bf16(
                    x, local_up, local_down, weights, ids, experts, prefill, shared_expert_func=shared_func
                )

            for prefill in (True, False):
                actual = run(prefill)
                torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.03)
                torch.testing.assert_close(x, saved, atol=0, rtol=0)
            if concentrated:
                with patch.object(
                    dist_group_manager, "get_default_group", return_value=SimpleNamespace(autotune_group=autotune_group)
                ), Autotuner.autotune_warmup():
                    actual = run(True)
                torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.03)
            original_ids, original_weights = ids.clone(), weights.clone()
            for prefill, overlap in ((False, False), (True, False), (False, True), (True, True)):
                ids.copy_(original_ids)
                weights.copy_(original_weights)
                shared_func = (lambda: x * 0.25) if overlap else None
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        actual = run(prefill)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = run(prefill)
                x.mul_(0.5)
                expected = reference(x, ids, weights)
                if overlap:
                    expected += shared_func()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured, expected, atol=0.002, rtol=0.03)
                ids.fill_(-1)
                weights.zero_()
                if rank == 0:
                    ids[0] = torch.tensor([0, experts - 1], device="cuda")
                    weights[0] = 0.5
                expected = reference(x, ids, weights)
                if overlap:
                    expected += shared_func()
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(captured, expected, atol=0.002, rtol=0.03)
                ids.fill_(-1)
                weights.zero_()
                graph.replay()
                torch.cuda.synchronize()
                empty_expected = shared_func() if overlap else torch.zeros_like(captured)
                torch.testing.assert_close(captured, empty_expected, atol=0, rtol=0)
                graph.reset()
                dist.barrier(device_ids=[rank])
    finally:
        dist_group_manager.ep_buffer = None
        del buffer
        dist.destroy_process_group()
