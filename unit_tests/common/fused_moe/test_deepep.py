import pytest

pytest.skip(reason="need special env, install deep_ep and deep_gemm", allow_module_level=True)

import os
import torch
import torch.distributed as dist
import deep_ep
import random
import numpy as np
from lightllm.common.fused_moe.grouped_fused_moe_ep import fused_experts_impl
from lightllm.common.fused_moe.deepep_scatter_gather import ep_scatter, ep_gather
from typing import Tuple
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    from deep_gemm import ceil_div

    x_padded = torch.zeros((ceil_div(m, 128) * 128, ceil_div(n, 128) * 128), dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def init_dist(local_rank: int, num_local_ranks: int):
    # NOTES: you may rewrite this function with your own cluster settings
    ip = os.getenv("MASTER_ADDR", "127.0.0.1")
    port = int(os.getenv("MASTER_PORT", "8361"))
    num_nodes = int(os.getenv("WORLD_SIZE", 1))
    node_rank = int(os.getenv("RANK", 0))
    assert (num_local_ranks < 8 and num_nodes == 1) or num_local_ranks == 8

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{ip}:{port}",
        world_size=num_nodes * num_local_ranks,
        rank=node_rank * num_local_ranks + local_rank,
    )
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.cuda.set_device(local_rank)

    return dist.get_rank(), dist.get_world_size(), dist.new_group(list(range(num_local_ranks * num_nodes)))


def fused_experts_impl_ref(
    x: torch.Tensor,  # [M, K]
    w1: torch.Tensor,  # [group, N, K]
    w2: torch.Tensor,  # [group, K, N/2]
    topk_weight: torch.Tensor,  # [M, topk]
    topk_ids: torch.Tensor,  # [M, topk]
    num_experts: int,
):
    N = w1.shape[1]
    ep_size = torch.distributed.get_world_size()
    experts_per_rank = num_experts // ep_size

    cnts = topk_ids.new_zeros((topk_ids.shape[0], num_experts))
    cnts.scatter_(1, topk_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = topk_ids.view(-1).argsort()
    sorted_tokens = x[idxs // topk_ids.shape[1]]
    sorted_tokens_shape = sorted_tokens.shape

    if ep_size > 1:
        tokens_per_ep_rank = tokens_per_expert.view(ep_size, -1).sum(dim=1)
        tokens_per_expert_group = tokens_per_expert.new_empty(tokens_per_expert.shape[0])
        dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
        output_splits = tokens_per_expert_group.view(ep_size, -1).sum(1).cpu().numpy().tolist()
        gathered_tokens = sorted_tokens.new_empty(
            tokens_per_expert_group.sum(dim=0).cpu().item(), sorted_tokens.shape[1]
        )
        input_split_sizes = tokens_per_ep_rank.cpu().numpy().tolist()
        dist.all_to_all(
            list(gathered_tokens.split(output_splits)),
            list(sorted_tokens.split(input_split_sizes)),
        )
        tokens_per_expert_post_gather = tokens_per_expert_group.view(ep_size, experts_per_rank).sum(dim=0)
        gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
        s = 0
        for i, k in enumerate(tokens_per_expert_group.cpu().numpy()):
            gatherd_idxs[s : s + k] = i % experts_per_rank
            s += k
        gatherd_idxs = gatherd_idxs.argsort()
        sorted_tokens = gathered_tokens[gatherd_idxs]
        tokens_per_expert = tokens_per_expert_post_gather
    tokens_per_expert = tokens_per_expert.cpu().numpy()

    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
        w1_out = torch.matmul(tokens_for_this_expert, w1[i, : N // 2, :].T)
        w2_out = torch.matmul(tokens_for_this_expert, w1[i, N // 2 :, :].T)
        tmp = torch.nn.functional.silu(w1_out)
        tmp = tmp * w2_out
        expert_out = torch.matmul(tmp, w2[i].T)
        outputs.append(expert_out)
        start_idx = end_idx

    outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

    if ep_size > 1:
        new_x = torch.empty_like(outs)
        new_x[gatherd_idxs] = outs
        gathered_tokens = new_x.new_empty(*sorted_tokens_shape)
        dist.all_to_all(
            list(gathered_tokens.split(input_split_sizes)),
            list(new_x.split(output_splits)),
        )
        outs = gathered_tokens

    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    final_out = (
        new_x.view(*topk_ids.shape, -1)
        .type(topk_weight.dtype)
        .mul_(topk_weight.unsqueeze(dim=-1))
        .sum(dim=1)
        .type(new_x.dtype)
    )

    return final_out


def case1(local_rank: int, num_local_ranks: int):
    # Init dist
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)

    torch.manual_seed(rank)

    # Construct inputs
    seqlen = 16
    hidden_states = torch.randn((seqlen, 7168), device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn((256 // num_local_ranks, 4096, 7168), device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn((256 // num_local_ranks, 7168, 2048), device="cuda", dtype=torch.bfloat16)

    w1_fp8 = torch.empty_like(w1, dtype=torch.float8_e4m3fn)
    w2_fp8 = torch.empty_like(w2, dtype=torch.float8_e4m3fn)
    w1_scale = torch.empty((256 // num_local_ranks, 4096 // 128, 7168 // 128), device="cuda", dtype=torch.float)
    w2_scale = torch.empty((256 // num_local_ranks, 7168 // 128, 2048 // 128), device="cuda", dtype=torch.float)

    for i in range(256 // num_local_ranks):
        w1_fp8[i], w1_scale[i] = per_block_cast_to_fp8(w1[i])
        w2_fp8[i], w2_scale[i] = per_block_cast_to_fp8(w2[i])

    topk_weights = torch.randn((seqlen, 8), device="cuda", dtype=torch.float32)
    topk_weights = torch.softmax(topk_weights, dim=-1)  # 对每行进行softmax归一化
    topk_weights = torch.tensor(topk_weights, device="cuda", dtype=torch.float32)
    topk_ids = torch.zeros((seqlen, 8), device="cuda", dtype=torch.int64)
    for i in range(seqlen):
        topk_ids[i] = torch.randperm(254, device="cuda")[:8] + 1

    # Init buffer
    test_ll_compatibility, num_rdma_bytes = True, 0
    num_max_dispatch_tokens_per_rank = 512
    if test_ll_compatibility:
        ll_num_tokens, ll_hidden, ll_num_experts, _ = num_max_dispatch_tokens_per_rank, 7168, 256, 8
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            ll_num_tokens, ll_hidden, num_ranks, ll_num_experts
        )

    buffer = deep_ep.Buffer(
        group,
        int(1e9),
        num_rdma_bytes,
        low_latency_mode=test_ll_compatibility,
        num_qps_per_rank=(ll_num_experts // num_ranks if test_ll_compatibility else 1),
    )

    # Test normal
    ref_output = fused_experts_impl_ref(
        x=hidden_states, w1=w1, w2=w2, topk_weight=topk_weights, topk_ids=topk_ids, num_experts=256
    )

    output = fused_experts_impl(
        hidden_states=hidden_states,
        w1=w1_fp8,
        w2=w2_fp8,
        topk_weights=topk_weights,
        topk_idx=topk_ids,
        num_experts=256,
        buffer=buffer,
        is_prefill=True,
        use_fp8_w8a8=True,
        use_fp8_all2all=True,
        use_int8_w8a16=False,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        previous_event=None,
    )

    # Test ll
    if test_ll_compatibility:
        buffer.clean_low_latency_buffer(ll_num_tokens, ll_hidden, ll_num_experts)
        ll_output = fused_experts_impl(
            hidden_states=hidden_states,
            w1=w1_fp8,
            w2=w2_fp8,
            topk_weights=topk_weights,
            topk_idx=topk_ids,
            num_experts=256,
            buffer=buffer,
            is_prefill=False,
            use_fp8_w8a8=True,
            use_fp8_all2all=True,
            use_int8_w8a16=False,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            previous_event=None,
        )

    # Check
    norm_sim = torch.nn.functional.cosine_similarity(ref_output, output).mean()
    ll_sim = torch.nn.functional.cosine_similarity(ref_output, ll_output).mean()

    assert torch.allclose(norm_sim, torch.ones(1), atol=1e-2, rtol=0)
    assert torch.allclose(ll_sim, torch.ones(1), atol=1e-2, rtol=0)
    logger.info(f"deepep cosine {norm_sim}")
    logger.info(f"deepep ll cosine {ll_sim}")

    # Profile
    profile = os.getenv("PROFILE", False)
    if profile:
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for i in range(10):
                output = fused_experts_impl(
                    hidden_states=hidden_states,
                    w1=w1_fp8,
                    w2=w2_fp8,
                    topk_weights=topk_weights,
                    topk_idx=topk_ids,
                    num_experts=256,
                    buffer=buffer,
                    is_prefill=True,
                    use_fp8_w8a8=True,
                    use_fp8_all2all=True,
                    use_int8_w8a16=False,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    previous_event=None,
                )
            # prof.step()
        if rank == 0:
            print(prof.key_averages().table(sort_by="cuda_time_total"))
            prof.export_chrome_trace("normal_trace.json")

        if test_ll_compatibility:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for i in range(10):
                    ll_output = fused_experts_impl(
                        hidden_states=hidden_states,
                        w1=w1_fp8,
                        w2=w2_fp8,
                        topk_weights=topk_weights,
                        topk_idx=topk_ids,
                        num_experts=256,
                        buffer=buffer,
                        is_prefill=False,
                        use_fp8_w8a8=True,
                        use_fp8_all2all=True,
                        use_int8_w8a16=False,
                        w1_scale=w1_scale,
                        w2_scale=w2_scale,
                        previous_event=None,
                    )
            if rank == 0:
                print(prof.key_averages().table(sort_by="cuda_time_total"))
                prof.export_chrome_trace("ll_trace.json")

    dist.barrier()


def test_end2end():
    num_processes = 8
    torch.multiprocessing.spawn(case1, args=(num_processes,), nprocs=num_processes)


def test_scatter_gather():
    block_size = 128
    num_recv_tokens_per_expert_list = [0] * 32
    num_recv_tokens_per_expert_list[6] = 128
    num_recv_tokens_per_expert_list[7] = 128
    num_recv_tokens_per_expert_list[8] = 128
    num_recv_tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list, dtype=torch.int, device="cuda")

    all_tokens = sum(num_recv_tokens_per_expert_list)
    m_indices_ref = torch.empty(all_tokens, device="cuda", dtype=torch.int32)
    m_indices = torch.empty(all_tokens, device="cuda", dtype=torch.int32)

    recv_x = torch.randn((7, 4096), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    recv_x_scale = torch.randn((7, 4096 // block_size), device="cuda", dtype=torch.float32)

    recv_topk_id = torch.ones((7, 8), device="cuda", dtype=torch.int32) * -1
    recv_topk_weights = torch.zeros((7, 8), device="cuda", dtype=torch.float)
    for i in range(7):
        for j in range(4):
            idx = random.randint(0, 7)
            expert_id = random.randint(6, 8)
            recv_topk_id[i][idx] = expert_id
            recv_topk_weights[i][idx] = random.randint(0, 10) / 10.0

    output_indexs = torch.zeros_like(recv_topk_id)
    output_tensor = torch.zeros((all_tokens, 4096), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    output_tensor_ref = torch.zeros((all_tokens, 4096), device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)

    output_tensor_scale = torch.zeros((all_tokens, 4096 // block_size), device="cuda", dtype=torch.float32)
    output_tensor_scale_ref = torch.zeros((all_tokens, 4096 // block_size), device="cuda", dtype=torch.float32)

    expert_start_loc = torch.cumsum(torch.tensor([0] + num_recv_tokens_per_expert_list[:-1], device="cuda"), dim=0)

    cur = 0
    for i, k in enumerate(num_recv_tokens_per_expert_list):
        m_indices_ref[cur : cur + k] = i
        cur += k

    ep_scatter(
        recv_x,
        recv_x_scale,
        recv_topk_id,
        num_recv_tokens_per_expert,
        expert_start_loc,
        output_tensor,
        output_tensor_scale,
        m_indices,
        output_indexs,
    )
    assert torch.allclose(m_indices, m_indices_ref, atol=1e-2, rtol=0)

    for i in range(recv_topk_id.shape[0]):
        for j in range(recv_topk_id.shape[1]):
            if recv_topk_id[i][j] >= 0:
                dst = output_indexs[i][j]
                output_tensor_ref[dst][:] = recv_x[i][:]
                output_tensor_scale_ref[dst][:] = recv_x_scale[i][:]

    assert torch.allclose(output_tensor.to(torch.float), output_tensor_ref.to(torch.float), atol=1e-2, rtol=0)
    assert torch.allclose(output_tensor_scale, output_tensor_scale_ref, atol=1e-2, rtol=0)

    #### gather

    gather_out_ref = torch.zeros_like(recv_x, device="cuda", dtype=torch.bfloat16)
    gather_out = torch.empty_like(recv_x, device="cuda", dtype=torch.bfloat16)
    gather_input = torch.zeros((all_tokens, 4096), device="cuda", dtype=torch.bfloat16)
    for i in range(recv_topk_id.shape[0]):
        for j in range(recv_topk_id.shape[1]):
            if recv_topk_id[i][j] >= 0:
                dst = output_indexs[i][j]
                gather_out_ref[i][:] += gather_input[dst][:] * recv_topk_weights[i][j]
    ep_gather(gather_input, recv_topk_id, recv_topk_weights, output_indexs, gather_out)
    assert torch.allclose(gather_out, gather_out_ref, atol=1e-2, rtol=0)


if __name__ == "__main__":
    pytest.main()
