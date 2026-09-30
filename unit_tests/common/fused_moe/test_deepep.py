"""Tests for the ElasticBuffer expanded-layout DeepEP path."""

import copy
import importlib.util
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import (
    ep_compact_metadata,
    ep_gather_chunk,
    ep_reduce_decode_output,
)


def _reference_reduce_decode_output(
    expert_output: torch.Tensor,
    route_weights: torch.Tensor,
    recv_src_metadata: torch.Tensor,
    num_valid_recv_tokens: int,
):
    metadata = recv_src_metadata[:num_valid_recv_tokens]
    expert_rows = metadata[:, 2:].to(torch.long)
    valid_expert_rows = expert_rows >= 0
    safe_expert_rows = expert_rows.clamp_min(0)

    contributions = expert_output[safe_expert_rows].float()
    contributions *= route_weights[safe_expert_rows, None].float()
    contributions *= valid_expert_rows[:, :, None]
    dense_output = contributions.sum(dim=1).to(expert_output.dtype)

    compact_metadata = metadata.clone()
    compact_metadata[:, 2:] = -1
    compact_metadata[:, 2] = torch.arange(
        num_valid_recv_tokens,
        dtype=compact_metadata.dtype,
        device=compact_metadata.device,
    )
    return dense_output, compact_metadata


def _build_reduce_inputs(
    num_recv_tokens: int,
    num_valid_recv_tokens: int,
    topk: int,
    hidden_size: int,
):
    num_expanded_rows = max(32, num_valid_recv_tokens * topk // 2)
    generator = torch.Generator(device="cuda").manual_seed(1234)
    expert_output = torch.randn(
        (num_expanded_rows, hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    route_weights = torch.rand(
        (num_expanded_rows,),
        dtype=torch.float32,
        device="cuda",
        generator=generator,
    )

    recv_src_metadata = torch.full(
        (num_recv_tokens, topk + 2),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    token_ids = torch.arange(num_valid_recv_tokens, dtype=torch.int32, device="cuda")
    recv_src_metadata[:num_valid_recv_tokens, 0] = 10_000 + token_ids
    recv_src_metadata[:num_valid_recv_tokens, 1] = token_ids % 8
    for topk_index in range(topk):
        expert_rows = (token_ids * topk + topk_index * 7) % num_expanded_rows
        # 同时覆盖普通映射、部分 top-k 不属于当前 rank，以及所有 top-k
        # 都不属于当前 rank 的 token。
        if topk > 1:
            expert_rows[(token_ids + topk_index) % 5 == 0] = -1
            expert_rows[token_ids % 17 == 0] = -1
        recv_src_metadata[:num_valid_recv_tokens, topk_index + 2] = expert_rows

    num_valid_recv_tokens_tensor = torch.tensor(
        [num_valid_recv_tokens],
        dtype=torch.int32,
        device="cuda",
    )
    return expert_output, route_weights, recv_src_metadata, num_valid_recv_tokens_tensor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "num_recv_tokens,num_valid_recv_tokens,topk,hidden_size",
    [
        # 无 hidden 尾块、top-k=1 的基础路径。
        (16, 7, 1, 1024),
        # 非对齐 hidden size、无效 expert 行和全无效 token。
        (64, 17, 4, 1031),
        # 接收 buffer 容量远大于真实 token 数。
        (4096, 17, 8, 257),
        # 有效 token 数超过 1024，覆盖一个 program 处理多行的循环路径。
        (1031, 1031, 8, 128),
    ],
)
def test_ep_reduce_decode_output_matches_reference(
    num_recv_tokens,
    num_valid_recv_tokens,
    topk,
    hidden_size,
):
    inputs = _build_reduce_inputs(
        num_recv_tokens=num_recv_tokens,
        num_valid_recv_tokens=num_valid_recv_tokens,
        topk=topk,
        hidden_size=hidden_size,
    )
    expert_output, route_weights, recv_src_metadata, num_valid_recv_tokens_tensor = inputs

    expected_output, expected_metadata = _reference_reduce_decode_output(
        expert_output=expert_output,
        route_weights=route_weights,
        recv_src_metadata=recv_src_metadata,
        num_valid_recv_tokens=num_valid_recv_tokens,
    )
    output, compact_metadata = ep_reduce_decode_output(
        expert_output=expert_output,
        route_weights=route_weights,
        recv_src_metadata=recv_src_metadata,
        num_valid_recv_tokens=num_valid_recv_tokens_tensor,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        output[:num_valid_recv_tokens],
        expected_output,
        atol=1e-2,
        rtol=1e-2,
    )
    assert torch.equal(
        compact_metadata[:num_valid_recv_tokens],
        expected_metadata,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _per_block_cast_to_fp8(weight: torch.Tensor):
    """按 DeepGEMM 的 128x128 权重 scale 布局量化二维权重。"""
    rows, columns = weight.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 127) // 128 * 128
    padded_weight = torch.zeros(
        (padded_rows, padded_columns),
        dtype=weight.dtype,
        device=weight.device,
    )
    padded_weight[:rows, :columns] = weight
    weight_blocks = padded_weight.view(
        padded_rows // 128,
        128,
        padded_columns // 128,
        128,
    )
    block_absmax = weight_blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp_min(1e-10)
    quantized_weight = (weight_blocks * (448.0 / block_absmax)).to(torch.float8_e4m3fn)
    return (
        quantized_weight.view_as(padded_weight)[:rows, :columns].contiguous(),
        (block_absmax / 448.0).view(padded_rows // 128, padded_columns // 128),
    )


def _build_local_expert_weights(rank: int, num_local_experts: int, hidden_size: int, intermediate_size: int):
    w1 = []
    w1_scale = []
    w2 = []
    w2_scale = []
    for local_expert_index in range(num_local_experts):
        global_expert_index = rank * num_local_experts + local_expert_index
        generator = torch.Generator(device="cuda").manual_seed(1000 + global_expert_index)
        expert_w1 = (
            torch.randn(
                (intermediate_size * 2, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            / hidden_size ** 0.5
        )
        expert_w2 = (
            torch.randn(
                (hidden_size, intermediate_size),
                dtype=torch.bfloat16,
                device="cuda",
                generator=generator,
            )
            / intermediate_size ** 0.5
        )
        quantized_w1, quantized_w1_scale = _per_block_cast_to_fp8(expert_w1)
        quantized_w2, quantized_w2_scale = _per_block_cast_to_fp8(expert_w2)
        w1.append(quantized_w1)
        w1_scale.append(quantized_w1_scale)
        w2.append(quantized_w2)
        w2_scale.append(quantized_w2_scale)
    return tuple(torch.stack(tensors) for tensors in (w1, w1_scale, w2, w2_scale))


def _decode_dispatch_reduce_combine_worker(rank: int, port: int) -> None:
    import deep_ep
    from lightllm.common.basemodel.triton_kernel.fused_moe.grouped_fused_moe_ep import (
        decode_masked_group_gemm,
        set_mk_alignment_for_contiguous_layout,
    )
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl import (
        FuseMoeDeepGEMM,
    )
    from lightllm.distributed import dist_group_manager
    from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import (
        per_token_group_quant_fp8,
    )

    world_size = 2
    num_tokens = 4
    num_experts = 4
    num_local_experts = num_experts // world_size
    topk = 2
    hidden_size = 256
    intermediate_size = 128
    num_max_tokens_per_rank = 8

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # 单机测试只覆盖 NVLink 通信，不依赖测试机是否配置 NCCL GIN/RDMA。
    os.environ["EP_DISABLE_GIN"] = "1"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    group = dist.new_group(list(range(world_size)), backend="nccl")
    dist.barrier(group=group, device_ids=[rank])

    buffer = None
    try:
        buffer = deep_ep.ElasticBuffer(
            group,
            num_max_tokens_per_rank=num_max_tokens_per_rank,
            hidden=hidden_size,
            num_topk=topk,
            use_fp8_dispatch=True,
            deterministic=True,
            allow_multiple_reduction=True,
            prefer_overlap_with_compute=False,
            explicitly_destroy=True,
        )
        set_mk_alignment_for_contiguous_layout(128)
        torch.manual_seed(100 + rank)
        hidden_states = torch.randn(
            (num_tokens, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        # 每个 rank 都包含：两个 expert 位于同一个远端 rank、两个 expert
        # 位于本地，以及分别命中两个 rank 的 token。
        if rank == 0:
            topk_idx = torch.tensor(
                [[2, 3], [0, 1], [0, 2], [1, 3]],
                dtype=torch.int64,
                device="cuda",
            )
        else:
            topk_idx = torch.tensor(
                [[0, 1], [2, 3], [0, 2], [1, 3]],
                dtype=torch.int64,
                device="cuda",
            )
        topk_weights = torch.tensor(
            [[0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75]],
            dtype=torch.float32,
            device="cuda",
        )
        w1, w1_scale, w2, w2_scale = _build_local_expert_weights(
            rank=rank,
            num_local_experts=num_local_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

        def dispatch():
            qinput_tensor, input_scale = per_token_group_quant_fp8(
                hidden_states,
                group_size=128,
                dtype=torch.float8_e4m3fn,
            )
            recv_x, _, recv_topk_weights, handle, event = buffer.dispatch(
                (qinput_tensor, input_scale),
                topk_idx=topk_idx,
                topk_weights=topk_weights,
                num_experts=num_experts,
                num_max_tokens_per_rank=num_max_tokens_per_rank,
                expert_alignment=128,
                async_with_compute_stream=False,
                allocate_on_comm_stream=False,
                do_cpu_sync=False,
                do_handle_copy=False,
                do_expand=True,
                do_zero_padding=True,
                use_tma_aligned_col_major_sf=True,
            )
            assert event.event is None
            return recv_x, recv_topk_weights, handle

        def run_experts(recv_x, handle):
            return decode_masked_group_gemm(
                recv_x=recv_x,
                expert_token_psum=handle.psum_num_recv_tokens_per_expert,
                expert_alignment=handle.expert_alignment,
                dtype=hidden_states.dtype,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                expected_m=1,
            )

        recv_x, recv_topk_weights, handle = dispatch()

        num_valid_recv_tokens = handle.psum_num_recv_tokens_per_scaleup_rank[-1:]
        # 每个来源 rank 各有 3 个 token 命中当前 rank。若 dispatch 没有按
        # 目标 rank 去重，这里会得到 8 个 expert 路由项而不是 6 个 token。
        assert num_valid_recv_tokens.item() == 6

        expert_output = run_experts(recv_x, handle)
        reference_expert_output = expert_output.clone()
        reference_recv_topk_weights = recv_topk_weights.clone()
        reference_handle = copy.copy(handle)
        reference_handle.recv_src_metadata = handle.recv_src_metadata.clone()
        reference_dense_output = torch.zeros(
            (reference_handle.recv_src_metadata.shape[0], hidden_size),
            dtype=reference_expert_output.dtype,
            device=reference_expert_output.device,
        )
        ep_gather_chunk(
            chunk=reference_expert_output,
            chunk_start=0,
            weights=reference_recv_topk_weights,
            recv_src_metadata=reference_handle.recv_src_metadata,
            output=reference_dense_output,
        )
        ep_compact_metadata(reference_handle.recv_src_metadata)
        expected_output, _, event = buffer.combine(
            reference_dense_output,
            handle=reference_handle,
            topk_weights=None,
            async_with_compute_stream=True,
            allocate_on_comm_stream=True,
        )
        event.current_stream_wait()
        expected_output = expected_output.clone()

        sync_handle = copy.copy(handle)
        sync_handle.recv_src_metadata = handle.recv_src_metadata.clone()
        sync_dense_output, sync_metadata = ep_reduce_decode_output(
            expert_output=expert_output,
            route_weights=recv_topk_weights,
            recv_src_metadata=sync_handle.recv_src_metadata,
            num_valid_recv_tokens=sync_handle.psum_num_recv_tokens_per_scaleup_rank[-1:],
        )
        sync_handle.recv_src_metadata = sync_metadata
        sync_output, _, sync_event = buffer.combine(
            sync_dense_output,
            handle=sync_handle,
            topk_weights=None,
            num_sms=0,
            async_with_compute_stream=False,
            allocate_on_comm_stream=False,
        )
        assert sync_event.event is None
        sync_output = sync_output.clone()

        dist_group_manager.ep_buffer = buffer
        moe_impl = object.__new__(FuseMoeDeepGEMM)
        overlap_output, hook = moe_impl.decode_combine(
            expert_output=expert_output,
            ep_handle=handle,
            recv_topk_weights=recv_topk_weights,
        )
        hook()
        torch.cuda.synchronize()

        # 对比已有 gather + metadata compact 两步参考实现与新的融合归约。
        # 三条路径共用相同路由和真实 FP8 grouped GEMM，仅归约和同步方式不同。
        torch.testing.assert_close(
            sync_output[:num_tokens],
            expected_output[:num_tokens],
            atol=2e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            overlap_output[:num_tokens],
            expected_output[:num_tokens],
            atol=2e-2,
            rtol=2e-2,
        )
    finally:
        try:
            if buffer is not None:
                buffer.destroy()
        finally:
            dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or importlib.util.find_spec("deep_ep") is None
    or importlib.util.find_spec("deep_gemm") is None,
    reason="requires DeepEP, DeepGEMM, and two CUDA GPUs",
)
def test_decode_dispatch_reduce_combine_two_gpu_correctness():
    mp.spawn(
        _decode_dispatch_reduce_combine_worker,
        args=(_free_port(),),
        nprocs=2,
        join=True,
    )
