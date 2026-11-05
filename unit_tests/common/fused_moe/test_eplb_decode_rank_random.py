"""共同 rank 随机优先级的 uint32 一致性、分布及允许同分的副本选择检查。"""

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_hash_targets import (
    _eplb_hash_targets_topk_ids_kernel,
)
from unit_tests.common.fused_moe.test_eplb_decode_hash_targets import rank_priority_reference


@triton.jit
def _rank_random_probe(
    tokens_ptr,
    ranks_ptr,
    output_ptr,
    current_rank,
    num_tokens,
    num_ranks,
    RANK_BITS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_RANKS: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_TOKENS + tl.arange(start=0, end=BLOCK_TOKENS)  # [BT]：输出行。
    rank_offsets = tl.arange(start=0, end=BLOCK_RANKS)  # [BR]：输出列。
    token_offsets = tl.load(pointer=tokens_ptr + offsets, mask=offsets < num_tokens, other=0)  # [BT]：实际 token 行号。
    ranks = tl.load(pointer=ranks_ptr + rank_offsets, mask=rank_offsets < num_ranks, other=0)
    initial_ranks = ranks[None, :] + tl.full(shape=(BLOCK_TOKENS, 1), value=0, dtype=tl.int64)  # [BT, BR]：目标 rank。
    # 以下三条与生产一致：AST 校验防止测试探针仍观测旧公式。
    rank_priority_seed = tl.cast(input=current_rank, dtype=tl.uint64) | (1 << 32)
    rank_random_offsets = (token_offsets[:, None].to(tl.uint32) << RANK_BITS) | initial_ranks.to(tl.uint32)
    initial_rank_priorities = tl.randint(seed=rank_priority_seed, offset=rank_random_offsets, n_rounds=10).to(tl.uint32)
    tl.static_assert(rank_random_offsets.dtype == tl.uint32)
    valid = (offsets[:, None] < num_tokens) & (rank_offsets[None, :] < num_ranks)
    tl.store(
        pointer=output_ptr + offsets[:, None] * num_ranks + rank_offsets[None, :],
        value=initial_rank_priorities,
        mask=valid,
    )


def test_probe_observes_production_uint32_rank_random():
    variables = {"rank_priority_seed", "rank_random_offsets", "initial_rank_priorities"}

    def expressions(kernel):
        return [
            ast.dump(node)
            for node in ast.walk(ast.parse(kernel.src))
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in variables for target in node.targets)
        ]

    assert len(expressions(_rank_random_probe)) == 3
    assert expressions(_rank_random_probe) == expressions(_eplb_hash_targets_topk_ids_kernel)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("source_rank", [0, 3, 7])
def test_rank_random_matches_independent_reference_at_uint32_boundaries(source_rank):
    tokens = np.concatenate(
        (np.arange(513, dtype=np.uint32), np.asarray([0, (1 << 31) - 1, 1 << 31, (1 << 32) - 1], dtype=np.uint32))
    )
    ranks = np.asarray([0, 1, 2, 7, 31, 255, 1 << 16, 1 << 31, (1 << 32) - 1], dtype=np.uint32)
    expected = np.asarray(
        [
            [rank_priority_reference(source_rank, int(token), int(rank), world_size=256) for rank in ranks]
            for token in tokens
        ],
        dtype=np.uint32,
    )
    output = torch.empty(expected.shape, dtype=torch.int64, device="cuda")
    _rank_random_probe[(triton.cdiv(x=len(tokens), y=32),)](
        tokens_ptr=torch.tensor(tokens.astype(np.int64), device="cuda"),
        ranks_ptr=torch.tensor(ranks.astype(np.int64), device="cuda"),
        output_ptr=output,
        current_rank=source_rank,
        num_tokens=len(tokens),
        num_ranks=len(ranks),
        RANK_BITS=8,
        BLOCK_TOKENS=32,
        BLOCK_RANKS=triton.next_power_of_2(n=len(ranks)),
    )
    np.testing.assert_array_equal(output.cpu().numpy(), expected)


@pytest.mark.parametrize("world_size", [3, 8, 32])
@pytest.mark.parametrize("token_pattern", ["consecutive", "stride_128", "uint32_wrap"])
def test_rank_random_winner_and_bit_distributions(monkeypatch, world_size, token_pattern):
    benchmark_path = Path(__file__).resolve().parents[3] / "test/benchmark/kernel"
    monkeypatch.syspath_prepend(str(benchmark_path))
    from eplb_decode_benchmark_utils import token_random_values

    tokens = np.arange(65536, dtype=np.uint32)
    if token_pattern == "stride_128":
        tokens *= np.uint32(128)
    elif token_pattern == "uint32_wrap":
        tokens += np.uint32(0xFFFF8000)
    ranks = np.arange(world_size, dtype=np.uint32)[None, :]
    offsets = (tokens[:, None] << np.uint32((world_size - 1).bit_length())) | ranks
    for source_rank in (0, 3, 7):
        scores = token_random_values(source_rank=source_rank | (1 << 32), token_indices=offsets)
        # 仅观测首选分布，不要求实际算子在同分时选最小 rank。
        counts = np.bincount(scores.argmin(axis=1), minlength=world_size)
        probability = 1 / world_size
        tolerance = 6 * np.sqrt(len(tokens) * probability * (1 - probability)) + 1
        assert np.all(np.abs(counts - len(tokens) * probability) <= tolerance), counts
        for bit in range(32):
            deviation = np.abs(((scores >> bit) & 1).mean(axis=0) - 0.5)
            assert deviation.max() < 6 * np.sqrt(0.25 / len(tokens))


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8, 12, 16, 32, 127])
def test_shifted_token_rank_offsets_do_not_overlap(world_size):
    rank_bits = (world_size - 1).bit_length()
    tokens = np.arange(257, dtype=np.uint32)[:, None]
    ranks = np.arange(world_size, dtype=np.uint32)[None, :]
    offsets = (tokens << np.uint32(rank_bits)) | ranks
    assert offsets.dtype == np.uint32
    assert len(np.unique(offsets)) == offsets.size
    np.testing.assert_array_equal(offsets, tokens * np.uint32(1 << rank_bits) + ranks)
    if world_size & (world_size - 1) == 0:
        # 二次幂 world_size 与原来的 token * world_size + rank 编码完全一致。
        np.testing.assert_array_equal(offsets, tokens * np.uint32(world_size) + ranks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("all_tied", [True, False])
def test_gpu_routing_with_forced_random_collisions_keeps_logical_ids(monkeypatch, all_tied):
    from lightllm.common.basemodel.triton_kernel.fused_moe import decode_eplb_topk_ids_hash_targets as hash_targets
    from lightllm.server.router.model_infer.mode_backend.eplb.placement import (
        build_logical_to_rank_physical_map,
        build_logical_to_physical_map,
    )
    from unit_tests.common.fused_moe import test_eplb_decode_hash_targets as reference

    # expert 1/3 分别锚定 rank 1/0，expert 0 在 rank 0/1/2 有副本。
    # 分数全相同则保留初选；rank 0/1 同为最小且初选为 rank 2 时，则保留先遇到的最小分副本。
    placement = [[0, 3, 4], [0, 1, 5], [0, 2, 6], [5, 6, 7]]
    logical = torch.tensor([[1, 3, 0]], dtype=torch.int32, device="cuda").expand(129, 3).contiguous()
    kernel = triton.JITFunction(_eplb_hash_targets_topk_ids_kernel.fn)
    original = (
        "initial_rank_priorities = tl.randint(seed=rank_priority_seed, "
        "offset=rank_random_offsets, n_rounds=10).to(tl.uint32)"
    )
    assert kernel.src.count(original) == 1
    forced_score = (
        "tl.full(shape=(BLOCK_TOKENS, BLOCK_TOP_K), value=0, dtype=tl.uint32)"
        if all_tied
        else "tl.where(condition=initial_ranks == 2, x=1, y=0).to(tl.uint32)"
    )
    kernel._unsafe_update_src(kernel.src.replace(original, "initial_rank_priorities = " + forced_score))
    # 直接通过生产接口测试同分行为，不依赖已经移除的轮数实验脚本及其启动器。
    monkeypatch.setattr(hash_targets, "_eplb_hash_targets_topk_ids_kernel", kernel)
    candidates = build_logical_to_physical_map(placement, 8, current_rank=3, node_world_size=4)
    logical_to_rank_physical_map = build_logical_to_rank_physical_map(placement, 8)
    monkeypatch.setattr(
        reference,
        "rank_priority_reference",
        lambda source_rank, token_index, target_rank, world_size: 0 if all_tied else int(target_rank == 2),
    )
    kwargs = dict(
        logical_topk_ids=logical,
        logical_to_physical_map=torch.tensor(candidates, dtype=torch.int32, device="cuda"),
        logical_to_rank_physical_map=torch.tensor(logical_to_rank_physical_map, dtype=torch.int32, device="cuda"),
        num_physical_experts_per_rank=3,
        current_rank=3,
        node_world_size=4,
    )
    output = hash_targets.eplb_hash_targets_topk_ids(**kwargs)
    initial, expected = reference.routing_reference(logical.tolist(), candidates, logical_to_rank_physical_map, 3, 3)
    torch.testing.assert_close(output.cpu(), torch.tensor(expected, dtype=logical.dtype), rtol=0, atol=0)
    physical_to_logical = torch.tensor(placement, dtype=torch.int64, device="cuda").flatten()
    torch.testing.assert_close(physical_to_logical[output.long()], logical.long(), rtol=0, atol=0)
    torch.testing.assert_close(hash_targets.eplb_hash_targets_topk_ids(**kwargs), output, rtol=0, atol=0)
    reversed_output = hash_targets.eplb_hash_targets_topk_ids(
        **dict(kwargs, logical_topk_ids=logical.flip(dims=(1,)).contiguous())
    )
    torch.testing.assert_close(
        physical_to_logical[reversed_output.long()], logical.flip(dims=(1,)).long(), rtol=0, atol=0
    )
    _, reversed_expected = reference.routing_reference(
        logical.flip(dims=(1,)).tolist(), candidates, logical_to_rank_physical_map, 3, 3
    )
    torch.testing.assert_close(
        reversed_output.cpu(), torch.tensor(reversed_expected, dtype=logical.dtype), rtol=0, atol=0
    )
    if all_tied:
        assert output.tolist() == initial
        torch.testing.assert_close(reversed_output.flip(dims=(1,)), output, rtol=0, atol=0)
    else:
        assert torch.any(reversed_output.flip(dims=(1,)) != output)
