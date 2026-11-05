"""Token 级 Philox 初选的独立参考、设备一致性和统计分散性测试。"""

import ast
from functools import lru_cache

import numpy as np
import pytest
import torch
import triton
import triton.language as tl

from lightllm.common.basemodel.triton_kernel.fused_moe.decode_eplb_topk_ids_hash_targets import (
    _eplb_hash_targets_topk_ids_kernel,
)


@lru_cache(maxsize=32768)
def token_random_reference(source_rank, token_index):
    """纯 Python Philox4x32-10，支持 uint64 offset，独立验证 tl.randint 的第一个输出。"""
    mask = (1 << 32) - 1
    counters = [token_index & mask, (token_index >> 32) & mask, 0, 0]
    key = [source_rank & mask, (source_rank >> 32) & mask]
    for _ in range(10):
        product0 = counters[0] * 0xD2511F53
        product2 = counters[2] * 0xCD9E8D57
        counters = [
            (product2 >> 32) ^ counters[1] ^ key[0],
            product2 & mask,
            (product0 >> 32) ^ counters[3] ^ key[1],
            product0 & mask,
        ]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return counters[0]


@triton.jit
def _token_random_probe(
    token_indices_ptr,
    output_ptr,
    current_rank,
    num_tokens,
    BLOCK_TOKENS: tl.constexpr,
):
    offsets = tl.program_id(axis=0) * BLOCK_TOKENS + tl.arange(start=0, end=BLOCK_TOKENS)  # [BT]
    token_offsets = tl.load(pointer=token_indices_ptr + offsets, mask=offsets < num_tokens, other=0).to(tl.uint32)
    # [BT, 1]：与生产算子的调用一致，独立观测初选随机数，不依赖最终共址结果。
    route_random = tl.randint(seed=current_rank, offset=token_offsets[:, None], n_rounds=10).to(tl.uint32)
    tl.store(pointer=output_ptr + offsets[:, None], value=route_random, mask=offsets[:, None] < num_tokens)


def probe(source_rank, token_indices):
    tokens = torch.tensor(np.asarray(token_indices, dtype=np.uint32).astype(np.int64), device="cuda")
    output = torch.empty(tokens.shape, dtype=torch.int64, device="cuda")
    _token_random_probe[(triton.cdiv(x=tokens.numel(), y=256),)](
        token_indices_ptr=tokens,
        output_ptr=output,
        current_rank=source_rank,
        num_tokens=tokens.numel(),
        BLOCK_TOKENS=256,
    )
    return output.cpu().numpy()


def test_probe_observes_production_random_call():
    def expressions(kernel):
        return [
            ast.dump(node.value)
            for node in ast.walk(ast.parse(kernel.src))
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "route_random" for target in node.targets)
        ]

    production = expressions(_eplb_hash_targets_topk_ids_kernel)
    assert len(production) == 1
    assert expressions(_token_random_probe) == production


def test_philox_zero_counter_and_key_known_answer():
    assert token_random_reference(source_rank=0, token_index=0) == 0x6627E8D5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("source_rank", [0, 3, 7])
def test_randint_matches_reference_at_uint32_boundaries(source_rank):
    # 非整块长度、连续行号、固定步长及最高位，避免有符号取模或边界 mask 错误。
    tokens = np.concatenate(
        (
            np.arange(513, dtype=np.uint32),
            np.arange(513, dtype=np.uint32) * np.uint32(128),
            np.asarray([0, (1 << 31) - 1, 1 << 31, (1 << 32) - 1], dtype=np.uint32),
        )
    )
    expected = np.asarray([token_random_reference(source_rank, int(token)) for token in tokens], dtype=np.int64)
    np.testing.assert_array_equal(probe(source_rank=source_rank, token_indices=tokens), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("source_rank", [0, 3, 7])
def test_randint_is_deterministic_and_spreads_replica_choices(source_rank):
    tokens = np.arange(65536, dtype=np.uint32)
    random_values = probe(source_rank=source_rank, token_indices=tokens)
    np.testing.assert_array_equal(random_values, probe(source_rank=source_rank, token_indices=tokens))
    assert not np.array_equal(random_values, probe(source_rank=source_rank + 1, token_indices=tokens))
    for replica_count in (2, 3, 4, 8, 16, 32):
        counts = np.bincount(random_values % replica_count, minlength=replica_count)
        probability = 1 / replica_count
        tolerance = 6 * np.sqrt(len(tokens) * probability * (1 - probability)) + 1
        assert np.all(np.abs(counts - len(tokens) * probability) <= tolerance), counts
    bit_means = ((random_values[:, None] >> np.arange(32)) & 1).mean(axis=0)
    assert np.max(np.abs(bit_means - 0.5)) <= 6 * np.sqrt(0.25 / len(tokens))
