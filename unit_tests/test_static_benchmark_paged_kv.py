import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_BENCHMARK_DIR = REPO_ROOT / "test" / "benchmark" / "static_inference"
for path in (REPO_ROOT, STATIC_BENCHMARK_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from static_benchmark import (
    BenchmarkCase,
    StaticBenchmarkExecutor,
    decode_profile_batch_divisor,
    resolve_batch_max_prefill_cases,
)


class FakeMemManager:
    def __init__(self, page_size: int):
        self.page_size = page_size
        self.next_index = 0
        self.alloc_sizes = []

    def alloc(self, size: int) -> torch.Tensor:
        assert size % self.page_size == 0
        self.alloc_sizes.append(size)
        indexes = torch.arange(self.next_index, self.next_index + size, dtype=torch.int32)
        self.next_index += size
        return indexes


def make_executor(page_size: int = 4):
    mem_manager = FakeMemManager(page_size=page_size)
    req_manager = SimpleNamespace(
        mem_manager=mem_manager,
        req_to_token_indexs=torch.full((4, 32), -1, dtype=torch.int32),
    )
    model = SimpleNamespace(mem_manager=mem_manager, req_manager=req_manager)
    args = SimpleNamespace(mtp_mode=None, mtp_step=0)
    executor = StaticBenchmarkExecutor(args=args, model=model, draft_models=[], token_source=None)
    return executor, mem_manager, req_manager


def assert_complete_physical_pages(indexes: torch.Tensor, page_size: int):
    for start in range(0, indexes.numel(), page_size):
        page = indexes[start : start + page_size]
        assert torch.all(page // page_size == page[0] // page_size)


def test_prefill_allocates_and_reuses_complete_pages_per_request():
    executor, mem_manager, req_manager = make_executor(page_size=4)
    req_idx = torch.tensor([0, 1], dtype=torch.int32)

    model_input = executor._make_prefill_input(
        token_chunk=np.arange(6, dtype=np.int64).reshape(2, 3),
        req_idx=req_idx,
        ready_cache_len=0,
    )

    assert mem_manager.alloc_sizes == [4, 4]
    assert model_input.b_is_decode_req.tolist() == [False, False]
    assert executor._hold_kv_len_by_req == {0: 4, 1: 4}
    assert_complete_physical_pages(req_manager.req_to_token_indexs[0, :4], page_size=4)
    assert_complete_physical_pages(req_manager.req_to_token_indexs[1, :4], page_size=4)

    # 两个请求都还在已有页内继续 decode，不应产生新的分配。
    executor._make_decode_input(
        batch_size=2,
        req_idx=req_idx,
        mtp_index=torch.zeros(2, dtype=torch.int32),
        seq_len=torch.tensor([4, 4], dtype=torch.int32),
        input_ids=torch.tensor([1, 2], dtype=torch.int64),
        max_kv_seq_len=4,
    )
    assert mem_manager.alloc_sizes == [4, 4]

    # 跨过页边界后，每个请求分别追加一整页。
    executor._make_decode_input(
        batch_size=2,
        req_idx=req_idx,
        mtp_index=torch.zeros(2, dtype=torch.int32),
        seq_len=torch.tensor([5, 5], dtype=torch.int32),
        input_ids=torch.tensor([3, 4], dtype=torch.int64),
        max_kv_seq_len=5,
    )
    assert mem_manager.alloc_sizes == [4, 4, 4, 4]
    assert executor._hold_kv_len_by_req == {0: 8, 1: 8}
    assert_complete_physical_pages(req_manager.req_to_token_indexs[0, :8], page_size=4)
    assert_complete_physical_pages(req_manager.req_to_token_indexs[1, :8], page_size=4)


def test_mtp_expanded_rows_allocate_once_per_request():
    executor, mem_manager, req_manager = make_executor(page_size=4)
    executor.args.mtp_mode = "eagle_with_att"
    executor.args.mtp_step = 3

    model_input = executor._make_decode_input(
        batch_size=4,
        req_idx=torch.tensor([0, 0, 1, 1], dtype=torch.int32),
        mtp_index=torch.tensor([0, 1, 0, 1], dtype=torch.int32),
        seq_len=torch.tensor([3, 4, 3, 4], dtype=torch.int32),
        input_ids=torch.tensor([1, 2, 3, 4], dtype=torch.int64),
        max_kv_seq_len=4,
        extra_kv_len=2,
    )

    assert model_input.batch_size == 4
    assert mem_manager.alloc_sizes == [8, 8]
    assert executor._hold_kv_len_by_req == {0: 8, 1: 8}
    assert_complete_physical_pages(req_manager.req_to_token_indexs[0, :8], page_size=4)
    assert_complete_physical_pages(req_manager.req_to_token_indexs[1, :8], page_size=4)


def test_profile_capacity_uses_page_aligned_per_request_cost():
    args = SimpleNamespace(
        mtp_mode=None,
        mtp_step=0,
        page_size=4,
        max_batch_size=0,
        batch_max_tokens=64,
    )
    prefill_case = BenchmarkCase(
        name="prefill",
        stage="prefill",
        batch_size=8,
        context_len=5,
        output_len=0,
        prefill_uncached_len=5,
        prefill_batch_size_by_batch_max_tokens=8,
    )

    resolved = resolve_batch_max_prefill_cases(args, [prefill_case], profiled_max_total_token_num=24)

    assert resolved[0].batch_size == 3
    assert resolved[0].profiled_batch_divisor == 8

    decode_case = BenchmarkCase(
        name="decode",
        stage="decode",
        batch_size=1,
        context_len=6,
        output_len=2,
    )
    assert decode_profile_batch_divisor(args, decode_case) == 20

    args.mtp_mode = "eagle_with_att"
    args.mtp_step = 3
    assert decode_profile_batch_divisor(args, decode_case) == 24
