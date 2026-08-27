import json
from types import SimpleNamespace

import pytest
import torch

from lightllm.utils.envs_utils import _get_mtp_draft_backbone_layer_num


def test_deepseek_v4_dspark_layer_count_uses_trailing_swa_layers(tmp_path):
    config = {
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "compress_ratios": [4] * 43 + [0, 0, 0],
        "dspark_block_size": 5,
        "dspark_target_layer_ids": [40, 41, 42],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    assert _get_mtp_draft_backbone_layer_num(str(tmp_path)) == 3


def test_dspark_scratch_cleanup_keeps_page_ids_on_cpu():
    from lightllm.server.router.model_infer.mtp_speculative import utils as mtp_utils
    from lightllm.server.router.model_infer.mtp_speculative.proposers.base import (
        MtpMemIndexesToFree,
    )

    class FakeMemManager:
        def __init__(self):
            self.scratch_frees = []
            self.normal_frees = []

        def free_dspark_swa_block(self, mem_indexes_cpu, pages_cpu):
            self.scratch_frees.append((mem_indexes_cpu.clone(), pages_cpu.clone()))

        def free(self, mem_indexes_cpu):
            self.normal_frees.append(mem_indexes_cpu.clone())

    mem_manager = FakeMemManager()
    backend = SimpleNamespace(model=SimpleNamespace(req_manager=SimpleNamespace(mem_manager=mem_manager)))
    scratch_full = torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
    scratch_pages = torch.tensor([7], dtype=torch.int32)
    normal_full = torch.tensor([20, 21], dtype=torch.int32)

    mtp_utils.free_mem_indexes(
        backend=backend,
        extra_mem_indexes_cpu=[
            MtpMemIndexesToFree(
                mem_indexes_cpu=scratch_full,
                swa_pages_cpu=scratch_pages,
            ),
            MtpMemIndexesToFree(mem_indexes_cpu=normal_full),
        ],
    )

    assert len(mem_manager.scratch_frees) == 1
    torch.testing.assert_close(mem_manager.scratch_frees[0][0], scratch_full)
    torch.testing.assert_close(mem_manager.scratch_frees[0][1], scratch_pages)
    assert len(mem_manager.normal_frees) == 1
    torch.testing.assert_close(mem_manager.normal_frees[0], normal_full)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_build_dspark_swa_index_exposes_history_and_complete_block():
    from lightllm.models.deepseek_v4.triton_kernel.build_dspark_swa_index import (
        build_dspark_swa_index,
    )

    block_size = 3
    window = 4
    req_to_token = torch.tensor(
        [
            list(range(0, 10)),
            list(range(10, 20)),
            [20] * 10,
        ],
        dtype=torch.int32,
        device="cuda",
    )
    full_to_swa = torch.arange(21, dtype=torch.int32, device="cuda") + 100
    req_idx = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32, device="cuda")
    positions = torch.tensor([4, 5, 6, 2, 3, 4], dtype=torch.int32, device="cuda")
    padded_width = 8
    indices = torch.empty((6, padded_width), dtype=torch.int32, device="cuda")
    lengths = torch.empty((6,), dtype=torch.int32, device="cuda")
    write_slots = torch.empty((6,), dtype=torch.int32, device="cuda")
    scratch_pages = torch.tensor([2, 3], dtype=torch.int32, device="cuda")

    build_dspark_swa_index(
        req_idx=req_idx,
        positions=positions,
        req_to_token_indexs=req_to_token,
        full_to_swa_indexs=full_to_swa,
        scratch_pages=scratch_pages,
        swa_index=indices,
        swa_length=lengths,
        swa_write_slots=write_slots,
        window=window,
        block_size=block_size,
        page_size=128,
        hold_req_id=2,
        hold_full_slot=20,
        hold_swa_slot=120,
    )

    expected_indices = torch.tensor(
        [
            [103, 102, 101, 100, 256, 257, 258, -1],
            [103, 102, 101, 100, 256, 257, 258, -1],
            [103, 102, 101, 100, 256, 257, 258, -1],
            [111, 110, 384, 385, 386, -1, -1, -1],
            [111, 110, 384, 385, 386, -1, -1, -1],
            [111, 110, 384, 385, 386, -1, -1, -1],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    torch.testing.assert_close(indices, expected_indices)
    torch.testing.assert_close(
        lengths, torch.tensor([7, 7, 7, 5, 5, 5], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(
        write_slots,
        torch.tensor([256, 257, 258, 384, 385, 386], dtype=torch.int32, device="cuda"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_dspark_swa_block_uses_one_scratch_page_per_request():
    from lightllm.common.kv_cache_mem_manager.deepseek4_mem_manager import (
        DeepseekV4MemoryManager,
    )

    manager = DeepseekV4MemoryManager.__new__(DeepseekV4MemoryManager)
    manager.full_to_swa_indexs = torch.full((32,), -1, dtype=torch.int32, device="cuda")
    manager.swa_page_live_count = torch.zeros((4,), dtype=torch.int32, device="cuda")
    manager._alloc_swa_pages = lambda count: torch.tensor(
        [2, 0],
        dtype=torch.int32,
        pin_memory=True,
    )
    mem_indexes = torch.tensor([3, 4, 5, 8, 9, 10], dtype=torch.int64, device="cuda")

    pages_cpu, pages = manager.alloc_dspark_swa_block(mem_indexes=mem_indexes, block_size=3)

    torch.testing.assert_close(
        manager.full_to_swa_indexs[mem_indexes],
        torch.full((6,), -1, dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(
        manager.swa_page_live_count,
        torch.zeros((4,), dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(pages_cpu, torch.tensor([2, 0], dtype=torch.int32))
