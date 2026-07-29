import sys
from types import SimpleNamespace

import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.moonep_impl import (
    FuseMoeMoonEP,
)


def test_moonep_padding_uses_zero_weight_unique_dummy_routes():
    hidden = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    weights = torch.full((3, 2), 0.5)
    expert_ids = torch.tensor([[2, 3], [4, 5], [6, 7]], dtype=torch.int64)

    padded_hidden, padded_weights, padded_ids = FuseMoeMoonEP._pad_inputs(
        None,
        hidden,
        weights,
        expert_ids,
        capacity=5,
    )

    assert padded_hidden.shape == (5, 4)
    assert padded_weights.shape == (5, 2)
    assert padded_ids.shape == (5, 2)
    assert padded_hidden.dtype == torch.bfloat16
    assert padded_weights.dtype == torch.float32
    assert padded_ids.dtype == torch.int32
    assert torch.equal(padded_hidden[:3], hidden)
    assert torch.count_nonzero(padded_hidden[3:]) == 0
    assert torch.count_nonzero(padded_weights[3:]) == 0
    assert torch.equal(padded_ids[3:], torch.tensor([[0, 1], [0, 1]], dtype=torch.int32))


def test_moonep_vm_group_indices_split_sources_slots_and_tail():
    cu_seqlens = torch.tensor([0, 128, 128, 256, 384], dtype=torch.int32)

    sources, slots = FuseMoeMoonEP._vm_group_indices(
        cu_seqlens,
        num_rows=512,
        num_experts=3,
    )

    assert torch.equal(sources[:128], torch.ones(128, dtype=torch.int32))
    assert torch.count_nonzero(sources[128:] != -1) == 0
    assert torch.count_nonzero(slots[:128] != -1) == 0
    assert torch.equal(slots[128:256], torch.zeros(128, dtype=torch.int32))
    assert torch.equal(slots[256:384], torch.ones(128, dtype=torch.int32))
    assert torch.count_nonzero(slots[384:] != -1) == 0


def test_moonep_rejects_more_tokens_than_the_configured_buffer():
    hidden = torch.zeros((3, 4), dtype=torch.bfloat16)
    weights = torch.zeros((3, 2), dtype=torch.float32)
    expert_ids = torch.zeros((3, 2), dtype=torch.int32)

    try:
        FuseMoeMoonEP._pad_inputs(None, hidden, weights, expert_ids, capacity=2)
    except ValueError as exc:
        assert "exceeding its configured capacity" in str(exc)
    else:
        raise AssertionError("expected MoonEP capacity validation to fail")


def test_moonep_grouped_gemm_masks_inactive_regions_without_negative_deepgemm_ids(monkeypatch):
    def grouped_gemm(a, b, output, group_ids):
        assert torch.all(group_ids >= 0)
        for row, group_id in enumerate(group_ids.tolist()):
            output[row] = a[row] @ b[group_id].T

    monkeypatch.setitem(
        sys.modules,
        "deep_gemm",
        SimpleNamespace(m_grouped_bf16_gemm_nt_contiguous=grouped_gemm),
    )
    hidden = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.bfloat16)
    source_weight = torch.tensor(
        [[[1, 0], [0, 1]], [[2, 0], [0, 2]]],
        dtype=torch.bfloat16,
    )
    slot_weight = torch.tensor([[[3, 0], [0, 3]]], dtype=torch.bfloat16)
    source_groups = torch.tensor([0, 1, -1, -1], dtype=torch.int32)
    slot_groups = torch.tensor([-1, -1, 0, -1], dtype=torch.int32)
    output = torch.empty_like(hidden)

    FuseMoeMoonEP._grouped_gemm(
        hidden,
        source_weight,
        slot_weight,
        source_groups,
        slot_groups,
        output,
    )

    expected = torch.tensor([[1, 2], [6, 8], [15, 18], [0, 0]], dtype=torch.bfloat16)
    assert torch.equal(output, expected)
