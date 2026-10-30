"""Verification commits State-owned recurrent data without model cursor aliases."""

from __future__ import annotations


from types import SimpleNamespace

import pytest
import torch

from lightllm.common.att_resources import AttResourceManager
from lightllm.common.att_resources.bounded_att import GDNState
from lightllm.common.att_resources.allocators import StatePageAllocator
from lightllm.server.router.model_infer.mtp_speculative.utils import verify_mtp_tokens
from unit_tests.common.att_resources.test_bounded_state import make_state


def make_manager(device: str | torch.device | int):
    manager = AttResourceManager(request_capacity=3)
    manager.set_att_state("main", make_state(device=device))
    manager.set_att_state("main", make_state(GDNState, device=device))
    manager.set_att_state("draft0", make_state(GDNState, device=device))
    manager.bounded_att_big_page_allocator = StatePageAllocator(3)
    manager.bounded_att_small_page_allocator = StatePageAllocator(2)
    manager.finish_state_creation()
    return manager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_commit_selects_state_for_one_role():
    manager = make_manager("cuda")
    for state in manager.get_bounded_attention_states():
        for slot in range(3):
            state.ssm_state[:, slot].fill_(slot + 10)
        state.mtp_state_index.fill_(1)
    manager.commit_bounded_att_state(
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([3], dtype=torch.int32, device="cuda"),
        role="main",
    )
    for state in (manager.main_kda_conv_fp32_ssm_fp32_state, manager.main_gdn_conv_fp32_ssm_fp32_state):
        assert state.mtp_state_index.tolist() == [2, 1, 1]
        selected = state.mtp_state_index[0].item()
        assert (state.ssm_state[:, selected] == 12).all()
    assert manager.draft0_gdn_conv_fp32_ssm_fp32_state.mtp_state_index.tolist() == [1, 1, 1]
    manager.commit_bounded_att_state(
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([1], dtype=torch.int32, device="cuda"),
        role="draft0",
    )
    assert manager.draft0_gdn_conv_fp32_ssm_fp32_state.mtp_state_index.tolist() == [0, 1, 1]
    assert manager.main_kda_conv_fp32_ssm_fp32_state.mtp_state_index.tolist() == [2, 1, 1]
    assert manager.main_gdn_conv_fp32_ssm_fp32_state.mtp_state_index.tolist() == [2, 1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_commit_without_recurrent_state_and_invalid_role():
    manager = AttResourceManager(request_capacity=3)
    manager.finish_state_creation()
    manager.commit_bounded_att_state(
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([3], dtype=torch.int32, device="cuda"),
        role="main",
    )
    manager.commit_bounded_att_state(
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([3], dtype=torch.int32, device="cuda"),
        role="draft0",
    )
    with pytest.raises(AssertionError, match="unknown model role"):
        manager.commit_bounded_att_state(
            torch.tensor([0], dtype=torch.int32, device="cuda"),
            torch.tensor([3], dtype=torch.int32, device="cuda"),
            role="target",
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("widths", [(3, 3, 3), (3, 1, 2)])
def test_verify_commits_main_states_for_fixed_and_dynamic_widths(widths: tuple[int, ...]):
    manager = make_manager("cuda")
    backend = SimpleNamespace(
        model=SimpleNamespace(
            att_resource_manager=manager,
            req_manager=SimpleNamespace(
                req_sampling_params_manager=SimpleNamespace(
                    req_to_next_token_ids=torch.tensor([[-1, 10, 11]] * 3, device="cuda")
                )
            ),
        )
    )
    reqs, starts, tokens, expected_mask = [], [], [], []
    for req, width, accepted in zip((2, 0, 1), widths, (3, 1, 2)):
        starts.append(len(reqs))
        reqs.extend([req] * width)
        tokens.extend(99 if offset == accepted - 1 else 10 + offset for offset in range(width))
        expected_mask.extend(int(offset < accepted) for offset in range(width))
    accept_lengths, accepted_index = verify_mtp_tokens(
        backend=backend,
        next_token_ids=torch.tensor(tokens, device="cuda"),
        b_req_idx=torch.tensor(reqs, dtype=torch.int32, device="cuda"),
        b_req_mtp_start_loc=torch.tensor(starts, dtype=torch.int32, device="cuda"),
        b_mtp_index=torch.tensor([step for width in widths for step in range(width)], dtype=torch.int32, device="cuda"),
    )
    assert accept_lengths.tolist() == [3, 1, 2]
    assert accepted_index.tolist() == expected_mask
    assert manager.main_kda_conv_fp32_ssm_fp32_state.mtp_state_index.tolist() == [0, 1, 2]
    assert manager.main_gdn_conv_fp32_ssm_fp32_state.mtp_state_index.tolist() == [0, 1, 2]
    assert manager.draft0_gdn_conv_fp32_ssm_fp32_state.mtp_state_index.tolist() == [0, 0, 0]
