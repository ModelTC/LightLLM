from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import full_state_test_fields

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from typing import TYPE_CHECKING
import pytest
import torch

from unit_tests.common.att_resources.model_helpers import main_linear_state
from unit_tests.common.att_resources.test_model_resources import make_resources

if TYPE_CHECKING:
    from easydict import EasyDict


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("glm", [False, True])
def test_main_and_draft_transfers_copy_pd_tails_and_reset_dp_tails(
    glm: bool, pd_args: EasyDict, monkeypatch: pytest.MonkeyPatch,
):
    pd_args.pd_kv_page_size = 128
    source, _ = make_resources(
        glm=glm,
        tp=1,
        rank=0,
        mtp=2,
        draft_layers=2,
    )
    target, _ = make_resources(
        glm=glm,
        tp=1,
        rank=0,
        mtp=2,
        draft_layers=2,
    )
    for index, role in enumerate(("main", "draft0")):
        state = getattr(source, f"{role}_mla_packed_bf16_state" if glm else f"{role}_gqa_bf16_state")
        for field in full_state_test_fields(state, full_state_test_buffers(state)).values():
            if "token" in field.axes:
                field.tensor.fill_(11 + index)
        if glm:
            getattr(source, f"{role}_kpool_c4_bf16_state").tail_buffer[:, 0].fill_(7 + index)
            getattr(target, f"{role}_kpool_c4_bf16_state").tail_buffer[:, 2].fill_(19)
    main_linear_state(source).conv_state.fill_(3)
    main_linear_state(source).ssm_state.fill_(5)
    page = torch.empty(source.get_pd_page_bytes(), dtype=torch.uint8, device="cuda")
    for kind, getter in (
        ("kv", "get_pd_token_page_layouts"),
        ("att_state", "get_pd_runtime_page_layouts"),
    ):
        source.write_to_pd_page(
            pd_page_buffers=page.unsqueeze(0),
            pd_page_index=0,
            kind=kind,
            layouts=getattr(source, getter)(),
            **({"token_indices": torch.tensor([0, 1], dtype=torch.long, device="cuda")} if kind == "kv" else {"req_idx": 0, "state_token_len": 8}),
        )
        target.read_from_pd_page(
            pd_page_buffers=page.unsqueeze(0),
            pd_page_index=0,
            kind=kind,
            layouts=getattr(target, getter)(),
            **({"token_indices": torch.tensor([8, 9], dtype=torch.long, device="cuda")} if kind == "kv" else {"req_idx": 1, "state_token_len": 8}),
        )
    src_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    dst_indices = torch.tensor([10, 11], dtype=torch.int64, device="cuda")

    def reject_host_list(tensor: torch.Tensor) -> list[int]:
        pytest.fail("DP manager and States must preserve Tensor indices")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", reject_host_list)
        target.copy_from_dp(
            remote_att_manager=source,
            src_token_indices=src_indices,
            dst_token_indices=dst_indices,
            src_big_page_ids=[],
            dst_big_page_ids=[],
            src_req_idx=0,
            dst_req_idx=2,
        )
    for index, role in enumerate(("main", "draft0")):
        state = getattr(target, f"{role}_mla_packed_bf16_state" if glm else f"{role}_gqa_bf16_state")
        for field in full_state_test_fields(state, full_state_test_buffers(state)).values():
            if "token" in field.axes:
                assert (field.tensor[:, 8:12] == 11 + index).all()
        if glm:
            tail = getattr(target, f"{role}_kpool_c4_bf16_state").tail_buffer
            expected_tail = torch.zeros_like(tail[:, 1])
            for position in range(4, 8):
                expected_tail[:, position % tail.shape[2]].fill_(7 + index)
            assert torch.equal(tail[:, 1], expected_tail)  # PD restores only the prefill logical window.
            assert not tail[:, 2].any()  # DP fetch resets the aligned prefix.

    slot = source.bounded_att_big_page_allocator.alloc(1)[0]
    source.save_bounded_att_runtime_to_pages(
        req_indices=[0],
        state_token_lens=[8],
        page_kind="big",
        page_slot_ids=[slot],
    )
    torch.cuda.synchronize()
    cpu = torch.empty((1, source.get_cpu_page_bytes()), dtype=torch.uint8, pin_memory=True)
    source.write_req_to_cpu(
        token_indices=torch.arange(8, dtype=torch.long, device="cuda"),
        cpu_page_buffers=cpu,
        cpu_page_indices=[0],
        layouts=source.get_cpu_page_layouts(),
        big_page_slot_ids=[slot],
    )
    torch.cuda.synchronize()
    dst_slot = target.bounded_att_big_page_allocator.alloc(1)[0]
    target.read_req_from_cpu(
        cpu_page_buffers=cpu,
        cpu_page_indices=[0],
        token_indices=torch.arange(
            start=8,
            end=16,
            dtype=torch.long,
            device="cuda",
        ),
        state_token_len=8,
        layouts=target.get_cpu_page_layouts(),
        big_page_slot_ids=[dst_slot],
        req_idx=1,
    )
    for index, role in enumerate(("main", "draft0")):
        state = getattr(target, f"{role}_mla_packed_bf16_state" if glm else f"{role}_gqa_bf16_state")
        for field in full_state_test_fields(state, full_state_test_buffers(state)).values():
            if "token" in field.axes:
                assert (field.tensor[:, 8:16] == 11 + index).all()
        if glm:
            assert not getattr(target, f"{role}_kpool_c4_bf16_state").tail_buffer[:, 1].any()
            assert not getattr(target, f"{role}_kpool_c4_bf16_state").tail_buffer[:, 2].any()
