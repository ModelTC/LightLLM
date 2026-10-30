from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import bind_test_fp8_calibration

from unit_tests.common.att_resources.model_helpers import full_state_test_fields

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from unit_tests.common.att_resources.model_helpers import read_kv_storage


from typing import TYPE_CHECKING
from math import prod

import pytest
import torch

from lightllm.common.att_resources.full_att import (
    GroupQuantizedAttentionConfig,
    GroupQuantizedFullAttentionState,
    StaticFP8AttentionConfig,
    StaticFP8FullAttentionState,
    PackedMLAAttentionConfig,
    PackedMLAFullAttentionState,
)
from lightllm.common.att_resources.types import (
    AxisSlice, StateTransferLayout, TransferRegion,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from lightllm.common.att_resources.full_att.base_state import FullAttentionState


KINDS = ("int4", "int8", "fp8_tensor", "fp8_head", "packed", "packed_fp8")


def make_quant_state(kind: str, device: str | torch.device | int):
    if kind in ("int4", "int8"):
        config = GroupQuantizedAttentionConfig(
            logical_layer_groups=((1,), (3,)),
            global_head_num=2,
            head_dim=16,
            dtype=torch.float16,
            bits=int(kind[-1]),
        )
        prepared_full_state = GroupQuantizedFullAttentionState(config, page_size=4, device=device)
        prepared_full_state.initialize_storage(8)
        return prepared_full_state
    if kind.startswith("fp8_"):
        per_head = kind == "fp8_head"
        config = StaticFP8AttentionConfig(
            logical_layer_groups=((1,), (3,)),
            global_head_num=2,
            head_dim=16,
            dtype=torch.uint8,
            scale_mode="per_head" if per_head else "per_tensor",
        )
        scales = torch.arange(
            start=1,
            end=9 if per_head else 5,
            dtype=torch.float32,
            device=device,
        ).reshape(2, -1) / 16
        prepared_full_state = StaticFP8FullAttentionState(config, page_size=4, device=device)
        bind_test_fp8_calibration(
            state=prepared_full_state, scales=scales,
            q_scales=torch.ones((2, config.head_num), dtype=torch.float32) if per_head else None,
        )
        prepared_full_state.initialize_storage(8)
        return prepared_full_state
    fp8 = kind == "packed_fp8"
    config = PackedMLAAttentionConfig(
        logical_layer_groups=((1,), (3,)),
        global_head_num=1,
        head_dim=800 if fp8 else 648,
        dtype=torch.uint8 if fp8 else torch.bfloat16,
        fp8_nope_dim=512 if fp8 else 0,
    )
    prepared_full_state = PackedMLAFullAttentionState(config, page_size=4, device=device)
    prepared_full_state.initialize_storage(8)
    return prepared_full_state


def transfer_layout(state: FullAttentionState):
    offset, regions = 0, []
    for name, field in full_state_test_fields(state, full_state_test_buffers(state)).items():
        shape = tuple(4 if axis == "token" else dim for axis, dim in zip(field.axes, field.tensor.shape))
        itemsize = field.tensor.element_size()
        offset = (offset + itemsize - 1) // itemsize * itemsize
        regions.append(
            TransferRegion(
                field=name,
                byte_start=offset,
                byte_end=offset + prod(shape) * itemsize,
                dtype=field.tensor.dtype,
                axes=field.axes,
                shape=shape,
                axis_slices=tuple(
                    AxisSlice(
                        axis=axis,
                        src_start=0,
                        dst_start=0,
                        length=dim,
                    )
                    for axis, dim in zip(field.axes, shape) if axis not in ("layer", "token")
                ),
            )
        )
        offset += prod(shape) * itemsize
    return StateTransferLayout(
        resource_id=state.resource_id,
        tokens_per_page=4,
        byte_start=0,
        byte_end=(offset + 15) // 16 * 16,
        regions=tuple(regions),
    )


def assert_token_fields_equal(
    source: FullAttentionState,
    source_ids: Sequence[int] | torch.Tensor,
    target: FullAttentionState,
    target_ids: Sequence[int] | torch.Tensor,
):
    src = full_state_test_fields(source, full_state_test_buffers(source))
    dst = full_state_test_fields(target, full_state_test_buffers(target))
    for name in src:
        assert torch.equal(
            src[name].tensor[:, source_ids].contiguous().view(torch.uint8),
            dst[name].tensor[:, target_ids].contiguous().view(torch.uint8).to(src[name].tensor.device),
        ), name


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_quantized_cpu_pd_dp_and_token_copy_include_every_field(
    kind: str, device: str | torch.device | int, monkeypatch: pytest.MonkeyPatch,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    source, target = make_quant_state(kind, device), make_quant_state(kind, device)
    for i, field in enumerate(full_state_test_fields(source, full_state_test_buffers(source)).values()):
        field.tensor.fill_(i + 1)
        field.tensor[:, 0].fill_(i + 11)
    if device == "cuda":
        source.copy_full_att_tokens(torch.tensor([0, 2], dtype=torch.long, device=device), torch.tensor([4, 5], dtype=torch.long, device=device))
        assert_token_fields_equal(
            source=source,
            source_ids=[0, 2],
            target=source,
            target_ids=[4, 5],
        )
    layout = transfer_layout(source)
    cpu = torch.zeros((2, layout.byte_end), dtype=torch.uint8, pin_memory=device == "cuda")
    if device == "cpu":
        with pytest.raises(AssertionError, match="CUDA State storage"):
            source.write_to_cpu(
                token_indices=torch.tensor([0, 1, 2, 3, 4], device=device),
                cpu_page_buffers=cpu,
                cpu_page_indices=[1, 0],
                layout=layout,
            )
    else:
        source.write_to_cpu(
            token_indices=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long, device=device),
            cpu_page_buffers=cpu,
            cpu_page_indices=[1, 0],
            layout=layout,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        target.read_from_cpu(
            cpu_page_buffers=cpu,
            cpu_page_indices=[1, 0],
            token_indices=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long, device=device),
            layout=layout,
        )
        assert_token_fields_equal(
            source=source,
            source_ids=[0, 1, 2, 3, 4],
            target=target,
            target_ids=[1, 2, 3, 4, 5],
        )
    page = torch.empty(layout.byte_end, dtype=torch.uint8, device=device)
    if device == "cpu":
        with pytest.raises(AssertionError, match="PD pages must reside on CUDA"):
            source.write_to_pd_buffer(
                token_indices=torch.tensor([0, 3], dtype=torch.long, device=device),
                pd_page_buffers=page.unsqueeze(0),
                pd_page_index=0,
                layout=layout,
            )
        with pytest.raises(AssertionError, match="PD pages must reside on CUDA"):
            target.read_from_pd_buffer(
                pd_page_buffers=page.unsqueeze(0),
                token_indices=torch.tensor([6, 7], dtype=torch.long, device=device),
                pd_page_index=0,
                layout=layout,
            )
    else:
        source.write_to_pd_buffer(
            token_indices=torch.tensor([0, 3], dtype=torch.long, device=device),
            pd_page_buffers=page.unsqueeze(0),
            pd_page_index=0,
            layout=layout,
        )
        target.read_from_pd_buffer(
            pd_page_buffers=page.unsqueeze(0),
            token_indices=torch.tensor([6, 7], dtype=torch.long, device=device),
            pd_page_index=0,
            layout=layout,
        )
        assert_token_fields_equal(
            source=source,
            source_ids=[0, 3],
            target=target,
            target_ids=[6, 7],
        )
    src_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    dst_indices = torch.tensor([0, 2], dtype=torch.int64, device=device)

    def reject_host_list(tensor: torch.Tensor) -> list[int]:
        pytest.fail("DP token indices must not be converted to Python lists")

    tensor_to = torch.Tensor.to

    def reject_index_conversion(tensor: torch.Tensor, *args, **kwargs):
        if tensor is src_indices or tensor is dst_indices:
            pytest.fail("DP State must use the supplied CUDA indices directly")
        return tensor_to(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "tolist", reject_host_list)
        patch.setattr(torch.Tensor, "to", reject_index_conversion)
        if device == "cpu":
            with pytest.raises(AssertionError, match="DP token indices must reside on CUDA"):
                target.copy_tokens_from_other_dp(
                    src_token_indices=src_indices, dst_token_indices=dst_indices, remote_dp_full_att_state=source,
                )
            return
        target.copy_tokens_from_other_dp(
            src_token_indices=src_indices,
            dst_token_indices=dst_indices,
            remote_dp_full_att_state=source,
        )
    assert_token_fields_equal(
        source=source,
        source_ids=[0, 1],
        target=target,
        target_ids=[0, 2],
    )


@pytest.mark.parametrize("kind", KINDS)
def test_cuda_compute_write_preserves_unrelated_fields(kind: str):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    state = make_quant_state(kind, "cuda")
    for storage_buffer in full_state_test_buffers(state).values():
        storage_buffer.zero_()
    if isinstance(state, PackedMLAFullAttentionState):
        state.get_layer_view(1).indexer_k.fill_(29)
    dims = state.config.kv_dim if isinstance(state, PackedMLAFullAttentionState) else state.config.head_dim
    heads = 1 if isinstance(state, PackedMLAFullAttentionState) else 2 * state.config.head_num
    kv = torch.randn((2, heads, dims), dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([1, 6], dtype=torch.int32, device="cuda")
    state.write_full_att_kv(1, ids, kv)
    if isinstance(state, GroupQuantizedFullAttentionState):
        quant = read_kv_storage(state)[0, ids].reshape(
            shape=(2, heads, -1, state.config.group_size * state.config.bits // 8),
        )
        if state.config.bits == 4:
            raw = quant.to(torch.uint8)
            quant = torch.stack(((raw & 15).to(torch.int16) - 7, (raw >> 4).to(torch.int16) - 7), -1).flatten(-2)
        dequant = (quant.float() * torch.cat((state.k_scales, state.v_scales), dim=2)[0, ids, ..., None]).reshape_as(kv)
        assert torch.all((dequant - kv.float()).abs() <= torch.cat((state.k_scales, state.v_scales), dim=2)[0, ids].amax() * 2)
    elif isinstance(state, StaticFP8FullAttentionState):
        scales = state.scales[0]
        if state.config.scale_mode == "per_tensor":
            scales = scales.repeat_interleave(state.config.head_num)
        dequant = read_kv_storage(state)[0, ids].view(torch.float8_e4m3fn).float() * scales[None, :, None]
        torch.testing.assert_close(
            actual=dequant,
            expected=kv.float(),
            rtol=0.13,
            atol=0.03,
        )
    else:
        view = state.get_layer_view(1)
        assert (view.indexer_k == 29).all()
        if not state.config.fp8_nope_dim:
            assert torch.equal(view.kv[ids], kv)
        else:
            fields = full_state_test_fields(state)
            assert torch.isfinite(fields["kv_scale"].tensor[0, ids]).all()
            assert torch.equal(fields["kv_rope"].tensor[0, ids], kv[..., 512:])
    assert not read_kv_storage(state)[1].any()


@pytest.mark.parametrize("kind", ["fp8_tensor", "fp8_head"])
def test_deferred_storage_preserves_real_fp8_calibration(kind: str):
    source = make_quant_state(kind, "cpu")
    state = StaticFP8FullAttentionState(
        source.config, page_size=source.page_size, device="meta",
    )
    assert state.scales is None and state.q_scales is None
    bind_test_fp8_calibration(state=state, scales=source.scales, q_scales=source.q_scales)
    assert read_kv_storage(state) is None and not state.scales.is_meta
    assert torch.equal(state.scales, source.scales)
    state.device = torch.device("cpu")
    state.initialize_storage(source.capacity)
    assert torch.equal(state.scales, source.scales)
    assert torch.equal(torch.from_numpy(state.cpu_scales), source.scales)
