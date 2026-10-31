from __future__ import annotations

from unit_tests.common.att_resources.model_helpers import full_state_test_fields

from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

from unit_tests.common.att_resources.model_helpers import read_kv_storage


from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state.full_att import (
    FullAttentionState,
    GQAFullAttentionState,
    MLAFullAttentionState,
    GQAAttentionConfig,
    MLAAttentionConfig,
)
from lightllm.common.attention_state.types import (
    AxisMapping, StateTransferLayout, TransferRegion,
)


def full_state(cls: type[FullAttentionState] = GQAFullAttentionState, device: str | torch.device | int = "cpu"):
    if cls is MLAFullAttentionState:
        config = MLAAttentionConfig(
            logical_layer_groups=((2,), (5,)), latent_dim=4, dtype=torch.float32,
        )
    else:
        config = GQAAttentionConfig(
            logical_layer_groups=((2,), (5,)), global_head_num=1, head_dim=4, dtype=torch.float32,
        )
    prepared_full_state = cls(config, page_size=4, device=device)
    prepared_full_state.allocate_storage(12)
    return prepared_full_state


def layout_for(state: FullAttentionState):
    regions = []
    offset = 16
    for name in full_state_test_fields(state, full_state_test_buffers(state)):
        regions.append(
            TransferRegion(
                field=name,
                byte_start=offset,
                byte_end=offset + 128,
                dtype=torch.float32,
                axes=("token", "layer", "head", "dim"),
                shape=(4, 2, 1, 4),
                axis_mappings=(
                    AxisMapping(
                        axis="head",
                        local_start=0,
                        external_start=0,
                        length=1,
                    ),
                    AxisMapping(
                        axis="dim",
                        local_start=0,
                        external_start=0,
                        length=4,
                    ),
                ),
            )
        )
        offset += 128
    return StateTransferLayout(
        state_name=state.state_name,
        tokens_per_page=4,
        byte_start=0,
        byte_end=offset + 16,
        regions=tuple(regions),
    )


@pytest.mark.parametrize("cls", [GQAFullAttentionState, MLAFullAttentionState])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_full_attention_write_copy_and_pd(
    cls: type[FullAttentionState], device: str | torch.device | int, monkeypatch: pytest.MonkeyPatch
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    state = full_state(cls, device)
    for storage_buffer in full_state_test_buffers(state).values():
        storage_buffer.zero_()
    ids = torch.tensor([1, 3], dtype=torch.int32, device=device)
    data = torch.arange(2 * read_kv_storage(state).shape[2] * 4, dtype=torch.float32, device=device).reshape(
        2, read_kv_storage(state).shape[2], 4
    )
    if device == "cuda":
        state.write_full_att_kv(2, ids, data)
        state.write_full_att_kv(5, ids, data + 100)
    else:
        with pytest.raises(AssertionError, match="token indices must reside on CUDA"):
            state.write_full_att_kv(2, ids, data)
        # CPU 仅构造复制和传输测试的数据，不执行推理 KV 写入。
        for layer, values in ((0, data), (1, data + 100)):
            head_start = 0
            for buffer in full_state_test_buffers(state).values():
                head_end = head_start + buffer.shape[2]
                buffer[layer, ids.long()] = values[:, head_start:head_end]
                head_start = head_end
    def reject_host_list(tensor: torch.Tensor):
        raise AssertionError("token copies must not convert indices to Python lists")

    if device == "cuda":
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "tolist", reject_host_list)
            state.copy_full_att_tokens(ids, torch.tensor([8, 9], dtype=torch.int32, device=device))
        assert torch.equal(read_kv_storage(state)[:, [1, 3]], read_kv_storage(state)[:, [8, 9]])
    else:
        with pytest.raises(AssertionError, match="token indices must reside on CUDA"):
            state.copy_full_att_tokens(ids, torch.tensor([8, 9], dtype=torch.int32, device=device))
    layout = layout_for(state)
    pages = torch.full(
        size=(3, layout.byte_end),
        fill_value=255,
        dtype=torch.uint8,
        device=device,
    )
    if device == "cpu":
        with pytest.raises(AssertionError, match="PD pages must reside on CUDA"):
            state.write_to_pd_page(
                token_indices=ids,
                pd_page_buffers=pages,
                pd_page_index=1,
                layout=layout,
            )
        with pytest.raises(AssertionError, match="PD pages must reside on CUDA"):
            state.read_from_pd_page(
                pd_page_buffers=pages,
                token_indices=ids,
                pd_page_index=1,
                layout=layout,
            )
        return
    state.write_to_pd_page(
        token_indices=ids,
        pd_page_buffers=pages,
        pd_page_index=1,
        layout=layout,
    )
    target = full_state(cls, device)
    for storage_buffer in full_state_test_buffers(target).values():
        storage_buffer.zero_()
    target.read_from_pd_page(
        pd_page_buffers=pages,
        token_indices=torch.tensor([5, 6], dtype=torch.long, device=device),
        pd_page_index=1,
        layout=layout,
    )
    assert torch.equal(read_kv_storage(target)[:, [5, 6]], read_kv_storage(state)[:, [1, 3]])
    assert (pages[0] == 255).all() and (pages[2] == 255).all()
    assert not read_kv_storage(target)[:, 0].any()


@pytest.mark.parametrize("state_offset, extra_page_bytes", [(0, 0), (32, 64)])
def test_cpu_partial_last_page(state_offset: int, extra_page_bytes: int):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    state = full_state(device="cuda")
    for storage_buffer in full_state_test_buffers(state).values():
        storage_buffer.copy_(torch.arange(storage_buffer.numel()).reshape_as(storage_buffer))
    layout = layout_for(state)
    layout = replace(
        layout,
        byte_start=layout.byte_start + state_offset,
        byte_end=layout.byte_end + state_offset,
        regions=tuple(
            replace(r, byte_start=r.byte_start + state_offset, byte_end=r.byte_end + state_offset)
            for r in layout.regions
        ),
    )
    buffer = torch.full(
        size=(4, layout.byte_end + extra_page_bytes),
        fill_value=252,
        dtype=torch.uint8,
        pin_memory=True,
    )
    state.write_to_cpu(
        token_indices=torch.tensor([0, 1, 2, 3, 7], dtype=torch.long, device="cuda"),
        cpu_page_buffers=buffer,
        cpu_page_indices=[2, 0],
        layout=layout,
    )
    target = full_state(device="cuda")
    for storage_buffer in full_state_test_buffers(target).values():
        storage_buffer.zero_()
    target.read_from_cpu(
        cpu_page_buffers=buffer,
        cpu_page_indices=[2, 0],
        token_indices=torch.tensor([6, 7, 8, 9, 10], dtype=torch.long, device="cuda"),
        layout=layout,
    )
    assert torch.equal(read_kv_storage(target)[:, [6, 7, 8, 9, 10]], read_kv_storage(state)[:, [0, 1, 2, 3, 7]])
    assert (buffer[[1, 3]] == 252).all()
    assert (buffer[:, :layout.byte_start] == 252).all() and (buffer[:, layout.byte_end:] == 252).all()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_invalid_late_region_shape_cannot_partially_write_pd_page():
    state = full_state(device="cuda")
    layout = layout_for(state)
    region = layout.regions[-1]
    region = replace(region, byte_end=region.byte_end - region.dtype.itemsize)
    bad = replace(layout, regions=(*layout.regions[:-1], region))
    # 直接构造 Tensor 视图时，字节数与 shape 不匹配仍会在提交复制前失败。
    buffer = torch.full(
        size=(layout.byte_end + 256,),
        fill_value=53,
        dtype=torch.uint8,
        device="cuda",
    )
    with pytest.raises(RuntimeError, match="shape"):
        state.write_to_pd_page(
            token_indices=torch.tensor([0], dtype=torch.long, device="cuda"),
            pd_page_buffers=buffer.unsqueeze(0),
            pd_page_index=0,
            layout=bad,
        )
    assert (buffer == 53).all()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dp_copies_all_layers_in_storage_order(device: str | torch.device | int):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    config = GQAAttentionConfig(
        logical_layer_groups=((1,), (3,), (5,), (7,), (9,), (11,)),
        global_head_num=1,
        head_dim=4,
        dtype=torch.float32,
    )
    source = GQAFullAttentionState(config, page_size=4, device=device)
    source.allocate_storage(12)
    target = GQAFullAttentionState(config, page_size=4, device=device)
    target.allocate_storage(12)
    for storage_buffer in full_state_test_buffers(source).values():
        storage_buffer.copy_(torch.arange(storage_buffer.numel(), device=device).reshape_as(storage_buffer))
    for storage_buffer in full_state_test_buffers(target).values():
        storage_buffer.fill_(-1)
    if device == "cpu":
        with pytest.raises(AssertionError, match="DP token indices must reside on CUDA"):
            target.copy_tokens_from_dp(
                src_token_indices=torch.tensor([2, 0]), dst_token_indices=torch.tensor([6, 7]),
                remote_dp_full_att_state=source,
            )
        return
    target.copy_tokens_from_dp(
        src_token_indices=torch.tensor([2, 0], dtype=torch.long, device=device),
        dst_token_indices=torch.tensor([6, 7], dtype=torch.long, device=device),
        remote_dp_full_att_state=source,
    )
    torch.testing.assert_close(read_kv_storage(target)[:, [6, 7]], read_kv_storage(source)[:, [2, 0]])
    assert (read_kv_storage(target)[:, 8:] == -1).all()
    assert (read_kv_storage(target)[:, :6] == -1).all()

    # 拒绝将单层数据广播到多层，失败时目标 token 保持原值。
    single_layer_source = GQAFullAttentionState(
        replace(config, logical_layer_groups=config.logical_layer_groups[:1]), page_size=4, device=device
    )
    single_layer_source.allocate_storage(12)
    for storage_buffer in full_state_test_buffers(single_layer_source).values():
        storage_buffer.fill_(9)
    with pytest.raises(AssertionError, match="equivalent TP state configurations"):
        target.copy_tokens_from_dp(
            src_token_indices=torch.tensor([2, 0], dtype=torch.long, device=device),
            dst_token_indices=torch.tensor([8, 9], dtype=torch.long, device=device),
            remote_dp_full_att_state=single_layer_source,
        )
    assert (read_kv_storage(target)[:, 8:] == -1).all()
