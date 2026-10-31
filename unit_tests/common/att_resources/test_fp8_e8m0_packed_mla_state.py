"""独立 FP8/E8M0 MLA：显式选择、固定编码、完整 payload 搬运和 padding 保留。"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import torch

from lightllm.common.attention_state.full_att import (
    FP8E8M0PackedMLAConfig,
    FP8E8M0PackedMLAState,
    MLAAttentionConfig,
    FullAttentionState,
    PackedMLAFullAttentionState,
    create_mla_state,
)
from lightllm.common.attention_state.types import StateTransferLayout


if TYPE_CHECKING:
    from easydict import EasyDict


def make_state(device: str = "meta", capacity: int = 16, tp_rank: int = 0) -> FP8E8M0PackedMLAState:
    state = create_mla_state(
        config=MLAAttentionConfig(
            logical_layer_groups=((0, 4), (8,)),
            latent_dim=512,
            dtype=torch.bfloat16,
            tp_rank=tp_rank,
            tp_world_size=2,
        ),
        kv_type="fp8kv_dsa",
        packed=True,
        page_size=4,
        device=device,
        fp8_format="group64_e8m0",
    )
    state.allocate_storage(token_capacity=capacity)
    return state


def test_factory_selects_independent_encoding_and_meta_geometry(pd_args: EasyDict) -> None:
    pd_args.cpu_cache_token_page_size = 4
    state = make_state()
    assert type(state) is FP8E8M0PackedMLAState
    assert FP8E8M0PackedMLAState.__bases__ == (FullAttentionState,)
    assert not isinstance(state, PackedMLAFullAttentionState)
    assert isinstance(state.config, FP8E8M0PackedMLAConfig)
    assert type(state.config).__bases__ == (object,)
    assert not hasattr(state.config, "head_dim")
    assert state.get_state_type() == "mla_packed_fp8_group64_e8m0"
    assert state.storage.shape == (2, 20, 1, 720) and state.storage.is_meta
    assert state.config.payload_bytes == 716 and state.get_token_bytes() == 1440
    assert state.get_storage_layer_index(logical_layer_id=0) == state.get_storage_layer_index(logical_layer_id=4)
    assert state.get_layer_view(logical_layer_id=8).kv.shape == (20, 1, 584)
    assert state.get_layer_view(logical_layer_id=8).indexer_k.shape == (20, 1, 132)
    (region,) = state.get_cpu_transfer_regions()
    assert region.shape == (2, 4, 1, 716) and region.byte_end == 2 * 4 * 716

    # 默认仍是原来的 group128/FP32 编码，不能再通过 512 维自动猜测 GLM 格式。
    with pytest.raises(AssertionError, match="FP8 MLA kernel requires"):
        create_mla_state(
            config=MLAAttentionConfig(
                logical_layer_groups=((0,),), latent_dim=512, dtype=torch.bfloat16
            ),
            kv_type="fp8kv_dsa",
            packed=True,
            device="meta",
        )
    with pytest.raises(AssertionError, match="512-dimensional"):
        create_mla_state(
            config=MLAAttentionConfig(
                logical_layer_groups=((0,),), latent_dim=576, dtype=torch.bfloat16
            ),
            kv_type="fp8kv_dsa",
            packed=True,
            fp8_format="group64_e8m0",
            device="meta",
        )


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("kv_type", ["None", "fp8kv_dsa"])
def test_glm_main_and_draft_explicitly_select_encoding(draft: bool, kv_type: str) -> None:
    from lightllm.models.glm5_next.model import Glm5NextTpPartModel
    from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel

    config = dict(
        num_hidden_layers=4,
        num_attention_heads=4,
        hidden_size=2048,
        kv_lora_rank=512,
        index_kpool=4,
        index_head_dim=128,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        linear_attn_config=dict(num_heads=4, head_dim=128, short_conv_kernel_size=4),
    )
    args = SimpleNamespace(
        tp=2, dp=1, llm_kv_type=kv_type, page_size=4, data_type="bfloat16", linear_att_ssm_data_type="float32"
    )
    model_class = Glm5NextMTPModel if draft else Glm5NextTpPartModel
    logical_ids = (14,) if draft else (10, 11, 12, 13)
    (state,) = model_class.create_full_att_states(
        model_config=config,
        args=args,
        model_logical_layer_ids=logical_ids,
        tp_rank=1,
        tp_world_size=2,
    )
    assert type(state) is (FP8E8M0PackedMLAState if kv_type == "fp8kv_dsa" else PackedMLAFullAttentionState)
    assert state.config.logical_layer_groups == ((logical_ids[-1],),)
    assert state.storage is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fixed_quantizer_preserves_indexer_and_other_rows() -> None:
    from lightllm.common.attention_state.full_att.fp8_e8m0_packed_mla_write import write_fp8_e8m0_packed_kv

    state = make_state(device="cuda")
    state.storage.fill_(173)
    indices = torch.tensor([1, 7, 12], dtype=torch.int32, device="cuda")
    kv = torch.randn((3, 1, 512), dtype=torch.bfloat16, device="cuda")
    expected = torch.empty((3, 1, 584), dtype=torch.uint8, device="cuda")
    write_fp8_e8m0_packed_kv(
        kv=kv, token_indices=torch.arange(3, dtype=torch.int32, device="cuda"), packed_kv=expected
    )
    state.write_full_att_kv(logical_layer_id=4, token_indices=indices, kv=kv)
    view = state.get_layer_view(logical_layer_id=0)
    torch.testing.assert_close(view.kv[indices], expected, rtol=0, atol=0)
    assert view.indexer_k.eq(173).all()
    assert state.storage[..., state.config.payload_bytes :].eq(173).all()
    assert state.storage[1].eq(173).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("transfer", ["cpu", "pd"])
@pytest.mark.parametrize("tp_rank", [0, 1])
@pytest.mark.parametrize("reorder_axes", [False, True])
def test_full_payload_roundtrip_skips_padding(
    transfer: str,
    tp_rank: int,
    reorder_axes: bool,
    pd_args: EasyDict,
) -> None:
    pd_args.cpu_cache_token_page_size, pd_args.pd_kv_page_size = 4, 16
    source = make_state(device="cuda")
    target = make_state(device="cuda", tp_rank=tp_rank)
    source.storage.random_(0, 256)
    target.storage.fill_(173)
    regions = source.get_cpu_transfer_regions() if transfer == "cpu" else source.get_pd_transfer_regions()
    if reorder_axes:
        regions = tuple(
            replace(
                region,
                axes=("token", "layer", "head", "dim"),
                shape=(region.shape[1], region.shape[0], *region.shape[2:]),
            )
            for region in regions
        )
    page_tokens = 4 if transfer == "cpu" else 16
    layout = StateTransferLayout(
        state_name=source.state_name,
        tokens_per_page=page_tokens,
        byte_start=0,
        byte_end=regions[0].byte_end,
        regions=regions,
    )
    pages = torch.full(
        (5, layout.byte_end + 32),
        99,
        dtype=torch.uint8,
        device="cpu" if transfer == "cpu" else "cuda",
        pin_memory=transfer == "cpu",
    )
    page_view = pages[:, 16:-16]
    src_ids = torch.tensor([11, 0, 4, 3, 13, 6, 5, 8, 2], dtype=torch.int32, device="cuda")
    dst_ids = torch.tensor([15, 2, 3, 7, 12, 9, 1, 8, 0], dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        if transfer == "cpu":
            source.write_to_cpu(
                token_indices=src_ids, cpu_page_buffers=page_view, cpu_page_indices=[3, 0, 4], layout=layout
            )
            target.read_from_cpu(
                cpu_page_buffers=page_view, cpu_page_indices=[3, 0, 4], token_indices=dst_ids, layout=layout
            )
        else:
            source.write_to_pd_page(token_indices=src_ids, pd_page_buffers=page_view, pd_page_index=3, layout=layout)
            target.read_from_pd_page(pd_page_buffers=page_view, token_indices=dst_ids, pd_page_index=3, layout=layout)
    stream.synchronize()
    torch.testing.assert_close(
        target.storage[:, dst_ids, :, :716], source.storage[:, src_ids, :, :716], rtol=0, atol=0
    )
    assert target.storage[..., 716:].eq(173).all()
    assert target.storage[:, 16:].eq(173).all()
    assert pages[:, :16].eq(99).all() and pages[:, -16:].eq(99).all()
    assert pages[1].eq(99).all() and pages[2].eq(99).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cpu_ready_pages_remain_unchanged(pd_args: EasyDict) -> None:
    pd_args.cpu_cache_token_page_size = 4
    source = make_state(device="cuda")
    source.storage.fill_(7)
    regions = source.get_cpu_transfer_regions()
    layout = StateTransferLayout(
        state_name=source.state_name, tokens_per_page=4, byte_start=0, byte_end=regions[0].byte_end, regions=regions
    )
    pages = torch.full((3, layout.byte_end), 99, dtype=torch.uint8, pin_memory=True)
    source.write_to_cpu(
        token_indices=torch.arange(9, dtype=torch.int32, device="cuda"),
        cpu_page_buffers=pages,
        cpu_page_indices=[2, 0, 1],
        layout=layout,
        cpu_page_ready_flags=[False, True, False],
    )
    torch.cuda.synchronize()
    assert pages[0].eq(99).all()
    assert pages[2].eq(7).all()
    partial_page = pages[1].view(2, 4, 1, 716)
    assert partial_page[:, 0].eq(7).all()
    assert partial_page[:, 1:].eq(99).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_local_and_dp_copy_include_entire_physical_row() -> None:
    source = make_state(device="cuda")
    target = make_state(device="cuda", capacity=32)
    source.storage.random_(0, 256)
    target.storage.fill_(173)
    src_ids = torch.tensor([0, 3, 11], dtype=torch.int32, device="cuda")
    dst_ids = torch.tensor([5, 20, 31], dtype=torch.int64, device="cuda")
    target.copy_tokens_from_dp(src_token_indices=src_ids, dst_token_indices=dst_ids, remote_dp_full_att_state=source)
    torch.testing.assert_close(target.storage[:, dst_ids], source.storage[:, src_ids], rtol=0, atol=0)
    source.copy_full_att_tokens(
        src_indices=src_ids, dst_indices=torch.tensor([4, 6, 8], dtype=torch.int64, device="cuda")
    )
    torch.testing.assert_close(source.storage[:, [4, 6, 8]], source.storage[:, src_ids], rtol=0, atol=0)
