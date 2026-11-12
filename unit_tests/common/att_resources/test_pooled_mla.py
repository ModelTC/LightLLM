"""Pooled indexer storage preserves the legacy encoding and cache lifecycle."""

from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state.full_att import PackedMLALogicalConfig, create_packed_mla_state
from lightllm.common.attention_state.types import StateTransferLayout
from lightllm.models.glm5_next.triton_kernel.kpool import compress_pools, gather_pools, gather_paged_pools


FORMATS = ("packed_bf16", "packed_fp16", "packed_group64_e8m0", "packed_nvfp4")


def make_state(format, page_size=4, device="cuda"):
    config = PackedMLALogicalConfig(
        logical_layer_groups=((3, 7), (11,)),
        latent_dim=512,
        rope_dim=0,
        format=format,
        indexer_pool_size=4,
    )
    state = create_packed_mla_state(config, page_size=page_size, device=device)
    state.allocate_storage(256)
    return state


@pytest.mark.parametrize("format,bytes_per_token", zip(FORMATS, (1060, 1060, 628, 324)))
@pytest.mark.parametrize("page_size", [4, 16])
def test_physical_capacity_includes_both_stores_and_hold(format, bytes_per_token, page_size):
    state = make_state(format, page_size, device="cpu")
    tensors = (state.storage, state.indexer_storage)
    assert sum(t.numel() * t.element_size() for t in tensors) == (256 + page_size) * state.get_token_bytes()
    assert state.get_token_bytes() == 2 * bytes_per_token
    assert state.indexer_storage.shape == (2, (256 + page_size) // 4, 1, 144)
    assert state.get_layer_view(3).kv.data_ptr() == state.get_layer_view(7).kv.data_ptr()
    assert state.get_layer_view(3).indexer_token_stride == 4


@pytest.mark.parametrize("page_size", [1, 2, 3, 6])
def test_reject_unaligned_pages(page_size):
    with pytest.raises(AssertionError, match="divisible by 4"):
        make_state("packed_nvfp4", page_size, device="cpu")


def test_reject_pooled_rope_config():
    config = PackedMLALogicalConfig(
        logical_layer_groups=((3,),),
        latent_dim=512,
        format="packed_nvfp4",
        indexer_pool_size=4,
    )
    with pytest.raises(AssertionError, match="NoPE"):
        create_packed_mla_state(config, page_size=4, device="meta")


@pytest.mark.parametrize(
    "model_type,page,expected", [("glm5_next", 1, 4), ("glm5_next", 16, 16), ("deepseek_v4", 1, 256), ("qwen3", 1, 1)]
)
def test_default_page_alignment(monkeypatch, model_type, page, expected):
    from types import SimpleNamespace
    from lightllm.utils.config_utils import auto_set_page_size

    monkeypatch.setattr("lightllm.utils.config_utils.get_model_type", lambda _: model_type)
    args = SimpleNamespace(page_size=page, model_dir="unused")
    auto_set_page_size(args)
    assert args.page_size == expected


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
@pytest.mark.parametrize("format", FORMATS)
def test_kv_write_is_byte_identical_to_legacy_and_leaves_indexer_untouched(format):
    compact = make_state(format)
    legacy = create_packed_mla_state(
        replace(
            PackedMLALogicalConfig(logical_layer_groups=((3, 7), (11,)), latent_dim=512, rope_dim=0, format=format),
            indexer_pool_size=1,
        ),
        page_size=4,
    )
    legacy.allocate_storage(256)
    compact.indexer_storage.random_(0, 256)
    before = compact.indexer_storage.clone()
    slots = torch.tensor([19, 4, 122, 7, 253], device="cuda", dtype=torch.int32)
    kv = torch.randn(5, 1, 512, device="cuda", dtype=torch.float16 if format == "packed_fp16" else torch.bfloat16)
    for state in (legacy, compact):
        state.write_full_att_kv(3, slots, kv)
    torch.testing.assert_close(
        compact.get_layer_view(3).kv[slots].view(torch.uint8),
        legacy.get_layer_view(3).kv[slots].view(torch.uint8),
        rtol=0,
        atol=0,
    )
    assert torch.equal(compact.indexer_storage, before)


def physical_slots(page_size, pages, tail):
    values = [page * page_size + offset for page in pages for offset in range(page_size)]
    values += [15 * page_size + i for i in range(tail)]
    # Exercise actual non-contiguous int64 token-index tensors.
    storage = torch.zeros(len(values) * 2, device="cuda", dtype=torch.int64)
    storage[::2] = torch.tensor(values, device="cuda", dtype=torch.int64)
    return storage[::2]


@cuda
@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("page_size", [4, 16])
@pytest.mark.parametrize("tail", [0, 1, 2, 3])
def test_same_and_remote_copy_preserve_pools_and_partial_tail(format, page_size, tail):
    source, target = make_state(format, page_size), make_state(format, page_size)
    source.storage.view(torch.uint8).random_(0, 256)
    source.indexer_storage.random_(0, 256)
    target.storage.zero_()
    target.indexer_storage.zero_()
    src = physical_slots(page_size, [2, 0], tail)
    dst = physical_slots(page_size, [7, 4], tail)
    # Move the tail to a disjoint physical page as well.
    if tail:
        dst[-tail:] = torch.arange(12 * page_size, 12 * page_size + tail, device="cuda")
    target.copy_tokens_from_dp(src, dst, source)
    torch.testing.assert_close(
        target.storage[:, dst].view(torch.uint8), source.storage[:, src].view(torch.uint8), rtol=0, atol=0
    )
    closed = src % 4 == 3
    torch.testing.assert_close(
        target.indexer_storage[:, dst[closed] // 4], source.indexer_storage[:, src[closed] // 4], rtol=0, atol=0
    )
    untouched = torch.ones(target.indexer_storage.shape[1], device="cuda", dtype=torch.bool)
    untouched[dst[closed] // 4] = False
    assert not target.indexer_storage[:, untouched].any()
    source.copy_full_att_tokens(src, dst)
    torch.testing.assert_close(
        source.indexer_storage[:, dst[closed] // 4], target.indexer_storage[:, dst[closed] // 4], rtol=0, atol=0
    )


@cuda
@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("page_size", [4, 16])
@pytest.mark.parametrize("tail", [0, 1, 2, 3])
@pytest.mark.parametrize("cpu", [False, True])
def test_page_transfer_roundtrip_ready_pages_and_untouched_tail(format, page_size, tail, cpu):
    source, target = make_state(format, page_size), make_state(format, page_size)
    source.storage.view(torch.uint8).random_(0, 256)
    source.indexer_storage.random_(0, 256)
    target.storage.zero_()
    target.indexer_storage.zero_()
    src = physical_slots(page_size, [2, 0], tail)
    dst = physical_slots(page_size, [7, 4], tail)
    if tail:
        dst[-tail:] = torch.arange(12 * page_size, 12 * page_size + tail, device="cuda")
    regions = source._get_transfer_regions(page_size)
    layout = StateTransferLayout("test", page_size, 0, regions[-1].byte_end, regions)
    buffers = torch.full(
        (5, layout.byte_end + 16), 91, dtype=torch.uint8, device="cpu" if cpu else "cuda", pin_memory=cpu
    )
    ids = [3, 1, 4] if tail else [3, 1]
    # A ready middle page must remain untouched for both KV and indexer.
    source._transfer(src, buffers, ids, layout, True, [False, True] + ([False] if tail else []))
    torch.cuda.synchronize()
    assert (buffers[1] == 91).all()
    source._transfer(src, buffers, ids, layout, True)
    target._transfer(dst, buffers, ids, layout, False)
    torch.cuda.synchronize()
    width = source.config.kv_bytes // source.config.dtype.itemsize
    torch.testing.assert_close(
        target.storage[:, dst, ..., :width].view(torch.uint8),
        source.storage[:, src, ..., :width].view(torch.uint8),
        rtol=0,
        atol=0,
    )
    closed = src % 4 == 3
    torch.testing.assert_close(
        target.indexer_storage[:, dst[closed] // 4, ..., :132],
        source.indexer_storage[:, src[closed] // 4, ..., :132],
        rtol=0,
        atol=0,
    )
    untouched = torch.ones(target.indexer_storage.shape[1], device="cuda", dtype=torch.bool)
    untouched[dst[closed] // 4] = False
    assert not target.indexer_storage[:, untouched].any()
    assert (buffers[:, layout.byte_end :] == 91).all()


@cuda
def test_compression_and_both_gathers_match_legacy_after_chunked_tail_and_graph_replay():
    torch.manual_seed(194)
    # Deliberately shuffle physical pages while preserving each page's offsets.
    slots = torch.cat([torch.arange(p * 16, (p + 1) * 16, device="cuda") for p in (3, 0, 5, 2)])
    legacy = torch.zeros(128, 1, 132, device="cuda", dtype=torch.uint8)
    compact = torch.zeros(32, 1, 144, device="cuda", dtype=torch.uint8)[..., :132]
    raw = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(4, 128, device="cuda", dtype=torch.bfloat16)
    tails = [torch.zeros(2, 6, 256, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    req = torch.zeros(1, device="cuda", dtype=torch.int32)
    lengths = torch.zeros(1, device="cuda", dtype=torch.int32)
    table = slots.to(torch.int32).view(1, -1)
    for begin, end in ((0, 5), (5, 11), (11, 13), (13, 64)):
        visible = torch.arange(begin + 1, end + 1, device="cuda", dtype=torch.int32)
        for buffer, tail, divisor in zip((legacy, compact), tails, (1, 4)):
            compress_pools(
                raw[begin:end],
                tail,
                buffer,
                ape,
                visible,
                slots[begin:end].to(torch.int32),
                req,
                torch.tensor([0, end - begin], device="cuda", dtype=torch.int32),
                torch.tensor([end], device="cuda", dtype=torch.int32),
                end - begin,
                indexer_token_stride=divisor,
            )
        torch.testing.assert_close(compact[slots[3:end:4] // 4], legacy[slots[3:end:4]], rtol=0, atol=0)
        assert torch.equal(tails[0], tails[1])
    for divisor, buffer in ((1, legacy), (4, compact)):
        gather_pools(buffer, table, req, lengths, 16, indexer_token_stride=divisor)
        gather_paged_pools(buffer, table, req, lengths, 16, indexer_token_stride=divisor)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        results = [
            (
                gather_pools(buffer, table, req, lengths * 4, 16, indexer_token_stride=divisor),
                gather_paged_pools(buffer, table, req, lengths, 16, indexer_token_stride=divisor),
            )
            for divisor, buffer in ((1, legacy), (4, compact))
        ]
    for count in (0, 1, 7, 16, 3):
        lengths.fill_(count)
        graph.replay()
        for left, right in zip(results[0], results[1]):
            for actual, expected in zip(left, right):
                torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
