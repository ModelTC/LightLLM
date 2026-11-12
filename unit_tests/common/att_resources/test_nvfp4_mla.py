"""NVFP4 MLA storage, independent quantization/attention references and transfers."""

from dataclasses import replace
from types import SimpleNamespace
import weakref

import pytest
import torch
import triton

from lightllm.common.attention_state.full_att import (
    NVFP4MLAConfig,
    NVFP4MLAState,
    PackedMLALogicalConfig,
    create_packed_mla_state,
    MLAConfig,
    create_mla_state,
)
from lightllm.common.attention_state.types import StateTransferLayout
from lightllm.common.basemodel.triton_kernel.kv_copy.nvfp4_mla import quantize_nvfp4_mla
from lightllm.common.basemodel.triton_kernel.mla_att.nvfp4 import nvfp4_mla_attention
from unit_tests.common.att_resources.test_nvfp4 import encode_reference, decode_reference


@pytest.mark.parametrize("kv_type", ["None", "nvfp4kv"])
@pytest.mark.parametrize("layer_kind", ["dense", "glm_lite", "sparse"])
def test_dense_prefill_binding_preserves_projection_and_sparse_overrides(monkeypatch, kv_type, layer_kind):
    from lightllm.models.deepseek2.layer_infer import transformer_layer_infer as dense_module
    from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer
    from lightllm.models.glm4_moe_lite.layer_infer.transformer_layer_infer import Glm4MoeLiteTransformerLayerInfer

    dense_cls = dense_module.Deepseek2TransformerLayerInfer
    layer_cls = {
        "dense": dense_cls,
        "glm_lite": Glm4MoeLiteTransformerLayerInfer,
        "sparse": Deepseek3_2TransformerLayerInfer,
    }[layer_kind]
    monkeypatch.setattr(dense_module.LlamaTransformerLayerInfer, "_bind_func", lambda self: None)
    monkeypatch.setattr(dense_cls, "_bind_ffn", lambda self: None)
    monkeypatch.setattr(dense_module, "get_env_start_args", lambda: SimpleNamespace(llm_kv_type=kv_type))
    layer = object.__new__(layer_cls)
    layer._bind_func()
    if layer_kind == "sparse":
        assert layer._context_attention_kernel.__func__ is Deepseek3_2TransformerLayerInfer._context_attention_kernel
        return

    torch.manual_seed(192)
    layer.qk_nope_head_dim, layer.qk_rope_head_dim = 32, 16
    layer.layer_num_, layer.softmax_scale = 3, 0.125
    q = torch.randn(4, 2, 48)
    projection = torch.randn(2, 32, 64)
    kv = torch.randn(4, 1, 80)
    packed = torch.zeros(16, 1, 68, dtype=torch.uint8)
    decompressed = (torch.randn(4, 2, 32), torch.randn(4, 1, 16), torch.randn(4, 2, 64))
    monkeypatch.setattr(layer, "_decompress_kv", lambda **kwargs: decompressed)
    captured = {}
    result = torch.randn(4, 2, 64)

    def prefill_att(**kwargs):
        captured.update(kwargs)
        return result

    state = SimpleNamespace(get_layer_view=lambda logical_layer_id: packed)
    infer = SimpleNamespace(
        att_state_manager=SimpleNamespace(get_state=lambda **kwargs: state),
        att_state_role="main",
        prefill_att_state=SimpleNamespace(prefill_att=prefill_att),
    )
    weights = SimpleNamespace(k_b_proj_=SimpleNamespace(bmm=lambda x: torch.bmm(x, projection)))
    assert layer._context_attention_kernel(q, kv, infer, weights) is result
    assert captured["att_control"].mla_prefill_dict["softmax_scale"] == 0.125
    if kv_type == "nvfp4kv":
        torch.testing.assert_close(captured["q"][0], torch.einsum("thd,hdk->thk", q[..., :32], projection))
        torch.testing.assert_close(captured["q"][1], q[..., 32:])
        assert captured["k"] is packed and captured["v"] is None
        assert captured["att_control"].mla_prefill_dict["prefill_cache_kv"] is kv
    else:
        assert captured["q"] is q and captured["v"] is decompressed[2]
        assert captured["k"][0] is decompressed[0] and captured["k"][1] is decompressed[1]


def make_state(device="meta", rank=0, rope=0, indexer=128):
    state = NVFP4MLAState(
        NVFP4MLAConfig(((3, 7), (11,)), rope_dim=rope, indexer_dim=indexer, tp_rank=rank, tp_world_size=2),
        page_size=4,
        device=device,
    )
    state.allocate_storage(24)
    return state


def dequantize(packed, dim=512, dtype=torch.bfloat16):
    values = decode_reference(packed[..., : dim // 2])
    scales = packed[..., dim // 2 : dim // 2 + dim // 16].contiguous().view(torch.float8_e4m3fn).float()
    return (values * scales.repeat_interleave(16, -1)).to(dtype)


@pytest.mark.parametrize("draft", [False, True])
def test_glm_main_and_mtp_select_nvfp4(draft):
    from lightllm.models.glm5_next.model import Glm5NextTpPartModel
    from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel
    from lightllm.common.attention_state.full_att.packed_mla.pooled_mla_state import PooledMLAState

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
        tp=2, dp=1, llm_kv_type="nvfp4kv", page_size=4, data_type="bfloat16", linear_att_ssm_data_type="float32"
    )
    cls = Glm5NextMTPModel if draft else Glm5NextTpPartModel
    ids = (14,) if draft else (10, 11, 12, 13)
    (state,) = cls.create_full_att_states(config, args, ids, tp_rank=1, tp_world_size=2)
    assert type(state) is PooledMLAState and state.config.logical_layer_groups == ((ids[-1],),)
    assert state.config.format == "packed_nvfp4" and state.config.kv_bytes == 288
    assert state.config.token_bytes == 324


def test_packed_factory_geometry_and_replication(pd_args):
    state = create_packed_mla_state(
        PackedMLALogicalConfig(((3, 7), (11,)), 512, "packed_nvfp4", rope_dim=0), page_size=4, device="meta"
    )
    state.allocate_storage(24)
    assert state.storage.shape == (2, 28, 1, 432)
    assert state.get_token_bytes() == 864
    assert state.get_layer_view(7).kv.shape == (28, 1, 288)
    assert state.get_layer_view(11).indexer_k.shape == (28, 1, 132)
    assert state.get_storage_layer_index(3) == state.get_storage_layer_index(7)
    for rank in (0, 1):
        other = make_state(rank=rank)
        (region,) = other.get_pd_transfer_regions()
        assert region.shape == (2, 8, 1, 420) and region.write_enabled == (rank == 0)


def test_plain_mla_factory_and_invalid_indexer_alignment():
    state = create_mla_state(MLAConfig(((0,),), 576, torch.float16), kv_type="nvfp4kv", rope_dim=64, device="meta")
    state.allocate_storage(12)
    assert type(state) is NVFP4MLAState and state.config.dtype == torch.float16
    assert state.get_layer_view(0).shape == (13, 1, 416) and state.get_token_bytes() == 416
    assert state.config.indexer_dim == 0
    with pytest.raises(AssertionError, match="aligned"):
        NVFP4MLAConfig(((0,),), latent_dim=32, indexer_dim=128)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rope", [0, 64])
def test_quantizer_strides_scales_and_indexer_preservation(dtype, rope):
    torch.manual_seed(53)
    state = make_state("cuda", rope=rope)
    state.storage.fill_(173)
    source = torch.randn(13, 1, (512 + rope) * 2, device="cuda", dtype=dtype)[..., ::2]
    source[0].zero_()
    source[1].mul_(1e-5)
    source[2].mul_(2048)
    indices = torch.randperm(28, device="cuda")[::2][:13]
    out = state.get_layer_view(3).kv
    quantize_nvfp4_mla(source, indices, out, rope)
    groups = source[..., :512].float().reshape(13, 1, 32, 16)
    amax = groups.abs().amax(-1)
    scales = (amax * (1 / 6)).clamp(max=448).to(torch.float8_e4m3fn)
    inverse = torch.where(amax > 0, scales.float().reciprocal().clamp(max=torch.finfo(torch.float32).max), 0)
    codes = encode_reference((groups * inverse[..., None]).clamp(-6, 6).flatten(-2))
    assert torch.equal(out[indices, :, :256], codes)
    assert torch.equal(out[indices, :, 256:288], scales.view(torch.uint8))
    if rope:
        assert torch.equal(out[..., 288:].view(dtype)[indices], source[..., 512:])
    assert state.get_layer_view(3).indexer_k.eq(173).all()
    assert state.storage[..., state.config.payload_bytes :].eq(173).all()
    untouched = torch.ones(28, dtype=torch.bool, device="cuda")
    untouched[indices] = False
    assert out[untouched].eq(173).all() and state.storage[1].eq(173).all()


def reference(q, kv, indices, scale, qr=None, kr=None):
    output = torch.zeros_like(q)
    for row in range(len(q)):
        ids = indices[row][(indices[row] >= 0) & (indices[row] < kv.shape[0])].long()
        if not ids.numel():
            continue
        values = kv[ids, 0].float()
        score = q[row].float() @ values.T
        if qr is not None:
            score += qr[row].float() @ kr[ids, 0].float().T
        output[row] = (score * scale).softmax(-1) @ values
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("heads", [8, 16, 64])
@pytest.mark.parametrize("rows,rope,current", [(9, 0, False), (19, 0, True), (3, 64, False)])
def test_sparse_attention_reference_and_graph_replay(heads, rows, rope, current):
    torch.manual_seed(17)
    source = torch.randn(4096, 1, 512 + rope, dtype=torch.bfloat16, device="cuda")
    packed = torch.zeros(4096, 1, 432 + rope * 2, device="cuda", dtype=torch.uint8)
    view = packed[..., : 288 + rope * 2]
    quantize_nvfp4_mla(source, torch.arange(4096, device="cuda"), view, rope)
    q = torch.randn(heads, rows, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    qr = torch.randn(rows, heads, rope, device="cuda", dtype=q.dtype) if rope else None
    indices = torch.randint(0, 4096, (rows, 2176), device="cuda", dtype=torch.int32)
    lengths = torch.tensor(([0, 1, 3, 127, 128, 511, 2048, 2049, 2051] * 3)[:rows], device="cuda")
    indices.masked_fill_(torch.arange(2176, device="cuda")[None, :] >= lengths[:, None], -1)
    row_table, new_kv = None, None
    if current:
        new_kv = source[:17].clone().add_(0.5)
        row_table = torch.full((4096,), -1, device="cuda", dtype=torch.int32)
        row_table[:17] = torch.arange(17, device="cuda", dtype=torch.int32)
        indices[1, 0] = 4

    def run():
        return nvfp4_mla_attention(q, view, indices, 0.0625, qr, new_kv, row_table)

    def check(output):
        decoded = dequantize(view)
        if current:
            decoded[:17] = new_kv
        expected = reference(q, decoded, indices, 0.0625, qr, source[..., 512:] if rope else None)
        torch.testing.assert_close(output, expected, atol=0.016, rtol=0.02)
        assert output.isfinite().all() and output[0].eq(0).all()

    check(run())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    graph.replay()
    check(output)
    q.mul_(0.7)
    indices[1, 0] = 1023
    graph.replay()
    check(output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("transfer", ["cpu", "pd"])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("reorder", [False, True])
def test_payload_roundtrip_includes_scales_and_skips_padding(transfer, rank, reorder, pd_args):
    source, target = make_state("cuda"), make_state("cuda", rank=rank)
    source.storage.random_(0, 256)
    target.storage.fill_(173)
    regions = source.get_cpu_transfer_regions() if transfer == "cpu" else source.get_pd_transfer_regions()
    if reorder:
        regions = tuple(
            replace(r, axes=("token", "layer", "head", "dim"), shape=(r.shape[1], r.shape[0], *r.shape[2:]))
            for r in regions
        )
    layout = StateTransferLayout(
        state_name=source.state_name, tokens_per_page=8, byte_start=0, byte_end=regions[0].byte_end, regions=regions
    )
    pages = torch.full(
        (3, layout.byte_end + 32),
        99,
        dtype=torch.uint8,
        device="cpu" if transfer == "cpu" else "cuda",
        pin_memory=transfer == "cpu",
    )
    page_view = pages[:, 16:-16]
    src = torch.tensor([11, 0, 4, 3, 13, 6, 5], device="cuda", dtype=torch.int32)
    dst = torch.tensor([15, 2, 3, 7, 12, 9, 1], device="cuda", dtype=torch.int64)
    if transfer == "cpu":
        source.write_to_cpu(src, page_view, [1], layout)
        target.read_from_cpu(page_view, [1], dst, layout)
    else:
        source.write_to_pd_page(src, page_view, 1, layout)
        target.read_from_pd_page(page_view, dst, 1, layout)
    torch.cuda.synchronize()
    assert torch.equal(source.storage[:, src, :, :420], target.storage[:, dst, :, :420])
    assert target.storage[..., 420:].eq(173).all()
    assert pages[:, :16].eq(99).all() and pages[:, -16:].eq(99).all()
    assert pages[0].eq(99).all() and pages[2].eq(99).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_token_and_dp_copies_preserve_entire_physical_row():
    source, target = make_state("cuda"), make_state("cuda")
    source.storage.random_(0, 256)
    target.storage.fill_(173)
    src = torch.tensor([0, 4, 7], device="cuda")
    dst = torch.tensor([11, 16, 19], device="cuda")
    expected = source.storage[:, src].clone()
    source.copy_full_att_tokens(src, dst)
    assert torch.equal(source.storage[:, dst], expected)
    target.copy_tokens_from_dp(src, dst, source)
    assert torch.equal(target.storage[:, dst], expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_state_rejects_mismatched_rope_dtype_and_skips_ready_cpu_page(pd_args):
    state = make_state("cuda", rope=64)
    ids = torch.arange(9, device="cuda", dtype=torch.int32)
    values = torch.randn(9, 1, 576, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="computation dtype"):
        state.write_full_att_kv(3, ids, values.half())
    state.write_full_att_kv(3, ids, values)
    assert torch.equal(state.get_layer_view(3).kv[..., 288:].view(torch.bfloat16)[ids], values[..., 512:])
    state.storage.fill_(7)
    regions = state.get_cpu_transfer_regions()
    layout = StateTransferLayout(
        state_name=state.state_name, tokens_per_page=8, byte_start=0, byte_end=regions[0].byte_end, regions=regions
    )
    pages = torch.full((3, layout.byte_end), 99, dtype=torch.uint8, pin_memory=True)
    state.write_to_cpu(ids, pages, [2, 0], layout, cpu_page_ready_flags=[False, True])
    torch.cuda.synchronize()
    assert pages[0].eq(99).all() and pages[1].eq(99).all() and pages[2].eq(7).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dense_mla_paged_causal_metadata_and_graph(dtype):
    from lightllm.common.basemodel.triton_kernel.mla_att.nvfp4 import prepare_prefill_queries

    torch.manual_seed(71)
    rows, heads, dim, rope = 7, 8, 512, 64
    source = torch.randn(128, 1, dim + rope, device="cuda", dtype=dtype)
    packed = torch.zeros(128, 1, 416, device="cuda", dtype=torch.uint8)
    quantize_nvfp4_mla(source, torch.arange(128, device="cuda"), packed, rope)
    table = torch.randperm(128, device="cuda").reshape(4, 32).int()
    req = torch.tensor([3, 1], device="cuda", dtype=torch.int32)
    cuq = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int32)
    seq = torch.tensor([19, 25], device="cuda", dtype=torch.int32)
    query_req, lengths = prepare_prefill_queries(req, cuq, seq, rows, 4, True)
    assert query_req.tolist() == [3, 3, 3, 1, 1, 1, 1]
    assert lengths.tolist() == [17, 18, 19, 22, 23, 24, 25]
    q = torch.randn(rows, heads, dim, device="cuda", dtype=dtype)
    qr = torch.randn(rows, heads, rope, device="cuda", dtype=dtype)

    def run():
        return nvfp4_mla_attention(
            q, packed, table, 0.0625, q_rope=qr, query_req=query_req, lengths=lengths, max_seq_len=25
        )

    def check(out):
        indices = table[query_req.long()].clone()
        indices.masked_fill_(torch.arange(32, device="cuda")[None, :] >= lengths[:, None], -1)
        decoded = dequantize(packed, dtype=dtype)
        expected = reference(q, decoded, indices, 0.0625, qr, source[..., 512:])
        torch.testing.assert_close(out, expected, atol=0.016, rtol=0.02)

    check(run())
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = run()
    lengths.sub_(2)
    table.copy_(table.roll(1, 1))
    graph.replay()
    check(out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_dense_runtime_lengths_reuse_compiled_attention(monkeypatch):
    source = torch.randn(128, 1, 576, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty(128, 1, 416, device="cuda", dtype=torch.uint8)
    quantize_nvfp4_mla(source, torch.arange(128, device="cuda"), packed, 64)
    table = torch.arange(128, device="cuda", dtype=torch.int32).reshape(2, 64)
    req = torch.tensor([1], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([17], device="cuda", dtype=torch.int32)
    q = torch.randn(1, 8, 512, device="cuda", dtype=torch.bfloat16)
    qr = torch.randn(1, 8, 64, device="cuda", dtype=q.dtype)

    def run(width):
        return nvfp4_mla_attention(
            q, packed, table, 0.0625, q_rope=qr, query_req=req, lengths=lengths, max_seq_len=width
        )

    run(17)
    misses = []

    def on_miss(*args, **kwargs):
        misses.append(1)
        return False

    monkeypatch.setattr(triton.knobs.runtime, "jit_cache_hook", on_miss)
    for length in (33, 63, 18):
        lengths.fill_(length)
        out = run(length)
        ids = table[1, :length].long()
        decoded = dequantize(packed)[ids, 0].float()
        score = q[0].float() @ decoded.T + qr[0].float() @ source[ids, 0, 512:].float().T
        expected = ((score * 0.0625).softmax(-1) @ decoded).bfloat16()
        torch.testing.assert_close(out[0], expected, atol=0.016, rtol=0.02)
    assert not misses, "changing sequence length must not compile new attention kernels"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_rope_current_override_survives_unreplayed_shape_probe():
    from lightllm.common.basemodel.attention.triton.nvfp4_mla import NVFP4MlaPrefillState
    from lightllm.common.basemodel.attention.base_att import AttControl
    from lightllm.common.basemodel.infer_struct import InferStateInfo

    torch.manual_seed(9)
    raw = torch.randn(64, 1, 576, device="cuda", dtype=torch.bfloat16)
    packed = torch.empty(64, 1, 416, device="cuda", dtype=torch.uint8)
    quantize_nvfp4_mla(raw, torch.arange(64, device="cuda"), packed, 64)
    current = raw[:9].clone().add_(0.25)
    infer = InferStateInfo()
    infer.mem_index = torch.arange(9, device="cuda", dtype=torch.int32)
    infer.max_kv_seq_len = 128  # Graph bound may exceed the request table width.
    table = torch.arange(64, device="cuda", dtype=torch.int32).reshape(1, 64)
    infer.req_manager = SimpleNamespace(req_to_token_indexs=table)
    state = NVFP4MlaPrefillState(
        _infer_state_weak_ref=weakref.ref(infer),
        query_req=torch.zeros(2, device="cuda", dtype=torch.int32),
        lengths=torch.tensor([13, 17], device="cuda", dtype=torch.int32),
    )
    q = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
    qr = torch.randn(2, 8, 64, device="cuda", dtype=q.dtype)
    control = AttControl(mla_prefill=True, mla_prefill_dict=dict(softmax_scale=0.0625, prefill_cache_kv=current))

    def run():
        return state.prefill_att((q, qr), packed, None, control)

    run()
    state.current_row_table = None
    torch.cuda.synchronize()
    with torch.cuda.graph(torch.cuda.CUDAGraph()):
        run()
    assert state.current_row_table is None
    decoded, kr = dequantize(packed), raw[..., 512:].clone()
    decoded[:9], kr[:9] = current[..., :512], current[..., 512:]
    indices = table.expand(2, -1).clone()
    indices.masked_fill_(torch.arange(64, device="cuda")[None, :] >= state.lengths[:, None], -1)
    torch.testing.assert_close(run(), reference(q, decoded, indices, 0.0625, qr, kr), atol=0.016, rtol=0.02)
