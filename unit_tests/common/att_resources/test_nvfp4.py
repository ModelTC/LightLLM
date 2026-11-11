"""Check dynamic FP4 against independent encoders and dequantized attention."""

import math
import os
from types import SimpleNamespace

import pytest
import torch
from dataclasses import replace

from lightllm.common.attention_state.full_att import GQAAttentionConfig, create_gqa_state
from lightllm.common.basemodel.triton_kernel.kv_copy.nvfp4 import quantize_nvfp4_kv
from lightllm.common.basemodel.triton_kernel.att.prefill_att.nvfp4 import context_attention_nvfp4
from lightllm.common.basemodel.triton_kernel.att.decode_att.nvfp4 import decode_attention_nvfp4


def decode_reference(packed):
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=packed.device)
    codes = torch.stack((packed & 15, packed >> 4), -1).long()
    return values[codes].flatten(-2)


def encode_reference(values):
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=values.device)
    distances = (values.float().abs()[..., None] - levels).abs()
    nearest = distances == distances.amin(-1, keepdim=True)
    indices = torch.arange(8, device=values.device).expand_as(distances)
    even = torch.where(nearest & (indices % 2 == 0), indices, 8).amin(-1)
    any_nearest = torch.where(nearest, indices, 8).amin(-1)
    codes = torch.where(even < 8, even, any_nearest).to(torch.uint8) | (torch.signbit(values).to(torch.uint8) << 3)
    return codes[..., ::2] | (codes[..., 1::2] << 4)


def reference_attention(q, kd, vd, table, req, visible, positions, sliding_window=(-1, -1)):
    outputs = []
    for row, query in enumerate(q.float()):
        slots = table[int(req[row]), : int(visible[row])].long()
        keys, values = kd[slots], vd[slots]
        repeat = query.shape[0] // keys.shape[1]
        keys, values = keys.repeat_interleave(repeat, 1), values.repeat_interleave(repeat, 1)
        scores = torch.einsum("hd,thd->ht", query, keys) / math.sqrt(query.shape[-1])
        token_positions = torch.arange(len(slots), device=q.device)
        left, right = sliding_window
        mask = torch.ones(len(slots), dtype=torch.bool, device=q.device)
        if left >= 0:
            mask &= token_positions >= positions[row] - left
        if right >= 0:
            mask &= token_positions <= positions[row] + right
        probabilities = scores.masked_fill(~mask, -float("inf")).softmax(-1)
        outputs.append(torch.einsum("ht,thd->hd", probabilities, values))
    return torch.stack(outputs)


def state(dim=128, device="cpu"):
    return create_gqa_state(
        GQAAttentionConfig(((3, 7),), 2, dim, torch.bfloat16),
        kv_type="nvfp4kv",
        page_size=4,
        device=device,
    )


def dequantize(rows, dim, dtype):
    values = decode_reference(rows[..., : dim // 2])
    encoded = rows[..., dim // 2 :].contiguous()
    sf = encoded.view(torch.float8_e4m3fn).float()
    return (values * sf.repeat_interleave(16, -1)).to(dtype).float()


def test_layout_accounts_for_scales_and_transfer_bytes():
    s = state()
    s.allocate_storage(20)
    assert s.k_storage.shape == (1, 24, 2, 72)
    assert s.get_token_bytes() == 2 * 2 * 72
    regions = s._get_transfer_regions(4)
    assert sum(r.byte_end - r.byte_start for r in regions) == 4 * s.get_token_bytes()
    assert regions[0].shape[-1] == regions[1].shape[-1] == 72
    assert s.get_storage_layer_index(3) == s.get_storage_layer_index(7)


def test_hybrid_full_layers_accept_nvfp4():
    from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
    from lightllm.server.core.objs.start_args_type import StartArgs

    config = dict(
        num_hidden_layers=4,
        hidden_size=128,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=128,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        layer_types=("linear_attention",) * 3 + ("full_attention",),
    )
    args = StartArgs(llm_kv_type="nvfp4kv", data_type="bfloat16")
    (s,) = Qwen3NextTpPartModel.create_full_att_states(config, args, (0, 1, 2, 3))
    assert s.config.logical_layer_groups == ((3,),)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_midpoint_ties_and_negative_zero_follow_nvfp4():
    values = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5, -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5, -0.0, 6],
        device="cuda",
        dtype=torch.bfloat16,
    )
    x = values.repeat(2).reshape(1, 2, 16)
    k = torch.empty(1, 1, 9, dtype=torch.uint8, device="cuda")
    v = torch.empty_like(k)
    quantize_nvfp4_kv(x, torch.zeros(1, dtype=torch.int64, device="cuda"), k, v)
    codes = [0, 2, 2, 4, 4, 6, 6, 8, 10, 10, 12, 12, 14, 14, 8, 7]
    expected = torch.tensor([codes[i] | (codes[i + 1] << 4) for i in range(0, 16, 2)], device="cuda")
    assert torch.equal(k[0, 0, :8], expected) and torch.equal(v, k)
    assert int(k[0, 0, 8]) == 56


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dim", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packing_sparse_slots_strides_zero_and_extreme_blocks(dim, dtype):
    torch.manual_seed(11)
    # Vary block amplitudes, including FP8 scale underflow and saturation.
    x = torch.randn(19, dim, 4, device="cuda", dtype=dtype).transpose(1, 2)
    x[0] = 0
    x[1] = -0.0
    x[2] *= 1e-5
    x[3] *= 2048
    k = torch.full((45, 2, dim // 2 + dim // 16), 0xAA, device="cuda", dtype=torch.uint8)
    v = torch.full_like(k, 0x55)
    indices = torch.randperm(45, device="cuda")[::2][:19]
    assert not indices.is_contiguous()
    quantize_nvfp4_kv(x, indices, k, v)
    groups = x.float().reshape(19, 4, dim // 16, 16)
    sf = (groups.abs().amax(-1) * (1 / 6)).clamp(max=448).to(torch.float8_e4m3fn)
    encoded = sf.view(torch.uint8)
    inverse = sf.float().reciprocal().clamp(max=torch.finfo(torch.float32).max)
    inverse = torch.where(groups.abs().amax(-1) > 0, inverse, 0)
    codes = encode_reference((groups * inverse[..., None]).clamp(-6, 6).flatten(-2))
    actual = torch.cat((k[indices], v[indices]), 1)
    assert torch.equal(actual[..., : dim // 2], codes)
    assert torch.equal(actual[..., dim // 2 :], encoded)
    untouched = torch.ones(45, device="cuda", dtype=torch.bool)
    untouched[indices] = False
    assert (k[untouched] == 0xAA).all() and (v[untouched] == 0x55).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_nvfp4_matches_flashinfer_upstream_encoder(dtype):
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("installed FlashInfer FP4 CUDA encoder requires SM100+")
    flashinfer = pytest.importorskip("flashinfer")
    torch.manual_seed(5)
    x = torch.randn(37, 4, 128, device="cuda", dtype=dtype)
    k = torch.empty(37, 2, 72, device="cuda", dtype=torch.uint8)
    v = torch.empty_like(k)
    quantize_nvfp4_kv(x, torch.arange(37, device="cuda"), k, v)
    packed, sf = flashinfer.fp4_quantize(
        x.reshape(-1, 128),
        torch.ones(1, device="cuda"),
        is_sf_swizzled_layout=False,
        nvfp4_4over6=None,
    )
    actual = torch.cat((k, v), 1)
    assert torch.equal(actual[..., :64], packed.view(torch.uint8).reshape(37, 4, 64))
    assert torch.equal(actual[..., 64:], sf.view(torch.uint8).reshape(37, 4, 8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dim", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal,window", [(True, (-1, -1)), (True, (31, 0)), (False, (-1, -1))])
def test_prefill_and_split_decode_match_dequantized_reference(dim, dtype, causal, window):
    torch.manual_seed(17)
    x = torch.randn(600, 4, dim, device="cuda", dtype=dtype)
    k = torch.empty(600, 2, dim // 2 + dim // 16, device="cuda", dtype=torch.uint8)
    v = torch.empty_like(k)
    quantize_nvfp4_kv(x, torch.arange(600, device="cuda"), k, v)
    kd, vd = dequantize(k, dim, dtype), dequantize(v, dim, dtype)
    table = torch.randperm(600, device="cuda")[None, :]
    req = torch.zeros(1, device="cuda", dtype=torch.int32)
    ready = torch.tensor([13], device="cuda", dtype=torch.int32)
    length = torch.tensor([272], device="cuda", dtype=torch.int32)
    starts = torch.zeros_like(req)
    q = torch.randn(259, 8, dim, device="cuda", dtype=dtype)
    out = torch.empty_like(q)
    context_attention_nvfp4(
        q,
        k,
        v,
        out,
        req,
        starts,
        length,
        ready,
        table,
        259,
        causal=causal,
        sliding_window=window,
    )
    positions = torch.arange(13, 272, device="cuda")
    visible = positions + 1 if causal else length.expand(259)
    expected = reference_attention(q, kd, vd, table, req.expand(259), visible, positions, window)
    tol = 0.014 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(out.float(), expected, atol=tol, rtol=tol)
    req = torch.zeros(3, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([255, 257, 513], device="cuda", dtype=torch.int32)
    q = torch.randn(3, 8, dim, device="cuda", dtype=dtype)
    out = decode_attention_nvfp4(
        q,
        k,
        v,
        table,
        req,
        lengths,
        lengths,
        513,
        sliding_window=window,
    )
    expected = reference_attention(q, kd, vd, table, req, lengths, lengths - 1, window)
    torch.testing.assert_close(out.float(), expected, atol=tol, rtol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_copy_and_cuda_graph_replay_preserve_and_update_block_scales():
    s = state(device="cuda")
    s.allocate_storage(32)
    idx = torch.tensor([1, 3, 7], device="cuda")
    x = torch.randn(3, 4, 128, device="cuda", dtype=torch.bfloat16)
    s.write_full_att_kv(3, idx, x)
    s.copy_full_att_tokens(idx, idx + 16)
    assert torch.equal(s.k_storage[:, idx], s.k_storage[:, idx + 16])
    assert torch.equal(s.v_storage[:, idx], s.v_storage[:, idx + 16])
    old = s.k_storage[:, idx].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        s.write_full_att_kv(7, idx, x)
    x.mul_(8)
    graph.replay()
    torch.cuda.synchronize()
    assert not torch.equal(old[..., 64:], s.k_storage[:, idx, :, 64:])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dim", [16, 128])
@pytest.mark.parametrize("transport", ["cpu", "pd", "dp"])
def test_transfers_preserve_packed_data_and_scale_bytes(dim, transport):
    from lightllm.common.attention_state.types import StateTransferLayout

    source = state(dim=dim, device="cuda")
    config = source.config if transport == "dp" else replace(source.config, tp_world_size=2, tp_rank=1)
    target = type(source)(config, page_size=4, device="cuda")
    for s in (source, target):
        s.allocate_storage(8)
    for t in (source.k_storage, source.v_storage):
        t.random_(0, 256)
    for t in (target.k_storage, target.v_storage):
        t.fill_(173)
    src_ids = torch.tensor([6, 1, 7], device="cuda", dtype=torch.int32)
    dst_ids = torch.tensor([2, 4, 5], device="cuda", dtype=torch.int64)
    expected = []
    for src, dst in ((source.k_storage, target.k_storage), (source.v_storage, target.v_storage)):
        snapshot = dst.clone()
        snapshot[:, dst_ids] = src[:, src_ids] if transport == "dp" else src[:, src_ids, 1:]
        expected.append(snapshot)
    if transport == "dp":
        target.copy_tokens_from_dp(src_ids, dst_ids, source)
    else:

        def layout(s):
            regions = s._get_transfer_regions(4)
            return StateTransferLayout(
                state_name=s.state_name,
                tokens_per_page=4,
                byte_start=0,
                byte_end=max(r.byte_end for r in regions),
                regions=regions,
            )

        source_layout, target_layout = layout(source), layout(target)
        pages = torch.full(
            (2, source_layout.byte_end),
            91,
            dtype=torch.uint8,
            device="cpu" if transport == "cpu" else "cuda",
            pin_memory=transport == "cpu",
        )
        if transport == "cpu":
            source.write_to_cpu(src_ids, pages, [1], source_layout)
            target.read_from_cpu(pages, [1], dst_ids, target_layout)
        else:
            source.write_to_pd_page(src_ids, pages, 1, source_layout)
            target.read_from_pd_page(pages, dst_ids, 1, target_layout)
        torch.cuda.synchronize()
        assert (pages[0] == 91).all()
    for actual, reference in zip((target.k_storage, target.v_storage), expected):
        assert torch.equal(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gqa5_mtp_graph_replay_updates_queries_lengths_and_request_slots():
    from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers

    torch.manual_seed(31)
    dim = 128
    k = torch.empty(600, 2, 72, device="cuda", dtype=torch.uint8)
    v = torch.empty_like(k)
    x = torch.randn(600, 4, dim, device="cuda", dtype=torch.bfloat16)
    quantize_nvfp4_kv(x, torch.arange(600, device="cuda"), k, v)
    kd, vd = dequantize(k, dim, x.dtype), dequantize(v, dim, x.dtype)
    table = torch.randperm(600, device="cuda").expand(3, -1).contiguous()
    req = torch.tensor([0] * 3 + [1] * 3 + [2] * 3, device="cuda", dtype=torch.int32)
    lengths = torch.tensor([1, 2, 3] * 2 + [1] * 3, device="cuda", dtype=torch.int32)
    markers = build_mtp_shared_group_markers(req, hold_req_id=2)
    q = torch.randn(9, 10, dim, device="cuda", dtype=torch.bfloat16)

    def attention():
        return decode_attention_nvfp4(
            q,
            k,
            v,
            table,
            req,
            lengths,
            lengths,
            256,
            markers=markers,
            max_group_size=3,
        )

    attention()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = attention()
    lengths.copy_(torch.tensor([511, 512, 513] + [1] * 6, device="cuda"))
    req.copy_(torch.tensor([0] * 3 + [2] * 6, device="cuda"))
    markers.copy_(build_mtp_shared_group_markers(req, hold_req_id=2))
    q.mul_(0.5)
    table.copy_(table.roll(17, 1))
    graph.replay()
    expected = reference_attention(q, kd, vd, table, req, lengths, lengths - 1)
    torch.testing.assert_close(output.float(), expected, atol=0.014, rtol=0.014)


def test_nvfp4_backend_selection_is_limited_to_gqa(monkeypatch):
    from lightllm.common.basemodel.attention import create_utils
    from lightllm.common.basemodel.attention.triton.nvfp4 import NVFP4TritonAttBackend
    from lightllm.server.core.objs.start_args_type import StartArgs

    args = StartArgs(llm_kv_type="nvfp4kv", llm_prefill_att_backend=["triton"], llm_decode_att_backend=["triton"])
    monkeypatch.setattr(create_utils, "get_env_start_args", lambda: args)
    assert create_utils.get_prefill_att_backend_class() is NVFP4TritonAttBackend
    assert create_utils.get_decode_att_backend_class() is NVFP4TritonAttBackend
    for mapping in (
        create_utils.mla_data_type_to_backend,
        create_utils.nsa_data_type_to_backend,
        create_utils.neo_data_type_to_backend,
    ):
        assert "nvfp4kv" not in mapping


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dim", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("causal,window", [(True, (-1, -1)), (True, (31, 0)), (False, (-1, -1))])
def test_once_dequantized_varlen_prefill_matches_reference(dim, dtype, causal, window):
    from lightllm.common.basemodel.triton_kernel.kv_copy.dequantize_nvfp4 import dequantize_nvfp4_kv
    from lightllm.utils.sgl_utils import flash_attn_varlen_func

    if torch.cuda.get_device_capability()[0] < 9 or flash_attn_varlen_func is None:
        pytest.skip("FA3 requires SM90+")
    torch.manual_seed(67)
    raw = torch.randn(600, 4, dim, device="cuda", dtype=dtype)
    k = torch.empty(600, 2, dim // 2 + dim // 16, device="cuda", dtype=torch.uint8)
    v = torch.empty_like(k)
    quantize_nvfp4_kv(raw, torch.arange(600, device="cuda"), k, v)
    kd, vd = dequantize(k, dim, dtype), dequantize(v, dim, dtype)
    table = torch.stack([torch.randperm(600, device="cuda") for _ in range(3)])
    requests = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([73, 272, 1], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 73, 345, 346], device="cuda", dtype=torch.int32)
    cu_q = torch.tensor([0, 31, 160, 161], device="cuda", dtype=torch.int32)
    staged_k = torch.empty(346, 2, dim, device="cuda", dtype=dtype)
    staged_v = torch.empty_like(staged_k)
    dequantize_nvfp4_kv(k, v, table, requests, lengths, cu_k, 272, staged_k, staged_v)
    expected_k = torch.cat([kd[table[r, :n]] for r, n in zip([2, 0, 1], [73, 272, 1])])
    expected_v = torch.cat([vd[table[r, :n]] for r, n in zip([2, 0, 1], [73, 272, 1])])
    assert torch.equal(staged_k.float(), expected_k) and torch.equal(staged_v.float(), expected_v)
    q = torch.randn(161, 10, dim, device="cuda", dtype=dtype)
    output = flash_attn_varlen_func(
        q,
        staged_k,
        staged_v,
        cu_q,
        cu_k,
        129,
        272,
        causal=causal,
        window_size=window,
    )
    positions = torch.cat([torch.arange(42, 73), torch.arange(143, 272), torch.arange(1)]).cuda()
    req = requests.repeat_interleave(torch.tensor([31, 129, 1], device="cuda"))
    visible = positions + 1 if causal else lengths.repeat_interleave(torch.tensor([31, 129, 1], device="cuda"))
    expected = reference_attention(q, kd, vd, table, req, visible, positions, window)
    tol = 0.014 if dtype == torch.bfloat16 else 0.003
    torch.testing.assert_close(output.float(), expected, atol=tol, rtol=tol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_nvfp4_graph_replay_updates_paged_lengths_requests_and_scales(monkeypatch, dim, dtype):
    pytest.importorskip("flashinfer")
    import lightllm.common.basemodel.attention.flashinfer.fp as fp_backend
    from lightllm.common.basemodel.flashinfer_kernel.nvfp4_decode import _has_repack_support
    from lightllm.common.basemodel.attention.flashinfer.nvfp4 import NVFP4FlashInferAttBackend

    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", str(torch.cuda.current_device()))
    monkeypatch.setattr(fp_backend, "get_dp_world_size", lambda: 2)
    monkeypatch.setattr(fp_backend, "get_current_device_id", lambda: torch.cuda.current_device())

    class Model(SimpleNamespace):
        __hash__ = object.__hash__
        __eq__ = object.__eq__

    capacity = 32768 if dim == 128 and dtype == torch.bfloat16 else 1024
    model = Model(
        config=dict(
            num_attention_heads=20,
            num_key_value_heads=2,
            hidden_size=20 * dim,
            head_dim=dim,
            sliding_window=131072,
            use_sliding_window=False,
        ),
        args=SimpleNamespace(page_size=16),
        max_seq_length=capacity,
        graph_max_batch_size=8,
        graph_max_len_in_batch=capacity,
        data_type=dtype,
        is_mtp_draft_model=False,
        mtp_manager=SimpleNamespace(get_decode_tokens_per_request=lambda _: 1),
        att_state_role="main",
        att_state_manager=SimpleNamespace(
            get_full_attention_states=lambda role: (SimpleNamespace(config=SimpleNamespace(head_num=2, head_dim=dim)),),
        ),
    )
    backend = NVFP4FlashInferAttBackend(model)
    monkeypatch.setattr(backend, "uses_causal_attention", lambda: True)
    monkeypatch.setattr(backend.fallback, "uses_causal_attention", lambda: True)
    torch.manual_seed(71)
    raw = torch.randn(capacity, 4, dim, device="cuda", dtype=dtype)
    k = torch.empty(capacity, 2, dim // 2 + dim // 16, device="cuda", dtype=torch.uint8)
    v = torch.empty_like(k)
    slots = torch.arange(capacity, device="cuda")
    quantize_nvfp4_kv(raw, slots, k, v)
    pages = torch.randperm(capacity // 16, device="cuda")
    row = (pages[:, None] * 16 + torch.arange(16, device="cuda")).flatten()
    table = torch.stack([row, row.roll(16), row.roll(32)]).int()

    class Infer(SimpleNamespace):
        pass

    def infer(lengths, requests, capture=False, replay=False):
        return Infer(
            batch_size=3,
            max_kv_seq_len=max(lengths),
            microbatch_index=0,
            input_ids=torch.zeros(3, device="cuda", dtype=torch.int64),
            b_seq_len=torch.tensor(lengths, device="cuda", dtype=torch.int32),
            b_req_idx=torch.tensor(requests, device="cuda", dtype=torch.int32),
            req_manager=SimpleNamespace(req_to_token_indexs=table),
            is_cuda_graph_capture=capture,
            is_cuda_graph_replay=replay,
        )

    initial = infer([1, 255, 17], [0, 1, 2], capture=True)
    captured = backend.create_att_decode_state(initial)
    assert type(captured).__name__ == "NVFP4FlashInferDecodeAttState"
    captured.init_state()
    if (
        torch.cuda.get_device_capability() == (9, 0)
        and _has_repack_support()
        and not os.environ.get("FLASHINFER_DISABLE_JIT")
    ):
        assert captured.decode_wrapper._cached_module.paged_run.__module__.endswith(".nvfp4_decode")
    q = torch.randn(3, 10, dim, device="cuda", dtype=dtype)
    captured.decode_att(q, k, v)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = captured.decode_att(q, k, v)
    window_graph = None
    if dim == 128 and dtype == torch.bfloat16:
        from lightllm.common.basemodel.attention.base_att import AttControl

        window_control = AttControl(use_sliding_window=True, sliding_window=(31, 0))
        captured.decode_att(q, k, v, window_control)
        window_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(window_graph):
            window_output = captured.decode_att(q, k, v, window_control)
    longer = [32001, 16, 16385] if capacity > 1024 else [513, 16, 257]
    for lengths, requests in [(longer, [2, 0, 1]), ([1, 33, 7], [1, 2, 0])]:
        updated = infer(lengths, requests, replay=True)
        update = backend.create_att_decode_state(updated)
        update.init_state()
        initial.b_seq_len.copy_(updated.b_seq_len)
        initial.b_req_idx.copy_(updated.b_req_idx)
        captured.copy_for_decode_cuda_graph(update)
        raw.mul_(0.5)
        quantize_nvfp4_kv(raw, slots, k, v)
        graph.replay()
        kd, vd = dequantize(k, dim, q.dtype), dequantize(v, dim, q.dtype)
        expected = reference_attention(q, kd, vd, table, updated.b_req_idx, updated.b_seq_len, updated.b_seq_len - 1)
        tol = 0.014 if dtype == torch.bfloat16 else 0.003
        torch.testing.assert_close(output.float(), expected, atol=tol, rtol=tol)
        if window_graph is not None:
            window_graph.replay()
            expected_window = reference_attention(
                q,
                kd,
                vd,
                table,
                updated.b_req_idx,
                updated.b_seq_len,
                updated.b_seq_len - 1,
                (31, 0),
            )
            torch.testing.assert_close(window_output.float(), expected_window, atol=tol, rtol=tol)

    # Exercise the complete prefill state and both sides of its workspace limit.
    from lightllm.common.basemodel.attention.fa3.nvfp4 import NVFP4Fa3AttBackend

    prefill = infer([73, 272, 513], [2, 0, 1])
    prefill.b_ready_cache_len = prefill.b_seq_len - torch.tensor([3, 2, 1], device="cuda")
    prefill.b_q_start_loc = torch.tensor([0, 3, 5], device="cuda", dtype=torch.int32)
    prefill.b1_cu_q_seq_len = torch.tensor([0, 3, 5, 6], device="cuda", dtype=torch.int32)
    prefill.b1_cu_kv_seq_len = torch.tensor([0, 73, 345, 858], device="cuda", dtype=torch.int32)
    prefill.max_q_seq_len = 3
    prefill.total_token_num = 858
    query = torch.randn(6, 10, dim, device="cuda", dtype=dtype)
    req = prefill.b_req_idx.repeat_interleave(torch.tensor([3, 2, 1], device="cuda"))
    positions = torch.tensor([70, 71, 72, 270, 271, 512], device="cuda")
    expected = reference_attention(query, kd, vd, table, req, positions + 1, positions)
    if torch.cuda.get_device_capability()[0] >= 9:
        decode_index_buffers = tuple(backend.kv_indices_buffer)
        fa3 = NVFP4Fa3AttBackend(model)
        monkeypatch.setattr(fa3, "uses_causal_attention", lambda: True)
        for budget in (256 * 1024 * 1024, 0, 0):
            fa3.max_dequant_workspace_bytes = budget
            state = fa3.create_att_prefill_state(prefill)
            state.init_state()
            output = state.prefill_att(query, k, v)
            assert (state.paged_state is not None) == (budget == 0)
            torch.testing.assert_close(output.float(), expected, atol=tol, rtol=tol)
            assert all(actual is old for actual, old in zip(backend.kv_indices_buffer, decode_index_buffers))
        if dim == 128 and dtype == torch.bfloat16:
            state = fa3.create_att_prefill_state(prefill)
            state.init_state()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = state.prefill_att(query, k, v)
            assert type(state.paged_state).__name__ == "NVFP4PrefillAttState"
            graph.replay()
            torch.testing.assert_close(output.float(), expected, atol=tol, rtol=tol)


@pytest.mark.parametrize("dim,old_flashinfer", [(16, False), (64, False), (128, True)])
def test_native_nvfp4_capability_falls_back_without_allocating_gpu_buffers(monkeypatch, dim, old_flashinfer):
    flashinfer = pytest.importorskip("flashinfer")
    from lightllm.common.basemodel.attention.flashinfer.nvfp4 import NVFP4FlashInferAttBackend
    from lightllm.common.basemodel.attention.triton.nvfp4 import NVFP4DecodeAttState, NVFP4PrefillAttState

    if old_flashinfer:
        monkeypatch.setattr(flashinfer.BatchDecodeWithPagedKVCacheWrapper, "run", lambda self, q: q)

    class Model(SimpleNamespace):
        __hash__ = object.__hash__
        __eq__ = object.__eq__

    class Infer(SimpleNamespace):
        pass

    model = Model(config=dict(num_attention_heads=10, hidden_size=10 * dim, head_dim=dim))
    backend = NVFP4FlashInferAttBackend(model)
    assert not backend.native_supported
    infer = Infer()
    assert isinstance(backend.create_att_prefill_state(infer), NVFP4PrefillAttState)
    assert isinstance(backend.create_att_decode_state(infer), NVFP4DecodeAttState)


def test_missing_small_tile_device_helpers_preserve_the_native_module(monkeypatch):
    from lightllm.common.basemodel.attention.flashinfer.nvfp4_decode import use_small_tile_nvfp4_kernel
    from lightllm.common.basemodel.flashinfer_kernel import nvfp4_decode

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    monkeypatch.setattr(nvfp4_decode, "_has_repack_support", lambda: False)
    native = object()
    wrapper = SimpleNamespace(_cached_module=native)
    use_small_tile_nvfp4_kernel(wrapper, torch.bfloat16, 128)
    assert wrapper._cached_module is native


def test_disabled_jit_preserves_the_native_module(monkeypatch):
    from lightllm.common.basemodel.attention.flashinfer.nvfp4_decode import use_small_tile_nvfp4_kernel
    from lightllm.common.basemodel.flashinfer_kernel import nvfp4_decode

    monkeypatch.setenv("FLASHINFER_DISABLE_JIT", "1")

    def unexpected_jit(*args):
        raise AssertionError("disabled JIT must not initialize the custom module")

    monkeypatch.setattr(nvfp4_decode, "_get_module", unexpected_jit)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_jit)
    native = object()
    wrapper = SimpleNamespace(_cached_module=native)
    use_small_tile_nvfp4_kernel(wrapper, torch.bfloat16, 128)
    assert wrapper._cached_module is native


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("noncausal", [False, True])
def test_backend_cuda_graph_replays_updated_lengths_and_marker_layout(monkeypatch, noncausal):
    from lightllm.common.basemodel.attention.triton.nvfp4 import NVFP4TritonAttBackend
    from lightllm.common.basemodel.attention.base_att import AttControl

    torch.manual_seed(53)
    s = state(device="cuda")
    s.allocate_storage(600)
    raw = torch.randn(600, 4, 128, device="cuda", dtype=torch.bfloat16)
    s.write_full_att_kv(3, torch.arange(600, device="cuda"), raw)
    k, v = s.get_layer_view(3)
    kd, vd = dequantize(k, 128, raw.dtype), dequantize(v, 128, raw.dtype)
    slots = torch.arange(600, device="cuda")
    table = torch.stack((slots, slots.roll(17), slots.roll(31)))

    class Model(SimpleNamespace):
        __hash__ = object.__hash__
        __eq__ = object.__eq__

    model = Model(
        config={"architectures": ["LlamaForCausalLM"]},
        is_mtp_draft_model=True,
        mtp_manager=SimpleNamespace(get_decode_tokens_per_request=lambda _: 3),
        req_manager=SimpleNamespace(HOLD_REQUEST_ID=2),
    )
    backend = NVFP4TritonAttBackend(model)
    monkeypatch.setattr(backend, "uses_causal_attention", lambda: not noncausal)

    class Infer(SimpleNamespace):
        pass

    def infer(lengths, requests):
        return Infer(
            batch_size=9,
            max_kv_seq_len=256,
            b_seq_len=torch.tensor(lengths, device="cuda", dtype=torch.int32),
            b_req_idx=torch.tensor(requests, device="cuda", dtype=torch.int32),
            req_manager=SimpleNamespace(req_to_token_indexs=table),
        )

    initial = infer([1, 2, 3] * 2 + [1] * 3, [0] * 3 + [1] * 3 + [2] * 3)
    captured = backend.create_att_decode_state(initial)
    captured.init_state()
    q = torch.randn(9, 10, 128, device="cuda", dtype=torch.bfloat16)
    control = AttControl(logical_layer_id=3)
    captured.decode_att(q, k, v, control)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = captured.decode_att(q, k, v, control)
    updated = infer([1] * 3 + [511, 512, 513, 241, 242, 243], [2] * 3 + [0] * 3 + [1] * 3)
    update = backend.create_att_decode_state(updated)
    update.init_state()
    initial.b_seq_len.copy_(updated.b_seq_len)
    initial.b_req_idx.copy_(updated.b_req_idx)
    captured.copy_for_decode_cuda_graph(update)
    assert torch.equal(captured.markers, update.markers)
    q.mul_(0.5)
    graph.replay()
    expected = reference_attention(q, kd, vd, table, updated.b_req_idx, update.visible_lengths, updated.b_seq_len - 1)
    torch.testing.assert_close(output.float(), expected, atol=0.014, rtol=0.014)
