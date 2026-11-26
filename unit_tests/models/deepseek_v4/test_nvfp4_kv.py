"""V4 NVFP4 storage and attention against independent byte/numerical oracles."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state.packed_kv_layout import PackedKVLayout
from lightllm.common.attention_state.bounded_att import NVFP4SlidingWindowState, PackedSlidingWindowState
from lightllm.common.basemodel.attention.nsa.dsv4_nvfp4 import DeepseekV4Nvfp4SparseAttBackend
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from lightllm.models.deepseek_v4_mtp.model import DeepseekV4MTPModel
from lightllm.models.deepseek_v4_dspark.model import DeepseekV4DSparkModel
from unit_tests.common.att_resources.test_nvfp4 import encode_reference, decode_reference

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def model_config(heads=64):
    return dict(
        num_hidden_layers=3,
        compress_ratios=[4, 128, 0],
        head_dim=512,
        qk_rope_head_dim=64,
        index_head_dim=128,
        sliding_window=128,
        num_attention_heads=heads,
    )


@pytest.mark.parametrize("model_class", [DeepseekV4TpPartModel, DeepseekV4MTPModel, DeepseekV4DSparkModel])
def test_main_and_draft_factories_select_nvfp4(model_class):
    config = model_config()
    args = SimpleNamespace(llm_kv_type="nvfp4kv", page_size=256, mtp_step=3)
    ids = (0, 1, 2) if model_class is DeepseekV4TpPartModel else (3, 4)
    full = model_class.create_full_att_states(config, args, ids)
    assert [s.config.token_slot_bytes for s in full] == ([129, 3] if ids[0] == 0 else [])
    bounded = model_class.create_bounded_att_states(config, args, ids, request_capacity=3, device="meta")
    swa = next(s for s in bounded if isinstance(s, PackedSlidingWindowState))
    assert isinstance(swa, NVFP4SlidingWindowState)
    assert swa.config.kv_bytes == 384
    assert swa.config.logical_layer_ids == ids
    assert swa.swa_buffer.shape[-1] == 384
    assert swa.get_state_type().endswith("nvfp4_block16")
    for state in full:
        assert state.get_state_type().endswith("nvfp4_block16")
        state.allocate_storage(512)
        assert state.get_layer_view(ids[0] if state.config.compress_ratio == 4 else ids[1]).kv.dtype == torch.uint8


class Model(SimpleNamespace):
    __hash__ = object.__hash__
    __eq__ = object.__eq__


def make_backend(heads=16, overlap=False, role="nvfp4_test"):
    model = Model(
        config=model_config(heads),
        tp_rank_=0,
        tp_world_size_=1,
        layers_infer=[SimpleNamespace(layer_num_=i) for i in range(3)],
        max_req_num=3,
        batch_max_tokens=512,
        data_type=torch.bfloat16,
        att_state_role=role,
        args=SimpleNamespace(
            run_mode="normal", enable_prefill_microbatch_overlap=overlap, enable_decode_microbatch_overlap=overlap
        ),
    )
    return DeepseekV4Nvfp4SparseAttBackend(model)


def pack_reference(raw, page_tokens, padding=0):
    layout = PackedKVLayout(512, 64, "nvfp4_block16")
    count = raw.shape[0]
    values = raw[:, :448].float().reshape(count, 28, 16)
    scales = (values.abs().amax(-1) / 6).to(torch.float8_e4m3fn)
    inv = torch.where(scales.float() != 0, scales.float().reciprocal(), 0)
    codes = encode_reference(values * inv[..., None]).reshape(count, 224)
    tail = raw[:, 448:].bfloat16().contiguous()
    data = torch.cat((codes, tail.view(torch.uint8)), -1)
    scale_bytes = torch.zeros(count, 32, dtype=torch.uint8, device=raw.device)
    scale_bytes[:, :28] = scales.view(torch.uint8)
    pages = count // page_tokens
    packed = torch.full((pages, layout.packed_bytes(page_tokens) + padding), 171, dtype=torch.uint8, device=raw.device)
    packed[:, : page_tokens * 352].view(pages, page_tokens, 352).copy_(data.view(pages, page_tokens, 352))
    packed[:, page_tokens * 352 : page_tokens * 384].view(pages, page_tokens, 32).copy_(
        scale_bytes.view(pages, page_tokens, 32)
    )
    decoded = torch.cat(
        (
            (decode_reference(codes).reshape(count, 28, 16) * scales.float()[..., None]).reshape(count, 448),
            tail.float(),
        ),
        -1,
    ).bfloat16()
    return packed[:, : layout.packed_bytes(page_tokens)], decoded


@CUDA
@pytest.mark.parametrize("ratio", [4, 128])
def test_compressor_packing_matches_independent_bytes(ratio):
    from lightllm.models.deepseek_v4.triton_kernel.compressor import _compress_norm_rope_pack_kernel

    page = 256 // ratio
    layout = PackedKVLayout(512, 64, "nvfp4_block16")
    stride = layout.packed_bytes(page) + 64
    slots = torch.tensor([0, page - 1, -1, page + 1, 2 * page - 1], device="cuda", dtype=torch.int64)
    scratch = torch.ones(5, 512, device="cuda") * (2.0 ** torch.arange(5, device="cuda"))[:, None]
    weight = torch.linspace(-3, 3, 512, device="cuda").bfloat16()
    positions = torch.arange(5, device="cuda", dtype=torch.int32) * ratio
    cos = torch.ones(5 * ratio, 32, device="cuda")
    sin = torch.zeros_like(cos)
    actual = torch.full((3, stride), 171, dtype=torch.uint8, device="cuda")
    _compress_norm_rope_pack_kernel[(5,)](
        scratch,
        weight,
        0.0,
        positions,
        slots,
        cos,
        *cos.stride(),
        sin,
        *sin.stride(),
        actual,
        HEAD_DIM=512,
        COMPRESS_RATIO=ratio,
        ROPE_HEAD_DIM=64,
        FP8_MAX=448.0,
        AMAX_MIN=1e-4,
        NOPE_DIM=448,
        QUANT_BLOCK=64,
        SCALE_BYTES=32,
        PAGE_SIZE=page,
        BYTES_PER_PAGE=stride,
        BLOCK=512,
        IS_IN_INDEXER=False,
        NVFP4=True,
    )
    packed, _ = pack_reference(weight.repeat(3 * page, 1), page)
    expected = torch.full_like(actual, 171)
    for slot in slots.tolist():
        if slot < 0:
            continue
        block, token = divmod(slot, page)
        data = token * layout.data_bytes
        scale = page * layout.data_bytes + token * layout.scale_bytes
        expected[block, data : data + layout.data_bytes] = packed[block, data : data + layout.data_bytes]
        expected[block, scale : scale + layout.scale_bytes] = packed[block, scale : scale + layout.scale_bytes]
    assert torch.equal(actual, expected)


def reference(q, decoded, extra_decoded, params, heads):
    results = []
    for row in range(q.shape[0]):
        chosen = []
        for source, name, length_name in [
            (decoded, "swa_indices", "swa_lengths"),
            (extra_decoded, "extra_indices", "extra_lengths"),
        ]:
            if source is None:
                continue
            index = params[name][row, 0, : int(params[length_name][row])].long()
            chosen.append(source[index[index >= 0]])
        kv = torch.cat(chosen).float()
        scores = q[row, :heads].float() @ kv.T * params["softmax_scale"]
        scores = torch.cat((scores, params["attn_sink"][:heads, None]), -1)
        probabilities = scores.softmax(-1)[:, :-1]
        results.append((probabilities @ kv).to(q.dtype))
    return torch.stack(results)


def attention_case(heads, ratio, rows=5, overlap=False):
    torch.manual_seed(740 + ratio + heads + rows)
    backend = make_backend(heads, overlap, role=f"nvfp4_{heads}_{rows}_{overlap}")
    q = torch.randn(rows, backend.padded_q_head_num, 512, device="cuda", dtype=torch.bfloat16) * 0.5
    pool, decoded = pack_reference(torch.randn(256, 512, device="cuda", dtype=q.dtype) * 0.3, 128)
    page = 256 // ratio if ratio else 1
    extra, extra_decoded = (
        pack_reference(
            torch.randn(3 * page, 512, device="cuda", dtype=q.dtype) * 0.3,
            page,
            padding=128 * 64 // ratio if ratio else 0,
        )
        if ratio
        else (None, None)
    )
    indices = torch.randint(0, 256, (rows, 1, 128), device="cuda", dtype=torch.int32)
    indices[:, :, ::7] = -1
    indices[0] = -1
    lengths = torch.arange(rows, device="cuda", dtype=torch.int32).mul_(17).clamp_max_(128)
    extra_indices = (
        torch.randint(0, extra_decoded.shape[0], (rows, 1, 128), device="cuda", dtype=torch.int32) if ratio else None
    )
    extra_lengths = lengths.clone() if ratio else None
    if ratio:
        extra_indices[:, :, ::5] = -1
    params = dict(
        compress_ratio=ratio,
        head_dim_v=512,
        softmax_scale=512 ** -0.5,
        swa_indices=indices,
        swa_lengths=lengths,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        attn_sink=torch.linspace(-3, 3, backend.padded_q_head_num, device="cuda"),
    )
    return backend, q, (pool, extra), decoded, extra_decoded, params


@CUDA
@pytest.mark.parametrize("heads", [8, 16, 64])
@pytest.mark.parametrize("ratio", [0, 4, 128])
@pytest.mark.parametrize("rows", [5, 65])
def test_prefill_and_decode_match_reference(heads, ratio, rows):
    backend, q, pool, decoded, extra_decoded, params = attention_case(heads, ratio, rows)
    infer = SimpleNamespace(microbatch_index=0)
    prefill = backend.create_att_prefill_state(infer)
    decode = backend.create_att_decode_state(infer)
    prefill.init_state()
    decode.init_state()
    out = torch.empty_like(q)
    actual = prefill.prefill_att(q, pool, None, AttControl(nsa_prefill=True, nsa_prefill_dict=params), out=out)
    assert actual.data_ptr() == out.data_ptr()
    expected = reference(q, decoded, extra_decoded, params, heads)
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    torch.testing.assert_close(
        decode.decode_att(q, pool, None, AttControl(nsa_decode=True, nsa_decode_dict=params)),
        expected,
        rtol=0.02,
        atol=0.002,
    )


@CUDA
@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_two_microbatch_graphs_replay_changed_inputs_without_shared_scratch(ratio):
    from unit_tests.models.glm5_next.test_fp8_kv import _capture_cuda_graph

    cases = []
    backend = None
    for slot in range(2):
        owner, q, pool, decoded, extra_decoded, params = attention_case(16, ratio, overlap=True)
        backend = backend or owner
        state = backend.create_att_decode_state(SimpleNamespace(microbatch_index=slot))
        state.init_state()
        control = AttControl(nsa_decode=True, nsa_decode_dict=params)

        def run(state=state, q=q, pool=pool, control=control):
            return state.decode_att(q, pool, None, control)

        graph, out = _capture_cuda_graph(run)
        cases.append((run, q, decoded, extra_decoded, params, graph, out, torch.cuda.Stream()))
    pointers = [{k: v.data_ptr() for k, v in w.items()} for w in backend._workspaces]
    for name in ("kv", "selected", "lengths"):
        assert pointers[0][name] != pointers[1][name]
    for _ in range(3):
        for run, q, decoded, extra_decoded, params, graph, out, stream in cases:
            q.normal_()
            params["swa_indices"].random_(0, 256)
            params["swa_lengths"].fill_(128)
            if ratio:
                params["extra_indices"].random_(0, extra_decoded.shape[0])
                params["extra_lengths"].fill_(128)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                graph.replay()
        for run, q, decoded, extra_decoded, params, graph, out, stream in cases:
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(out, reference(q, decoded, extra_decoded, params, 16), rtol=0.02, atol=0.003)
    assert [{k: v.data_ptr() for k, v in w.items()} for w in backend._workspaces] == pointers


@CUDA
def test_fused_swa_nvfp4_bytes_and_runtime_tail_match_independent_encoder():
    from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_kv_norm_rope_to_swa
    from unit_tests.models.deepseek_v4.test_kernels import rope_reference

    tokens = 257
    positions = torch.arange(tokens, device="cuda", dtype=torch.int32)
    kv = torch.ones(tokens, 512, dtype=torch.bfloat16, device="cuda")
    kv[0].zero_()
    weight = torch.linspace(-1.5, 1.5, 512, device="cuda").bfloat16()
    angles = torch.arange(tokens, device="cuda")[:, None].float() * torch.linspace(0.001, 0.05, 32, device="cuda")
    freqs = torch.complex(angles.cos(), angles.sin())
    values = kv.float() * torch.rsqrt(kv.float().square().mean(-1, keepdim=True) + 1e-6) * weight.float()
    raw = values.clone()
    raw[:, 448:] = rope_reference(values, freqs, positions).bfloat16().float()
    padded = torch.cat((raw, torch.zeros(384 - tokens, 512, device="cuda")))
    expected, _ = pack_reference(padded, 128)
    buffer = torch.full_like(expected, 73)
    runtime = torch.full((2, 128 * 384), 19, dtype=torch.uint8, device="cuda")
    runtime_slots = torch.where(positions >= tokens - 256, positions % 256, -1)
    fused_kv_norm_rope_to_swa(
        kv, weight, 1e-6, freqs, positions, positions, buffer, 128, runtime_slots, runtime, kv_format="nvfp4_block16"
    )
    # Codes, scales and padding are exact. RoPE keeps the existing fused-FMA
    # numerical tolerance; a separate PyTorch rotation can round one BF16 ULP
    # differently. Runtime and prefill copies must still be byte-identical.
    for token in range(tokens):
        page, offset = divmod(token, 128)
        for begin, count in [(offset * 352, 352), (128 * 352 + offset * 32, 32)]:
            actual_bytes = buffer[page, begin : begin + count]
            expected_bytes = expected[page, begin : begin + count]
            exact = 224 if count == 352 else count
            different = torch.nonzero(actual_bytes[:exact] != expected_bytes[:exact]).flatten()
            assert not different.numel(), (
                token,
                begin,
                different.tolist(),
                actual_bytes[different].tolist(),
                expected_bytes[different].tolist(),
            )
            if count == 352:
                torch.testing.assert_close(
                    actual_bytes[224:].view(torch.bfloat16),
                    expected_bytes[224:].view(torch.bfloat16),
                    rtol=0.01,
                    atol=0.02,
                )
            if token >= tokens - 256:
                runtime_page, runtime_offset = divmod(token % 256, 128)
                target = runtime_offset * 352 if count == 352 else 128 * 352 + runtime_offset * 32
                assert torch.equal(runtime[runtime_page, target : target + count], actual_bytes)


@CUDA
def test_long_hca_history_uses_bounded_staging_and_rebuilds_lengths():
    backend, q, pool, decoded, _, params = attention_case(16, 128, rows=17)
    extra, extra_decoded = pack_reference(torch.randn(8192, 512, dtype=torch.bfloat16, device="cuda") * 0.3, 2)
    params["extra_indices"] = torch.arange(8192, dtype=torch.int32, device="cuda").view(1, 1, -1).expand(17, 1, -1)
    params["extra_lengths"] = torch.full((17,), 8192, dtype=torch.int32, device="cuda")
    state = backend.create_att_decode_state(SimpleNamespace(microbatch_index=0))
    state.init_state()
    pointer = backend._workspaces[0]["kv"].data_ptr()
    for length in (8192, 1, 4097):
        params["extra_lengths"].fill_(length)
        out = state.decode_att(q, (pool[0], extra), None, AttControl(nsa_decode=True, nsa_decode_dict=params))
        torch.testing.assert_close(out, reference(q, decoded, extra_decoded, params, 16), rtol=0.02, atol=0.002)
    assert backend._workspaces[0]["kv"].data_ptr() == pointer
    assert backend._workspaces[0]["kv"].numel() * 2 == 64 * 1024 * 1024
