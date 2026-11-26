from unit_tests.common.attention.backends.step_fixture import create_att_state
from lightllm.common.attention.backends.gqa.fa3 import FA3DecodeRuntimeConfig

"""GQA 跨量化的 KV 缓存：FP8（FlashInfer 每个张量，FA3 每个注意力头）和 int8/int4 组（Triton）。"""

from unit_tests.common.attention.backends.groups_runner import run_attention

import pytest
import torch

from unit_tests.common.attention.backends.step_fixture import make_infer_state
from lightllm.common.attention.backends.gqa import GQAKVScales, GQAAttConfig, GQAAttInputs
from lightllm.common.attention.backends.gqa.fa3 import FA3GQADecodeBackend, FA3GQAPrefillBackend
from lightllm.common.attention.backends.gqa.flashinfer import (
    FlashInferGQADecodeBackend,
    FlashInferGQAPrefillBackend,
)
from lightllm.common.attention.backends.gqa.triton import TritonGQADecodeBackend, TritonGQAPrefillBackend
from lightllm.common.attention.backends.gqa.triton_quant import (
    TritonQuantGQADecodeBackend,
    TritonQuantGQAPrefillBackend,
)

HEADS, KV_HEADS, DIM, CAP = 8, 2, 64, 512
LENGTHS, READY, REQ_IDS = [33, 9, 259], [16, 5, 128], [2, 0, 1]


def config(**fields):
    return GQAAttConfig(
        num_heads=fields.get("num_heads", HEADS),
        num_kv_heads=fields.get("num_kv_heads", KV_HEADS),
        head_dim=fields.get("head_dim", DIM),
        dtype=fields.get("dtype", torch.bfloat16),
        sliding_window=fields.get("sliding_window", (-1, -1)),
        softmax_scale=fields.get("softmax_scale", None),
        causal=fields.get("causal", True),
        has_sinks=fields.get("has_sinks", False),
        kv_quant=fields.get("kv_quant", None),
        kv_quant_group_size=fields.get("kv_quant_group_size", None),
        has_q_scale=fields.get("has_q_scale", False),
        has_alibi=fields.get("has_alibi", False),
        bidirectional_image_spans=fields.get("bidirectional_image_spans", False),
    )


@pytest.mark.parametrize(
    "fields,match",
    [
        (dict(kv_quant="fp4"), "kv_quant must be one of"),
        (dict(kv_quant="int8_group"), "kv_quant_group_size must be given"),
        (dict(kv_quant="fp8_per_head", kv_quant_group_size=8), "kv_quant_group_size must be given"),
        (dict(kv_quant="int8_group", kv_quant_group_size=7), "must divide head_dim"),
        (dict(kv_quant="fp8_per_tensor", has_q_scale=True), "has_q_scale applies only"),
    ],
)
def test_layer_config_rejects_malformed_kv_quant(fields, match):
    with pytest.raises(ValueError, match=match):
        config(**fields)


@pytest.mark.parametrize(
    "backends,supported",
    [
        ((TritonGQAPrefillBackend, TritonGQADecodeBackend), [None]),
        ((TritonQuantGQAPrefillBackend, TritonQuantGQADecodeBackend), ["int8_group", "int4_group"]),
        ((FlashInferGQAPrefillBackend, FlashInferGQADecodeBackend), [None, "fp8_per_tensor"]),
        ((FA3GQAPrefillBackend, FA3GQADecodeBackend), [None, "fp8_per_head"]),
    ],
)
def test_impls_accept_exactly_their_kv_quant_kinds(backends, supported):
    for kind in (None, "fp8_per_tensor", "fp8_per_head", "int8_group", "int4_group"):
        att_config = config(kv_quant=kind, kv_quant_group_size=8 if kind and kind.endswith("group") else None)
        for backend in backends:
            if kind in supported:
                backend(att_config)
            else:
                with pytest.raises(NotImplementedError):
                    backend(att_config)


def test_fa3_fp8_rejects_window_and_sinks():
    for fields in (dict(sliding_window=(15, 0)), dict(has_sinks=True), dict(softmax_scale=1.0)):
        with pytest.raises(NotImplementedError, match="FP8"):
            FA3GQADecodeBackend(config(kv_quant="fp8_per_head", **fields))


def test_fa3_fp8_needs_a_bf16_model():
    # FA3 只将 FP8 注意力输出写为 BF16，因此 FP16 模型将获得 BF16 激活。
    att_config = GQAAttConfig(
        num_heads=HEADS, num_kv_heads=KV_HEADS, head_dim=DIM, dtype=torch.float16, kv_quant="fp8_per_head"
    )
    for backend in (FA3GQAPrefillBackend, FA3GQADecodeBackend):
        with pytest.raises(NotImplementedError, match="BF16 model"):
            backend(att_config)
    FA3GQADecodeBackend(config(kv_quant="fp8_per_head"))


# ---- numerics on GPU ----

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def make_batch(token_map, *, prefill):
    lengths = torch.tensor(LENGTHS, device="cuda", dtype=torch.int32)
    q_lens = [n - r for n, r in zip(LENGTHS, READY)] if prefill else [1] * len(LENGTHS)
    cu_q = torch.tensor([0, *q_lens], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    return make_infer_state(
        is_prefill=prefill,
        req_to_token_indexs=token_map,
        b_req_idx=torch.tensor(REQ_IDS, device="cuda", dtype=torch.int32),
        b_seq_len=lengths,
        batch_size=len(LENGTHS),
        max_kv_seq_len=max(LENGTHS),
        total_token_num=sum(LENGTHS),
        b1_cu_q_seq_len=cu_q,
        b1_cu_kv_seq_len=torch.tensor([0, *LENGTHS], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32),
        b_q_start_loc=cu_q[:-1],
        b_ready_cache_len=torch.tensor(READY if prefill else [n - 1 for n in LENGTHS], device="cuda").int(),
        max_q_seq_len=max(q_lens),
    )


def reference(q, k, v, token_map, *, prefill):
    """Causal attention of each query over its request's dequantized keys, in batch order."""
    out, row = [], 0
    for req_id, length, ready in zip(REQ_IDS, LENGTHS, READY if prefill else [n - 1 for n in LENGTHS]):
        slots = token_map[req_id, :length].long()
        keys = k[slots].float().repeat_interleave(HEADS // KV_HEADS, dim=1)
        values = v[slots].float().repeat_interleave(HEADS // KV_HEADS, dim=1)
        for pos in range(ready, length):
            scores = torch.einsum("hd,khd->hk", q[row].float(), keys[: pos + 1]) * DIM ** -0.5
            out.append(torch.einsum("hk,khd->hd", scores.softmax(-1), values[: pos + 1]))
            row += 1
    return torch.stack(out)


def queries(prefill, seed=3):
    rows = sum(n - r for n, r in zip(LENGTHS, READY)) if prefill else len(LENGTHS)
    return torch.randn(rows, HEADS, DIM, device="cuda", dtype=torch.bfloat16, generator=gen(seed))


def group_cache(bits):
    """Return (k, v, kv_scales, dequantized k, dequantized v) written through the real cache state."""
    from lightllm.common.attention_state.full_att.gqa.group_quantized_state import (
        GroupQuantizedAttentionConfig,
        GroupQuantizedFullAttentionState,
    )

    state = GroupQuantizedFullAttentionState(
        GroupQuantizedAttentionConfig(((0,),), KV_HEADS, DIM, torch.bfloat16, bits=bits, group_size=8)
    )
    state.allocate_storage(3 * CAP)
    kv = torch.randn(3 * CAP, 2 * KV_HEADS, DIM, device="cuda", dtype=torch.bfloat16, generator=gen(1))
    state.write_full_att_kv(0, torch.arange(3 * CAP, device="cuda", dtype=torch.int32), kv)
    k, v = state.get_layer_view(0)
    scales = state.get_layer_kv_scales(0)

    def dequant(data, scale):
        if bits == 4:
            packed = data.view(torch.uint8)
            data = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).float() - 7
        return data.float() * scale.float().repeat_interleave(8, dim=-1)

    return k, v, scales, dequant(k, scales.k_scale), dequant(v, scales.v_scale)


@cuda
@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("prefill", [True, False])
def test_triton_group_quant_matches_dequantized_reference(bits, prefill):
    k, v, scales, k_ref, v_ref = group_cache(bits)
    token_map = torch.randperm(3 * CAP, device="cuda", dtype=torch.int32, generator=gen(0)).view(3, CAP)
    backend = TritonQuantGQAPrefillBackend if prefill else TritonQuantGQADecodeBackend
    impl = backend(config(kv_quant=f"int{bits}_group", kv_quant_group_size=8))
    att_state = create_att_state(backend=impl, infer_state=make_batch(token_map, prefill=prefill))
    q = queries(prefill)
    out = torch.empty_like(q)
    assert (
        run_attention(
            backend=impl,
            att_state=att_state,
            layer_inputs=GQAAttInputs(q=q, k_cache=k, v_cache=v, kv_scales=scales),
            out=out,
        )
        is out
    )
    torch.testing.assert_close(
        out.float(), reference(q, k_ref, v_ref, token_map, prefill=prefill), atol=0.03, rtol=0.03
    )


def fp8_cache():
    kv = torch.randn(3 * CAP, 2 * KV_HEADS, DIM, device="cuda", generator=gen(1)) * 4
    data = kv.to(torch.float8_e4m3fn).view(torch.uint8)
    return data[:, :KV_HEADS], data[:, KV_HEADS:]


@cuda
@pytest.mark.parametrize("static_q", [False, True])
@pytest.mark.parametrize("prefill", [True, False])
def test_fa3_fp8_per_head_matches_dequantized_reference(static_q, prefill):
    from lightllm.utils.sgl_utils import flash_attn_with_kvcache

    if flash_attn_with_kvcache is None:
        pytest.skip("FA3 (sgl_kernel) is required")
    k, v = fp8_cache()
    k_scale = torch.tensor([0.03, 0.05], device="cuda")
    v_scale = torch.tensor([0.04, 0.02], device="cuda")
    q_scale = torch.tensor([0.02, 0.03], device="cuda") if static_q else None
    token_map = torch.randperm(3 * CAP, device="cuda", dtype=torch.int32, generator=gen(0)).view(3, CAP)
    batch = make_batch(token_map, prefill=prefill)
    att_config = config(kv_quant="fp8_per_head", has_q_scale=static_q)
    if prefill:
        impl = FA3GQAPrefillBackend(att_config=att_config)
    else:
        runtime = FA3DecodeRuntimeConfig(max_seq_len=CAP, max_decode_rows=3)
        impl = FA3GQADecodeBackend(att_config=att_config, runtime_config=runtime)
    att_state = create_att_state(backend=impl, infer_state=batch)
    q = queries(prefill)
    out = torch.empty_like(q)
    scales = GQAKVScales(k_scale=k_scale, v_scale=v_scale, q_scale=q_scale)
    run_attention(
        backend=impl,
        att_state=att_state,
        layer_inputs=GQAAttInputs(q=q, k_cache=k, v_cache=v, kv_scales=scales),
        out=out,
    )
    k_ref = k.view(torch.float8_e4m3fn).float() * k_scale[None, :, None]
    v_ref = v.view(torch.float8_e4m3fn).float() * v_scale[None, :, None]
    # q is quantized to FP8 as well, so compare against the full-precision reference loosely.
    torch.testing.assert_close(out.float(), reference(q, k_ref, v_ref, token_map, prefill=prefill), atol=0.1, rtol=0.1)


@cuda
@pytest.mark.parametrize("prefill", [True, False])
def test_flashinfer_fp8_per_tensor_matches_dequantized_reference(prefill):
    pytest.importorskip("flashinfer")
    k, v = fp8_cache()
    token_map = torch.randperm(3 * CAP, device="cuda", dtype=torch.int32, generator=gen(0)).view(3, CAP)
    att_config = config(kv_quant="fp8_per_tensor")
    workspace_bytes = 32 * 1024 * 1024
    if prefill:
        init = FlashInferGQAPrefillBackend.RuntimeConfig(max_seq_len=CAP, workspace_bytes=workspace_bytes)
        impl = FlashInferGQAPrefillBackend(att_config=att_config, runtime_config=init)
    else:
        init = FlashInferGQADecodeBackend.RuntimeConfig(
            max_seq_len=CAP, max_batch_size=3, workspace_bytes=workspace_bytes
        )
        impl = FlashInferGQADecodeBackend(att_config=att_config, runtime_config=init)
    att_state = create_att_state(backend=impl, infer_state=make_batch(token_map, prefill=prefill))
    q = queries(prefill)
    out = torch.empty_like(q)
    scales = GQAKVScales(k_scale=0.03, v_scale=0.05)
    run_attention(
        backend=impl,
        att_state=att_state,
        layer_inputs=GQAAttInputs(q=q, k_cache=k, v_cache=v, kv_scales=scales),
        out=out,
    )
    k_ref, v_ref = k.view(torch.float8_e4m3fn).float() * 0.03, v.view(torch.float8_e4m3fn).float() * 0.05
    torch.testing.assert_close(
        out.float(), reference(q, k_ref, v_ref, token_map, prefill=prefill), atol=0.03, rtol=0.03
    )


@pytest.mark.parametrize("prefill", [True, False])
def test_quant_triton_rejects_non_causal_layers(prefill):
    # A non-causal (block-draft) layer must be rejected, not run causally over the int8 cache.
    backend = TritonQuantGQAPrefillBackend if prefill else TritonQuantGQADecodeBackend
    with pytest.raises(NotImplementedError, match="causal"):
        backend(config(kv_quant="int8_group", kv_quant_group_size=8, causal=False))


@cuda
@pytest.mark.parametrize("prefill", [True, False])
def test_unquantized_triton_rejects_a_quantized_cache(prefill):
    # Models that always use the unquantized Triton adapter get a clear error with an int8 cache.
    k, v, _, _, _ = group_cache(8)
    token_map = torch.randperm(3 * CAP, device="cuda", dtype=torch.int32, generator=gen(0)).view(3, CAP)
    backend = TritonGQAPrefillBackend if prefill else TritonGQADecodeBackend
    impl = backend(config())
    att_state = create_att_state(backend=impl, infer_state=make_batch(token_map, prefill=prefill))
    q = queries(prefill)
    with pytest.raises(AssertionError, match="KV cache of q's dtype"):
        run_attention(
            backend=impl,
            att_state=att_state,
            layer_inputs=GQAAttInputs(q=q, k_cache=k, v_cache=v),
            out=torch.empty_like(q),
        )
