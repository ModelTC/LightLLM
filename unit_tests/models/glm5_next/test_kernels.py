import dataclasses
from types import SimpleNamespace

import pytest
import torch
import triton

from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.envs_utils import set_env_start_args
from lightllm.models.glm5_next.triton_kernel.kda import chunk_kda_with_fused_gate
from lightllm.models.glm5_next.triton_kernel.kda_decode import fused_recurrent_kda
from lightllm.models.glm5_next.triton_kernel.kpool import compress_pools, gather_pools, expand_topk
from lightllm.models.glm5_next.triton_kernel.index_quant import hadamard_transform_quant_fp8
from lightllm.models.glm5_next.triton_kernel.mhc import hc_pre_norm, hc_pre_reference, hc_post, hc_post_reference
from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def setup():
    torch.manual_seed(1525)
    set_env_start_args(dataclasses.asdict(StartArgs()))
    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))


def _reference_kda(q, k, v, gate, beta, a, bias, state):
    q = q.float() * torch.rsqrt(q.float().square().sum(-1, keepdim=True) + 1e-6) / q.shape[-1] ** 0.5
    k = k.float() * torch.rsqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)
    decay = (-5 * torch.sigmoid(a.exp()[:, None] * (gate.float() + bias))).exp()
    state = state * decay[..., None]
    delta = (v.float() - torch.einsum("hkv,hk->hv", state, k)) * beta.float().sigmoid()[:, None]
    state = state + k[..., None] * delta[:, None, :]
    return torch.einsum("hkv,hk->hv", state, q), state


@pytest.mark.parametrize("tokens", [1, 3, 65, 129])
def test_kda_chunk_and_decode_match_recurrence(tokens):
    heads, dim = 2, 128
    rand = lambda *shape: torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    q, k, v, gate = [rand(1, tokens, heads, dim) for _ in range(4)]
    beta = rand(1, tokens, heads)
    a = torch.randn(heads, device="cuda")
    bias = torch.randn(heads, dim, device="cuda")
    initial = torch.randn(1, heads, dim, dim, device="cuda") * 0.1
    expected = []
    state = initial[0].clone()
    for i in range(tokens):
        out, state = _reference_kda(q[0, i], k[0, i], v[0, i], gate[0, i], beta[0, i], a, bias, state)
        expected.append(out)
    expected = torch.stack(expected).unsqueeze(0)
    actual, final = chunk_kda_with_fused_gate(
        q=q,
        k=k,
        v=v.clone(),
        raw_g=gate,
        beta=beta.float().sigmoid(),
        A_log=a,
        g_bias=bias.flatten(),
        initial_state=initial,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.tensor([0, tokens], dtype=torch.int32, device="cuda"),
        safe_gate=True,
    )
    torch.testing.assert_close(actual.float(), expected, atol=4e-3, rtol=3e-2)
    torch.testing.assert_close(final[0], state, atol=8e-3, rtol=3e-2)
    # Reuse a nonzero request slot; the neighboring requests must stay intact.
    states = torch.randn(4, heads, dim, dim, device="cuda")
    states[2] = initial[0]
    unchanged = states[[0, 1, 3]].clone()
    for i in range(tokens):
        out, _ = fused_recurrent_kda(
            q[:, i : i + 1],
            k[:, i : i + 1],
            v[:, i : i + 1],
            gate[:, i : i + 1].reshape(1, 1, -1),
            beta[:, i : i + 1],
            a,
            bias.flatten(),
            states,
            torch.tensor([2], device="cuda", dtype=torch.int32),
        )
        torch.testing.assert_close(out[0, 0].float(), expected[0, i], atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(states[2], state, atol=2e-5, rtol=2e-4)
    assert torch.equal(states[[0, 1, 3]], unchanged)


def test_kpool_chunk_boundaries_and_fragmented_token_kv():
    # Two unaligned requests, with a pool completed after restoring token KV.
    seqs = [9, 7]
    ragged = torch.randperm(40, device="cuda", dtype=torch.int32)[: sum(seqs)]
    req_idx = torch.tensor([1, 3], device="cuda", dtype=torch.int32)
    table = torch.zeros(4, 12, device="cuda", dtype=torch.int32)
    table[1, :9], table[3, :7] = ragged[:9], ragged[9:]
    source = torch.randn(sum(seqs), 256, device="cuda", dtype=torch.bfloat16)
    ape = torch.randn(4, 128, device="cuda")
    packed_storage = torch.zeros(40, 1, 904, device="cuda", dtype=torch.bfloat16)
    raw = packed_storage[:, :, 576:832]
    packed = packed_storage.view(torch.uint8)[:, :, -132:]
    for first, end in [(0, 3), (3, 9), (9, 11), (11, 16)]:
        raw[ragged[first:end].long(), 0] = source[first:end]
        start = 0 if first < 9 else 9
        lengths = torch.arange(first - start + 1, end - start + 1, device="cuda", dtype=torch.int32)
        starts = torch.full_like(lengths, start)
        compress_pools(raw, packed, ape, lengths, starts, ragged)
        # Model the full KV copy performed by cache offload/load or request move.
        packed_storage = packed_storage.clone()
        raw = packed_storage[:, :, 576:832]
        packed = packed_storage.view(torch.uint8)[:, :, -132:]
    keys, scales = gather_pools(packed, table, req_idx, torch.tensor(seqs, device="cuda", dtype=torch.int32), 3)
    for batch, start in enumerate([0, 9]):
        for group in range(seqs[batch] // 4):
            values = source[start + group * 4 : start + group * 4 + 4]
            expected = (values[:, :128].float() * (values[:, 128:].float() + ape).softmax(0)).sum(0).bfloat16()
            expected_key, expected_scale = hadamard_transform_quant_fp8(expected[None], scale=128 ** -0.5)
            torch.testing.assert_close(keys[batch * 3 + group].float(), expected_key[0].float(), atol=0, rtol=0)
            torch.testing.assert_close(scales[batch * 3 + group], expected_scale[0, 0], atol=0, rtol=0)
    lengths = torch.tensor(seqs, device="cuda", dtype=torch.int32)
    starts = torch.tensor([0, 9], device="cuda", dtype=torch.int32)
    groups = torch.tensor([[1, 0], [0, -1]], device="cuda", dtype=torch.int32)
    out, relative = expand_topk(groups, lengths, starts, ragged, topk=8)
    for batch, expected in enumerate([[4, 5, 6, 7, 0, 1, 2, 3, 8], list(range(7))]):
        assert relative[batch, : len(expected)].tolist() == expected
        assert (
            out[batch, : len(expected)].tolist()
            == ragged[starts[batch].item() + torch.tensor(expected, device="cuda")].tolist()
        )
        assert (out[batch, len(expected) :] == -1).all()


def test_mhc_matches_reference():
    streams, hidden = 4, 4096
    x = torch.randn(3, streams * hidden, device="cuda", dtype=torch.bfloat16)
    fn = torch.randn(24, streams * hidden, device="cuda") * 0.005
    scale = torch.randn(3, device="cuda")
    base = torch.randn(24, device="cuda")
    norm = torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
    expected = hc_pre_reference(x, fn, scale, base, streams, 1e-5, 1e-6, 20)
    actual = hc_pre_norm(x, fn, scale, base, norm, streams, 1e-5, 1e-5, 1e-6, 20)
    torch.testing.assert_close(actual[0], rmsnorm_forward(expected[0], norm, 1e-5), atol=0.04, rtol=0.03)
    for a, b in zip(actual[1:], expected[1:]):
        # DeepGEMM's mHC projection uses TF32 inputs and FP32 accumulation.
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)
    layer_out = torch.randn(3, hidden, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        hc_post(layer_out, x, *actual[1:], streams),
        hc_post_reference(layer_out, x, *actual[1:], streams),
        atol=0.04,
        rtol=0.02,
    )


def test_mhc_keeps_streams_through_decode_autotuning():
    from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType
    from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer

    hidden = 4096
    weight = SimpleNamespace(
        att_norm_weight_=SimpleNamespace(weight=torch.ones(hidden, device="cuda", dtype=torch.bfloat16)),
        ffn_norm_weight_=SimpleNamespace(weight=torch.ones(hidden, device="cuda", dtype=torch.bfloat16)),
    )
    for prefix in ("attn", "ffn"):
        setattr(weight, f"hc_{prefix}_fn", SimpleNamespace(weight=torch.randn(24, 4 * hidden, device="cuda") * 0.005))
        setattr(weight, f"hc_{prefix}_base", SimpleNamespace(weight=torch.zeros(24, device="cuda")))
        setattr(weight, f"hc_{prefix}_scale", SimpleNamespace(weight=torch.ones(3, device="cuda")))
    layer = object.__new__(Glm5NextTransformerLayerInfer)
    layer.embed_dim_, layer.mhc_streams = hidden, 4
    layer.num_hidden_layers, layer.autotune_layer_num = 5, 4
    layer.eps_, layer.hc_eps, layer.hc_sinkhorn_iters = 1e-5, 1e-6, 20
    layer.token_attention_forward = layer.context_attention_forward = layer._ffn = lambda x, *_: x * 0.1
    x = torch.randn(1, hidden, device="cuda", dtype=torch.bfloat16)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        for i in range(5):
            layer.layer_num_ = i
            x = layer.token_forward(x, None, weight)
            assert x.shape == (1, hidden if i == 4 else 4 * hidden)
    with Autotuner.autotune_warmup():
        for i in range(4):
            layer.layer_num_ = i
            x = layer.context_forward(x, None, weight)
            assert x.shape == (1, hidden if i == 3 else 4 * hidden)


def test_shared_chunk_kernel_preserves_gdn_natural_log_decay():
    from lightllm.common.basemodel.triton_kernel.linear_att.fla.ops import chunk_gated_delta_rule

    tokens, heads, dim = 67, 2, 128
    q, k, v = [torch.randn(1, tokens, heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    q = torch.nn.functional.normalize(q.float(), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(k.float(), dim=-1).bfloat16()
    gate = -torch.rand(1, tokens, heads, device="cuda")
    beta = torch.rand_like(gate)
    state = torch.randn(1, heads, dim, dim, device="cuda") * 0.1
    initial = state.clone()
    expected = []
    for i in range(tokens):
        state *= gate[:, i, :, None, None].exp()
        delta = (v[:, i].float() - torch.einsum("bhkv,bhk->bhv", state, k[:, i].float())) * beta[:, i, :, None]
        state += k[:, i, :, :, None].float() * delta[:, :, None, :]
        expected.append(torch.einsum("bhkv,bhk->bhv", state, q[:, i].float()) / dim ** 0.5)
    out, final = chunk_gated_delta_rule(
        q,
        k,
        v,
        gate,
        beta,
        initial_state=initial,
        output_final_state=True,
        cu_seqlens=torch.tensor([0, tokens], device="cuda", dtype=torch.int32),
    )
    torch.testing.assert_close(out.float(), torch.stack(expected, 1), atol=4e-3, rtol=3e-2)
    torch.testing.assert_close(final, state, atol=8e-3, rtol=3e-2)


def test_nope_attention_with_existing_image_kernels():
    from lightllm.common.basemodel.attention.base_att import AttControl
    from lightllm.models.glm5_next.attention import Glm5NextSparsePrefillState, Glm5NextSparseDecodeState

    packed = torch.randn(32, 1, 904, dtype=torch.bfloat16, device="cuda")
    packed[:, :, 512:576] = 0
    kv = packed[:, :, :576]
    q = torch.randn(3, 16, 512, dtype=torch.bfloat16, device="cuda")
    indexes = torch.full((3, 128), -1, dtype=torch.int32, device="cuda")
    selected = [[7, 2, 9], [8, 19, 3, 5, 1], [12, 24]]
    expected = []
    for i, locs in enumerate(selected):
        indexes[i, : len(locs)] = torch.tensor(locs, device="cuda")
        keys = kv[locs, 0, :512].float()
        expected.append((q[i].float() @ keys.T * 0.0625).softmax(-1) @ keys)
    expected = torch.stack(expected)
    control = AttControl(nsa_prefill_dict={"topk_mem_indices": indexes, "softmax_scale": 0.0625})
    prefill = Glm5NextSparsePrefillState()._nsa_prefill_att(q, kv, control)
    torch.testing.assert_close(prefill.float(), expected, atol=0.012, rtol=0.015)
    lengths = torch.tensor([len(x) for x in selected], dtype=torch.int32, device="cuda")
    decode = Glm5NextSparseDecodeState(
        infer_state=SimpleNamespace(b1_cu_q_seq_len=torch.arange(4, dtype=torch.int32, device="cuda"), max_q_seq_len=1),
        nsa_cache_seqlens=lengths,
        nsa_cu_seqlens_k_new=torch.nn.functional.pad(lengths.cumsum(0, dtype=torch.int32), (1, 0)),
    )
    control.nsa_decode_dict = control.nsa_prefill_dict
    out = decode._nsa_decode_att((q, q[..., :0]), kv, control)
    torch.testing.assert_close(out.float(), expected, atol=0.012, rtol=0.015)


def test_kpool_indexer_long_prefill_and_cached_decode():
    from lightllm.models.glm5_next.indexer import Glm5NextNsaInfer

    tokens, heads, dim = 2061, 32, 128
    hidden = torch.randn(tokens, dim, device="cuda", dtype=torch.bfloat16)
    q_weight = torch.randn(dim, heads * dim, device="cuda", dtype=torch.bfloat16) * 0.1
    ape = torch.randn(4, dim, device="cuda")
    weights = SimpleNamespace(
        wk_proj_=SimpleNamespace(mm=lambda x: x),
        k_norm_=lambda x, eps: x,
        index_kpool_compress_gate=SimpleNamespace(mm=lambda x: x * 0.25),
        index_kpool_compress_ape=SimpleNamespace(weight=ape),
        wq_b_proj_=SimpleNamespace(mm=lambda x: x @ q_weight),
        weights_proj_=SimpleNamespace(mm=lambda x: torch.ones(x.shape[0], heads, device=x.device)),
    )
    storage = torch.zeros(tokens + 9, 1, 904, device="cuda", dtype=torch.bfloat16)
    ragged = torch.randperm(tokens + 9, device="cuda", dtype=torch.int32)[:tokens]
    manager = SimpleNamespace(
        get_indexer_raw_buffer=lambda _: storage[:, :, 576:832],
        get_indexer_k_buffer=lambda _: storage.view(torch.uint8)[:, :, -132:],
    )
    infer = SimpleNamespace(
        mem_manager=manager,
        mem_index=ragged,
        max_kv_seq_len=tokens,
        req_manager=SimpleNamespace(req_to_token_indexs=ragged[None]),
        b_req_idx=torch.zeros(1, device="cuda", dtype=torch.int32),
        b_seq_len=torch.tensor([tokens], device="cuda", dtype=torch.int32),
    )
    state = SimpleNamespace(
        lengths=torch.arange(1, tokens + 1, device="cuda", dtype=torch.int32),
        ks=torch.zeros(tokens, device="cuda", dtype=torch.int32),
        ragged_mem_index=ragged,
        query_batch=torch.zeros(tokens, device="cuda", dtype=torch.int32),
    )
    indexer = Glm5NextNsaInfer(
        0, {"index_topk": 2048, "index_n_heads": heads, "index_head_dim": dim, "rms_norm_eps": 1e-5}, 1
    )
    _, full = indexer._get_indices(hidden, hidden, infer, state, weights)
    assert full[-1, 2048].item() == tokens - 1  # Always-selected incomplete tail.
    assert full[0, 0].item() == 0 and (full[0, 1:] == -1).all()
    assert full[-1, :2049].unique().numel() == 2049
    assert (full[-1, :2049] < tokens).all()
    # A later decode reads pooled history from restored token KV.
    storage = storage.clone()
    infer.mem_index = ragged[-1:]
    state.lengths = state.lengths[-1:]
    state.ks = state.ks[-1:]
    state.query_batch = state.query_batch[-1:]
    _, decoded = indexer._get_indices(hidden[-1:], hidden[-1:], infer, state, weights)
    assert set(decoded[0, :2049].tolist()) == set(full[-1, :2049].tolist())

    pool_count = tokens // 4
    pool_values = hidden[: pool_count * 4].view(pool_count, 4, dim)
    pooled = (pool_values.float() * (pool_values.float() * 0.25 + ape).softmax(1)).sum(1).bfloat16()
    k_fp8, k_scale = hadamard_transform_quant_fp8(pooled, dim ** -0.5)
    query = (hidden[-1:] @ q_weight).view(heads, dim)
    q_fp8, q_scale = hadamard_transform_quant_fp8(query, dim ** -0.5)
    logits = (q_fp8.float() @ k_fp8.float().T * k_scale.flatten()).clamp_min(0)
    scores = (logits * q_scale * (heads ** -0.5 * dim ** -0.5)).sum(0)
    expected_groups = set(scores.topk(512).indices.tolist())
    assert set((decoded[0, :2048:4] // 4).tolist()) == expected_groups
