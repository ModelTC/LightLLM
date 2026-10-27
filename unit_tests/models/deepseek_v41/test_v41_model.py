from types import SimpleNamespace

import pytest
import torch

from lightllm.models.deepseek_v41.triton_kernel.attention import (
    pool_pairs,
    destindex_copy_kv_dsv41,
    index_logits,
)
from lightllm.common.state_cache_manager.deepseek4 import DeepseekV4CacheConfig
from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_q_indexer_rope_quant
from lightllm.models.deepseek_v4.triton_kernel.cache_io import get_compressed_kv_buffer
from lightllm.models.deepseek_v4.triton_kernel.destindex_copy_indexer_k_dsv4 import destindex_copy_indexer_k_dsv4

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for key in (
        "GLOBAL_RANK",
        "CURRENT_RANK_IN_DP",
        "CURRENT_DEVICE_ID",
        "CURRENT_RANK_IN_NODE",
        "GLOBAL_DP_RANK",
        "DP_RANK_IN_NODE",
    ):
        monkeypatch.setenv("LIGHTLLM_" + key, "0")
    for key in ("GLOBAL_WORLD_SIZE", "DP_WORLD_SIZE", "NODE_WORLD_SIZE", "DP_SIZE"):
        monkeypatch.setenv("LIGHTLLM_" + key, "1")
    torch.manual_seed(410)


def test_config_flattens_text_without_losing_model_identity(tmp_path):
    import json
    from lightllm.models.deepseek_v41.model import DeepseekV41TpPartModel
    from lightllm.utils.config_utils import get_config_json, get_hidden_size
    from lightllm.models.registry import ModelRegistry

    assert ModelRegistry.get_model_class({"model_type": "deepseek_v41"}) is DeepseekV41TpPartModel
    assert DeepseekV41TpPartModel.__module__ == "lightllm.models.deepseek_v41.model"

    config = {
        "model_type": "deepseek_v41",
        "text_config": {"model_type": "deepseek_v41_text", "hidden_size": 5120},
        "vision_config": {"hidden_size": 1024},
        "quantization_config": {"expert_dtype": "fp4"},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert get_config_json(str(tmp_path)) == config
    assert get_hidden_size(str(tmp_path)) == 5120
    model = DeepseekV41TpPartModel.__new__(DeepseekV41TpPartModel)
    model.weight_dir_ = str(tmp_path)
    model.finetune_config = None
    model._init_config()
    config = model.config
    assert config["model_type"] == "deepseek_v41"
    assert config["hidden_size"] == 5120
    assert config["expert_dtype"] == "fp4"
    assert config["n_embd"] == 5120


@pytest.mark.parametrize("expert_dtype", [None, "fp4"])
def test_v41_mxfp4_uses_quant_selector_on_hopper(expert_dtype):
    from lightllm.common.quantization import Quantcfg

    config = dict(
        model_type="deepseek_v41",
        n_layer=2,
        expert_dtype="fp4",
        quantization_config=dict(quant_method="fp8", weight_block_size=[32, 32]),
    )
    quant = Quantcfg(config, expert_dtype=expert_dtype)
    assert quant.get_quant_method(0, "fused_moe").method_name == "mxfp4w4a16-b32-marlin"
    assert quant.get_quant_method(0, "wq_a").method_name == "fp8w8a8-b32-triton"


def test_fp8_history_capacity_counts_shared_sources_and_indexer_scales():
    config = DeepseekV4CacheConfig.from_model_config(
        dict(
            model_type="deepseek_v41",
            num_hidden_layers=40,
            compress_ratios=[0, 0, *([2] * 18), *([1] * 19), 0],
            kv_source_layer_ids=[2, 8, 14, 20],
            head_dim=512,
            index_head_dim=128,
            sliding_window=128,
            qk_rope_head_dim=64,
        ),
        40,
        2048,
    )
    assert config.indexer_bytes == 132
    assert config.history_widths == (360, 360, 360, 720)
    assert sum(config.history_widths) == 1800
    assert config.layer_to_history_idx[38] == config.layer_to_history_idx[20]


def _rope_reference(x, freqs, positions):
    rotated = x.clone()
    tail = torch.view_as_complex(x[..., -64:].float().unflatten(-1, (-1, 2)))
    frequency = freqs[positions.long()]
    if x.ndim == 3:
        frequency = frequency[:, None, :]
    rotated[..., -64:] = torch.view_as_real(tail * frequency).flatten(-2).bfloat16()
    return rotated


def _indexer_fp8_reference(x):
    values = x.float()
    scale = values.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
    return (values / scale).to(torch.float8_e4m3fn).float() * scale


def _assert_native_fp8_gemm_accuracy(actual, expected):
    # Hopper FP8 WGMMA accumulates with less precision than the former FP16 fallback.
    # Bound its error against an FP64 product of the decoded quantized inputs.
    expected = expected.double()
    error = actual.double() - expected
    assert error.norm() <= 1e-4 * expected.norm()
    assert error.abs().max() <= 1e-3 * expected.square().mean().sqrt()


@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("heads", [1, 32])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_indexer_fp8_rope_matches_reference(tokens, heads, position_dtype):
    x = torch.randn((tokens, heads, 128), device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    angles = torch.randn((32, 32), device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    positions = torch.randint(0, 32, (tokens,), dtype=position_dtype, device="cuda")
    weights = torch.ones(tokens, heads, device="cuda")
    q, scale = fused_q_indexer_rope_quant(x, weights, 1.0, freqs, positions, apply_hadamard=False)
    assert q.dtype == torch.float8_e4m3fn
    assert q.shape == x.shape
    expected = _indexer_fp8_reference(_rope_reference(x, freqs, positions))
    torch.testing.assert_close(q.float() * scale, expected, atol=0, rtol=1e-6)


@pytest.mark.parametrize("tokens", [1, 17, 513])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("runtime_tail", [False, True])
@pytest.mark.parametrize("eps", [1e-6, 1e-20])
def test_swa_fused_norm_rope_fp8_scatter_matches_reference(tokens, position_dtype, runtime_tail, eps):
    from lightllm.common.req_manager.deepseek4 import DeepseekV4ReqManager
    from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_forward
    from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd
    from lightllm.models.deepseek_v4.triton_kernel.cache_io import view_swa_kv

    raw = torch.randn(tokens, 640, device="cuda", dtype=torch.bfloat16)
    kv = raw[:, 128:]
    kv[0].zero_()
    weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(1024, 32, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    positions = torch.randint(0, 1024, (tokens,), device="cuda", dtype=position_dtype)
    config = DeepseekV4CacheConfig((0,))
    pages = (tokens + 127) // 128
    slots = torch.randperm(pages * 128, device="cuda")[:tokens].int()
    runtime = None
    if runtime_tail:
        runtime = torch.randperm(pages * 128, device="cuda")[:tokens].int() + pages * 128
        runtime[::3] = -1
    requests = DeepseekV4ReqManager.__new__(DeepseekV4ReqManager)
    requests.cache_config = config
    requests.swa_buffer = torch.full(
        (1, pages * (2 if runtime_tail else 1) * 128, config.swa_storage_width),
        165,
        dtype=torch.uint8,
        device="cuda",
    )
    expected = torch.full_like(view_swa_kv(requests.swa_buffer[0], config), 165)
    # Reference the former separate RMSNorm/RoPE path and independently build its packed bytes.
    normalized = rmsnorm_forward(kv, weight, eps)
    cos, sin = freqs.real[positions.long()].contiguous(), freqs.imag[positions.long()].contiguous()
    rotary_emb_fwd(normalized[:, None, -64:], None, cos, sin)
    groups = normalized[:, :448].float().reshape(tokens, 7, 64)
    raw_scale = groups.abs().amax(-1).clamp_min(1e-4) / 448
    bits = raw_scale.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
    scale64 = torch.exp2(exponent.float() - 127)
    fp8 = (groups / scale64[..., None]).to(torch.float8_e4m3fn).reshape(tokens, 448)
    data = torch.cat((fp8.view(torch.uint8), normalized[:, 448:].contiguous().view(torch.uint8)), dim=-1)
    for targets in (slots, runtime) if runtime_tail else (slots,):
        valid = targets >= 0
        page, offset = targets[valid].long() // 128, targets[valid].long() % 128
        expected[page[:, None], offset[:, None] * 576 + torch.arange(576, device="cuda")] = data[valid]
        expected[page[:, None], 128 * 576 + offset[:, None] * 8 + torch.arange(7, device="cuda")] = exponent[
            valid
        ].byte()
    requests.pack_mla_kv_to_cache_fused_norm_rope(
        0,
        slots,
        kv,
        weight,
        eps,
        freqs,
        positions,
        runtime,
        bf16_rounding=True,
    )
    torch.testing.assert_close(view_swa_kv(requests.swa_buffer[0], config), expected, atol=0, rtol=0)


@pytest.mark.parametrize("tokens", [1, 17, 513])
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("position_dtype", [torch.int32, torch.int64])
def test_q_rope_writes_padded_buffer_without_per_head_norm(tokens, heads, position_dtype):
    from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_q_norm_rope
    from lightllm.models.deepseek2.triton_kernel.rotary_emb import rotary_emb_fwd

    q = (torch.randn(tokens, heads + 2, 512, device="cuda", dtype=torch.bfloat16) * 3)[:, :heads]
    angles = torch.randn(1024, 32, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    positions = torch.randint(0, 1024, (tokens,), device="cuda", dtype=position_dtype)
    reference = q.clone()
    rotary_emb_fwd(
        reference[..., -64:],
        None,
        freqs.real[positions.long()].contiguous(),
        freqs.imag[positions.long()].contiguous(),
    )
    padded = torch.full((tokens, 64, 512), 42, device="cuda", dtype=torch.bfloat16)
    expected = padded.clone()
    expected[:, :heads] = reference
    fused_q_norm_rope(q, padded[:, :heads], 1e-20, freqs, positions, apply_norm=False)
    torch.testing.assert_close(padded[:, :heads, :448], expected[:, :heads, :448], atol=0, rtol=0)
    torch.testing.assert_close(padded[:, heads:], expected[:, heads:], atol=0, rtol=0)
    # RoPE kernels can choose different FP32 FMA order at BF16 rounding boundaries.
    torch.testing.assert_close(padded[:, :heads, 448:], expected[:, :heads, 448:], atol=0.015625, rtol=0.01)


def test_pure_swa_uses_base_rope_while_compressed_attention_keeps_yarn():
    from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
    from lightllm.models.deepseek_v41.model import DeepseekV41TpPartModel

    config = dict(
        qk_rope_head_dim=64,
        max_position_embeddings=32,
        rope_theta=10000,
        compress_rope_theta=160000,
        rope_scaling=dict(rope_type="yarn", factor=16, original_max_position_embeddings=16, beta_fast=32, beta_slow=1),
    )
    models = []
    for cls, model_type in ((DeepseekV4TpPartModel, "deepseek_v4"), (DeepseekV41TpPartModel, "deepseek_v41")):
        model = cls.__new__(cls)
        model.config, model.max_seq_length = dict(config, model_type=model_type), 32
        model.layers_infer = [SimpleNamespace(compress_ratio=ratio, index_infer=SimpleNamespace()) for ratio in (0, 2)]
        model._init_to_get_rotary()
        models.append(model)
    base, actual = models
    angles = torch.outer(
        torch.arange(32, device="cuda", dtype=torch.float32),
        1.0 / 10000 ** (torch.arange(0, 64, 2, device="cuda", dtype=torch.float32) / 64),
    )
    torch.testing.assert_close(actual._freqs_cis_sliding, torch.complex(angles.cos(), angles.sin()))
    torch.testing.assert_close(actual._freqs_cis_compress, base._freqs_cis_compress)
    assert (actual._freqs_cis_sliding - base._freqs_cis_sliding).abs().max() > 1e-4
    assert actual.layers_infer[0].freqs_cis is actual._freqs_cis_sliding


@pytest.mark.parametrize("split", [1, 3, 5])
def test_c2_carries_partial_groups_across_chunk_boundaries(split):
    kv, scores = torch.randn(7, 1024, device="cuda").split(512, -1)
    positions = torch.arange(7, dtype=torch.int32, device="cuda")
    req = torch.zeros(7, dtype=torch.int32, device="cuda")
    state = torch.zeros(2, 4, 2, 512, device="cuda")
    first = pool_pairs(kv[:split], scores[:split], req[:split], positions[:split], state, 1)
    second = pool_pairs(kv[split:], scores[split:], req[split:], positions[split:], state, 1)
    output = torch.cat((first, second))
    assert output[::2].count_nonzero() == 0
    actual = output[1::2]
    expected = (kv[:6].view(3, 2, 512) * scores[:6].view(3, 2, 512).softmax(1)).sum(1).bfloat16()
    torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.008)
    torch.testing.assert_close(state[0, 6 % 4, 0], kv[-1])
    torch.testing.assert_close(state[0, 6 % 4, 1], scores[-1])


def test_c2_does_not_cross_request_boundaries_or_commit_hold():
    kv = torch.randn(4, 512, device="cuda")
    scores = torch.randn_like(kv)
    state = torch.randn(3, 4, 2, 512, device="cuda")
    previous = state.clone()
    req = torch.tensor([0, 0, 1, 2], dtype=torch.int32, device="cuda")
    pos = torch.tensor([0, 1, 1, 0], dtype=torch.int32, device="cuda")
    out = pool_pairs(kv, scores, req, pos, state, 2)
    expected = previous[1, 0, 0] + (kv[2] - previous[1, 0, 0]) * torch.sigmoid(scores[2] - previous[1, 0, 1])
    torch.testing.assert_close(out[2], expected.bfloat16(), atol=0.015625, rtol=0.008)
    torch.testing.assert_close(state[2], previous[2])


@pytest.mark.parametrize("state_slots", [7, 8])
@pytest.mark.parametrize("start", [8, 9])
@pytest.mark.parametrize("accepted", [1, 2, 3, 4])
def test_c2_keeps_accepted_history_after_rejecting_draft_rows(state_slots, start, accepted):
    kv, scores = torch.randn(start + 4, 1024, device="cuda").split(512, -1)
    state = torch.zeros(2, state_slots, 2, 512, device="cuda")

    def run(kv, scores, position):
        count = kv.shape[0]
        return pool_pairs(
            kv,
            scores,
            torch.zeros(count, dtype=torch.int32, device="cuda"),
            torch.arange(position, position + count, dtype=torch.int32, device="cuda"),
            state,
            1,
        )

    run(kv[:start], scores[:start], 0)
    run(kv[start:], scores[start:], start)
    position = start + accepted
    new_kv, new_scores = torch.randn(2, 1024, device="cuda").split(512, -1)
    previous_kv, previous_score = kv[position - 1], scores[position - 1]
    for offset in range(2):
        actual = run(new_kv[offset : offset + 1], new_scores[offset : offset + 1], position + offset)
        if (position + offset) % 2:
            expected = previous_kv + (new_kv[offset] - previous_kv) * torch.sigmoid(new_scores[offset] - previous_score)
            torch.testing.assert_close(actual[0], expected.bfloat16(), atol=0.015625, rtol=0.008)
        previous_kv, previous_score = new_kv[offset], new_scores[offset]


@pytest.mark.parametrize("ratio", [1, 2])
def test_indexer_reads_paged_source_keys_with_causal_mask(ratio):
    n, heads = 5, 2
    cache = torch.zeros((1, (256 // ratio) * 132), dtype=torch.uint8, device="cuda")
    raw = torch.randn(n, 128, dtype=torch.bfloat16, device="cuda")
    slots = torch.arange(n, dtype=torch.int32, device="cuda") * ratio + ratio - 1
    freqs = torch.ones((16, 32), dtype=torch.complex64, device="cuda")
    destindex_copy_indexer_k_dsv4(raw, slots, slots, cache, 256 // ratio, ratio, freqs_cis=freqs)
    queries = torch.randn(2, heads, 128, dtype=torch.bfloat16, device="cuda")
    weights = torch.randn(2, heads, device="cuda")
    req = torch.zeros(2, dtype=torch.int32, device="cuda")
    positions = torch.tensor([ratio - 1, n * ratio - 1], dtype=torch.int32, device="cuda")
    table = torch.arange(256, dtype=torch.int32, device="cuda")[None, :]
    query, folded_weights = fused_q_indexer_rope_quant(queries, weights, 1.0, freqs, positions, apply_hadamard=False)
    actual = index_logits(query, folded_weights.squeeze(-1), cache, req, positions, table, ratio, n)
    expected = torch.einsum("thd,kd->thk", _indexer_fp8_reference(queries), _indexer_fp8_reference(raw)).relu()
    expected = (expected * weights[..., None]).sum(1)
    expected[0, 1:] = -float("inf")
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.01)


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("use_candidate_blocks", [False, True])
def test_fp8_indexer_page_scatter_candidate_blocks_and_cuda_graph(ratio, use_candidate_blocks):
    tokens, heads = 259, 32
    page_size = 256 // ratio
    cache = torch.full((3, page_size * 132), 165, dtype=torch.uint8, device="cuda")
    positions = torch.arange(tokens * ratio, dtype=torch.int32, device="cuda")
    physical = torch.randperm(3, device="cuda").repeat_interleave(256) * 256
    physical += torch.arange(256, device="cuda").repeat(3)
    slots = physical[: tokens * ratio].int()
    raw = torch.randn((tokens * ratio, 128), dtype=torch.bfloat16, device="cuda")
    angles = torch.randn(tokens * ratio, 32, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    destindex_copy_indexer_k_dsv4(raw, slots, positions, cache, page_size, ratio, freqs_cis=freqs)
    closed = torch.arange(ratio - 1, tokens * ratio, ratio, device="cuda")
    expected_cache = torch.full_like(cache, 165)
    compressed = slots[closed].long() // ratio
    pages, offsets = compressed // page_size, compressed % page_size
    keys = _rope_reference(raw[closed], freqs, positions[closed] - ratio + 1).float()
    scales = keys.abs().amax(-1).clamp_min(1e-4) / 448
    quantized = (keys / scales[:, None]).to(torch.float8_e4m3fn).view(torch.uint8)
    expected_cache[pages[:, None], offsets[:, None] * 128 + torch.arange(128, device="cuda")] = quantized
    expected_cache[pages[:, None], page_size * 128 + offsets[:, None] * 4 + torch.arange(4, device="cuda")] = (
        scales[:, None].contiguous().view(torch.uint8)
    )
    torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)

    query = torch.randn((3, heads, 128), dtype=torch.bfloat16, device="cuda")
    weights = torch.randn((3, heads), device="cuda")
    query_pos = torch.tensor([0, 129 * ratio - 1, tokens * ratio - 1], dtype=torch.int32, device="cuda")
    req = torch.zeros(3, dtype=torch.int32, device="cuda")
    table = physical.int()[None, :]
    candidate_blocks = (
        torch.tensor([[0, 1, -1], [0, 16, -1], [32, 0, 16]], device="cuda") if use_candidate_blocks else None
    )
    count = 24 if use_candidate_blocks else tokens

    def run():
        q, folded_weights = fused_q_indexer_rope_quant(query, weights, 1.0, freqs, query_pos, apply_hadamard=False)
        return index_logits(q, folded_weights.squeeze(-1), cache, req, query_pos, table, ratio, count, candidate_blocks)

    def reference():
        q = _indexer_fp8_reference(_rope_reference(query, freqs, query_pos))
        k = _indexer_fp8_reference(_rope_reference(raw[closed], freqs, positions[closed] - ratio + 1))
        score = torch.einsum("thd,kd->thk", q, k)
        score = (score.relu() * weights[..., None]).sum(1)
        if use_candidate_blocks:
            logical = (candidate_blocks[..., None] * 8 + torch.arange(8, device="cuda")).flatten(1)
            score = score.gather(1, logical.clamp(0, tokens - 1))
        else:
            logical = torch.arange(tokens, device="cuda")[None, :]
        return score.masked_fill((logical < 0) | (logical >= ((query_pos + 1) // ratio)[:, None]), -float("inf"))

    torch.testing.assert_close(run(), reference(), atol=0.03, rtol=0.01)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    query.normal_()
    graph.replay()
    torch.testing.assert_close(actual, reference(), atol=0.03, rtol=0.01)


@pytest.mark.parametrize("ratio", [1, 2])
def test_main_cache_writer_preserves_rope_and_layer_ownership(ratio):
    page_size = 256 // ratio
    cache = torch.zeros(
        (2, 2, DeepseekV4CacheConfig((0, 0)).packed_kv_bytes(page_size)), dtype=torch.uint8, device="cuda"
    )
    kv = torch.randn(3, 512, dtype=torch.bfloat16, device="cuda")
    kv[0].zero_()
    kv[:, 0] = -1e-5
    slots = torch.tensor([ratio - 1, 256 + ratio - 1, 511], dtype=torch.int32, device="cuda")
    positions = torch.tensor([0, 5, 127], dtype=torch.int32, device="cuda")
    angles = torch.randn(128, 32, device="cuda")
    freqs = torch.polar(torch.ones_like(angles), angles)
    destindex_copy_kv_dsv41(kv, freqs, slots, positions, cache[1], ratio, page_size)
    valid = (positions + 1) % ratio == 0
    rounded = _rope_reference(kv[valid], freqs, positions[valid] - ratio + 1)
    count = int(valid.sum())
    blocks = rounded[:, :448].float().reshape(count, 7, 64)
    scale = blocks.abs().amax(-1).clamp_min(1e-4) / 448
    bits = scale.contiguous().view(torch.int32)
    exponent = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).int()
    scale = torch.exp2(exponent.float() - 127)
    fp8 = (blocks / scale[..., None]).to(torch.float8_e4m3fn).reshape(count, 448)
    expected = torch.zeros_like(cache[1])
    compressed = slots[valid].long() // ratio
    pages, offsets = compressed // page_size, compressed % page_size
    expected[pages[:, None], offsets[:, None] * 576 + torch.arange(576, device="cuda")] = torch.cat(
        (fp8.view(torch.uint8), rounded[:, 448:].contiguous().view(torch.uint8)), -1
    )
    expected[pages[:, None], page_size * 576 + offsets[:, None] * 8 + torch.arange(7, device="cuda")] = exponent.byte()
    torch.testing.assert_close(cache[1], expected, atol=0, rtol=0)
    assert cache[0].count_nonzero() == 0


def test_shifted_mhc_uses_previous_pre_and_returns_new_coefficients(monkeypatch):
    from lightllm.common.basemodel.triton_kernel.mhc import pre_norm

    monkeypatch.setattr(pre_norm, "LIGHTLLM_DISABLE_DEEPGEMM_MHC", True)
    t, h = 3, 128
    x = torch.randn(t, 4 * h, dtype=torch.bfloat16, device="cuda")
    fn = torch.randn(24, 4 * h, device="cuda") * 0.01
    scale, base = torch.randn(3, device="cuda"), torch.randn(24, device="cuda")
    norm = torch.randn(h, dtype=torch.bfloat16, device="cuda")
    previous = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda").expand(t, -1).contiguous()
    actual, _, _, new_pre = pre_norm.hc_pre_norm_shifted(x, previous, fn, scale, base, norm, 1e-20, 1e-20, 1e-6, 20)
    selected = x.view(t, 4, h)[:, 0].float()
    expected = (selected * torch.rsqrt(selected.square().mean(-1, keepdim=True) + 1e-20) * norm).bfloat16()
    projection = x.float() @ fn.T * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20)
    expected_pre = torch.sigmoid(projection[:, :4] * scale[0] + base[:4]) + 1e-6
    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.01)
    torch.testing.assert_close(new_pre, expected_pre, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("scale_first", [False, True])
@pytest.mark.parametrize("tokens", [1, 17, 256])
def test_block32_fp8_loader_and_native_gemm(scale_first, tokens):
    from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight
    from lightllm.common.quantization.w8a8 import FP8w8a8B32TritonQuantizationMethod
    from lightllm.common.basemodel.triton_kernel.quantization.fp8w8a8_block_gemm_kernel import (
        w8a8_block_fp8_matmul,
    )

    method = FP8w8a8B32TritonQuantizationMethod()
    method.hf_quantization_config = {"scale_fmt": "ue8m0"}
    layer = ROWMMWeight(128, [32, 32], ["gate.weight", "up.weight"], torch.bfloat16, quant_method=method)
    pack, parts = layer.mm_param, layer.mm_param_list
    raw = (torch.randn(64, 128) * 10).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-5, 0, (2, 4)).float()).to(torch.float8_e8m0fnu)
    for part, w, ws in zip(parts, raw.split(32), scale.split(1)):
        loaders = [(method.load_weight, w), (method.load_weight_scale, ws)]
        if scale_first:
            loaders.reverse()
        loaders[0][0](loaders[0][1], part)
        assert not all(part.load_ok)
        loaders[1][0](loaders[1][1], part)
        assert all(part.load_ok)
    assert pack.weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(pack.weight.float().cpu(), raw.float(), atol=0, rtol=0)
    x = torch.randn(tokens, 128, device="cuda", dtype=torch.bfloat16)
    x[:, :32] = 0
    blocks = x.float().unflatten(-1, (4, 32))
    sx = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1e-4) / 448)))
    qx = (blocks / sx[..., None]).to(torch.float8_e4m3fn).flatten(-2)
    dw = raw.float().cuda() * scale.float().cuda().repeat_interleave(32, 0).repeat_interleave(32, 1)
    dx = qx.float() * sx.repeat_interleave(32, -1)
    expected = (dx.double() @ dw.double().T).float()
    qa, _, sa, _, _ = method._dynamic_quant_input(x, pack, use_custom_tensor_mananger=False)
    torch.testing.assert_close(qa.float(), qx.float(), atol=0, rtol=0)
    torch.testing.assert_close(sa, sx, atol=0, rtol=0)
    out = torch.empty(tokens, 64, device="cuda", dtype=torch.float32)
    actual = method.apply(x, pack, out=out, use_custom_tensor_mananger=False)
    assert actual is out
    _assert_native_fp8_gemm_accuracy(actual, expected)
    # Autotune candidates with K=64/128 must still respect the checkpoint's K=32 scales.
    for tile_k in (32, 64, 128):
        w8a8_block_fp8_matmul(
            qx,
            pack.weight.T,
            sx,
            pack.weight_scale.T,
            out,
            (32, 32),
            run_config=dict(BLOCK_M=64, BLOCK_N=64, BLOCK_K=tile_k, GROUP_M=8, num_warps=4, num_stages=3),
        )
        _assert_native_fp8_gemm_accuracy(out, expected)


@pytest.mark.parametrize("m,n,k", [(1, 1792, 5120), (17, 4096, 1280), (256, 288, 5120)])
def test_block32_native_gemm_model_shapes_and_batch_invariance(m, n, k):
    from lightllm.common.basemodel.layer_weights.meta_weights import ROWMMWeight
    from lightllm.common.quantization.w8a8 import FP8w8a8B32TritonQuantizationMethod

    method = FP8w8a8B32TritonQuantizationMethod()
    method.hf_quantization_config = {"scale_fmt": "ue8m0"}
    layer = ROWMMWeight(k, [n], "proj.weight", torch.bfloat16, quant_method=method)
    method.load_weight(torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02, layer.mm_param)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    qa, qw, sa, sw, _ = method._dynamic_quant_input(x, layer.mm_param, use_custom_tensor_mananger=False)
    dx = qa.float() * sa.repeat_interleave(32, -1)
    dw = qw.float() * sw.repeat_interleave(32, 0).repeat_interleave(32, 1)
    expected = (dx.double() @ dw.double()).float()
    out = torch.empty(m, n, device="cuda", dtype=torch.float32)
    layer.mm(x, out=out, use_custom_tensor_mananger=False)
    _assert_native_fp8_gemm_accuracy(out, expected)
    single = torch.empty_like(out[:1])
    layer.mm(x[:1], out=single, use_custom_tensor_mananger=False)
    torch.testing.assert_close(single, out[:1], atol=0, rtol=0)


def test_c2_first_token_has_no_compressed_history():
    from lightllm.models.deepseek_v4.layer_infer.indexer import DeepseekV4IndexInfer

    indexer = DeepseekV4IndexInfer(
        0,
        dict(
            model_type="deepseek_v41",
            rms_norm_eps=1e-20,
            compress_ratios=[2],
            index_topk=512,
            index_head_dim=128,
            qk_rope_head_dim=64,
            index_n_heads=2,
            index_source_layer_ids=[0],
            candidate_source_layer_id=20,
            candidate_topk_blocks=2048,
        ),
        1,
    )
    indexer.freqs_cis = torch.ones((1, 32), dtype=torch.complex64, device="cuda")
    config = DeepseekV4CacheConfig((2,), short_compress_ratio=2, long_compress_ratio=1)
    cache = torch.zeros((256, 1, config.history_widths[0]), dtype=torch.uint8, device="cuda")
    state = SimpleNamespace(
        position_ids=torch.tensor([0], device="cuda"),
        dsv4_sparse_req_idx=torch.tensor([0], device="cuda"),
        v41_index_capacity=1,
        dsv4_swa_indices=torch.full((1, 128), -1, dtype=torch.int32, device="cuda"),
        dsv4_swa_lengths=torch.ones(1, dtype=torch.int32, device="cuda"),
        mem_manager=SimpleNamespace(cache_config=config, get_att_input_params=lambda layer: cache),
        req_manager=SimpleNamespace(
            cache_config=config,
            req_to_token_indexs=torch.full((2, 256), -1, dtype=torch.int32, device="cuda"),
            HOLD_REQUEST_ID=1,
        ),
    )
    weight = SimpleNamespace(
        idx_wq_b_=SimpleNamespace(
            mm=lambda x, **kwargs: torch.zeros((x.shape[0], 256), dtype=x.dtype, device=x.device)
        ),
        idx_weights_proj_=SimpleNamespace(mm=lambda x, out, **kwargs: out.zero_()),
    )
    x = torch.zeros((1, 128), dtype=torch.bfloat16, device="cuda")
    metadata = indexer.build_metadata(x, x[:, :32], state, weight)
    assert metadata["extra_indices"].shape == (1, 1, 512)
    assert (metadata["extra_indices"] == -1).all()
    assert metadata["extra_lengths"].item() == 0


@pytest.mark.parametrize("intermediate_size", [128, 288])
def test_hopper_mxfp4_marlin_padding_preserves_clamped_experts(intermediate_size):
    from lightllm.common.quantization.deepgemm import MXFP4MoEQuantizationMethod
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.marlin_impl import FuseMoeMarlin

    method = MXFP4MoEQuantizationMethod()
    w13, parts = method.create_moe_weight([intermediate_size, intermediate_size], 128, torch.bfloat16, 0, num_experts=2)
    w2, _ = method.create_moe_weight([128], intermediate_size, torch.bfloat16, 0, num_experts=2)
    table = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
    raw, decoded = [], []
    for n, k in [(intermediate_size, 128), (intermediate_size, 128), (128, intermediate_size)]:
        codes = torch.randint(0, 16, (2, n, k))
        raw.append((codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.int8))
        decoded.append((table[codes] * 0.03125).bfloat16().cuda())
    for e in range(2):
        for pack, source in [
            (parts[0].get_expert(e), raw[0][e]),
            (parts[1].get_expert(e), raw[1][e]),
            (w2.get_expert(e), raw[2][e]),
        ]:
            method.load_weight(source, pack)
            method.load_weight_scale(
                torch.full((source.shape[0], source.shape[1] // 16), 0.03125).to(torch.float8_e8m0fnu), pack
            )
    holder = SimpleNamespace(w13=w13, w2=w2, data_type_=torch.bfloat16)
    method.finalize_moe_weight(holder)
    expert = FuseMoeMarlin.__new__(FuseMoeMarlin)
    expert.quant_method = method
    expert.n_routed_experts = 2
    expert.workspace = expert.create_workspace()
    x = torch.randn(3, 128, dtype=torch.bfloat16, device="cuda") * 2
    ids = torch.tensor([[0, 1], [1, 0], [0, 1]], device="cuda")
    weights = torch.tensor([[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]], device="cuda")
    actual = expert._fused_experts(x, w13, w2, weights, ids, alpha=1.0, limit=1.0, clamp_up_add_one=False)
    expected = torch.zeros_like(x, dtype=torch.float32)
    for row in range(3):
        for i in range(2):
            e = ids[row, i]
            gate = torch.nn.functional.linear(x[row], decoded[0][e]).clamp(max=1.0)
            up = torch.nn.functional.linear(x[row], decoded[1][e]).clamp(-1.0, 1.0)
            hidden = torch.nn.functional.silu(gate.float()).bfloat16() * up
            expected[row] += torch.nn.functional.linear(hidden, decoded[2][e]).float() * weights[row, i]
    torch.testing.assert_close(actual.float(), expected, atol=0.05, rtol=0.03)


@pytest.mark.parametrize("block_size", [None, 32, 128])
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
def test_o_projection_loads_bf16_groups_and_preserves_tp_slices(block_size, tp_size, tp_rank):
    from lightllm.common.basemodel.layer_weights.meta_weights import ROWBMMWeight, COLMMWeight
    from lightllm.models.deepseek_v41.layer_weights.transformer_layer_weight import DeepseekV41TransformerLayerWeight
    from lightllm.models.deepseek_v41.layer_infer.transformer_layer_infer import DeepseekV41TransformerLayerInfer

    groups, rank, width, hidden = 4, 128, 256, 64
    name = "layers.0.attn.wo_a.weight"
    raw = torch.randn(groups * rank, width).bfloat16() * 0.02
    weights = {name: raw}
    if block_size is not None:
        raw = (raw * 128).to(torch.float8_e4m3fn)
        scale = torch.exp2(torch.randint(-8, -3, (groups * rank // block_size, width // block_size)).float())
        weights = {name: raw, "layers.0.attn.wo_a.scale": scale.to(torch.float8_e8m0fnu)}
        reference = (raw.float() * scale.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)).bfloat16()
    else:
        reference = raw
    reference = reference.view(groups, rank, width).cuda()
    wb = torch.randn(hidden, groups * rank).bfloat16() * 0.02
    weights["layers.0.attn.wo_b.weight"] = wb
    layer = DeepseekV41TransformerLayerWeight.__new__(DeepseekV41TransformerLayerWeight)
    layer.prefix, layer.o_proj_fp8, layer.data_type_ = "layers.0", False, torch.bfloat16
    layer.o_groups, layer.o_lora_rank, layer.n_heads, layer.head_dim = groups, rank, 8, 128
    layer.network_config_ = {"quantization_config": {"weight_block_size": [block_size, block_size]}}
    layer.experts = SimpleNamespace(quant_method=SimpleNamespace(weight_scale_suffix=None))
    layer.wo_a_ = ROWBMMWeight(groups, width, rank, name, torch.bfloat16, tp_rank=tp_rank, tp_world_size=tp_size)
    layer.wo_b_ = COLMMWeight(
        groups * rank, [hidden], "layers.0.attn.wo_b.weight", torch.bfloat16, tp_rank=tp_rank, tp_world_size=tp_size
    )
    layer.load_hf_weights(weights)
    assert layer.wo_a_.verify_load() and layer.wo_b_.verify_load()
    local_groups = groups // tp_size
    start = tp_rank * local_groups
    wa = reference[start : start + local_groups]
    torch.testing.assert_close(layer.wo_a_.weight, wa.transpose(1, 2), atol=0, rtol=0)
    wb = wb[:, start * rank : (start + local_groups) * rank].cuda()
    infer = DeepseekV41TransformerLayerInfer.__new__(DeepseekV41TransformerLayerInfer)
    infer.tp_groups = local_groups
    infer._tpsp_reduce = lambda input, infer_state: input
    for tokens in (1, 17):
        # FlashMLA's padded head dimension can leave a gap between token rows.
        o = torch.randn(tokens, local_groups * 2 + 1, 128, device="cuda", dtype=torch.bfloat16)[:, :-1]
        grouped = o.reshape(tokens, local_groups, width)
        projected = torch.cat([torch.nn.functional.linear(grouped[:, g], wa[g]) for g in range(local_groups)], -1)
        expected = torch.nn.functional.linear(projected, wb)
        actual = infer._project_o(o, None, layer)
        torch.testing.assert_close(actual, expected, atol=0.015625, rtol=0.01)


def test_decoder_replay_ragged_rows_and_swa_floor_preserve_encoder_state():
    from lightllm.models.deepseek_v41.infer_struct import DeepseekV41InferStateInfo

    state = DeepseekV41InferStateInfo()
    state.req_manager = SimpleNamespace(cache_config=SimpleNamespace(sliding_window=128))
    state.prefill_att_state = SimpleNamespace(
        backend=SimpleNamespace(create_att_prefill_state=lambda _: SimpleNamespace(init_state=lambda: None))
    )
    state.b_q_seq_len = torch.tensor([200, 3, 129], dtype=torch.int32, device="cuda")
    state.b_seq_len = torch.tensor([220, 1003, 257], dtype=torch.int32, device="cuda")
    state.b_ready_cache_len = state.b_seq_len - state.b_q_seq_len
    state.max_q_seq_len = 200
    state._dsv4_token_to_batch_idx = torch.repeat_interleave(torch.arange(3, device="cuda"), state.b_q_seq_len)
    state.position_ids = torch.cat(
        [torch.arange(a, b, device="cuda") for a, b in ((20, 220), (1000, 1003), (128, 257))]
    )
    for name in (
        "input_ids",
        "mem_index",
        "dsv4_sparse_req_idx",
        "dsv4_swa_write_slots",
        "dsv4_swa_runtime_write_slots",
    ):
        setattr(state, name, torch.arange(332, device="cuda"))
    for name in ("position_cos_sliding", "position_sin_sliding", "position_cos_compress", "position_sin_compress"):
        setattr(state, name, torch.randn(332, 32, device="cuda"))
    state.dsv4_swa_indices = torch.arange(332 * 128, dtype=torch.int32, device="cuda").view(332, 128)
    state.dsv4_swa_lengths = torch.full((332,), 128, dtype=torch.int32, device="cuda")
    state.v41_sparse_selection = {"extra_indices": torch.arange(332, device="cuda")[:, None, None]}
    state.candidate_blocks = torch.arange(332, device="cuda")[:, None]
    state.engram_hashes = torch.arange(332, device="cuda")[:, None, None]
    state._dsv4_prefill_pad_q_len = 129
    original_swa = state.dsv4_swa_indices.clone()
    rows = state.init_decoder_state()
    expected_rows = torch.cat((torch.arange(72, 203, device="cuda"), torch.arange(204, 332, device="cuda")))
    torch.testing.assert_close(rows, expected_rows)
    decoder = state.decoder_state
    assert decoder.req_manager is state.req_manager
    assert decoder.max_q_seq_len == 128
    assert decoder.b_q_seq_len.tolist() == [128, 3, 128]
    assert decoder.b_ready_cache_len.tolist() == [92, 1000, 129]
    assert decoder.b_q_start_loc.tolist() == [0, 128, 131]
    assert decoder._dsv4_prefill_pad_q_len == 128
    torch.testing.assert_close(decoder.engram_hashes[:, 0, 0], expected_rows)
    torch.testing.assert_close(decoder.candidate_blocks[:, 0], expected_rows)
    torch.testing.assert_close(decoder.v41_sparse_selection["extra_indices"][:, 0, 0], expected_rows)
    expected_lengths = torch.tensor([*range(1, 129), 1, 2, 3, *range(1, 129)], device="cuda", dtype=torch.int64)
    torch.testing.assert_close(decoder.dsv4_swa_lengths, expected_lengths)
    invalid = torch.arange(128, device="cuda")[None, :] >= expected_lengths[:, None]
    assert (decoder.dsv4_swa_indices[invalid] == -1).all()
    torch.testing.assert_close(state.dsv4_swa_indices, original_swa)
    assert state.b_ready_cache_len.tolist() == [20, 1000, 128]


@pytest.mark.parametrize(
    "expert_dtype,block_size,o_groups,vision,dspark",
    [
        ("fp4", 32, 1, False, False),
        ("fp4", 32, 2, False, False),
        ("fp8", 128, 1, False, False),
        ("fp8", 128, 2, False, False),
        ("fp8", 128, 1, True, False),
        ("fp8", 128, 1, False, True),
        ("fp8", 128, 1, True, True),
    ],
)
def test_text_model_constructs_and_chunked_prefill_matches_full(
    tmp_path, monkeypatch, o_groups, expert_dtype, block_size, vision, dspark
):
    """Exercise the real model input, paging, shared KV, attention, MoE and output paths."""
    import json
    import uuid
    from dataclasses import asdict
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import (
        set_env_start_args,
        get_env_start_args,
        get_unique_server_name,
        get_added_mtp_kv_layer_num,
    )
    from lightllm.models.deepseek_v41.model import DeepseekV41TpPartModel, DeepseekV41VisionTpPartModel
    from lightllm.models.deepseek_v41_dspark.model import DeepseekV41DSparkModel
    from lightllm.models.deepseek_v41_dspark.layer_infer.post_layer_infer import DeepseekV41DSparkPostLayerInfer
    from lightllm.models.deepseek_v41_dspark.layer_weights.pre_and_post_layer_weight import (
        DeepseekV41DSparkPreAndPostLayerWeight,
    )
    from lightllm.common.basemodel.batch_objs import ModelInput
    from lightllm.common.basemodel.layer_weights.meta_weights import MMWeightTpl, ROWBMMWeight
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
    from lightllm.common.basemodel.layer_weights.meta_weights.base_weight import BaseWeight
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.common.basemodel.triton_kernel.mhc import pre_norm

    config = dict(
        model_type="deepseek_v41",
        expert_dtype=expert_dtype,
        architectures=["DeepseekV41ForCausalLM"],
        quantization_config=dict(
            quant_method="fp8", weight_block_size=[block_size, block_size], scale_fmt="ue8m0", expert_dtype=expert_dtype
        ),
        text_config=dict(
            vocab_size=64,
            hidden_size=128,
            num_hidden_layers=5,
            num_attention_heads=o_groups,
            num_key_value_heads=1,
            head_dim=512,
            qk_rope_head_dim=64,
            q_lora_rank=128 if block_size == 128 else 32,
            o_lora_rank=128 if block_size == 128 else 32,
            o_groups=o_groups,
            rms_norm_eps=1e-20,
            sliding_window=128,
            compress_ratios=[0, 2, 2, 1, 1],
            kv_source_layer_ids=[1, 3],
            index_source_layer_ids=[1, 3, 4],
            candidate_source_layer_id=3,
            candidate_topk_blocks=2,
            candidate_block_size=8,
            index_n_heads=2,
            index_head_dim=128,
            index_topk=128,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            engram_layer_ids=[1],
            engram_compressed_vocab_size=64,
            engram_pad_token_id=2,
            engram_max_ngram_size=4,
            engram_vocab_size=101,
            engram_n_heads=2,
            engram_head_dim=64,
            engram_num_embeddings=[660],
            n_routed_experts=8,
            n_shared_experts=1,
            num_experts_per_tok=6,
            moe_intermediate_size=128,
            scoring_func="sqrtsoftplus",
            norm_topk_prob=True,
            routed_scaling_factor=1.5,
            swiglu_limit=10.0,
            max_position_embeddings=512,
            rope_theta=10000.0,
            compress_rope_theta=160000.0,
        ),
    )
    if vision:
        config["vision_config"] = dict(
            num_hidden_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            intermediate_size=12,
            patch_size=2,
            rope_theta=10000,
            downsample_ratio=1,
            max_image_tokens=64,
            min_pixels=0,
            max_wh_ratio=None,
        )
    if dspark:
        config["text_config"].update(
            compress_ratios=[0, 2, 2, 1, 1, 0, 0, 0],
            num_nextn_predict_layers=3,
            dspark_block_size=5,
            dspark_noise_token_id=0,
            dspark_target_layer_ids=[2, 3, 4],
            dspark_markov_rank=128,
            dspark_n_routed_experts=4,
            dspark_num_experts_per_tok=2,
        )
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from lightllm.models.deepseek_v41.engram import EngramEmbeddingWeight

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(64)}, unk_token="t0"))
    )
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(config))
    args = StartArgs(
        model_dir=str(tmp_path),
        data_type="bfloat16",
        page_size=256,
        batch_max_tokens=512,
        run_mode="normal",
        disable_dynamic_prompt_cache=True,
        disable_cudagraph=True,
        max_req_total_len=512,
        disable_symm_mem_allreduce=True,
        disable_flashinfer_allreduce=True,
        output_constraint_mode="none",
        mtp_mode="dspark" if dspark else None,
        mtp_step=3 if dspark else 0,
        mtp_draft_model_dir=[str(tmp_path)] if dspark else None,
    )
    set_env_start_args(asdict(args))
    get_env_start_args.cache_clear()
    get_added_mtp_kv_layer_num.cache_clear()
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test_v41_" + uuid.uuid4().hex)
    get_unique_server_name.cache_clear()
    monkeypatch.setattr(pre_norm, "LIGHTLLM_DISABLE_DEEPGEMM_MHC", True)
    monkeypatch.setattr(MtpManager, "_instance", None)
    torch.distributed.init_process_group("nccl", init_method=f"file://{tmp_path}/nccl", rank=0, world_size=1)
    dist_group_manager.create_groups(1)

    class TinyModel(DeepseekV41VisionTpPartModel if vision else DeepseekV41TpPartModel):
        def _autotune_warmup(self):
            pass

        def _check_max_len_infer(self):
            pass

        def load_weights(self, weight_dict):
            # Generate a small BF16 checkpoint; use the production weight loaders below.
            weights = {}
            for layer in [self.pre_post_weight, *self.trans_layers_weight]:
                for attr in vars(layer).values():
                    if self.is_mtp_draft_model and (
                        attr is self.pre_post_weight.wte_weight_ or attr is self.pre_post_weight.lm_head_weight_
                    ):
                        continue
                    if isinstance(attr, FusedMoeWeight):
                        for expert in range(attr.n_routed_experts):
                            for name, shape in [("w1", (128, 128)), ("w3", (128, 128)), ("w2", (128, 128))]:
                                prefix = f"{attr.weight_prefix}.{expert}.{name}"
                                if expert_dtype == "fp4":
                                    weights[prefix + ".weight"] = torch.randint(
                                        -128, 128, (shape[0], shape[1] // 2), dtype=torch.int8
                                    )
                                    scale_shape = (shape[0], shape[1] // 32)
                                else:
                                    weights[prefix + ".weight"] = (torch.randn(shape) * 2).to(torch.float8_e4m3fn)
                                    scale_shape = (shape[0] // block_size, shape[1] // block_size)
                                weights[prefix + ".scale"] = torch.full(scale_shape, 0.0078125).to(torch.float8_e8m0fnu)
                    elif isinstance(attr, ROWBMMWeight):
                        name = attr.weight_names[0]
                        value = torch.randn(o_groups * attr.dim2, attr.dim1).bfloat16() * 0.02
                        weights[name] = (value / 0.0078125).to(torch.float8_e4m3fn)
                        weights[name.removesuffix("weight") + "scale"] = torch.full(
                            (value.shape[0] // block_size, value.shape[1] // block_size), 0.0078125
                        ).to(torch.float8_e8m0fnu)
                    elif isinstance(attr, MMWeightTpl):
                        for name, part in zip(attr.weight_names, attr.mm_param_list):
                            value = torch.randn(part.weight.shape).bfloat16() * 0.02
                            if attr.quant_method.method_name.startswith("fp8w8a8-"):
                                value = (value / 0.0078125).to(torch.float8_e4m3fn)
                                weights[name.removesuffix("weight") + "scale"] = torch.full(
                                    (value.shape[0] // block_size, value.shape[1] // block_size), 0.0078125
                                ).to(torch.float8_e8m0fnu)
                            weights[name] = value
                    elif isinstance(attr, EngramEmbeddingWeight):
                        weights[attr.prefix + ".weight"] = torch.randn(660, 64).to(torch.float8_e4m3fn)
                        weights[attr.prefix + ".scale"] = torch.full((660, 2), 0.03125).to(torch.float8_e8m0fnu)
                    elif isinstance(attr, BaseWeight):
                        value = (
                            torch.ones_like(attr.weight, device="cpu")
                            if "norm" in attr.weight_name
                            else torch.randn_like(attr.weight, device="cpu") * 0.02
                        )
                        weights[attr.weight_name] = value
            if self.is_mtp_draft_model:
                DeepseekV41DSparkModel.load_weights(self, weights)
            else:
                super().load_weights(weights)

    model = None
    try:
        model = TinyModel(
            dict(
                weight_dir=str(tmp_path),
                max_total_token_num=1024,
                run_mode="normal",
                max_req_num=2,
                max_seq_length=512,
                batch_max_tokens=512,
                disable_cudagraph=True,
            )
        )
        with monkeypatch.context() as unsupported:
            unsupported.setattr(model.args, "enable_prefill_cudagraph", True)
            with pytest.raises(ValueError, match="prefill CUDA Graph"):
                model._verify_params()
        with monkeypatch.context() as cpu_cache:
            cpu_cache.setattr(model.args, "enable_cpu_cache", True)
            for role in ("normal", "prefill"):
                cpu_cache.setattr(model.args, "run_mode", role)
                for disk in (False, True):
                    cpu_cache.setattr(model.args, "enable_disk_cache", disk)
                    model._verify_params()
            cpu_cache.setattr(model.args, "run_mode", "decode")
            with pytest.raises(ValueError, match="CPU cache on Decode nodes"):
                model._verify_params()
        tokens = torch.arange(1, 301, dtype=torch.int64) % 64
        images = []
        if vision:
            from lightllm.server.router.model_infer.infer_batch import g_infer_context

            tokens[5:17] = torch.arange(1000, 1012)
            images = [dict(token_id=1000, token_num=12, start_idx=5, start_index_in_embed_cache=0)]
            embed_cache = torch.randn(12, 1, 128, device="cuda", dtype=torch.bfloat16)
            monkeypatch.setattr(
                g_infer_context, "cpu_embed_cache_client", SimpleNamespace(cpu_embed_cache_tensor=embed_cache)
            )
        model.req_manager.req_to_token_indexs[0, :512] = torch.arange(512, dtype=torch.int32, device="cuda")

        def prefill_input(start, end, req_idx=0):
            tensor = lambda value: torch.tensor([value], dtype=torch.int32)
            return ModelInput(
                batch_size=1,
                total_token_num=end,
                max_q_seq_len=end - start,
                max_kv_seq_len=end,
                max_cache_len=start,
                input_ids=tokens[start:end].clone(),
                b_req_idx=tensor(req_idx),
                b_mtp_index=tensor(0),
                b_seq_len=tensor(end),
                b_is_decode_req=torch.tensor([False]),
                b_ready_cache_len=tensor(start),
                b_prefill_start_loc=tensor(0),
                b_prefill_has_output_cpu=[True],
                is_prefill=True,
                multimodal_params=[{"images": images, "audios": []}],
                token_lookback=[(req_idx, start, tokens[max(0, start - model.token_lookback_depth) : start].tolist())]
                if start and model.token_lookback_depth
                else None,
            )

        def prefill(start, end, req_idx=0):
            return model.forward(prefill_input(start, end, req_idx)).logits.clone()

        if dspark:
            from copy import copy
            from lightllm.models.draft_registry import DraftModelRegistry
            from lightllm.common.basemodel.cuda_graph import CudaGraph

            assert DraftModelRegistry.get_model_class(config, "dspark") is DeepseekV41DSparkModel
            assert model.req_manager.cache_config.layer_num == 8
            captured = []
            for index in (2, 3, 4):
                layer = model.layers_infer[index]
                original = layer._hc_attn_in

                def record_attention_input(hidden, state, weight, original=original):
                    streams = hidden[0] if isinstance(hidden, tuple) else hidden
                    captured.append(streams.view(-1, 4, 128).mean(1).clone())
                    return original(hidden, state, weight)

                monkeypatch.setattr(layer, "_hc_attn_in", record_attention_input)
            inp = prefill_input(0, 256)
            out = model.forward(inp)
            assert out.mtp_collector.spec_hidden.shape == (128, 384)
            assert out.mtp_collector.spec_hidden_prefill_ready_cache_len.tolist() == [128]
            torch.testing.assert_close(
                out.mtp_collector.spec_hidden, torch.cat([hidden[-128:] for hidden in captured], -1), atol=0, rtol=0
            )

            class TinyDraft(DeepseekV41DSparkModel):
                load_weights = TinyModel.load_weights

                def _init_cudagraph(self):
                    self.graph = None

                def _check_max_len_infer(self):
                    pass

            draft = TinyDraft(
                dict(
                    weight_dir=str(tmp_path),
                    main_model=model,
                    mtp_previous_draft_models=[],
                    run_mode="normal",
                    max_total_token_num=1024,
                    max_req_num=2,
                    max_seq_length=512,
                    batch_max_tokens=512,
                    disable_cudagraph=True,
                )
            )
            assert draft.req_manager is model.req_manager
            assert draft.mem_manager is model.mem_manager
            assert draft.block_size == 3
            assert type(draft.post_infer) is DeepseekV41DSparkPostLayerInfer
            assert type(draft.pre_post_weight) is DeepseekV41DSparkPreAndPostLayerWeight
            assert [weight.n_routed_experts for weight in draft.trans_layers_weight] == [4, 4, 4]
            assert all(layer.num_experts_per_tok == 2 for layer in draft.layers_infer)
            assert draft.pre_post_weight.markov_w1_weight_.weight_name == "mtp.2.markov_head.embed.weight"
            assert not hasattr(draft.pre_post_weight, "hc_head_fn_")
            commit = prefill_input(128, 256)
            commit.mtp_draft_input_hiddens = out.mtp_collector.spec_hidden
            with monkeypatch.context() as kv_commit:
                kv_commit.setattr(
                    dist_group_manager,
                    "clear_deepep_buffer",
                    lambda: pytest.fail("KV-only commits must not clear the target's DeepEP buffer"),
                )
                draft.forward(commit)
            # Every draft stage owns target-hidden KV, without modifying target history.
            assert model.req_manager.swa_buffer[5:].count_nonzero() > 0

            def decode_input(ids, position):
                count = len(ids)
                return ModelInput(
                    batch_size=count,
                    total_token_num=count,
                    max_q_seq_len=1,
                    max_kv_seq_len=position + count,
                    max_cache_len=position,
                    input_ids=ids,
                    b_req_idx=torch.zeros(count, dtype=torch.int32),
                    b_mtp_index=torch.zeros(count, dtype=torch.int32),
                    b_seq_len=torch.arange(position + 1, position + count + 1, dtype=torch.int32),
                    b_position_delta=torch.zeros(count, dtype=torch.int32),
                    b_shared_seq_len=torch.zeros(count, dtype=torch.int32),
                    b_shared_radix_node_id=torch.zeros(count, dtype=torch.int32),
                    is_prefill=False,
                    multimodal_params=[{"images": [], "audios": []}] * count,
                )

            verify = decode_input(tokens[256:260], 256)
            verified = model.forward(verify)
            verify_logits = verified.logits.clone()
            assert verified.mtp_collector.spec_hidden.shape == (4, 384)
            commit = copy(verify)
            commit.mtp_draft_input_hiddens = verified.mtp_collector.spec_hidden
            draft.forward(commit)
            history = [buffer.clone() for buffer in model.mem_manager.kv_buffer]
            proposal = decode_input(torch.tensor([int(verified.logits[-1].argmax()), 0, 0]), 260)
            eager = draft.forward(proposal)
            expected_ids = eager.mtp_collector.draft_token_ids.clone()
            expected_confidence = eager.mtp_collector.confidence_logits.clone()
            assert expected_ids.shape == (3,)
            assert expected_confidence.shape == (1, 3)
            assert torch.isfinite(expected_confidence).all()
            draft.graph = CudaGraph(3, 3, 3, max_batch_size=3, max_len_in_batch=512)
            for _ in range(2):
                actual = draft.forward(proposal)
                torch.testing.assert_close(actual.mtp_collector.draft_token_ids, expected_ids, atol=0, rtol=0)
                torch.testing.assert_close(actual.mtp_collector.confidence_logits, expected_confidence, atol=0, rtol=0)
            assert len(draft.graph.graph) == 1
            for actual, expected in zip(model.mem_manager.kv_buffer, history):
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)

            # Target verification must match autoregressive decode, including after rejected future rows.
            model.req_manager.clear_runtime_state(0)
            model.forward(inp)
            sequential = torch.cat(
                [model.forward(decode_input(tokens[pos : pos + 1], pos)).logits.clone() for pos in range(256, 260)]
            )
            torch.testing.assert_close(verify_logits, sequential, atol=0.0625, rtol=0.03)
            # Match the verify layout to isolate rejected-state leakage from FlashMLA batch rounding.
            model.req_manager.clear_runtime_state(0)
            model.forward(inp)
            model.forward(verify)
            clean_resume = model.forward(decode_input(tokens[257:260], 257)).logits.clone()
            model.req_manager.clear_runtime_state(0)
            model.forward(inp)
            rejected_ids = tokens[256:260].clone()
            rejected_ids[1:] = (rejected_ids[1:] + 7) % 64
            model.forward(decode_input(rejected_ids, 256))
            resumed = model.forward(decode_input(tokens[257:260], 257)).logits
            torch.testing.assert_close(resumed, clean_resume, atol=0, rtol=0)

            # CED suffix commits must preserve draft SWA across a prefix-cache hit.
            from lightllm.server.router.model_infer.mtp_speculative.proposers.dspark import DSparkProposer

            proposer = DSparkProposer(backend=SimpleNamespace(draft_models=[draft]), enable_dynmaic_mtp=False)

            def prefill_and_commit(start, end, req_idx=0):
                model_input = prefill_input(start, end, req_idx)
                output = model.forward(model_input)
                assert proposer.fill_draft_model_kv_state(model_input, output, output.logits.argmax(-1)) is None

            model.req_manager.clear_runtime_state(0)
            prefill_and_commit(0, 256)
            checkpoints = model.req_manager.create_small_page_cache_manager(1)
            model.req_manager.save_state(0, 0, checkpoints, checkpoint_len=256)
            prefill_and_commit(256, 257)
            from lightllm.models.deepseek_v4.triton_kernel.swa_cache import gather_swa_tokens

            positions = torch.arange(129, 257, device="cuda")
            expected = gather_swa_tokens(model.req_manager, 0, positions)[5:]
            model.req_manager.req_to_token_indexs[1, :256] = model.req_manager.req_to_token_indexs[0, :256]
            model.req_manager.req_to_token_indexs[1, 256] = 512
            model.req_manager.restore_state(SimpleNamespace(req_idx=1, cur_kv_len=256), checkpoints, 0)
            prefill_and_commit(256, 257, req_idx=1)
            actual = gather_swa_tokens(model.req_manager, 1, positions)[5:]
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            return

        # Full prompt logits keep the complete decoder path; retain the exact
        # prefill/chunk/decode comparisons independently of approximate replay.
        model.return_all_prompt_logics = True
        full = prefill(0, 33)
        assert torch.isfinite(full).all()
        model.req_manager.clear_runtime_state(0)
        prefill(0, 7)
        chunked = prefill(7, 33)
        # FlashMLA's batch-dependent BF16 rounding can cross a subsequent FP8 bin boundary.
        torch.testing.assert_close(full, chunked, atol=0.0625, rtol=0.03)
        references = []
        for end in (34, 35, 36):
            model.req_manager.clear_runtime_state(0)
            references.append(prefill(0, end))

        def decode(position, req_idx=0):
            tensor = lambda value: torch.tensor([value], dtype=torch.int32)
            inp = ModelInput(
                batch_size=1,
                total_token_num=position + 1,
                max_q_seq_len=1,
                max_kv_seq_len=position + 1,
                max_cache_len=position,
                input_ids=tokens[position : position + 1].clone(),
                b_req_idx=tensor(req_idx),
                b_mtp_index=tensor(0),
                b_seq_len=tensor(position + 1),
                b_position_delta=tensor(0),
                b_shared_seq_len=tensor(0),
                b_shared_radix_node_id=tensor(0),
                is_prefill=False,
                multimodal_params=[{"images": [], "audios": []}],
                token_lookback=[
                    (req_idx, position, tokens[max(0, position - model.token_lookback_depth) : position].tolist())
                ]
                if req_idx and model.token_lookback_depth
                else None,
            )
            return model.forward(inp).logits.clone()

        # P/D restores C2 state; Engram predecessors are rebuilt from the
        # receiving request's tokens at its new physical KV pages.
        from lightllm.models.deepseek_v4.pd_cache import _copy_request_state, get_pd_page_nbytes
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_history_pages

        with monkeypatch.context() as pd:
            for role in ("prefill", "decode"):
                pd.setattr(model.args, "run_mode", role)
                model._verify_params()
            pd.setattr(model.args, "run_mode", "normal")
            pd.setattr(model, "return_all_prompt_logics", False)
            model.req_manager.clear_runtime_state(0)
            prefill(0, 257)
            model.req_manager.req_to_token_indexs[1, :512] = torch.arange(512, 1024, dtype=torch.int32, device="cuda")
            manager = SimpleNamespace(
                cache_config=model.req_manager.cache_config,
                req_to_token_indexs=model.req_manager.req_to_token_indexs,
                **{
                    name: getattr(model.req_manager, name)
                    for name in (
                        "swa_buffer",
                        "swa_capacity_tokens",
                        "HOLD_REQUEST_ID",
                        "HOLD_SWA_SLOT",
                        "req_to_c4_state",
                        "req_to_c4_indexer_state",
                        "req_to_c128_state",
                    )
                },
            )
            state_page = torch.empty(get_pd_page_nbytes(manager.cache_config, 256), dtype=torch.uint8, device="cuda")
            _copy_request_state(manager, state_page, 0, 257, write=True)
            history_page = torch.empty_like(state_page).view(1, -1)
            page_id = torch.zeros(1, dtype=torch.int32, device="cuda")
            for start, end in ((0, 256), (256, 257)):
                for req, write in ((0, True), (1, False)):
                    copy_history_pages(
                        model.mem_manager,
                        manager.req_to_token_indexs[req, start:end].long(),
                        page_id,
                        history_page,
                        to_pages=write,
                        page_tokens=256,
                        round_tail=True,
                    )
            expected = [decode(pos) for pos in (257, 258)]
            model.req_manager.clear_runtime_state(1)
            model.engram_hasher.history[512:769].fill_(-1)
            _copy_request_state(manager, state_page, 1, 257, write=False)
            for pos, reference in zip((257, 258), expected):
                torch.testing.assert_close(decode(pos, req_idx=1), reference, atol=0, rtol=0)

        model.req_manager.clear_runtime_state(0)
        prefill(0, 33)
        eager_decode = []
        for position, expected in zip((33, 34, 35), references):
            actual = decode(position)
            torch.testing.assert_close(actual, expected, atol=0.0625, rtol=0.03)
            eager_decode.append(actual)

        from lightllm.common.basemodel.cuda_graph import CudaGraph

        model.graph = CudaGraph(1, 1, 1, max_batch_size=1, max_len_in_batch=512)
        model.req_manager.clear_runtime_state(0)
        prefill(0, 33)
        for position, expected in zip((33, 34, 35), eager_decode):
            torch.testing.assert_close(decode(position), expected, atol=0, rtol=0)
        assert len(model.graph.graph) == 1

        # Reuse a 256-token prefix in another request slot, with different suffix slots.
        model.graph = None
        model.req_manager.req_to_token_indexs[0, :512] = torch.arange(512, dtype=torch.int32, device="cuda")
        model.req_manager.clear_runtime_state(0)
        prefill(0, 256)
        checkpoints = model.req_manager.create_small_page_cache_manager(1)
        model.req_manager.save_state(0, 0, checkpoints, checkpoint_len=256)
        assert checkpoints.c4_shape[0] == 0
        assert checkpoints.indexer_shape[0] == 0
        expected_prefill = prefill(256, 257)
        expected_decode = [decode(position) for position in (257, 258)]
        model.req_manager.req_to_token_indexs[1, :256] = model.req_manager.req_to_token_indexs[0, :256]
        model.req_manager.req_to_token_indexs[1, 256:259] = torch.arange(512, 515, dtype=torch.int32, device="cuda")
        model.req_manager.restore_state(SimpleNamespace(req_idx=1, cur_kv_len=256), checkpoints, 0)
        torch.testing.assert_close(prefill(256, 257, req_idx=1), expected_prefill, atol=0, rtol=0)
        for position, expected in zip((257, 258), expected_decode):
            torch.testing.assert_close(decode(position, req_idx=1), expected, atol=0, rtol=0)

        # CPU pages must resume at new physical slots after all source Engram
        # history is lost, rather than relying on a still-resident GPU prefix.
        from dataclasses import replace
        from lightllm.models.deepseek_v4.triton_kernel.cache_io import copy_checkpoint_pages

        with monkeypatch.context() as cpu_cache:
            manager, requests = model.mem_manager, model.req_manager
            layout = replace(manager.cache_config, cpu_cache_token_page_size=256)
            cpu_cache.setattr(manager, "cache_config", layout)
            torch.cuda.synchronize()
            manager.big_page_buffers.buffer[0].copy_(checkpoints.buffer[0])
            pages = torch.empty((1, layout.get_cpu_cache_big_page_bytes()), dtype=torch.uint8, pin_memory=True)
            page_ids = torch.zeros(1, dtype=torch.int32, device="cuda")
            checkpoint_ids = torch.zeros(1, dtype=torch.int64, device="cuda")
            copy_history_pages(manager, requests.req_to_token_indexs[0, :256].long(), page_ids, pages, to_pages=True)
            copy_checkpoint_pages(manager, checkpoint_ids, page_ids, pages, to_pages=True)
            torch.cuda.synchronize()
            manager.big_page_buffers.buffer[0].zero_()
            model.engram_hasher.history.fill_(-99)
            requests.req_to_token_indexs[1, :512] = torch.arange(512, 1024, dtype=torch.int32, device="cuda")
            copy_history_pages(manager, requests.req_to_token_indexs[1, :256].long(), page_ids, pages, to_pages=False)
            copy_checkpoint_pages(manager, checkpoint_ids, page_ids, pages, to_pages=False)
            requests.restore_big_page_state(0, SimpleNamespace(req_idx=1, cur_kv_len=256))
            torch.testing.assert_close(prefill(256, 257, req_idx=1), expected_prefill, atol=0, rtol=0)
            for position, expected in zip((257, 258), expected_decode):
                torch.testing.assert_close(decode(position, req_idx=1), expected, atol=0, rtol=0)
            requests.req_to_token_indexs[1, :256] = requests.req_to_token_indexs[0, :256]
            requests.req_to_token_indexs[1, 256:259] = torch.arange(512, 515, dtype=torch.int32, device="cuda")
        assert len(model.mem_manager.kv_buffer) == 2
        assert (
            get_compressed_kv_buffer(model.mem_manager, 1).data_ptr()
            == get_compressed_kv_buffer(model.mem_manager, 2).data_ptr()
        )
        assert (
            get_compressed_kv_buffer(model.mem_manager, 3).data_ptr()
            == get_compressed_kv_buffer(model.mem_manager, 4).data_ptr()
        )
        model.req_manager.req_to_token_indexs[0, :512] = torch.arange(512, dtype=torch.int32, device="cuda")
        model.req_manager.clear_runtime_state(0)
        prefill(0, 256)
        full_history = [buffer.clone() for buffer in model.mem_manager.kv_buffer]
        model.return_all_prompt_logics = False
        counts = []
        decoder = model.layers_infer[4]
        original_ffn = decoder._ffn

        def record_decoder_ffn(x, state, weight):
            counts.append(x.shape[0])
            return original_ffn(x, state, weight)

        monkeypatch.setattr(decoder, "_ffn", record_decoder_ffn)
        model.req_manager.clear_runtime_state(0)
        assert torch.isfinite(prefill(0, 256)).all()
        assert counts == [128]
        for actual, expected in zip(model.mem_manager.kv_buffer, full_history):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        model.req_manager.save_state(0, 0, checkpoints, checkpoint_len=256)
        expected = prefill(256, 257)
        assert counts[-1] == 1
        model.req_manager.restore_state(SimpleNamespace(req_idx=1, cur_kv_len=256), checkpoints, 0)
        # Decoder replay must not read saved SWA before the replay boundary.
        model.req_manager.req_to_swa_state[4, 1].fill_(255)
        torch.testing.assert_close(prefill(256, 257, req_idx=1), expected, atol=0, rtol=0)
        assert torch.isfinite(decode(257, req_idx=1)).all()
        model.req_manager.req_to_token_indexs[1, :512] = torch.arange(512, 1024, dtype=torch.int32, device="cuda")
        for request in (0, 1):
            model.req_manager.clear_runtime_state(request)
        batch = ModelInput(
            batch_size=2,
            total_token_num=203,
            max_q_seq_len=200,
            max_kv_seq_len=200,
            max_cache_len=0,
            input_ids=torch.cat((tokens[:200], tokens[:3])),
            b_req_idx=torch.tensor([0, 1], dtype=torch.int32),
            b_mtp_index=torch.zeros(2, dtype=torch.int32),
            b_seq_len=torch.tensor([200, 3], dtype=torch.int32),
            b_is_decode_req=torch.zeros(2, dtype=torch.bool),
            b_ready_cache_len=torch.zeros(2, dtype=torch.int32),
            b_prefill_start_loc=torch.tensor([0, 200], dtype=torch.int32),
            b_prefill_has_output_cpu=[True, True],
            is_prefill=True,
            multimodal_params=[{"images": images, "audios": []}] * 2,
        )
        batched = model.forward(batch).logits.clone()
        assert counts[-1] == 131
        for request, length in ((0, 200), (1, 3)):
            model.req_manager.clear_runtime_state(request)
            reference = prefill(0, length, req_idx=request)
            torch.testing.assert_close(batched[request], reference[0], atol=0.0625, rtol=0.03)
        prefill(200, 256)
        assert counts[-1] == 56
        assert torch.isfinite(decode(256)).all()
        # Startup captures decode graphs with repeated HOLD requests before any real prefill.
        model.req_manager.free_all()
        model.mem_manager.free_all()
        model._init_padded_req()
        model.graph = CudaGraph(1, 1, 1, max_batch_size=32, max_len_in_batch=512)
        model.graph.warmup(model)
    finally:
        if model is not None:
            model.req_manager.clear_runtime_state(0)
            for allocator in (model.mem_manager.allocator,):
                allocator.shared_can_use_token_num.shm.close()
                allocator.shared_can_use_token_num.shm.unlink()
        torch.distributed.destroy_process_group()
        dist_group_manager.groups.clear()
        get_env_start_args.cache_clear()
        get_added_mtp_kv_layer_num.cache_clear()
        get_unique_server_name.cache_clear()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [16, 5120])
@pytest.mark.parametrize("masked", [False, True])
def test_engram_gate_matches_reference_and_cuda_graph(dtype, dim, masked):
    from lightllm.models.deepseek_v41.engram import engram_forward

    tokens, mult = 7, 4
    # Both producers may expose row slices; no contiguous copies are needed.
    streams = torch.randn(tokens * 2, mult, dim, device="cuda", dtype=dtype)[::2]
    kv = torch.randn(tokens * 2, (mult + 1) * dim, device="cuda", dtype=dtype)[::2]
    streams[0].zero_()
    kv[0, : mult * dim].zero_()
    weight = SimpleNamespace(
        engram_embed_=SimpleNamespace(lookup=lambda _: None),
        engram_wkv_=SimpleNamespace(mm=lambda _: kv),
        engram_q_=SimpleNamespace(weight=torch.randn(mult, dim, device="cuda", dtype=dtype)),
        engram_k_=SimpleNamespace(weight=torch.randn(mult, dim, device="cuda", dtype=dtype)),
    )
    token_mask = torch.arange(tokens, device="cuda") % 2 == 0 if masked else None
    eps = 1e-20

    def reference():
        h = streams.float()
        key = kv[:, : mult * dim].float().reshape(tokens, mult, dim)
        value = kv[:, mult * dim :].float()
        dot = (h * key * weight.engram_q_.weight * weight.engram_k_.weight).sum(-1)
        dot *= torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps) * dim ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = torch.where(token_mask[:, None], gate, 0)
        return (h + gate[..., None] * value[:, None, :]).to(dtype)

    def check(actual):
        torch.testing.assert_close(actual, reference(), atol=0.015625 if dtype == torch.bfloat16 else 2e-6, rtol=1e-5)
        if masked:
            torch.testing.assert_close(actual[~token_mask], streams[~token_mask], atol=0, rtol=0)

    check(engram_forward(streams, None, weight, eps, token_mask))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = engram_forward(streams, None, weight, eps, token_mask)
    streams.normal_()
    kv.normal_()
    graph.replay()
    check(actual)


def test_engram_image_tokens_do_not_modify_residual_streams():
    from lightllm.models.deepseek_v41.engram import engram_forward

    streams = torch.randn(3, 4, 16, device="cuda", dtype=torch.bfloat16)
    weight = SimpleNamespace(
        engram_embed_=SimpleNamespace(lookup=lambda _: torch.ones(3, 16, device="cuda")),
        engram_wkv_=SimpleNamespace(mm=lambda _: torch.ones(3, 80, device="cuda")),
        engram_q_=SimpleNamespace(weight=torch.ones(4, 16, device="cuda")),
        engram_k_=SimpleNamespace(weight=torch.ones(4, 16, device="cuda")),
    )
    token_mask = torch.tensor([True, False, True], device="cuda")
    hashes = torch.zeros(3, 1, dtype=torch.int64, device="cuda")
    expected = engram_forward(streams, hashes, weight, 1e-6)
    actual = engram_forward(streams, hashes, weight, 1e-6, token_mask=token_mask)
    torch.testing.assert_close(actual[~token_mask], streams[~token_mask], atol=0, rtol=0)
    torch.testing.assert_close(actual[token_mask], expected[token_mask], atol=0, rtol=0)
    assert not torch.equal(actual[token_mask], streams[token_mask])


def test_engram_host_embedding_uses_fp8_scales_and_zeroes_other_tp_rows(monkeypatch):
    from lightllm.models.deepseek_v41.engram import EngramEmbeddingWeight

    for key in ("GLOBAL_WORLD_SIZE", "DP_WORLD_SIZE"):
        monkeypatch.setenv("LIGHTLLM_" + key, "2")
    embedding = EngramEmbeddingWeight("layers.1.engram.embed", rows=8, dim=64)
    raw = torch.randn(8, 64).to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-4, 0, (8, 2)).float()).to(torch.float8_e8m0fnu)
    embedding.load_hf_weights({embedding.prefix + ".weight": raw, embedding.prefix + ".scale": scale})
    assert embedding.verify_load()
    # Inspect this rank's contribution before the normal TP sum.
    embedding.tp_world_size_ = 1
    ids = torch.tensor([[0, 3, 4, 7]], dtype=torch.int64, device="cuda")
    actual = embedding.lookup(ids).view(4, 64)
    expected = (raw.float() * scale.float().repeat_interleave(32, -1)).bfloat16().cuda()
    torch.testing.assert_close(actual[:2], expected[[0, 3]], atol=0, rtol=0)
    assert actual[2:].count_nonzero() == 0


def test_engram_dp_ranks_share_host_table_and_load_once(monkeypatch):
    import uuid
    from lightllm.models.deepseek_v41.engram import EngramEmbeddingWeight
    from lightllm.utils.envs_utils import get_unique_server_name

    monkeypatch.setenv("LIGHTLLM_DP_SIZE", "2")
    monkeypatch.setenv("LIGHTLLM_UNIQUE_SERVICE_NAME_ID", "test_v41_engram_" + uuid.uuid4().hex)
    get_unique_server_name.cache_clear()
    first = EngramEmbeddingWeight("layers.1.engram.embed", rows=8, dim=64)
    second = EngramEmbeddingWeight("layers.1.engram.embed", rows=8, dim=64)
    try:
        raw = torch.randn(8, 64).to(torch.float8_e4m3fn)
        scales = torch.full((8, 2), 0.125).to(torch.float8_e8m0fnu)
        first.load_hf_weights({first.prefix + ".weight": raw, first.prefix + ".scale": scales})
        assert first._host_shm.name == second._host_shm.name
        assert second.verify_load()
        # A later rank must neither zero nor reload a table already used by another rank.
        second.load_hf_weights({second.prefix + ".weight": torch.zeros_like(raw)})
        indices = torch.tensor([[0, 3, 7]], dtype=torch.int64, device="cuda")
        expected = (raw.float()[[0, 3, 7]] * 0.125).bfloat16().cuda().view(1, -1)
        torch.testing.assert_close(first.lookup(indices), expected, atol=0, rtol=0)
        torch.testing.assert_close(second.lookup(indices), expected, atol=0, rtol=0)
    finally:
        first._host_registration()
        second._host_registration()
        first._host_shm.unlink()
        get_unique_server_name.cache_clear()


@pytest.mark.parametrize("vision", [False, True])
def test_engram_hashes_survive_chunk_and_swa_page_boundaries(monkeypatch, vision):
    from lightllm.models.deepseek_v41 import engram
    import numpy as np

    config = dict(
        vision_n_layers=int(vision),
        engram_compressed_vocab_size=64,
        engram_pad_token_id=2,
        engram_max_ngram_size=4,
        engram_layer_ids=[1, 14],
        engram_vocab_size=101,
        engram_n_heads=2,
    )
    monkeypatch.setattr(engram, "build_compressed_token_map", lambda _: (list(range(64)), 64))
    hasher = engram.EngramHasher(config, None, 512)
    ids = torch.arange(135, dtype=torch.int64, device="cuda") % 64
    if vision:
        ids[127:130] = torch.arange(1000, 1003, device="cuda")
    # Physical pages deliberately differ from logical sequence pages.
    pages = torch.tensor([[2, 0], [1, 3]], dtype=torch.int32, device="cuda")
    table = (pages[:, :, None] * 128 + torch.arange(128, device="cuda")).flatten(1)

    def state(start, end, req_idx=0):
        positions = torch.arange(start, end, dtype=torch.int32, device="cuda")
        return SimpleNamespace(
            position_ids=positions,
            input_ids=ids[start:end],
            dsv4_sparse_req_idx=torch.full_like(positions, req_idx),
            mem_index=table[req_idx, positions.long()],
            req_manager=SimpleNamespace(req_to_token_indexs=table),
        )

    full = hasher(state(0, 135))
    chunked = torch.cat([hasher(state(0, 127)), hasher(state(127, 129)), hasher(state(129, 135))])
    torch.testing.assert_close(full, chunked, atol=0, rtol=0)
    # Independent scalar n-gram reference at the sequence start and page transition.
    for pos in (0, 1, 127, 128, 130, 131, 134):
        for layer, lid in enumerate(config["engram_layer_ids"]):
            bound = (np.iinfo(np.int64).max // 64) // 2
            multipliers = np.random.default_rng(10007 * lid).integers(0, bound, size=4, dtype=np.int64) * 2 + 1
            rolling = 0
            blocked = False
            for shift in range(4):
                blocked |= pos < shift or int(ids[pos - shift]) >= 64
                token = 2 if blocked else int(ids[pos - shift])
                rolling ^= token * int(multipliers[shift])
                if shift:
                    for head in range(2):
                        col = (shift - 1) * 2 + head
                        expected = rolling % int(hasher.primes[layer, col]) + int(hasher.offsets[layer, col])
                        assert int(full[pos, layer, col]) == expected

    # Simulate GPU/CPU cache restoration or P/D transfer into unrelated physical slots.
    manager = SimpleNamespace(req_to_token_indexs=table)
    for start in (1, 2, 3, 127, 129, 130):
        hasher.history.fill_(-99)
        tail = ids[max(0, start - 3) : start].tolist()
        hasher.restore_history([(1, start, tail)], manager)
        restored = hasher(state(start, 135, req_idx=1))
        torch.testing.assert_close(restored, full[start:], atol=0, rtol=0)

    graph = torch.cuda.CUDAGraph()
    replay_state = state(130, 135, req_idx=1)
    with torch.cuda.graph(graph):
        replayed = hasher(replay_state)
    hasher.history.fill_(-99)
    hasher.restore_history([(1, 130, ids[127:130].tolist())], manager)
    graph.replay()
    torch.testing.assert_close(replayed, full[130:], atol=0, rtol=0)
