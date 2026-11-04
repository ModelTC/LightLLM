from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from lightllm.common.execution.dense import DenseAttentionBackend, DenseStaticParams
from lightllm.common.execution.specs import DenseAttentionSpec, DeviceCaps, ExecutionLimits, FormatSpec
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.gpt_oss_fused_moe_weight_tp import (
    GPTOSSFusedMoeWeightTP,
)


def packed_weights(shape, device="cpu"):
    assert shape[-1] % 32 == 0
    blocks = torch.randint(0, 256, (*shape[:-1], shape[-1] // 32, 16), dtype=torch.uint8, device=device)
    scales = torch.randint(120, 123, blocks.shape[:-1], dtype=torch.uint8, device=device)
    codes = torch.stack((blocks & 15, blocks >> 4), -1).flatten(-2).int()
    exponent, mantissa = (codes & 7) >> 1, (codes & 1).float()
    values = torch.where(exponent == 0, mantissa / 2, torch.ldexp(1 + mantissa / 2, exponent - 1))
    values = torch.where(codes & 8 != 0, -values, values)
    values = (values * torch.exp2(scales.float().sub(127).unsqueeze(-1))).flatten(-2)
    return blocks, scales, values


def gpt_oss_checkpoint(reference, packed):
    # Nonzero sinks and biases make missing static parameters observable.
    for name, parameter in reference.named_parameters():
        if name.endswith(".sinks"):
            parameter.copy_(torch.linspace(-1, 1, parameter.numel(), device=parameter.device))
        elif name.endswith((".bias", "_bias")):
            parameter.copy_(torch.randn_like(parameter) * 0.03)
    result = {name: value.detach().cpu() for name, value in reference.state_dict().items()}
    if packed:
        for name, parameter in reference.named_parameters():
            if name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
                blocks, scales, values = packed_weights(parameter.shape)
                parameter.copy_(values)
                del result[name]
                result[name + "_blocks"] = blocks
                result[name + "_scales"] = scales
    return result


@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_gpt_oss_checkpoint_shards_preserve_interleaving_and_bias(monkeypatch, tp, packed):
    from lightllm.common.basemodel.layer_weights.meta_weights import base_weight

    monkeypatch.setattr(base_weight, "get_dp_world_size", lambda: tp)
    monkeypatch.setattr(base_weight, "get_current_device_id", lambda: "cpu")
    torch.manual_seed(21)
    prefix = "model.layers.0.mlp.experts"
    checkpoint, expected = {}, {}
    for name, shape in (("gate_up_proj", (4, 64, 256)), ("down_proj", (4, 128, 64))):
        if packed:
            blocks, scales, tensor = packed_weights(shape)
            checkpoint[f"{prefix}.{name}_blocks"] = blocks
            checkpoint[f"{prefix}.{name}_scales"] = scales
        else:
            tensor = torch.randn(shape)
            checkpoint[f"{prefix}.{name}"] = tensor
        expected[name] = tensor.bfloat16()
        checkpoint[f"{prefix}.{name}_bias"] = torch.randn(4, shape[-1]).bfloat16()
    views = []
    for rank in range(tp):
        monkeypatch.setattr(base_weight, "get_current_rank_in_dp", lambda: rank)
        loader = GPTOSSFusedMoeWeightTP(prefix, 4, 64, 128, torch.bfloat16)
        with pytest.raises(ValueError, match="fully loaded"):
            loader.native_moe_view()
        items = list(checkpoint.items())
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(loader.load_hf_weights, [dict(items[::2]), dict(items[1::2])]))
        assert loader.verify_load()
        fmt, fields = loader.native_moe_view()
        assert fmt.shard == (rank, tp)
        start, stop = rank * 128 // tp, (rank + 1) * 128 // tp
        torch.testing.assert_close(
            fields["gate_up"], expected["gate_up_proj"][:, :, start * 2 : stop * 2].transpose(1, 2), rtol=0, atol=0
        )
        torch.testing.assert_close(fields["down"], expected["down_proj"][:, start:stop].transpose(1, 2), rtol=0, atol=0)
        torch.testing.assert_close(
            fields["gate_up_bias"], checkpoint[f"{prefix}.gate_up_proj_bias"][:, start * 2 : stop * 2], rtol=0, atol=0
        )
        views.append(fields)
    torch.testing.assert_close(
        sum(view["down_bias"].float() for view in views), checkpoint[f"{prefix}.down_proj_bias"].float(), rtol=0, atol=0
    )


def test_gpt_oss_declares_sink_window_and_routing():
    from lightllm.models.gpt_oss.model import GptOssModelAdapter

    adapter = GptOssModelAdapter(
        dict(
            hidden_size=256,
            intermediate_size=512,
            num_local_experts=8,
            num_experts_per_tok=2,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            sliding_window=4,
            layer_types=["sliding_attention", "full_attention"],
        ),
        role="main",
        dtype=torch.bfloat16,
        tp_world_size=2,
    )
    first, second = adapter.attention_specs().values()
    assert first.attention_sink and second.attention_sink
    assert first.window == (3, 0) and second.window == (-1, -1)
    assert first.q_heads == 4 and first.kv_heads == 1
    assert adapter.moe_specs()[0].scoring == "topk_softmax"
    assert adapter.moe_specs()[1].alpha == 1.702
    assert adapter.moe_specs()[0].round_intermediates
    fmt = FormatSpec("plain", torch.bfloat16, "token_head_dim", ("k", "v"), 1)
    assert (
        not DenseAttentionBackend("triton", "decode")
        .probe(first, fmt, DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(1 << 20))
        .offers
    )
    params = DenseStaticParams(first, torch.empty)
    with pytest.raises(ValueError, match="sink"):
        params.validate_sinks(torch.device("cpu"))
    params = replace(params, sinks=torch.ones(4, dtype=torch.bfloat16))
    params.validate_sinks(torch.device("cpu"))
    with pytest.raises(ValueError, match="not declared"):
        replace(params, spec=replace(first, attention_sink=False)).validate_sinks(torch.device("cpu"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("yarn", [False, True])
@torch.no_grad()
def test_gpt_oss_prefill_decode_prefix_and_graph(monkeypatch, tmp_path, dtype, packed, yarn):
    from unit_tests.common.execution.test_model_cuda import test_tiny_model_prefill_decode_prefix_and_graph

    test_tiny_model_prefill_decode_prefix_and_graph(
        "gpt_oss",
        "fa3",
        "fa3",
        "None",
        monkeypatch,
        tmp_path,
        model_dtype=dtype,
        gptoss_packed=packed,
        gptoss_yarn=yarn,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("packed", [False, True])
@torch.no_grad()
def test_gpt_oss_all_experts_prefill_decode_prefix_and_graph(monkeypatch, tmp_path, dtype, packed):
    from unit_tests.common.execution.test_model_cuda import test_tiny_model_prefill_decode_prefix_and_graph

    test_tiny_model_prefill_decode_prefix_and_graph(
        "gpt_oss",
        "fa3",
        "fa3",
        "None",
        monkeypatch,
        tmp_path,
        model_dtype=dtype,
        gptoss_packed=packed,
        gptoss_topk=4,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("window", [(-1, -1), (3, 0)])
@torch.no_grad()
def test_bound_sink_attention_matches_torch_and_updates_graph(fp8, phase, window):
    from types import SimpleNamespace
    from lightllm.common.execution.planner import Planner
    from lightllm.common.execution.runner import attention_batch
    from lightllm.common.execution.specs import StateSpec, StateView
    from lightllm.common.execution.static_fp8 import StaticFP8AttentionBackend

    torch.manual_seed(13)
    spec = DenseAttentionSpec(4, 2, 64, dtype=torch.bfloat16, window=window, attention_sink=True)
    tensors = [torch.randn(128, 2, 64, device="cuda", dtype=spec.dtype) for _ in range(2)]
    scale = torch.tensor([0.015625, 0.0078125], device="cuda")
    q_scale = torch.tensor([0.015625, 0.015625], device="cuda")
    if fp8:
        k, v = [(t.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn) for t, s in zip(tensors, scale)]
        fields = (k, v, scale, q_scale)
        fmt = FormatSpec(
            "fp8_per_tensor", k.dtype, "token_head_dim", ("k", "v", "kv_scale", "q_scale"), 1, scale_dtype=torch.float32
        )
        reference_k, reference_v = [t.float() * s for t, s in zip((k, v), scale)]
        backend = StaticFP8AttentionBackend(phase)
    else:
        fields = tuple(tensors)
        fmt = FormatSpec("plain", spec.dtype, "token_head_dim", ("k", "v"), 1)
        reference_k, reference_v = [t.float() for t in tensors]
        backend = DenseAttentionBackend("fa3", phase)
    view = StateView(StateSpec("main", "kv", 0, (0,), (2, 64)), fmt, fields)
    sinks = torch.tensor([-1, 0, 1, 2], device="cuda", dtype=torch.bfloat16)
    alloc = lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda")
    bound = (
        Planner((backend,), DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(1 << 20, 2, 64))
        .bind(
            [(0, phase, spec, view, DenseStaticParams(spec, alloc, sinks), "auto")],
            lambda size: alloc(size, torch.uint8),
        )
        .bindings[0]
        .execution
    )
    assert bound.sinks.data_ptr() == sinks.data_ptr()
    table = torch.randperm(128, device="cuda", dtype=torch.int32).view(2, 64)
    q_len = 2 if phase == "prefill" else 1
    q = torch.randn(2 * q_len, 4, 64, device="cuda", dtype=spec.dtype)

    def metadata(lengths):
        return bound.create_metadata(
            attention_batch(
                SimpleNamespace(
                    batch_size=2,
                    max_q_seq_len=q_len,
                    max_kv_seq_len=max(lengths),
                    microbatch_index=0,
                    req_manager=SimpleNamespace(req_to_token_indexs=table),
                    b_req_idx=torch.arange(2, device="cuda", dtype=torch.int32),
                    b_seq_len=torch.tensor(lengths, device="cuda", dtype=torch.int32),
                    b_q_start_loc=torch.tensor([0, q_len], device="cuda", dtype=torch.int32),
                    b_ready_cache_len=torch.tensor(lengths, device="cuda", dtype=torch.int32) - q_len,
                    b1_cu_q_seq_len=torch.tensor([0, q_len, 2 * q_len], device="cuda", dtype=torch.int32),
                    b1_cu_kv_seq_len=torch.tensor([0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32),
                )
            )
        )

    def reference(lengths):
        result = []
        for i, length in enumerate(lengths):
            query = q[i * q_len : (i + 1) * q_len].float()
            if fp8:
                query = (query / 0.015625).clamp(-448, 448).to(torch.float8_e4m3fn).float() * 0.015625
            key = reference_k[table[i, :length]].repeat_interleave(2, dim=1)
            value = reference_v[table[i, :length]].repeat_interleave(2, dim=1)
            scores = torch.einsum("qhd,khd->hqk", query, key) * spec.scale
            distance = torch.arange(length - q_len, length, device="cuda")[:, None] - torch.arange(
                length, device="cuda"
            )
            mask = distance >= 0
            if window != (-1, -1):
                mask &= distance <= window[0]
            scores.masked_fill_(~mask, -float("inf"))
            logits = torch.cat((scores, sinks.float()[:, None, None].expand(-1, q_len, 1)), -1)
            result.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1)[..., :-1], value))
        return torch.cat(result)

    lengths = [9, 6]
    batch = metadata(lengths)
    tolerance = 0.025 if fp8 else 0.006
    torch.testing.assert_close(bound.run(q, batch).float(), reference(lengths), rtol=tolerance, atol=tolerance)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = bound.run(q, batch)
    q.mul_(0.75)
    lengths = [12, 8]
    batch.copy_for_graph(metadata(lengths))
    graph.replay()
    torch.testing.assert_close(output.float(), reference(lengths), rtol=tolerance, atol=tolerance)
    torch.cuda.synchronize()
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("attention_factor", [None, 1.5])
def test_yarn_optional_parameters_match_reference(attention_factor):
    from transformers import GptOssConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from lightllm.models.gpt_oss.model import GptOssTpPartModel

    config = GptOssConfig(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=32,
        rope_parameters=dict(
            rope_type="yarn",
            rope_theta=150000.0,
            factor=4.0,
            original_max_position_embeddings=8,
            truncate=False,
            beta_fast=None,
            beta_slow=None,
            attention_factor=attention_factor,
        ),
    )
    model = GptOssTpPartModel.__new__(GptOssTpPartModel)
    model.config = model.normalize_model_config(config.to_dict())
    model.head_dim_, model.max_seq_length, model.data_type = 64, 32, torch.float32
    model._init_to_get_yarn_rotary()
    inv, scale = ROPE_INIT_FUNCTIONS["yarn"](config, torch.device("cuda"))
    frequencies = torch.outer(torch.arange(32, device="cuda", dtype=torch.float32), inv)
    torch.testing.assert_close(model._cos_cached, frequencies.cos() * scale)
    torch.testing.assert_close(model._sin_cached, frequencies.sin() * scale)
