import collections
import inspect
import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from lightllm.common.basemodel.attention.triton.int4kv import Int4kvTritonDecodeAttState
from lightllm.common.basemodel.triton_kernel.att.decode_att.int4kv import (
    int4kv_flash_decoding_stage1 as stage1_module,
    ppl_int4kv_flash_decoding as decode_module,
)
from lightllm.common.basemodel.triton_kernel.att.decode_att.int8kv.normal.int8kv_flash_decoding_stage2 import (
    flash_decode_stage2,
)
from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType, AutotuneLevel
from lightllm.utils.envs_utils import get_decode_attn_autotune_seq_len


@pytest.fixture(autouse=True)
def autotune_environment(monkeypatch):
    torch.manual_seed(42)
    monkeypatch.delenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", raising=False)
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(stage1_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    get_decode_attn_autotune_seq_len.cache_clear()
    yield
    get_decode_attn_autotune_seq_len.cache_clear()


def make_inputs(device="cpu", head_dim=64, group_size=8, dtype=torch.bfloat16, token_count=17, block_num=2):
    # 与内存管理器一致：K/V 和 scale 分别来自共享存储中的非连续视图。
    low = torch.randint(0, 15, (token_count, 4, head_dim // 2), device=device)
    high = torch.randint(0, 15, low.shape, device=device)
    kv = (low | (high << 4)).to(torch.int8)
    scale = (torch.rand(token_count, 4, head_dim // group_size, device=device) * 0.1 + 0.02).to(dtype)
    return dict(
        q=torch.randn(2, 8, head_dim, dtype=dtype, device=device),
        k=kv[:, :2],
        k_scale=scale[:, :2],
        v=kv[:, 2:],
        v_scale=scale[:, 2:],
        Req_to_tokens=torch.arange(5 * 16, dtype=torch.int32, device=device).view(5, 16) % max(1, token_count),
        B_req_idx=torch.tensor([4, 2], dtype=torch.int32, device=device),
        B_Seqlen=torch.tensor([3, 2], dtype=torch.int32, device=device),
        max_kv_seq_len=3,
        mid_out=torch.full((2, 8, block_num, head_dim), -100, dtype=dtype, device=device),
        mid_out_logsumexp=torch.full((2, 8, block_num), -100, dtype=dtype, device=device),
        block_seq=256,
    )


def rebuild(inputs):
    args, kwargs = stage1_module.rebuild_inputs(**inputs)
    return inspect.signature(stage1_module.int4kv_flash_decode_stage1.fn).bind(*args, **kwargs).arguments


@pytest.mark.parametrize("level", [AutotuneLevel.ADAPTIVE_AUTOTUNE, AutotuneLevel.FORCE_AUTOTUNE])
@pytest.mark.parametrize("configured, expected", [(None, 32768), ("8193", 8704)])
def test_tuning_key_uses_configured_length(monkeypatch, level, configured, expected):
    if configured is not None:
        monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", configured)
    monkeypatch.setattr(stage1_module, "get_triton_autotune_level", lambda: level)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert stage1_module.get_run_key(torch.empty(2, 8, 64), 32768) == 2_000_000_000 + expected


@pytest.mark.parametrize(
    "phase,level",
    [
        (None, AutotuneLevel.ADAPTIVE_AUTOTUNE),
        (None, AutotuneLevel.FORCE_AUTOTUNE),
        (AutotuneKernelType.GENERAL, AutotuneLevel.ADAPTIVE_AUTOTUNE),
        (AutotuneKernelType.DECODE_ATTENTION, AutotuneLevel.USE_AUTOTUNE_HIS_CONFIG),
    ],
)
def test_lookup_uses_actual_length(monkeypatch, phase, level):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", "invalid")
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", phase)
    monkeypatch.setattr(stage1_module, "get_triton_autotune_level", lambda: level)
    for actual, expected in [(2, 512), (511, 512), (512, 512), (513, 1024), (8193, 8704)]:
        assert stage1_module.get_run_key(torch.empty(2, 8, 64), actual) == 2_000_000_000 + expected


@pytest.mark.parametrize("token_count", [3, 128])
def test_rebuild_preserves_packed_cache_and_scales(monkeypatch, token_count):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", "17")
    inputs = make_inputs(token_count=token_count)
    snapshots = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    rebuilt = rebuild(inputs)
    torch.testing.assert_close(rebuilt["Req_to_tokens"].long(), torch.arange(34).view(2, 17) % token_count)
    assert rebuilt["B_req_idx"].tolist() == [0, 1]
    assert rebuilt["B_Seqlen"].tolist() == [17, 17]
    assert rebuilt["max_kv_seq_len"] == 17
    assert rebuilt["block_seq"] == inputs["block_seq"]
    for name in ["q", "k", "v", "k_scale", "v_scale", "mid_out", "mid_out_logsumexp"]:
        assert rebuilt[name] is inputs[name]
    for name, value in snapshots.items():
        torch.testing.assert_close(inputs[name], value)


@pytest.mark.parametrize("name", ["v", "k_scale", "v_scale"])
def test_rebuild_rejects_inconsistent_token_capacity(name):
    inputs = make_inputs()
    inputs[name] = inputs[name][:-1]
    with pytest.raises(AssertionError, match="same number of tokens"):
        rebuild(inputs)


def test_rebuild_rejects_empty_cache():
    with pytest.raises(ValueError, match="non-empty KV cache"):
        rebuild(make_inputs(token_count=0))


def test_static_key_separates_quant_groups_without_head_or_block_counts():
    kernel = stage1_module.int4kv_flash_decode_stage1
    inputs = make_inputs()
    key = kernel._static_key(**inputs)
    inputs["q"] = inputs["q"][:, :4]
    for name in ["k", "k_scale", "v", "v_scale"]:
        inputs[name] = inputs[name][:, :1]
    inputs["mid_out"] = torch.empty(2, 4, 128, 64, dtype=torch.bfloat16)
    assert kernel._static_key(**inputs) == key
    inputs["k_scale"] = inputs["k_scale"][:, :, :2]
    assert kernel._static_key(**inputs) != key


def test_decode_passes_saved_length_to_stage1(monkeypatch):
    inputs = make_inputs()
    state = Int4kvTritonDecodeAttState(
        backend=SimpleNamespace(),
        infer_state=SimpleNamespace(
            batch_size=2,
            max_kv_seq_len=8193,
            b_req_idx=inputs["B_req_idx"],
            b_seq_len=inputs["B_Seqlen"],
            req_manager=SimpleNamespace(req_to_token_indexs=inputs["Req_to_tokens"]),
        ),
    )
    state.init_state()
    state.infer_state.max_kv_seq_len = 32768
    calls = []
    monkeypatch.setattr(stage1_module, "int4kv_flash_decode_stage1", lambda **kwargs: calls.append(kwargs))
    stage2 = inspect.getmodule(flash_decode_stage2)
    monkeypatch.setattr(stage2, "flash_decode_stage2", lambda *args: None)
    state.decode_att(
        inputs["q"],
        (inputs["k"], inputs["k_scale"]),
        (inputs["v"], inputs["v_scale"]),
        alloc_func=lambda shape, dtype, device: torch.empty(shape, dtype=dtype),
    )
    assert calls[0]["max_kv_seq_len"] == 8193
    assert calls[0]["block_seq"] == 256


def reference_attention(inputs):
    def dequant(packed, scale):
        packed = packed.to(torch.uint8)
        unpacked = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).float() - 7
        group_size = unpacked.shape[-1] // scale.shape[-1]
        # 与算子一致，以 scale 的 dtype 产生解量化 K/V，再用 FP32 SDPA 验证 attention 本身。
        return (unpacked * scale.float().repeat_interleave(group_size, dim=-1)).to(scale.dtype).float()

    k, v = dequant(inputs["k"], inputs["k_scale"]), dequant(inputs["v"], inputs["v_scale"])
    outputs = []
    for row, length in enumerate(inputs["B_Seqlen"].tolist()):
        indices = inputs["Req_to_tokens"][inputs["B_req_idx"][row], :length].long()
        with sdpa_kernel(SDPBackend.MATH):
            outputs.append(
                F.scaled_dot_product_attention(
                    inputs["q"][row].float().unsqueeze(1),
                    k[indices].transpose(0, 1),
                    v[indices].transpose(0, 1),
                    enable_gqa=True,
                ).squeeze(1)
            )
    return torch.stack(outputs)


def reduce_output(inputs):
    output = torch.empty_like(inputs["q"])
    flash_decode_stage2(inputs["mid_out"], inputs["mid_out_logsumexp"], inputs["B_Seqlen"], output, inputs["block_seq"])
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kv_len", [257, 8192, 16384])
@pytest.mark.parametrize(
    "head_dim,group_size,dtype", [(64, 8, torch.bfloat16), (128, 32, torch.float16), (256, 8, torch.bfloat16)]
)
def test_long_kv_matches_dequantized_fp32_reference(monkeypatch, kv_len, head_dim, group_size, dtype):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", str(kv_len))
    inputs = rebuild(make_inputs("cuda", head_dim, group_size, dtype, token_count=2 * kv_len))
    inputs["B_Seqlen"][1] -= 1
    reference = reference_attention(inputs)
    # 两个中间分块迫使长请求循环处理多个 BLOCK_SEQ，同时验证末尾 mask。
    for block_n in [16, 32, 64, 128]:
        stage1_module.int4kv_flash_decode_stage1.fn(
            **inputs, run_config={"BLOCK_N": block_n, "num_warps": 4, "num_stages": 2}
        )
        output = reduce_output(inputs)
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), reference, atol=2e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kv_len", [8193, 16384])
def test_full_autotune_and_graph_reuse(tmp_path, monkeypatch, kv_len):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", str(kv_len))
    monkeypatch.setattr(autotuner_module.dist, "is_initialized", lambda: False)
    kernel = stage1_module.int4kv_flash_decode_stage1
    monkeypatch.setattr(kernel, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(kernel, "cached_configs", {})
    monkeypatch.setattr(kernel, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(kernel, "warmuped_configs_set", set())
    assert kernel.mutates_args == []
    inputs = make_inputs("cuda", token_count=2 * kv_len, block_num=128)
    inputs["Req_to_tokens"] = torch.arange(5 * kv_len, dtype=torch.int32, device="cuda").view(5, kv_len) % (2 * kv_len)
    snapshots = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    reference = reference_attention(inputs)
    rebuild_func, benchmark = kernel.rebuild_input_func, kernel._bench
    rebuild_count, timings = 0, []

    def checked_rebuild(*args, **kwargs):
        nonlocal rebuild_count
        rebuild_count += 1
        return rebuild_func(*args, **kwargs)

    def checked_bench(*args, **kwargs):
        rebuilt = inspect.signature(kernel.fn).bind(*args, **kwargs).arguments
        assert rebuilt["B_Seqlen"].tolist() == [kv_len, kv_len]
        assert rebuilt["Req_to_tokens"].shape == (2, kv_len)
        assert rebuilt["max_kv_seq_len"] == kv_len
        for name in ["q", "k", "v", "k_scale", "v_scale", "mid_out", "mid_out_logsumexp"]:
            assert rebuilt[name] is inputs[name]
        elapsed = benchmark(*args, **kwargs)
        timings.append(elapsed)
        return elapsed

    monkeypatch.setattr(kernel, "rebuild_input_func", checked_rebuild)
    monkeypatch.setattr(kernel, "_bench", checked_bench)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        kernel(**inputs)
    assert rebuild_count == 1 and len(timings) == 48 and any(math.isfinite(t) for t in timings)
    run_key = str(2_000_000_000 + (kv_len + 511) // 512 * 512)
    assert all(list(configs) == [run_key] for configs in kernel.cached_configs.values())
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert run_key in json.loads(next(tmp_path.glob("*.json")).read_text())
    torch.testing.assert_close(reduce_output(inputs).float(), reference, atol=2e-3, rtol=2e-2)

    # 清空内存配置后从文件重载，FORCE 下仍不能跨层重复搜索。
    monkeypatch.setattr(kernel, "cached_configs", {})
    monkeypatch.setattr(kernel, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.FORCE_AUTOTUNE)
    monkeypatch.setattr(stage1_module, "get_triton_autotune_level", lambda: AutotuneLevel.FORCE_AUTOTUNE)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        kernel(**inputs)
    assert rebuild_count == 1 and len(timings) == 48

    output = torch.empty_like(inputs["q"])

    def decode():
        kernel(**inputs)
        flash_decode_stage2(
            inputs["mid_out"], inputs["mid_out_logsumexp"], inputs["B_Seqlen"], output, inputs["block_seq"]
        )

    decode()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        decode()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), reference, atol=2e-3, rtol=2e-2)
    assert rebuild_count == 1 and len(timings) == 48
    for name, snapshot in snapshots.items():
        if name not in ["mid_out", "mid_out_logsumexp"]:
            torch.testing.assert_close(inputs[name], snapshot)

    # 捕获后更新为真正的长请求，验证 Graph 读取原始输入缓冲区，stage2 仍按 BLOCK_SEQ 归约。
    inputs["B_Seqlen"].fill_(kv_len)
    inputs["max_kv_seq_len"] = kv_len
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output.float(), reference_attention(inputs), atol=2e-3, rtol=2e-2)
    assert rebuild_count == 1 and len(timings) == 48
