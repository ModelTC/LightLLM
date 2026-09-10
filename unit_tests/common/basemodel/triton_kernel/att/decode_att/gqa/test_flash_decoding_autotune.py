import collections
import inspect
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from frozendict import frozendict
from torch.nn.attention import SDPBackend, sdpa_kernel

from lightllm.common.basemodel.attention.triton.fp import TritonDecodeAttState
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding import (
    gqa_flash_decoding as decode_module,
    gqa_flash_decoding_stage1 as stage1_module,
)
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.flash_decoding.gqa_flash_decoding_stage2 import (
    flash_decode_stage2,
)
from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType, AutotuneLevel
from lightllm.utils.envs_utils import get_decode_attn_autotune_seq_len


@pytest.fixture(autouse=True)
def autotune_environment(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", raising=False)
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(stage1_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    get_decode_attn_autotune_seq_len.cache_clear()
    yield
    get_decode_attn_autotune_seq_len.cache_clear()


@pytest.mark.parametrize("level", [AutotuneLevel.ADAPTIVE_AUTOTUNE, AutotuneLevel.FORCE_AUTOTUNE])
@pytest.mark.parametrize("configured, expected", [(None, 32768), ("8193", 8704)])
def test_tuning_key_uses_bucketed_configured_length(monkeypatch, level, configured, expected):
    if configured is not None:
        monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", configured)
    monkeypatch.setattr(stage1_module, "get_triton_autotune_level", lambda: level)
    q = torch.empty(2, 8, 64)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert stage1_module.get_run_key(q, 2) == 2_000_000_000 + expected


@pytest.mark.parametrize(
    "phase,level",
    [
        (None, AutotuneLevel.ADAPTIVE_AUTOTUNE),
        (None, AutotuneLevel.FORCE_AUTOTUNE),
        (AutotuneKernelType.GENERAL, AutotuneLevel.ADAPTIVE_AUTOTUNE),
        (AutotuneKernelType.DECODE_ATTENTION, AutotuneLevel.USE_AUTOTUNE_HIS_CONFIG),
    ],
)
def test_non_tuning_key_uses_actual_length(monkeypatch, phase, level):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", "invalid")
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", phase)
    monkeypatch.setattr(stage1_module, "get_triton_autotune_level", lambda: level)
    q = torch.empty(2, 8, 64)
    for actual, bucket in [(2, 512), (511, 512), (512, 512), (513, 1024), (8193, 8704)]:
        assert stage1_module.get_run_key(q, actual) == 2_000_000_000 + bucket


def make_inputs(device="cpu", head_dim=64, token_count=17, table_width=16):
    # K/V 是共享存储中的不连续视图；原请求索引也不是从 0 开始连续排列。
    kv = torch.randn(token_count, 4, head_dim, dtype=torch.bfloat16, device=device)
    return dict(
        q=torch.randn(2, 8, head_dim, dtype=kv.dtype, device=device),
        k=kv[:, :2],
        v=kv[:, 2:],
        Req_to_tokens=torch.arange(5 * table_width, dtype=torch.int32, device=device).view(5, table_width)
        % token_count,
        B_req_idx=torch.tensor([4, 2], dtype=torch.int32, device=device),
        B_Seqlen=torch.tensor([3, 2], dtype=torch.int32, device=device),
        max_len_in_batch=3,
        mid_out=torch.full((2, 8, 2, head_dim), -100, dtype=kv.dtype, device=device),
        mid_out_logsumexp=torch.full((2, 8, 2), -100, dtype=torch.float32, device=device),
        block_seq=256,
        sliding_window=(-1, -1),
    )


def test_static_key_shares_heads_and_blocks_but_separates_windows():
    kernel = stage1_module.flash_decode_stage1
    inputs = make_inputs()
    full_key = frozendict(kernel._static_key(**inputs))
    inputs.pop("sliding_window")
    assert frozendict(kernel._static_key(**inputs)) == full_key

    inputs["q"] = torch.empty(2, 4, 64, dtype=torch.bfloat16)
    inputs["k"] = torch.empty(17, 1, 64, dtype=torch.bfloat16)
    inputs["mid_out"] = torch.empty(2, 4, 128, 64, dtype=torch.bfloat16)
    assert frozendict(kernel._static_key(**inputs)) == full_key

    window_key = frozendict(kernel._static_key(**inputs, sliding_window=(511, 0)))
    assert window_key != full_key
    assert frozendict(kernel._static_key(**inputs, sliding_window=[511, 0])) == window_key
    assert frozendict(kernel._static_key(**inputs, sliding_window=(1023, 0))) != window_key


@pytest.mark.parametrize("num_tokens", [3, 128])
def test_rebuild_preserves_layout_and_bounds_indices(monkeypatch, num_tokens):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", "17")
    inputs = make_inputs(token_count=num_tokens)
    inputs["sliding_window"] = (7, 0)
    snapshots = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}

    args, kwargs = stage1_module.rebuild_inputs(**inputs)
    rebuilt = inspect.signature(stage1_module.flash_decode_stage1.fn).bind(*args, **kwargs).arguments
    assert rebuilt["Req_to_tokens"].shape == (2, 17)
    assert rebuilt["Req_to_tokens"].min() >= 0 and rebuilt["Req_to_tokens"].max() < num_tokens
    assert rebuilt["B_req_idx"].tolist() == [0, 1]
    assert rebuilt["B_Seqlen"].tolist() == [17, 17]
    assert rebuilt["max_len_in_batch"] == 17
    for name in ["q", "k", "v", "mid_out", "mid_out_logsumexp"]:
        assert rebuilt[name] is inputs[name]
    assert rebuilt["block_seq"] == 256
    assert rebuilt["sliding_window"] == (7, 0)
    for name, value in snapshots.items():
        torch.testing.assert_close(inputs[name], value)


def test_state_retains_actual_length_before_graph_capture(monkeypatch):
    state = TritonDecodeAttState(
        backend=SimpleNamespace(
            model=SimpleNamespace(
                is_mtp_draft_model=False, mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 0)
            )
        ),
        infer_state=SimpleNamespace(max_kv_seq_len=8193),
    )
    state.init_state()
    state.infer_state.max_kv_seq_len = 32768
    calls = []
    monkeypatch.setattr(
        decode_module, "gqa_token_decode_attention_flash_decoding", lambda **kwargs: calls.append(kwargs)
    )
    state.decode_att(
        q=torch.empty(2, 8, 64),
        k=torch.empty(16, 2, 64),
        v=torch.empty(16, 2, 64),
        alloc_func=lambda shape, dtype: torch.empty(shape, dtype=dtype),
    )
    assert calls[0]["max_len_in_batch"] == 8193


def reference_attention(inputs):
    outputs = []
    for i, length in enumerate(inputs["B_Seqlen"].tolist()):
        start = max(0, length - 1 - inputs["sliding_window"][0]) if inputs["sliding_window"][0] >= 0 else 0
        indices = inputs["Req_to_tokens"][inputs["B_req_idx"][i], start:length].long()
        with sdpa_kernel(SDPBackend.MATH):
            outputs.append(
                F.scaled_dot_product_attention(
                    inputs["q"][i].float().unsqueeze(1),
                    inputs["k"][indices].float().transpose(0, 1),
                    inputs["v"][indices].float().transpose(0, 1),
                    enable_gqa=True,
                ).squeeze(1)
            )
    return torch.stack(outputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kv_len", [8192, 16384])
@pytest.mark.parametrize("head_dim", [64, 256])
@pytest.mark.parametrize("sliding_window", [(-1, -1), (511, 0)])
def test_stage1_stage2_long_kv_matches_reference(kv_len, head_dim, sliding_window):
    inputs = make_inputs("cuda", head_dim=head_dim, token_count=kv_len * 2, table_width=kv_len)
    inputs["B_Seqlen"] = torch.tensor([kv_len, kv_len - 1], dtype=torch.int32, device="cuda")
    inputs["max_len_in_batch"] = kv_len
    inputs["sliding_window"] = sliding_window
    reference = reference_attention(inputs)
    # 只有两个 grid block，长 KV 需要每个 program 循环多次；stage2 仍归约同一布局。
    output = torch.empty_like(inputs["q"])
    for block_n in [16, 64, 128]:
        stage1_module.flash_decode_stage1.fn(**inputs, run_config={"BLOCK_N": block_n, "num_warps": 4, "num_stages": 2})
        flash_decode_stage2(
            inputs["mid_out"],
            inputs["mid_out_logsumexp"],
            inputs["B_Seqlen"],
            output,
            inputs["block_seq"],
            sliding_window=sliding_window,
        )
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), reference, atol=2e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kv_len", [8192, 16384])
def test_autotune_rebuilds_once_and_graph_uses_original_inputs(tmp_path, monkeypatch, kv_len):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", str(kv_len))
    monkeypatch.setattr(autotuner_module.dist, "is_initialized", lambda: False)
    kernel = stage1_module.flash_decode_stage1
    monkeypatch.setattr(kernel, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(kernel, "cached_configs", {})
    monkeypatch.setattr(kernel, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(kernel, "warmuped_configs_set", set())
    assert kernel.mutates_args == []
    monkeypatch.setattr(
        kernel,
        "configs_gen_func",
        lambda: [{"BLOCK_N": n, "num_warps": 4, "num_stages": 2} for n in [16, 64, 128]],
    )
    inputs = make_inputs("cuda")
    snapshots = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    reference = reference_attention(inputs)
    rebuild = kernel.rebuild_input_func
    benchmark = kernel._bench
    rebuild_count = 0
    benchmark_count = 0

    def checked_rebuild(*args, **kwargs):
        nonlocal rebuild_count
        rebuild_count += 1
        return rebuild(*args, **kwargs)

    def checked_bench(*args, **kwargs):
        nonlocal benchmark_count
        rebuilt = inspect.signature(kernel.fn).bind(*args, **kwargs).arguments
        assert rebuilt["B_Seqlen"].tolist() == [kv_len, kv_len]
        assert rebuilt["Req_to_tokens"].shape == (2, kv_len)
        assert rebuilt["B_req_idx"].tolist() == [0, 1]
        assert rebuilt["max_len_in_batch"] == kv_len
        assert rebuilt["mid_out"] is inputs["mid_out"]
        elapsed = benchmark(*args, **kwargs)
        assert math.isfinite(elapsed)
        benchmark_count += 1
        return elapsed

    monkeypatch.setattr(kernel, "rebuild_input_func", checked_rebuild)
    monkeypatch.setattr(kernel, "_bench", checked_bench)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        kernel(**inputs)
    assert rebuild_count == 1 and benchmark_count == 3
    assert all(list(configs) == [str(2_000_000_000 + kv_len)] for configs in kernel.cached_configs.values())
    # FORCE 在 decode 阶段仍复用缓存，不因模型层重复调用而再次调优。
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.FORCE_AUTOTUNE)
    monkeypatch.setattr(stage1_module, "get_triton_autotune_level", lambda: AutotuneLevel.FORCE_AUTOTUNE)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        kernel(**inputs)
    assert rebuild_count == 1 and benchmark_count == 3

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
    assert rebuild_count == 1 and benchmark_count == 3
    for name, value in snapshots.items():
        if name not in ["mid_out", "mid_out_logsumexp"]:
            torch.testing.assert_close(inputs[name], value)
