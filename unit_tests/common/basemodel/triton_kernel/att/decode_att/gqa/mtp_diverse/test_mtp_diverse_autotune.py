import collections
import inspect
import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from lightllm.common.basemodel import cuda_graph as graph_module
from lightllm.common.basemodel.attention.triton import fp as state_module
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse import (
    mtp_diverse_attn as decode_module,
    stage1_single_token as stage1_module,
    stage2_single_token as stage2_module,
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
    monkeypatch.setattr(stage2_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    get_decode_attn_autotune_seq_len.cache_clear()
    yield
    get_decode_attn_autotune_seq_len.cache_clear()


@pytest.mark.parametrize("level", [AutotuneLevel.ADAPTIVE_AUTOTUNE, AutotuneLevel.FORCE_AUTOTUNE])
@pytest.mark.parametrize("configured, expected", [(None, 32768), ("8193", 8704)])
def test_tuning_key_uses_configured_length(monkeypatch, level, configured, expected):
    if configured is not None:
        monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", configured)
    monkeypatch.setattr(stage1_module, "get_triton_autotune_level", lambda: level)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert stage1_module.get_run_key(torch.empty(7, 8, 64), 2) == 7_000_000_000 + expected


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
        assert stage1_module.get_run_key(torch.empty(7, 8, 64), actual) == 7_000_000_000 + expected


def make_inputs(device="cpu", batch=7, block_batch=3, head_dim=128, token_count=17, table_width=16, block_num=2):
    kv = torch.randn(token_count, 4, head_dim, dtype=torch.bfloat16, device=device)
    return dict(
        q=torch.randn(batch, 8, head_dim, dtype=kv.dtype, device=device),
        k=kv[:, :2],
        v=kv[:, 2:],
        Req_to_tokens=torch.arange(6 * table_width, dtype=torch.int32, device=device).view(6, table_width)
        % max(1, token_count),
        B_req_idx=torch.full((batch,), 4, dtype=torch.int32, device=device),
        b_seq_len=torch.full((batch,), 2, dtype=torch.int32, device=device),
        b_mark_shared_group=torch.ones(batch, dtype=torch.int32, device=device),
        max_kv_len=2,
        mid_out=torch.full((batch, 8, block_num, head_dim), -100, dtype=kv.dtype, device=device),
        mid_out_logsumexp=torch.full((batch, 8, block_num), -100, dtype=torch.float32, device=device),
        block_batch=block_batch,
    )


def rebuild(inputs):
    args, kwargs = stage1_module.rebuild_inputs(**inputs)
    return inspect.signature(stage1_module.mtp_diverse_stage1_single_token.fn).bind(*args, **kwargs).arguments


@pytest.mark.parametrize("token_count", [3, 128])
@pytest.mark.parametrize(
    "batch, block_batch, length, req_ids, lengths, marks",
    [
        (7, 3, 17, [0, 0, 0, 1, 1, 1, 2], [15, 16, 17, 15, 16, 17, 17], [0, 0, 3, 0, 0, 3, 1]),
        (5, 4, 4, [0, 0, 0, 0, 1], [1, 2, 3, 4, 4], [0, 0, 0, 4, 1]),
        (1, 4, 1, [0], [1], [1]),
    ],
)
def test_rebuild_preserves_inputs_and_shared_prefixes(
    monkeypatch, token_count, batch, block_batch, length, req_ids, lengths, marks
):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", str(length))
    inputs = make_inputs(batch=batch, block_batch=block_batch, token_count=token_count)
    snapshots = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    rebuilt = rebuild(inputs)
    assert rebuilt["B_req_idx"].tolist() == req_ids
    assert rebuilt["b_seq_len"].tolist() == lengths
    assert rebuilt["b_mark_shared_group"].tolist() == marks
    assert rebuilt["max_kv_len"] == length
    expected = torch.arange((max(req_ids) + 1) * length).view(-1, length) % token_count
    torch.testing.assert_close(rebuilt["Req_to_tokens"].long(), expected)
    for name in ["q", "k", "v", "mid_out", "mid_out_logsumexp"]:
        assert rebuilt[name] is inputs[name]
    assert rebuilt["block_batch"] == block_batch
    for name, value in snapshots.items():
        torch.testing.assert_close(inputs[name], value)


@pytest.mark.parametrize("batch, block_batch, length", [(7, 3, 2), (2, 4, 1)])
def test_rebuild_rejects_length_shorter_than_group(monkeypatch, batch, block_batch, length):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", str(length))
    with pytest.raises(ValueError, match="at least the largest MTP group size"):
        rebuild(make_inputs(batch=batch, block_batch=block_batch))


def test_rebuild_rejects_empty_or_mismatched_cache():
    with pytest.raises(ValueError, match="non-empty KV cache"):
        rebuild(make_inputs(token_count=0))
    inputs = make_inputs()
    inputs["v"] = inputs["v"][:-1]
    with pytest.raises(AssertionError, match="same number of tokens"):
        rebuild(inputs)


def test_decode_passes_saved_length_and_selected_block_n(monkeypatch):
    inputs = make_inputs()
    infer_state = SimpleNamespace(
        max_kv_seq_len=8193,
        b_req_idx=inputs["B_req_idx"],
        b_seq_len=inputs["b_seq_len"],
        req_manager=SimpleNamespace(req_to_token_indexs=inputs["Req_to_tokens"]),
    )
    model = SimpleNamespace(
        is_mtp_draft_model=False,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 2),
        req_manager=SimpleNamespace(HOLD_REQUEST_ID=-1),
    )
    monkeypatch.setattr(state_module, "build_mtp_shared_group_markers", lambda *a, **kw: inputs["b_mark_shared_group"])
    state = state_module.TritonDecodeAttState(backend=SimpleNamespace(model=model), infer_state=infer_state)
    state.init_state()
    infer_state.max_kv_seq_len = 32768
    calls = []

    def stage1(**kwargs):
        calls.append(kwargs)
        return 32

    monkeypatch.setattr(decode_module, "mtp_diverse_stage1_single_token", stage1)
    monkeypatch.setattr(decode_module, "mtp_diverse_stage2_single_token", lambda **kwargs: calls.append(kwargs))
    state.decode_att(q=inputs["q"], k=inputs["k"], v=inputs["v"])
    assert calls[0]["max_kv_len"] == 8193
    assert calls[0]["b_mark_shared_group"] is inputs["b_mark_shared_group"]
    assert calls[1]["block_n"] == 32
    assert calls[1]["max_kv_len"] == 8193


def reference_attention(inputs):
    results = []
    for row, length in enumerate(inputs["b_seq_len"].tolist()):
        indices = inputs["Req_to_tokens"][inputs["B_req_idx"][row], :length].long()
        with sdpa_kernel(SDPBackend.MATH):
            results.append(
                F.scaled_dot_product_attention(
                    inputs["q"][row].float().unsqueeze(1),
                    inputs["k"][indices].float().transpose(0, 1),
                    inputs["v"][indices].float().transpose(0, 1),
                    enable_gqa=True,
                ).squeeze(1)
            )
    return torch.stack(results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kv_len", [17, 8192, 16384])
@pytest.mark.parametrize("block_batch", [1, 3, 4])
@pytest.mark.parametrize("head_dim", [64, 128])
def test_shared_groups_match_fp32_reference(monkeypatch, kv_len, block_batch, head_dim):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", str(kv_len))
    inputs = rebuild(make_inputs("cuda", block_batch=block_batch, head_dim=head_dim, token_count=7 * kv_len))
    reference = reference_attention(inputs)
    out = torch.empty_like(inputs["q"])
    # 仅两个 grid block，长 KV 必须循环多次；17 token 同时覆盖末块 mask 和无效分块的归约边界。
    for block_n in [16, 32, 64]:
        selected_block_n = stage1_module.mtp_diverse_stage1_single_token.fn(
            **inputs,
            run_config={"BLOCK_N": block_n, "num_warps": 4, "num_stages": 2, "warp_specialize": False},
        )
        stage2_module.mtp_diverse_stage2_single_token.fn(
            inputs["mid_out"], inputs["mid_out_logsumexp"], inputs["b_seq_len"], out, selected_block_n, kv_len
        )
        torch.testing.assert_close(out.float(), reference, atol=2e-3, rtol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kv_len", [8192, 16384])
def test_full_autotune_before_graph_and_cached_reuse(tmp_path, monkeypatch, kv_len):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", str(kv_len))
    monkeypatch.setattr(autotuner_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.FORCE_AUTOTUNE)
    monkeypatch.setattr(stage1_module, "get_triton_autotune_level", lambda: AutotuneLevel.FORCE_AUTOTUNE)
    kernel = stage1_module.mtp_diverse_stage1_single_token
    monkeypatch.setattr(kernel, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(kernel, "cached_configs", {})
    monkeypatch.setattr(kernel, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(kernel, "warmuped_configs_set", set())
    assert kernel.mutates_args == []
    # 使用生产路径的小 batch 布局，让 stage2 在长请求下归约完整的 128 个分块。
    inputs = make_inputs("cuda", table_width=kv_len, token_count=3 * kv_len, block_num=128)
    stage2 = stage2_module.mtp_diverse_stage2_single_token
    stage2_cache = tmp_path / "stage2"
    stage2_cache.mkdir()
    monkeypatch.setattr(stage2, "_cache_dir", str(stage2_cache), raising=False)
    monkeypatch.setattr(stage2, "cached_configs", {})
    monkeypatch.setattr(stage2, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(stage2, "warmuped_configs_set", set())
    snapshots = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    original_rebuild, original_bench = kernel.rebuild_input_func, kernel._bench
    rebuild_count, benchmark_count, valid_benchmark_count = 0, 0, 0

    def checked_rebuild(*args, **kwargs):
        nonlocal rebuild_count
        assert not torch.cuda.is_current_stream_capturing()
        rebuild_count += 1
        return original_rebuild(*args, **kwargs)

    def checked_bench(*args, **kwargs):
        nonlocal benchmark_count, valid_benchmark_count
        rebuilt = inspect.signature(kernel.fn).bind(*args, **kwargs).arguments
        assert rebuilt["max_kv_len"] == kv_len
        assert rebuilt["b_seq_len"].tolist() == [kv_len - 2, kv_len - 1, kv_len] * 2 + [kv_len]
        assert rebuilt["b_mark_shared_group"].tolist() == [0, 0, 3, 0, 0, 3, 1]
        assert rebuilt["Req_to_tokens"].shape == (3, kv_len)
        elapsed = original_bench(*args, **kwargs)
        # 不可编译或资源不足的候选按 Autotuner 原有规则返回 inf；完整搜索必须选出有效配置。
        if math.isfinite(elapsed):
            valid_benchmark_count += 1
        benchmark_count += 1
        return elapsed

    monkeypatch.setattr(kernel, "rebuild_input_func", checked_rebuild)
    monkeypatch.setattr(kernel, "_bench", checked_bench)
    monkeypatch.setattr(
        graph_module,
        "get_env_start_args",
        lambda: SimpleNamespace(
            enable_decode_microbatch_overlap=False, enable_tpsp_mix_mode=False, enable_torch_memory_saver=False
        ),
    )
    out = torch.empty_like(inputs["q"])

    def decode(state):
        block_n = kernel(**inputs)
        stage2(inputs["mid_out"], inputs["mid_out_logsumexp"], inputs["b_seq_len"], out, block_n, inputs["max_kv_len"])
        return out

    graph = graph_module.CudaGraph(1, 7, 1, max_batch_size=7, max_len_in_batch=32768)
    graph.capture_decode(decode, SimpleNamespace(input_ids=torch.ones(7, device="cuda")))
    torch.cuda.synchronize()
    assert rebuild_count == 1 and benchmark_count == len(kernel.configs_gen_func())
    assert valid_benchmark_count > 0
    # 两个 KV 的短请求沿用原 MTP 测试的 BF16 容差；下面的长请求仍使用更严格的 FP32 对照。
    torch.testing.assert_close(out.float(), reference_attention(inputs), atol=1e-2, rtol=1e-2)
    for name, value in snapshots.items():
        if name not in ["mid_out", "mid_out_logsumexp"]:
            torch.testing.assert_close(inputs[name], value)
    cache_file = next(tmp_path.glob("*.json"))
    saved = cache_file.read_bytes()
    assert list(json.loads(saved)) == [str(7_000_000_000 + kv_len)]
    assert json.loads(saved)[str(7_000_000_000 + kv_len)] is not None
    stage2_configs = json.loads(next(stage2_cache.glob("*.json")).read_bytes())
    assert list(stage2_configs) == [str(7 * 8 * 1_000_000_000 + 128)]
    assert next(iter(stage2_configs.values())) is not None
    # 同一层配置以及从文件重新加载的配置，在 FORCE 模式下都不能再次搜索。
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        kernel(**inputs)
    kernel.cached_configs.clear()
    kernel.fast_match_configs.clear()
    kernel.warmuped_configs_set.clear()
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        kernel(**inputs)
    assert rebuild_count == 1 and benchmark_count == len(kernel.configs_gen_func())
    assert cache_file.read_bytes() == saved
    # 原缓冲区改为真实长请求和共享组后回放，验证 Graph 没有捕获 benchmark 的重建输入。
    inputs["B_req_idx"].copy_(torch.tensor([1, 1, 1, 3, 3, 3, 2], device="cuda", dtype=torch.int32))
    inputs["b_mark_shared_group"].copy_(torch.tensor([0, 0, 3, 0, 0, 3, 1], device="cuda", dtype=torch.int32))
    inputs["b_seq_len"].copy_(
        torch.tensor([kv_len - 2, kv_len - 1, kv_len] * 2 + [kv_len], device="cuda", dtype=torch.int32)
    )
    graph.graph[7][0].replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), reference_attention(inputs), atol=2e-3, rtol=2e-2)
