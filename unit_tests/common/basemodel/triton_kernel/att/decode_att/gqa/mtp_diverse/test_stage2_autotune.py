import collections
import inspect
import json
import math
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel import cuda_graph as graph_module
from lightllm.common.basemodel.triton_kernel.att.decode_att.gqa.mtp_diverse import stage2_single_token as stage2_module
from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import Autotuner, AutotuneKernelType, AutotuneLevel
from lightllm.utils.envs_utils import get_decode_attn_autotune_seq_len


@pytest.fixture(autouse=True)
def autotune_environment(monkeypatch):
    torch.manual_seed(42)
    monkeypatch.delenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", raising=False)
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(stage2_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    get_decode_attn_autotune_seq_len.cache_clear()
    yield
    get_decode_attn_autotune_seq_len.cache_clear()


@pytest.mark.parametrize("level", [AutotuneLevel.ADAPTIVE_AUTOTUNE, AutotuneLevel.FORCE_AUTOTUNE])
@pytest.mark.parametrize("configured, capacity, expected", [(None, 128, 128), ("8193", 256, 129), ("65", 128, 2)])
def test_tuning_key_uses_effective_reduction_blocks(monkeypatch, level, configured, capacity, expected):
    if configured is not None:
        monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", configured)
    monkeypatch.setattr(stage2_module, "get_triton_autotune_level", lambda: level)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        assert stage2_module.get_run_key(torch.empty(2, 4, capacity, 64), 64, 2) == 8_000_000_000 + expected


@pytest.mark.parametrize(
    "phase,level",
    [
        (None, AutotuneLevel.ADAPTIVE_AUTOTUNE),
        (None, AutotuneLevel.FORCE_AUTOTUNE),
        (AutotuneKernelType.GENERAL, AutotuneLevel.ADAPTIVE_AUTOTUNE),
        (AutotuneKernelType.DECODE_ATTENTION, AutotuneLevel.USE_AUTOTUNE_HIS_CONFIG),
        (AutotuneKernelType.DECODE_ATTENTION, AutotuneLevel.CLOSE_AUTOTUNE),
    ],
)
def test_lookup_counts_real_blocks_without_reading_gpu_lengths(monkeypatch, phase, level):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", "invalid")
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", phase)
    monkeypatch.setattr(stage2_module, "get_triton_autotune_level", lambda: level)
    mid = torch.empty(2, 4, 128, 64)
    for length, blocks in [(1, 1), (64, 1), (65, 2), (511, 8), (512, 8), (513, 9), (8192, 128), (16384, 128)]:
        assert stage2_module.get_run_key(mid, 64, length) == 8_000_000_000 + blocks


def test_key_reuses_equal_work_and_distinguishes_capacity_limits():
    key = stage2_module.get_run_key
    small, large = torch.empty(2, 4, 64, 64), torch.empty(2, 4, 128, 64)
    assert key(small, 64, 513) == key(large, 64, 513)
    assert key(small, 64, 8192) != key(large, 64, 8192)
    assert key(large, 64, 8192) == key(large, 64, 16384)


def make_inputs(device="cpu", capacity=128, block_n=64, dtype=torch.bfloat16, head_dim=128):
    # 模拟短请求：stage1 仅写入第一个分块，其余位置用 NaN 标记为未初始化。
    mid = torch.full((2, 4, capacity, head_dim), float("nan"), dtype=dtype, device=device)
    lse = torch.full((2, 4, capacity), float("nan"), dtype=torch.float32, device=device)
    mid[:, :, 0] = torch.randn(2, 4, head_dim, dtype=dtype, device=device)
    lse[:, :, 0] = 0
    return dict(
        mid_out=mid,
        mid_out_logsumexp=lse,
        B_Seqlen=torch.full((2,), 2, dtype=torch.int32, device=device),
        out=torch.full((2, 4, head_dim), -100, dtype=dtype, device=device),
        block_n=block_n,
        max_kv_len=2,
    )


@pytest.mark.parametrize("length", [65, 8193, 16384])
def test_rebuild_reuses_readonly_intermediate_inputs(monkeypatch, length):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", str(length))
    inputs = make_inputs()
    snapshots = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    args, kwargs = stage2_module.rebuild_inputs(**inputs)
    rebuilt = inspect.signature(stage2_module.mtp_diverse_stage2_single_token.fn).bind(*args, **kwargs).arguments
    assert rebuilt["max_kv_len"] == length
    assert rebuilt["B_Seqlen"].tolist() == [length, length]
    assert rebuilt["block_n"] == inputs["block_n"]
    assert rebuilt["out"] is inputs["out"]
    for name in ["mid_out", "mid_out_logsumexp"]:
        assert rebuilt[name] is inputs[name]
        assert rebuilt[name].shape == inputs[name].shape
        assert rebuilt[name].stride() == inputs[name].stride()
        assert rebuilt[name].dtype == inputs[name].dtype
        assert rebuilt[name].device == inputs[name].device
    assert rebuilt["B_Seqlen"] is not inputs["B_Seqlen"]
    assert torch.isfinite(rebuilt["B_Seqlen"]).all()
    for name, snapshot in snapshots.items():
        torch.testing.assert_close(inputs[name], snapshot, equal_nan=True)


def reduction_reference(inputs):
    results = []
    for row, length in enumerate(inputs["B_Seqlen"].tolist()):
        blocks = min((length + inputs["block_n"] - 1) // inputs["block_n"], inputs["mid_out"].shape[2])
        weights = torch.softmax(inputs["mid_out_logsumexp"][row, :, :blocks].float(), dim=-1)
        results.append((weights.unsqueeze(-1) * inputs["mid_out"][row, :, :blocks].float()).sum(dim=1))
    return torch.stack(results)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("block_n", [16, 64])
@pytest.mark.parametrize("capacity", [2, 128])
@pytest.mark.parametrize("dtype,head_dim", [(torch.bfloat16, 128), (torch.float16, 64)])
def test_all_configs_reduce_only_valid_blocks(block_n, capacity, dtype, head_dim):
    lengths = [1, block_n, block_n + 1, capacity * block_n, capacity * block_n + 8192]
    mid = torch.randn(len(lengths), 4, capacity, head_dim, dtype=dtype, device="cuda")
    lse = torch.randn(len(lengths), 4, capacity, dtype=torch.float32, device="cuda") * 20
    for row, length in enumerate(lengths):
        blocks = min((length + block_n - 1) // block_n, capacity)
        mid[row, :, blocks:] = float("nan")
        lse[row, :, blocks:] = float("nan")
    inputs = dict(
        mid_out=mid,
        mid_out_logsumexp=lse,
        B_Seqlen=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        out=torch.empty(len(lengths), 4, head_dim, dtype=dtype, device="cuda"),
        block_n=block_n,
        max_kv_len=max(lengths),
    )
    reference = reduction_reference(inputs)
    for config in stage2_module.get_test_configs():
        stage2_module.mtp_diverse_stage2_single_token.fn(**inputs, run_config=config)
        torch.testing.assert_close(inputs["out"].float(), reference, atol=2e-3, rtol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("length", [65, 8192, 16384])
def test_full_tuning_graph_and_cache_reuse(tmp_path, monkeypatch, length):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", str(length))
    monkeypatch.setattr(autotuner_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.FORCE_AUTOTUNE)
    monkeypatch.setattr(stage2_module, "get_triton_autotune_level", lambda: AutotuneLevel.FORCE_AUTOTUNE)
    kernel = stage2_module.mtp_diverse_stage2_single_token
    monkeypatch.setattr(kernel, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(kernel, "cached_configs", {})
    monkeypatch.setattr(kernel, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(kernel, "warmuped_configs_set", set())
    inputs = make_inputs("cuda")
    snapshots = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    original_rebuild, original_bench = kernel.rebuild_input_func, kernel._bench
    rebuild_count, benchmark_count = 0, 0

    def checked_rebuild(*args, **kwargs):
        nonlocal rebuild_count
        assert not torch.cuda.is_current_stream_capturing()
        rebuild_count += 1
        return original_rebuild(*args, **kwargs)

    def checked_bench(*args, **kwargs):
        nonlocal benchmark_count
        rebuilt = inspect.signature(kernel.fn).bind(*args, **kwargs).arguments
        assert rebuilt["B_Seqlen"].tolist() == [length, length]
        assert rebuilt["max_kv_len"] == length and rebuilt["block_n"] == 64
        for name in ["mid_out", "mid_out_logsumexp"]:
            assert rebuilt[name] is inputs[name]
            assert rebuilt[name].shape == inputs[name].shape
        elapsed = original_bench(*args, **kwargs)
        assert math.isfinite(elapsed)
        torch.testing.assert_close(inputs["out"], snapshots["out"])
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

    def decode(state):
        kernel(**inputs)
        return inputs["out"]

    graph = graph_module.CudaGraph(1, 2, 1, max_batch_size=2, max_len_in_batch=32768)
    graph.capture_decode(decode, SimpleNamespace(input_ids=torch.ones(2, device="cuda")))
    torch.cuda.synchronize()
    assert rebuild_count == 1 and benchmark_count == 20
    torch.testing.assert_close(inputs["out"].float(), reduction_reference(inputs), atol=2e-3, rtol=1e-2)
    for name in ["mid_out", "mid_out_logsumexp", "B_Seqlen"]:
        torch.testing.assert_close(inputs[name], snapshots[name], equal_nan=True)
    cache_file = next(tmp_path.glob("*.json"))
    saved = cache_file.read_bytes()
    expected_key = str(8_000_000_000 + min((length + 63) // 64, 128))
    assert list(json.loads(saved)) == [expected_key]
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        kernel(**inputs)
    kernel.cached_configs.clear()
    kernel.fast_match_configs.clear()
    kernel.warmuped_configs_set.clear()
    # 8K/16K 都归约 128 个块，切换代表性长度后也应复用同一个 key。
    if length >= 8192:
        monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", str(24576 - length))
        get_decode_attn_autotune_seq_len.cache_clear()
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        kernel(**inputs)
    assert rebuild_count == 1 and benchmark_count == 20
    assert cache_file.read_bytes() == saved
    inputs["mid_out"].normal_()
    inputs["mid_out_logsumexp"].normal_()
    inputs["B_Seqlen"].fill_(length)
    graph.graph[2][0].replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(inputs["out"].float(), reduction_reference(inputs), atol=2e-3, rtol=1e-2)
