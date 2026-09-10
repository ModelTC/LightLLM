import collections
import inspect
import math
from types import SimpleNamespace

import pytest
import torch
from frozendict import frozendict

from lightllm.common.basemodel.attention.fa3 import fp as fa3_module
from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import AutotuneKernelType, AutotuneLevel, Autotuner
from lightllm.utils import sgl_utils
from lightllm.utils.envs_utils import get_decode_attn_autotune_seq_len


@pytest.fixture(autouse=True)
def autotune_seq_len_environment(monkeypatch):
    monkeypatch.delenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", raising=False)
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    get_decode_attn_autotune_seq_len.cache_clear()
    yield
    get_decode_attn_autotune_seq_len.cache_clear()


@pytest.mark.parametrize("seq_len, expected_len", [(None, 32768), ("8192", 8192), ("16384", 16384), ("8193", 8704)])
@pytest.mark.parametrize("level", [AutotuneLevel.ADAPTIVE_AUTOTUNE, AutotuneLevel.FORCE_AUTOTUNE])
def test_fa3_run_key_uses_configured_length_during_tuning(monkeypatch, seq_len, expected_len, level):
    if seq_len is not None:
        monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", seq_len)
    monkeypatch.setattr(sgl_utils, "get_triton_autotune_level", lambda: level)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        for graph_capacity in [16, 32768]:
            page_table = torch.empty(2, graph_capacity, dtype=torch.int32)
            assert (
                sgl_utils._flash_attn_kvcache_run_key(page_table, 3, 2)
                == 2 * 10_000_000_000_000 + 3 * 10_000_000 + expected_len
            )


@pytest.mark.parametrize(
    "phase, level",
    [
        (None, AutotuneLevel.USE_AUTOTUNE_HIS_CONFIG),
        (None, AutotuneLevel.ADAPTIVE_AUTOTUNE),
        (None, AutotuneLevel.FORCE_AUTOTUNE),
        (AutotuneKernelType.GENERAL, AutotuneLevel.ADAPTIVE_AUTOTUNE),
        (AutotuneKernelType.GENERAL, AutotuneLevel.FORCE_AUTOTUNE),
        (AutotuneKernelType.DECODE_ATTENTION, AutotuneLevel.USE_AUTOTUNE_HIS_CONFIG),
    ],
)
def test_fa3_run_key_uses_actual_length_without_tuning(monkeypatch, phase, level):
    # 未调优时不应读取该环境变量，即使它无效也不影响正常配置查找。
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", "invalid")
    monkeypatch.setattr(sgl_utils, "get_triton_autotune_level", lambda: level)
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", phase)
    page_table = torch.empty(2, 32768, dtype=torch.int32)
    for actual_len, expected_len in [(1, 512), (512, 512), (513, 1024), (8192, 8192), (16384, 16384)]:
        assert (
            sgl_utils._flash_attn_kvcache_run_key(page_table, 3, actual_len)
            == 2 * 10_000_000_000_000 + 3 * 10_000_000 + expected_len
        )


def test_fa3_runtime_matches_cached_config_by_actual_length(monkeypatch):
    kernel = sgl_utils.flash_attn_with_kvcache_autotune
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.USE_AUTOTUNE_HIS_CONFIG)
    monkeypatch.setattr(autotuner_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(kernel, "fn", lambda **kwargs: kwargs["run_config"])
    monkeypatch.setattr(kernel, "fast_match_configs", collections.defaultdict(dict))
    inputs = dict(
        q=torch.empty(2, 4, 8),
        k_cache=torch.empty(1, 1, 2, 8),
        v_cache=torch.empty(1, 1, 2, 8),
        page_table=torch.empty(2, 32768, dtype=torch.int32),
        max_seqlen_q=1,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        sinks=None,
        k_descale=None,
        v_descale=None,
    )
    static_key = frozendict(kernel._static_key(**inputs))
    base_key = 2 * 10_000_000_000_000 + 10_000_000
    monkeypatch.setattr(
        kernel,
        "cached_configs",
        {static_key: {str(base_key + 8192): {"num_splits": 16}, str(base_key + 16384): {"num_splits": 32}}},
    )

    assert kernel(**inputs, max_seqlen_k=8192) == {"num_splits": 16}
    assert kernel(**inputs, max_seqlen_k=8191) == {"num_splits": 16}
    assert kernel(**inputs, max_seqlen_k=15000) == {"num_splits": 32}
    assert set(kernel.fast_match_configs[static_key]) == {str(base_key + 8192), str(base_key + 15360)}


def test_fa3_decode_preserves_actual_length_before_graph_capture(monkeypatch):
    graph = SimpleNamespace(can_run=lambda **kwargs: True, graph_max_len_in_batch=32768)
    model = SimpleNamespace(
        graph=graph,
        req_manager=SimpleNamespace(req_to_token_indexs=torch.zeros(2, 8192, dtype=torch.int32)),
        is_mtp_draft_model=False,
        mtp_manager=SimpleNamespace(get_decode_draft_step=lambda _: 0),
    )
    state = fa3_module.Fa3DecodeAttState(
        backend=SimpleNamespace(
            model=model,
            uses_causal_attention=lambda: True,
            uses_dynamic_spec_verify_layout=lambda: False,
            get_page_table_view=lambda att_batch_size, max_kv_len, microbatch_index: torch.zeros(
                att_batch_size, max_kv_len, dtype=torch.int32
            ),
        ),
        infer_state=SimpleNamespace(
            batch_size=2,
            max_kv_seq_len=8192,
            microbatch_index=0,
            b_req_idx=torch.tensor([0, 1], dtype=torch.int32),
            b_seq_len=torch.tensor([4096, 8192], dtype=torch.int32),
            b1_cu_q_seq_len=torch.tensor([0, 1, 2], dtype=torch.int32),
            b1_cu_kv_seq_len=torch.tensor([0, 4096, 12288], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(fa3_module, "page_table_copy", lambda **kwargs: None)
    state.init_state()
    # 模拟 CudaGraph._capture_decode 对 infer_state 的修改。
    state.infer_state.max_kv_seq_len = graph.graph_max_len_in_batch
    calls = []

    def attention(**kwargs):
        calls.append(kwargs)
        return kwargs["q"]

    monkeypatch.setattr(fa3_module, "flash_attn_with_kvcache_autotune", attention)
    state.decode_att(torch.empty(2, 4, 8), torch.empty(1, 2, 8), torch.empty(1, 2, 8))
    assert calls[0]["max_seqlen_k"] == 8192
    assert calls[0]["page_table"].shape == (2, 32768)


@pytest.mark.parametrize("seq_len", ["0", "-1", "invalid"])
def test_invalid_decode_autotune_seq_len_is_rejected(monkeypatch, seq_len):
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", seq_len)
    with pytest.raises(ValueError):
        get_decode_attn_autotune_seq_len()


@pytest.mark.parametrize("page_size, num_pages", [(1, 64), (1, 3), (256, 3)])
@pytest.mark.parametrize("query_lengths", [[1, 1], [3, 3], [0, 3]])
def test_fa3_rebuilds_valid_kv_metadata_without_changing_originals(monkeypatch, page_size, num_pages, query_lengths):
    kv_len = 17
    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", str(kv_len))
    max_pages = (kv_len + page_size - 1) // page_size
    q = torch.randn(sum(query_lengths), 4, 8)
    k = torch.randn(num_pages, page_size, 2, 8)
    v = torch.randn_like(k)
    page_table = torch.full((2, 8), -1, dtype=torch.int32)
    seq_lens = torch.full((2,), 2, dtype=torch.int32)
    cu_q = torch.tensor([0, query_lengths[0], sum(query_lengths)], dtype=torch.int32)
    original = (q, k, v, seq_lens, page_table, cu_q, max(query_lengths), 2)
    snapshots = [tensor.clone() for tensor in original[:6]]
    options = {
        "cu_seqlens_k_new": None,
        "causal": True,
        "window_size": (16, 0),
        "softmax_scale": 0.25,
        "return_softmax_lse": False,
    }

    args, kwargs = sgl_utils._flash_attn_kvcache_rebuild_inputs(*original, **options)

    assert all(args[i] is original[i] for i in [0, 1, 2, 5])
    assert args[6:] == (max(query_lengths), kv_len)
    assert kwargs == options
    assert args[4].shape == (2, max_pages)
    assert args[4].min() >= 0 and args[4].max() < num_pages
    if num_pages >= 2 * max_pages:
        assert args[4].unique().numel() == 2 * max_pages
    torch.testing.assert_close(args[3], torch.full_like(seq_lens, kv_len))
    for tensor, snapshot in zip(original[:6], snapshots):
        torch.testing.assert_close(tensor, snapshot)

    # Batched Q does not require cumulative query or KV lengths.
    batched_q = torch.randn(2, 1, 4, 8)
    args, kwargs = sgl_utils._flash_attn_kvcache_rebuild_inputs(
        q=batched_q,
        k_cache=k,
        v_cache=v,
        page_table=page_table,
        cache_seqlens=seq_lens,
        cu_seqlens_q=None,
        max_seqlen_q=1,
        max_seqlen_k=2,
    )
    assert args[0] is batched_q
    assert args[5:] == (None, 1, kv_len)


@pytest.mark.parametrize("kv_len", [8192, 16384])
@pytest.mark.parametrize("query_lengths", [[1, 1], [3, 3], [0, 3]])
def test_fa3_autotunes_long_kv_then_captures_original_inputs(tmp_path, monkeypatch, kv_len, query_lengths):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA3 requires a Hopper GPU")
    if sgl_utils.flash_attn_with_kvcache is None:
        pytest.skip("sgl_kernel FA3 is unavailable")

    monkeypatch.setenv("LIGHTLLM_DECODE_ATTN_AUTOTUNE_SEQ_LEN", str(kv_len))
    kernel = sgl_utils.flash_attn_with_kvcache_autotune
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    monkeypatch.setattr(sgl_utils, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    monkeypatch.setattr(autotuner_module.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(kernel, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(kernel, "cached_configs", {})
    monkeypatch.setattr(kernel, "fast_match_configs", collections.defaultdict(dict))
    monkeypatch.setattr(kernel, "warmuped_configs_set", set())

    q = torch.randn(sum(query_lengths), 4, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2 * kv_len, 1, 2, 64, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    inputs = dict(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table=torch.zeros(2, 32768, device="cuda", dtype=torch.int32),
        cache_seqlens=torch.full((2,), 2, device="cuda", dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, query_lengths[0], sum(query_lengths)], device="cuda", dtype=torch.int32),
        cu_seqlens_k_new=None,
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=2,
        causal=True,
        window_size=(-1, -1),
        softcap=0.0,
        sinks=None,
        k_descale=None,
        v_descale=None,
    )
    snapshots = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    reference = kernel.fn(**inputs)
    benchmark = kernel._bench
    timings = []

    def checked_bench(*args, **kwargs):
        bound = inspect.signature(kernel.fn).bind(*args, **kwargs).arguments
        assert bound["cache_seqlens"].tolist() == [kv_len, kv_len]
        assert bound["page_table"].shape == (2, kv_len)
        assert bound["cu_seqlens_k_new"] is None
        assert bound["max_seqlen_k"] == kv_len
        assert bound["q"] is q
        elapsed = benchmark(*args, **kwargs)
        assert math.isfinite(elapsed), f"FA3 benchmark failed for {kwargs['run_config']}"
        timings.append(elapsed)
        return elapsed

    monkeypatch.setattr(kernel, "_bench", checked_bench)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        output = kernel(**inputs)
    assert len(timings) == 3
    expected_key = str(2 * 10_000_000_000_000 + max(query_lengths) * 10_000_000 + kv_len)
    assert kernel._run_key(**inputs) == 2 * 10_000_000_000_000 + max(query_lengths) * 10_000_000 + 512
    assert all(list(configs) == [expected_key] for configs in kernel.cached_configs.values())
    torch.testing.assert_close(output, reference)

    def unexpected_rebuild(*args, **kwargs):
        pytest.fail("Cached execution and CUDA Graph capture must use the original inputs")

    monkeypatch.setattr(kernel, "rebuild_input_func", unexpected_rebuild)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        kernel(**inputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = kernel(**inputs)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured_output, reference)
    for name, snapshot in snapshots.items():
        torch.testing.assert_close(inputs[name], snapshot)
