import collections
import inspect
import math

import pytest
import torch

from lightllm.common.triton_utils import autotuner as autotuner_module
from lightllm.common.triton_utils.autotuner import AutotuneKernelType, AutotuneLevel, Autotuner
from lightllm.utils import sgl_utils


@pytest.mark.parametrize("page_size, num_pages", [(1, 32), (1, 3), (256, 3)])
@pytest.mark.parametrize("query_lengths", [[1, 1], [3, 3], [0, 3]])
def test_fa3_rebuilds_valid_kv_metadata_without_changing_originals(page_size, num_pages, query_lengths):
    q = torch.randn(sum(query_lengths), 4, 8)
    k = torch.randn(num_pages, page_size, 2, 8)
    v = torch.randn_like(k)
    page_table = torch.full((2, 8), -1, dtype=torch.int32)
    seq_lens = torch.full((2,), 2, dtype=torch.int32)
    cu_q = torch.tensor([0, query_lengths[0], sum(query_lengths)], dtype=torch.int32)
    cu_k = torch.tensor([0, 2, 4], dtype=torch.int32)
    original = (q, k, v, seq_lens, page_table, cu_q, cu_k, max(query_lengths))
    snapshots = [tensor.clone() for tensor in original[:-1]]
    options = {"softmax_scale": 0.25, "return_softmax_lse": False}

    args, kwargs = sgl_utils._flash_attn_kvcache_rebuild_inputs(*original, True, (16, 0), **options)

    assert all(args[i] is original[i] for i in [0, 1, 2, 5])
    assert args[7:] == (max(query_lengths), True, (16, 0))
    assert kwargs == options
    assert args[4].shape == page_table.shape
    assert args[4].min() >= 0 and args[4].max() < num_pages
    if num_pages >= page_table.numel():
        assert args[4].unique().numel() == page_table.numel()
    torch.testing.assert_close(args[3], torch.full_like(seq_lens, 8 * page_size))
    torch.testing.assert_close(args[6], torch.tensor([0, 8, 16], dtype=torch.int32) * page_size)
    for tensor, snapshot in zip(original[:-1], snapshots):
        torch.testing.assert_close(tensor, snapshot)

    # Batched Q does not require cumulative query or KV lengths.
    batched_q = torch.randn(2, 1, 4, 8)
    args, kwargs = sgl_utils._flash_attn_kvcache_rebuild_inputs(
        q=batched_q, k_cache=k, v_cache=v, page_table=page_table, cache_seqlens=seq_lens
    )
    assert args[0] is batched_q
    assert args[5:8] == (None, None, None)


@pytest.mark.parametrize("kv_len", [8192, 16384])
@pytest.mark.parametrize("query_lengths", [[1, 1], [3, 3], [0, 3]])
def test_fa3_autotunes_long_kv_then_captures_original_inputs(tmp_path, monkeypatch, kv_len, query_lengths):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("FA3 requires a Hopper GPU")
    if sgl_utils.flash_attn_with_kvcache is None:
        pytest.skip("sgl_kernel FA3 is unavailable")

    kernel = sgl_utils.flash_attn_with_kvcache_autotune
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(autotuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
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
        page_table=torch.zeros(2, kv_len, device="cuda", dtype=torch.int32),
        cache_seqlens=torch.full((2,), 2, device="cuda", dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, query_lengths[0], sum(query_lengths)], device="cuda", dtype=torch.int32),
        cu_seqlens_k_new=torch.tensor([0, 2, 4], device="cuda", dtype=torch.int32),
        max_seqlen_q=max(query_lengths),
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
        assert bound["cu_seqlens_k_new"].tolist() == [0, kv_len, 2 * kv_len]
        assert bound["q"] is q
        elapsed = benchmark(*args, **kwargs)
        assert math.isfinite(elapsed), f"FA3 benchmark failed for {kwargs['run_config']}"
        timings.append(elapsed)
        return elapsed

    monkeypatch.setattr(kernel, "_bench", checked_bench)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE_ATTENTION):
        output = kernel(**inputs)
    assert len(timings) == 3
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
