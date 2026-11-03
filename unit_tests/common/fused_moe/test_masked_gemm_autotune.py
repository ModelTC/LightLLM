import json
from collections import defaultdict

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe import grouped_fused_moe_ep as ep
from lightllm.common.triton_utils import autotuner as tuner_module
from lightllm.common.triton_utils.autotuner import AutotuneKernelType, AutotuneLevel, Autotuner


def test_tuning_counts_are_rebuilt_without_mutating_dispatch():
    counts = torch.tensor([0, 120], dtype=torch.int32)
    out = torch.empty(2, 128, 256)
    args, kwargs = ep._masked_gemm_tuning_inputs(None, None, out, counts, 16)
    torch.testing.assert_close(args[3], torch.tensor([16, 16], dtype=torch.int32))
    torch.testing.assert_close(counts, torch.tensor([0, 120], dtype=torch.int32))
    assert kwargs == {}


@pytest.mark.skipif(
    not torch.cuda.is_available() or not ep.HAS_DEEPGEMM or torch.cuda.get_device_capability()[0] != 9,
    reason="SM90, DeepEP and DeepGEMM required",
)
@pytest.mark.parametrize("expected_m", [1, 17, 65])
def test_masked_gemm_tunes_then_replays_with_changed_counts(tmp_path, monkeypatch, expected_m):
    torch.manual_seed(319)
    kernel = ep._deepgemm_grouped_fp8_nt_masked
    monkeypatch.setattr(kernel, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(kernel, "cached_configs", {})
    monkeypatch.setattr(kernel, "fast_match_configs", defaultdict(dict))
    monkeypatch.setattr(kernel, "_missing_config_keys", set())
    monkeypatch.setattr(Autotuner, "_autotune_warmup_kernel_type", None)
    monkeypatch.setattr(tuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    monkeypatch.setattr(tuner_module.dist, "is_initialized", lambda: False)
    a = torch.randn(2, 128, 128, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(2, 256, 128, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.ones(2, 128, 1, device="cuda")
    sb = torch.ones(2, 2, 1, device="cuda")
    out = torch.empty(2, 128, 256, device="cuda", dtype=torch.bfloat16)
    counts = torch.tensor([0, 125], device="cuda", dtype=torch.int32)

    def check():
        reference = (a.float() @ b.float().transpose(1, 2)).bfloat16()
        for i, rows in enumerate(counts.tolist()):
            torch.testing.assert_close(out[i, :rows], reference[i, :rows], atol=0.05, rtol=0.02)

    # A smaller expected_m must not truncate work specified by masked_m.
    for config in kernel.configs_gen_func():
        kernel((a, sa), (b, sb), out, counts, expected_m, run_config=config)
        check()
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE):
        kernel((a, sa), (b, sb), out, counts, expected_m)
    check()
    torch.testing.assert_close(counts, torch.tensor([0, 125], device="cuda", dtype=torch.int32))
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    saved = json.loads(files[0].read_text())
    assert saved[str(expected_m)] in kernel.configs_gen_func()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernel((a, sa), (b, sb), out, counts, expected_m)
    counts.copy_(torch.tensor([97, 1], device="cuda", dtype=torch.int32))
    a.copy_((a.float() * 2).to(a.dtype))
    graph.replay()
    check()
    assert json.loads(files[0].read_text()) == saved
