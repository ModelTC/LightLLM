import json
from collections import defaultdict

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import ep_aligned_expert_offsets
from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization.humming import (
    HUMMING_FP4_QUANT_METHOD,
    HUMMING_FP8_QUANT_METHOD,
    HummingExpertGemm,
    _humming_grouped_decode_gemm,
)
from lightllm.common.quantization.quantize_method import WeightPack
from lightllm.common.triton_utils import autotuner as tuner_module
from lightllm.common.triton_utils.autotuner import AutotuneKernelType, AutotuneLevel, Autotuner


@pytest.mark.parametrize("bits,method", [(8, HUMMING_FP8_QUANT_METHOD), (4, HUMMING_FP4_QUANT_METHOD)])
def test_humming_checkpoint_selection(bits, method):
    config = {
        "model_type": "deepseek_v41",
        "n_layer": 1,
        "expert_dtype": f"fp{bits}",
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32], "scale_fmt": "ue8m0"},
    }
    if bits == 8:
        config["quantization_config"]["expert_weight_block_size"] = [1, 32]
    quant_cfg = Quantcfg(config, enable_ep_moe=True, ep_moe_backend="deepep_v2_humming")
    assert quant_cfg.get_quant_type(0, "fused_moe") == method


@pytest.mark.parametrize("bits", [8, 4])
def test_humming_experts_tune_then_replay_deepep_offsets(bits, tmp_path, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires an SM90 GPU")
    pytest.importorskip("humming")
    kernel = _humming_grouped_decode_gemm
    monkeypatch.setattr(kernel, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(kernel, "cached_configs", {})
    monkeypatch.setattr(kernel, "fast_match_configs", defaultdict(dict))
    monkeypatch.setattr(kernel, "_missing_config_keys", set())
    monkeypatch.setattr(tuner_module, "get_triton_autotune_level", lambda: AutotuneLevel.ADAPTIVE_AUTOTUNE)
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(123)
    experts, rows, n, k = 3, 128, 256, 512
    x = torch.randn((rows, k), device=device, generator=gen).to(torch.float8_e4m3fn)
    input_scale = torch.full((k // 32, rows), 2.0 ** -6, device=device).T
    weight_scale = torch.randint(115, 131, (experts, n, k // 32), device=device, dtype=torch.uint8, generator=gen)
    if bits == 8:
        weight = torch.randn((experts, n, k), device=device, generator=gen).to(torch.float8_e4m3fn)
        dense_weight = weight.float()
    else:
        weight = torch.randint(0, 256, (experts, n, k // 2), device=device, dtype=torch.uint8, generator=gen)
        fp4 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=device)
        dense_weight = fp4[torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2).long()]
    dense_weight *= torch.exp2(weight_scale.float() - 127).repeat_interleave(32, dim=-1)
    pack = WeightPack(weight=weight, weight_scale=weight_scale)
    weight_ptr = weight.data_ptr()
    locks = torch.zeros(
        torch.cuda.get_device_properties(device).multi_processor_count * 8, device=device, dtype=torch.int32
    )
    gemm = HummingExpertGemm(pack, "float8e4m3" if bits == 8 else "float4e2m1", torch.bfloat16, locks)
    assert gemm.tensors["weight"].data_ptr() == weight_ptr

    ends = torch.tensor([17, 43, 83], device=device, dtype=torch.int32)
    output = torch.empty((rows, n), device=device, dtype=torch.bfloat16)

    def run():
        offsets = ep_aligned_expert_offsets(ends, alignment=32, rows=rows)
        gemm(x, input_scale, output, offsets, 72, expert_alignment=32)

    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE):
        run()
    (cache_file,) = tmp_path.glob("*.json")
    saved_config = json.loads(cache_file.read_text())
    assert saved_config["72"] in kernel.configs_gen_func()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # The second expert becomes empty, and the last expert moves to row 32.
    ends.copy_(torch.tensor([11, 32, 65], device=device, dtype=torch.int32))
    graph.replay()
    dense_input = x.float() * input_scale.repeat_interleave(32, dim=-1)
    for expert, start, end in ((0, 0, 11), (2, 32, 65)):
        expected = (dense_input[start:end] @ dense_weight[expert].T).bfloat16().float()
        error = (output[start:end].float() - expected).norm() / expected.norm()
        assert error.item() < 0.015
    assert json.loads(cache_file.read_text()) == saved_config
