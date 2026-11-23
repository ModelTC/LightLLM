import json
from collections import defaultdict

import pytest
import torch
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.utils.config_utils import auto_set_ep_moe_backend

from lightllm.common.basemodel.triton_kernel.fused_moe.deepep_expanded_layout_kernels import ep_aligned_expert_offsets
from lightllm.common.quantization import Quantcfg
from lightllm.common.quantization.humming import (
    HUMMING_FP4_QUANT_METHOD,
    HUMMING_FP8_QUANT_METHOD,
    HummingExpertGemm,
    HummingFP4MoEQuantizationMethod,
    HummingFP8MoEQuantizationMethod,
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
    assert quant_cfg.get_quant_type(0, "wq_a") == "fp8w8a8-b32-humming"
    assert quant_cfg.get_quant_type(0, "wq_b") == "fp8w8a8-b32-triton"


@pytest.mark.parametrize(
    "sm90,enable_ep_moe,backend,expert_dtype,expected",
    [
        (True, True, "auto", None, "deepep_v2_humming"),
        (True, True, "auto", "mxfp4", "deepep_v2_humming"),
        (True, True, "auto", "fp8", "auto"),
        (False, True, "auto", None, "auto"),
        (True, False, "auto", None, "auto"),
        (True, True, "deepep_v2", None, "deepep_v2"),
    ],
)
def test_native_mxfp4_hopper_backend_selection(
    tmp_path, monkeypatch, sm90, enable_ep_moe, backend, expert_dtype, expected
):
    config = {
        "model_type": "deepseek_v41",
        "text_config": {"num_hidden_layers": 40},
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr("lightllm.utils.device_utils.is_sm90_gpu", lambda: sm90)
    args = StartArgs(
        model_dir=str(tmp_path), enable_ep_moe=enable_ep_moe, ep_moe_backend=backend, expert_dtype=expert_dtype
    )
    auto_set_ep_moe_backend(args)
    assert args.ep_moe_backend == expected
    if expected == "deepep_v2_humming":
        config.update(n_layer=40, expert_dtype="fp4")
        quant = Quantcfg(config, expert_dtype=args.expert_dtype, enable_ep_moe=True, ep_moe_backend=args.ep_moe_backend)
        assert quant.get_quant_type(0, "fused_moe") == HUMMING_FP4_QUANT_METHOD


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
    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", str(torch.cuda.current_device()))
    method = HummingFP8MoEQuantizationMethod() if bits == 8 else HummingFP4MoEQuantizationMethod()
    pack, _ = method.create_moe_weight([n], k, torch.bfloat16, torch.cuda.current_device(), experts)
    # Native V4.1 serializes two FP4 values in signed int8 storage.
    method.load_weight(weight if bits == 8 else weight.view(torch.int8), pack)
    method.load_weight_scale(weight_scale.view(torch.float8_e8m0fnu), pack)
    assert all(pack.load_ok)
    assert pack.weight.numel() == experts * n * k // (1 if bits == 8 else 2)
    weight_ptr = pack.weight.data_ptr()
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

    offsets = ep_aligned_expert_offsets(ends, alignment=32, rows=rows)
    for rows_per_expert in (0, 256, 512):
        gemm(x, input_scale, output, offsets, 72, prefill_rows_per_expert=rows_per_expert)
        for expert, start, end in ((0, 0, 11), (2, 32, 65)):
            expected = (dense_input[start:end] @ dense_weight[expert].T).bfloat16().float()
            error = (output[start:end].float() - expected).norm() / expected.norm()
            assert error.item() < 0.015


@pytest.mark.parametrize(
    "rows,n,k", [(64, 256, 512), (128, 256, 512), (384, 256, 512), (16, 1792, 5120), (64, 1792, 5120)]
)
def test_humming_dense_b32_tune_and_replay(rows, n, k, tmp_path, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires an SM90 GPU")
    pytest.importorskip("humming")
    from lightllm.common.quantization.humming import HummingFP8B32QuantizationMethod, _humming_dense_b32
    from lightllm.common.basemodel.triton_kernel.quantization.fp8act_quant_kernel import per_token_group_quant_fp8

    monkeypatch.setattr(_humming_dense_b32, "_cache_dir", str(tmp_path), raising=False)
    monkeypatch.setattr(_humming_dense_b32, "cached_configs", {})
    monkeypatch.setattr(_humming_dense_b32, "fast_match_configs", defaultdict(dict))
    monkeypatch.setattr(_humming_dense_b32, "_missing_config_keys", set())
    method = HummingFP8B32QuantizationMethod()
    method.hf_quantization_config = {"scale_fmt": "ue8m0"}
    torch.manual_seed(351)
    weight = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.exp2(torch.randint(-9, -3, (n // 32, k // 32), device="cuda").float())
    pack = WeightPack(weight, scales)
    x = torch.randn((rows, k), device="cuda", dtype=torch.bfloat16)
    with Autotuner.autotune_warmup(AutotuneKernelType.DECODE):
        method.apply(x, pack, use_custom_tensor_mananger=False)
    assert _humming_dense_b32.cached_configs
    before = weight.clone()
    q, a_scale = per_token_group_quant_fp8(x, 32, eps=1e-4, use_ue8m0_scales=True)
    reference = (q.float() * a_scale.repeat_interleave(32, -1)) @ (
        weight.float() * scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
    ).T
    output = torch.empty((rows, n), device="cuda", dtype=torch.bfloat16)
    for backend in ("triton", "humming"):
        _humming_dense_b32.fn(q, a_scale, output, pack, pack.humming_dense, None, {"backend": backend})
        error = (output.float() - reference).norm() / reference.norm()
        assert error < 0.015
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = method.apply(x, pack, out=output, use_custom_tensor_mananger=False)
    x.mul_(0.5)
    graph.replay()
    expected = method.apply(x, pack, use_custom_tensor_mananger=False)
    # Stream-K reduction order can differ between graph and eager launches.
    assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.015
    assert (actual.float() - reference * 0.5).norm() / reference.norm() < 0.015
    assert torch.equal(weight.view(torch.uint8), before.view(torch.uint8))


def test_humming_dense_respects_per_layer_override(tmp_path):
    config = {
        "model_type": "deepseek_v41",
        "n_layer": 2,
        "expert_dtype": "fp8",
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [32, 32],
            "expert_weight_block_size": [1, 32],
            "scale_fmt": "ue8m0",
        },
    }
    custom = tmp_path / "quant.yaml"
    custom.write_text(
        "quant_type: fp8w8a8-b32-triton\nmix_bits:\n"
        "  - name: wq_a\n    layer_nums: [0]\n    quant_type: fp8w8a8-b32-triton\n"
    )
    quant = Quantcfg(config, custom_cfg_path=str(custom), enable_ep_moe=True, ep_moe_backend="deepep_v2_humming")
    assert quant.get_quant_type(0, "wq_a") == "fp8w8a8-b32-triton"
    assert quant.get_quant_type(1, "wq_a") == "fp8w8a8-b32-humming"
    assert quant.get_quant_type(1, "gate_up_proj") == "fp8w8a8-b32-triton"
