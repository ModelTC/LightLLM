import lightllm.common.basemodel  # noqa: F401

from lightllm.common.quantization import Quantcfg


def test_expert_dtype_overrides_unquantized_checkpoint_moe_only():
    quant_cfg = Quantcfg({"n_layer": 2}, quant_type="none", expert_dtype="fp8")

    assert quant_cfg.get_quant_type(0, "fused_moe") == "deepgemm-fp8w8a8-b128"
    assert quant_cfg.get_quant_type(1, "fused_moe") == "deepgemm-fp8w8a8-b128"
    assert quant_cfg.get_quant_type(0, "qkv_proj") == "none"
    assert not quant_cfg.quantized_weight
