from lightllm.common.quantization.quantize_method import QuantizationMethod


def select_fuse_moe_impl(quant_method: QuantizationMethod, enable_ep_moe: bool):
    if enable_ep_moe:
        from .deepgemm_impl import FuseMoeDeepGEMM

        return FuseMoeDeepGEMM

    if quant_method.method_name == "awq_marlin":
        from .marlin_impl import FuseMoeMarlin

        return FuseMoeMarlin
    else:
        from .triton_impl import FuseMoeTriton

        return FuseMoeTriton
