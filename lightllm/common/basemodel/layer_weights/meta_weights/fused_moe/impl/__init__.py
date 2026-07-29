from lightllm.common.quantization.quantize_method import QuantizationMethod
from lightllm.common.quant_type import QUANT_TYPE_AWQ_MARLIN
from .triton_impl import FuseMoeTriton
from .marlin_impl import FuseMoeMarlin
from .deepgemm_impl import FuseMoeDeepGEMM


def select_fuse_moe_impl(quant_method: QuantizationMethod, enable_ep_moe: bool):
    if enable_ep_moe:
        return FuseMoeDeepGEMM

    if quant_method.method_name == QUANT_TYPE_AWQ_MARLIN:
        return FuseMoeMarlin
    else:
        return FuseMoeTriton
