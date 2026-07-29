from lightllm.common.quantization.quantize_method import QuantizationMethod
from .triton_impl import FuseMoeTriton
from .marlin_impl import FuseMoeMarlin
from .deepgemm_impl import FuseMoeDeepGEMM
from .moonep_impl import FuseMoeMoonEP
from .mxfp4_impl import FuseMoeMXFP4
from lightllm.utils.envs_utils import get_env_start_args


def select_fuse_moe_impl(quant_method: QuantizationMethod, enable_ep_moe: bool):
    if enable_ep_moe:
        if getattr(get_env_start_args(), "moe_ep_backend", "deepep") == "moonep":
            return FuseMoeMoonEP
        return FuseMoeDeepGEMM

    if quant_method.method_name == "awq_marlin":
        return FuseMoeMarlin
    elif quant_method.method_name == "mxfp4":
        return FuseMoeMXFP4
    else:
        return FuseMoeTriton
