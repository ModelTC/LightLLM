import os
import torch
from .quantize_method import QuantizationMethod
from .registry import QUANTMETHODS
from lightllm.common.basemodel.layer_infer.cache_tensor_manager import g_cache_manager
import torch.nn.functional as F

try:
    HAS_TORCH_AO = True
    from torchao.dtypes import to_affine_quantized_intx, AffineQuantizedTensor
    from torchao.dtypes import TensorCoreTiledLayoutType
    from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain
    from torchao.quantization import (
        int4_weight_only,
        int8_weight_only,
        float8_weight_only,
        fpx_weight_only,
        int8_dynamic_activation_int8_weight,
        float8_dynamic_activation_float8_weight,
        quantize_,
    )
    from torchao.utils import (
        TORCH_VERSION_AT_LEAST_2_4,
        TORCH_VERSION_AT_LEAST_2_5,
    )
except:
    HAS_TORCH_AO = False


class AOBaseQuantizationMethod(QuantizationMethod):
    def __init__(self):
        super().__init__()
        assert HAS_TORCH_AO, "torchao is not installed, you can't use quant api of it"
        assert TORCH_VERSION_AT_LEAST_2_4, "torchao requires torch >=2.4"
        self.quant_func = None

    def quantize(self, weight: torch.Tensor):
        """ """
        dummy_linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        dummy_linear.weight = torch.nn.Parameter(weight.cuda(self.device_id_))
        quantize_(dummy_linear, self.quant_func)
        return dummy_linear.weight

    def apply(self, input_tensor, weights, bias=None, out=None, use_custom_tensor_mananger=True):
        return F.linear(input_tensor, weights, bias)


@QUANTMETHODS.register([f"ao-w4a16-{group_size}" for group_size in [32, 64, 128, 256]])
class AOW4A16QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self, group_size=128):
        super().__init__()
        assert group_size in [
            32,
            64,
            128,
            256,
        ], f"torchao int4-weightonly requires groupsize in [32,64,128,256], but gets {group_size}"
        self.group_size = group_size
        self.quant_func = int4_weight_only(group_size=self.group_size)


@QUANTMETHODS.register("ao-w8a8")
class AOW8A8QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.quant_func = int8_dynamic_activation_int8_weight()


@QUANTMETHODS.register("ao-w8a16")
class AOW8A16QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.quant_func = int8_weight_only()


@QUANTMETHODS.register("ao-fp8w8a16")
class AOFP8W8A16QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        is_cuda_8_9 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
        assert is_cuda_8_9, "FP8 requires GPU with compute capability >= 8.9"
        self.quant_func = float8_weight_only()


@QUANTMETHODS.register("ao-fp6w6a16")
class AOFP6W6A16QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        assert TORCH_VERSION_AT_LEAST_2_5, "torchao fp6 requires torch >=2.5"
        self.quant_func = fpx_weight_only(3, 2)
