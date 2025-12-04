import os
import torch
from .quantize_method import QuantizationMethod
from .registry import QUANTMETHODS
import torch.nn.functional as F
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from lightllm.common.basemodel.layer_weights.meta_weights.mm_weight.mm_weight import MMWeightPack

try:
    HAS_TORCH_AO = True
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
        return dummy_linear.weight, None, None

    def apply(
        self,
        input_tensor: torch.Tensor,
        weight_pack: "MMWeightPack",
        out: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
        use_custom_tensor_mananger: bool = True,
    ) -> torch.Tensor:
        weights = weight_pack.weight
        bias = weight_pack.bias
        return F.linear(input_tensor, weights, bias)

    @property
    def method_name(self):
        return "ao-base"


@QUANTMETHODS.register(["ao-w4a16-256"])
class AOW4A16QuantizationMethodGroup256(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.group_size = 256
        self.quant_func = int4_weight_only(group_size=self.group_size)
        self.has_weight_scale = False
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "ao-w4a16-256"


@QUANTMETHODS.register(["ao-w4a16-128"])
class AOW4A16QuantizationMethodGroup128(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.group_size = 128
        self.quant_func = int4_weight_only(group_size=self.group_size)
        self.has_weight_scale = False
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "ao-w4a16-128"


@QUANTMETHODS.register(["ao-w4a16-64"])
class AOW4A16QuantizationMethodGroup64(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.group_size = 64
        self.quant_func = int4_weight_only(group_size=self.group_size)
        self.has_weight_scale = False
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "ao-w4a16-64"


@QUANTMETHODS.register(["ao-w4a16-32"])
class AOW4A16QuantizationMethodGroup32(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.group_size = 32
        self.quant_func = int4_weight_only(group_size=self.group_size)
        self.has_weight_scale = False
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "ao-w4a16-32"


@QUANTMETHODS.register("ao-w8a8")
class AOW8A8QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.quant_func = int8_dynamic_activation_int8_weight()
        self.has_weight_scale = False
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "ao-w8a8"


@QUANTMETHODS.register("ao-w8a16")
class AOW8A16QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        self.quant_func = int8_weight_only()
        self.has_weight_scale = False
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "ao-w8a16"


@QUANTMETHODS.register("ao-fp8w8a16")
class AOFP8W8A16QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        is_cuda_8_9 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
        assert is_cuda_8_9, "FP8 requires GPU with compute capability >= 8.9"
        self.quant_func = float8_weight_only()
        self.has_weight_scale = False
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "ao-fp8w8a16"


@QUANTMETHODS.register("ao-fp6w6a16")
class AOFP6W6A16QuantizationMethod(AOBaseQuantizationMethod):
    def __init__(self):
        super().__init__()
        assert TORCH_VERSION_AT_LEAST_2_5, "torchao fp6 requires torch >=2.5"
        self.quant_func = fpx_weight_only(3, 2)
        self.has_weight_scale = False
        self.has_weight_zero_point = False

    @property
    def method_name(self):
        return "ao-fp6w6a16"
