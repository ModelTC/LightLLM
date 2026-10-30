from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .quantize_method import QuantizationMethod


class QuantMethodFactory:
    def __init__(self):
        self._quant_methods = {}

    def register(self, names, platform="cuda"):
        def decorator(cls):
            local_names = names
            if isinstance(local_names, str):
                local_names = [local_names]
            for n in local_names:
                if n not in self._quant_methods:
                    self._quant_methods[n] = {}
                self._quant_methods[n][platform] = cls
            return cls

        return decorator

    def supported_methods(self, platform="cuda"):
        return tuple(
            sorted(name for name, methods in self._quant_methods.items() if platform is None or platform in methods)
        )

    def validate(self, key, platform="cuda"):
        if not isinstance(key, str) or platform not in self._quant_methods.get(key, {}):
            raise ValueError(
                f"QuantMethod '{key}' for platform '{platform}' not supported. "
                f"Supported methods: {', '.join(self.supported_methods(platform))}"
            )

    def get(self, key, platform="cuda", *args, **kwargs) -> "QuantizationMethod":
        self.validate(key, platform)
        quant_method_class = self._quant_methods[key][platform]
        if isinstance(quant_method_class, str):
            module, class_name = quant_method_class.rsplit(".", 1)
            quant_method_class = getattr(import_module(f".{module}", __package__), class_name)
            self._quant_methods[key][platform] = quant_method_class
        return quant_method_class()


QUANTMETHODS = QuantMethodFactory()

for names, method in {
    ("w8a8-vllm", "w8a8"): "w8a8.w8a8QuantizationMethod",
    ("fp8w8a8-vllm", "fp8w8a8"): "w8a8.FP8w8a8QuantizationMethod",
    "fp8w8a8-pt-vllm": "w8a8.FP8w8a8PerTensorVllmQuantizationMethod",
    "fp8w8a8-pt-sgl": "w8a8.FP8w8a8PerTensorSglQuantizationMethod",
    ("fp8w8a8-pt-triton", "fp8w8a8-pt"): "w8a8.FP8w8a8PerTensorTritonQuantizationMethod",
    "fp8w8a8-b128-vllm": "w8a8.FP8w8a8B128VllmQuantizationMethod",
    ("fp8w8a8-b128-triton", "fp8w8a8-b128"): "w8a8.FP8w8a8B128TritonQuantizationMethod",
    ("fp8w8a8g128-triton", "fp8w8a8g128"): "w8a8gx.FP8w8a8g128QuantizationMethod",
    ("fp8w8a8g64-triton", "fp8w8a8g64"): "w8a8gx.FP8w8a8g64QuantizationMethod",
    "fp8w8a8-b128-deepgemm": "deepgemm.DeepGEMMFP8w8a8B128QuantizationMethod",
    "fp4fp8-b32-deepgemm": "deepgemm.DeepGEMMFP8FP4B32QuantizationMethod",
    "awq": "awq.AWQW4A16QuantizationMethod",
    "awq_marlin": "awq.AWQMARLINW4A16QuantizationMethod",
    "none": "no_quant.NoQuantization",
}.items():
    QUANTMETHODS.register(names)(method)
QUANTMETHODS.register("none", platform="musa")("no_quant.NoQuantization")
