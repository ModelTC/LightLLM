from .quantize_method import QuantizationMethod
from lightllm.common.quant_type import normalize_quant_type


class QuantMethodFactory:
    def __init__(self):
        self._quant_methods = {}

    def register(self, names, platform="cuda"):
        def decorator(cls):
            local_names = names
            if isinstance(local_names, str):
                local_names = [local_names]
            for n in local_names:
                canonical_name = normalize_quant_type(n)
                if canonical_name != n:
                    raise ValueError(f"QuantMethod must register canonical name `{canonical_name}`, not alias `{n}`.")
                if n not in self._quant_methods:
                    self._quant_methods[n] = {}
                self._quant_methods[n][platform] = cls
            return cls

        return decorator

    def get(self, key, platform="cuda", *args, **kwargs) -> "QuantizationMethod":
        canonical_name = normalize_quant_type(key)
        quant_method_class_dict = self._quant_methods.get(canonical_name)
        if not quant_method_class_dict:
            raise ValueError(f"QuantMethod '{canonical_name}' not registered.")

        quant_method_class = quant_method_class_dict.get(platform)
        if quant_method_class is None:
            raise ValueError(f"QuantMethod '{canonical_name}' for platform '{platform}' not supported.")
        return quant_method_class()


QUANTMETHODS = QuantMethodFactory()
