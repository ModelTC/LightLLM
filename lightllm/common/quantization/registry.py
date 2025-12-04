from .quantize_method import QuantizationMethod
from typing import Type


class QuantMethodFactory:
    def __init__(self):
        self._quant_methods = {}

    def register(self, names):
        def decorator(cls):
            local_names = names
            if isinstance(local_names, str):
                local_names = [local_names]
            for n in local_names:
                self._quant_methods[n] = cls
            return cls

        return decorator

    def get(self, key, *args, **kwargs) -> Type[QuantizationMethod]:
        if key == "none":
            return None
        quant_method_class = self._quant_methods.get(key)
        if not quant_method_class:
            raise ValueError(f"QuantMethod '{key}' not supported.")
        return quant_method_class()


QUANTMETHODS = QuantMethodFactory()
