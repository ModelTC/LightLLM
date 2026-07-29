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

    def get(self, key, platform="cuda", *args, **kwargs) -> "QuantizationMethod":
        quant_method_class_dict = self._quant_methods.get(key)
        if not quant_method_class_dict:
            raise ValueError(f"QuantMethod '{key}' not supported.")

        quant_method_class = quant_method_class_dict.get(platform)
        if quant_method_class is None:
            raise ValueError(f"QuantMethod '{key}' for platform '{platform}' not supported.")
        return quant_method_class()


QUANTMETHODS = QuantMethodFactory()
