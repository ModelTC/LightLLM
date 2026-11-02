from .quantize_method import QuantizationMethod
from .plan import WeightSelection


class QuantMethodFactory:
    def __init__(self):
        self._quant_methods = {}
        self._selections = {}
        self._selection_keys = {}
        self._content_keys = {}

    def register(
        self,
        names,
        platform="cuda",
        *,
        weight_format=None,
        activation_format=None,
        implementation=None,
        content_key=None,
    ):
        def decorator(cls):
            local_names = names
            if isinstance(local_names, str):
                local_names = [local_names]
            for n in local_names:
                if n not in self._quant_methods:
                    self._quant_methods[n] = {}
                self._quant_methods[n][platform] = cls
                self._content_keys[n, platform] = content_key or local_names[0]
            if any(value is not None for value in (weight_format, activation_format, implementation)):
                if not all((weight_format, activation_format, implementation)):
                    raise ValueError(
                        "quantization providers must declare weight, activation and implementation together"
                    )
                selection = WeightSelection(weight_format, activation_format, implementation)
                key = selection, platform
                if key in self._selection_keys:
                    raise ValueError(f"duplicate quantization provider for {selection} on {platform}")
                self._selection_keys[key] = local_names[0]
                for name in local_names:
                    self._selections[name, platform] = selection
            return cls

        return decorator

    def resolve(self, value, platform="cuda"):
        if isinstance(value, str):
            return value
        if not isinstance(value, dict) or set(value) != {"weight_format", "activation_format", "implementation"}:
            raise ValueError("quantization selection requires weight_format, activation_format and implementation")
        selection = WeightSelection(**value)
        key = self._selection_keys.get((selection, platform))
        if key is None:
            raise ValueError(f"no quantization provider for {selection} on {platform}")
        return key

    def describe(self, key, platform="cuda"):
        return self._selections.get((key, platform))

    def content_key(self, key, platform="cuda"):

        return self._content_keys.get((key, platform), key)

    def get(self, key, platform="cuda", *args, **kwargs) -> "QuantizationMethod":
        quant_method_class_dict = self._quant_methods.get(key)
        if not quant_method_class_dict:
            raise ValueError(f"QuantMethod '{key}' not supported.")

        quant_method_class = quant_method_class_dict.get(platform)
        if quant_method_class is None:
            raise ValueError(f"QuantMethod '{key}' for platform '{platform}' not supported.")
        return quant_method_class()


QUANTMETHODS = QuantMethodFactory()
