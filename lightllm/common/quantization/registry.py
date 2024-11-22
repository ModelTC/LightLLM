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

    def get(self, key, *args, **kwargs):
        if key == "no_quant":
            return None
        quant_method_class = self._quant_methods.get(key)
        if not quant_method_class:
            raise ValueError(f"QuantMethod '{key}' not supported.")
        tmp_key = key.split("-")
        if len(tmp_key) == 2:
            return quant_method_class()
        else:
            group_size = int(tmp_key[-1])
            return quant_method_class(group_size)


QUANTMETHODS = QuantMethodFactory()
