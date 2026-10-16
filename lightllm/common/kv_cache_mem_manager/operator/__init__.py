from importlib import import_module

_EXPORTS = {
    "BaseMemManagerOperator": (".base", "BaseMemManagerOperator"),
    "NormalMemOperator": (".normal", "NormalMemOperator"),
    "QuantScaleMemOperator": (".quant", "QuantScaleMemOperator"),
    "PPLInt4KVMemOperator": (".quant", "PPLInt4KVMemOperator"),
    "PPLInt8KVMemOperator": (".quant", "PPLInt8KVMemOperator"),
    "LinearAttMemOperator": (".linear_att", "LinearAttMemOperator"),
    "Deepseek2MemOperator": (".deepseek", "Deepseek2MemOperator"),
    "Deepseek3_2MemOperator": (".deepseek", "Deepseek3_2MemOperator"),
    "FP8PerTokenGroupQuantDeepseek3_2MemOperator": (".deepseek", "FP8PerTokenGroupQuantDeepseek3_2MemOperator"),
    "FP8StaticPerHeadQuantMemOperator": (".fp8_quant", "FP8StaticPerHeadQuantMemOperator"),
    "FP8StaticPerTensorQuantMemOperator": (".fp8_quant", "FP8StaticPerTensorQuantMemOperator"),
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attr)
    globals()[name] = value
    return value
