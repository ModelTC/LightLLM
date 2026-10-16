from importlib import import_module

_EXPORTS = {
    "ReqManager": (".base", "ReqManager"),
    "ReqManagerForMamba": (".linear_att", "ReqManagerForMamba"),
    "Glm5NextReqManager": (".glm5_next", "Glm5NextReqManager"),
    "HybridAttentionReqManager": (".hybrid_base", "HybridAttentionReqManager"),
    "ReqSamplingParamsManager": (".req_sampling_params", "ReqSamplingParamsManager"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attr)
    globals()[name] = value
    return value
