"""Public contract exports without importing cyclic implementation modules."""

from importlib import import_module


_EXPORTS = {
    "ModelAdapter": (".model", "ModelAdapter"),
    "StateSpec": (".state", "StateSpec"),
    "FormatSpec": (".state", "FormatSpec"),
    "StateView": (".state", "StateView"),
    "LayerStates": (".state", "LayerStates"),
    "DeviceCaps": (".device", "DeviceCaps"),
    "ExecutionLimits": (".device", "ExecutionLimits"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
