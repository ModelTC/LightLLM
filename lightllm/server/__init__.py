from importlib import import_module

_EXPORTS = {"TokenLoad": (".router.token_load", "TokenLoad")}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attr)
    globals()[name] = value
    return value
