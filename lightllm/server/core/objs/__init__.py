from importlib import import_module

_EXPORTS = {
    "ConstraintOutParams": (".sampling_params", "ConstraintOutParams"),
    "SamplingParams": (".sampling_params", "SamplingParams"),
    "StopStrMatchMode": (".sampling_params", "StopStrMatchMode"),
    "Req": (".req", "Req"),
    "FinishStatus": (".req", "FinishStatus"),
    "ShmReqManager": (".shm_req_manager", "ShmReqManager"),
    "StartArgs": (".start_args_type", "StartArgs"),
    "AtomicShmLock": (".atomic_lock", "AtomicShmLock"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attr = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attr)
    globals()[name] = value
    return value
