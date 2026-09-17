from dataclasses import dataclass, field
from typing import Callable

# The default order of fallback implementations.
DEFAULT_ORDER = ("triton", "torch")

OPS_TABLE: dict[str, "OpEntry"] = {}


@dataclass
class OpEntry:
    fns: dict[str, Callable] = field(default_factory=dict)
    extras: list[str] = field(default_factory=list)

    @property
    def impls(self) -> tuple[str, ...]:
        extras = tuple(name for name in self.extras if name in self.fns)
        defaults = tuple(kind for kind in DEFAULT_ORDER if kind in self.fns)
        return extras + defaults


def register_op(op_name: str, *, impl: str):
    def decorator(fn: Callable) -> Callable:
        entry = OPS_TABLE.setdefault(op_name, OpEntry())
        if impl in entry.fns:
            raise RuntimeError(f"The {op_name} op has already been registered with the {impl!r} implementation.")
        entry.fns[impl] = fn
        if impl not in DEFAULT_ORDER:
            entry.extras.insert(0, impl)
        return fn

    return decorator


def get_op(op_name: str) -> Callable:
    entry = OPS_TABLE.get(op_name)
    if entry is None:
        raise NotImplementedError(f"The {op_name!r} op is not registered.")
    for impl in entry.impls:
        fn = entry.fns.get(impl)
        if fn is not None:
            return fn
    raise NotImplementedError(f"The {op_name!r} op has no usable implementation in {entry.impls}.")
