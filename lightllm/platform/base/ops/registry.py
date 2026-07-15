from typing import Callable, TypeVar

from lightllm.platform.base.ops.out import wrap_out

F = TypeVar("F", bound=Callable)


class OpRegistry:

    def __init__(self) -> None:
        self._ops: dict[str, dict[str, Callable]] = {}

    def register(self, impl_family: str, op_name: str, impl: Callable) -> None:
        family_ops = self._ops.setdefault(impl_family, {})
        if op_name in family_ops:
            raise ValueError(f"Op '{op_name}' already registered for impl_family '{impl_family}'")
        family_ops[op_name] = impl

    def get(self, impl_family: str, op_name: str) -> Callable | None:
        return self._ops.get(impl_family, {}).get(op_name)

    def has_impl_family(self, impl_family: str) -> bool:
        return bool(self._ops.get(impl_family))

    def resolve(
        self,
        op_name: str,
        *,
        fallback_chain: tuple[str, ...],
    ) -> Callable:
        for family in fallback_chain:
            impl = self.get(family, op_name)
            if impl is not None:
                return impl
        raise KeyError(
            f"Op '{op_name}' not found for "
            f"fallback_chain={fallback_chain!r}"
        )


op_registry = OpRegistry()


def register_op(
    impl_family: str,
    *,
    name: str | None = None,
    out: Callable | None = None,
) -> Callable[[F], F]:

    def decorator(fn: F) -> F:
        op_name = name or fn.__name__
        impl = wrap_out(fn, out) if out is not None else fn
        op_registry.register(impl_family, op_name, impl)
        return fn

    return decorator
