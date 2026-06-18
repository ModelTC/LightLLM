from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable)


class SamplingRegistry:

    def __init__(self) -> None:
        self._ops: dict[str, dict[str, dict[str | None, Callable]]] = {}

    def register(
        self, 
        impl_family: str, 
        op_name: str, 
        impl: Callable,
        *,
        sampling_backend: str | None = None,
    ) -> None:
        backends = self._ops.setdefault(impl_family, {}).setdefault(op_name, {})
        if sampling_backend in backends:
            raise ValueError(
                f"Sampling op {op_name!r} already registered for "
                f"impl_family={impl_family!r}, sampling_backend={sampling_backend!r}"
            )
        backends[sampling_backend] = impl

    def get(
        self,
        impl_family: str,
        op_name: str,
        sampling_backend: str | None,
    ) -> Callable | None:
        return self._ops.get(impl_family, {}).get(op_name, {}).get(sampling_backend)

    def has_impl_family(self, impl_family: str) -> bool:
        return bool(self._ops.get(impl_family))

    def resolve(
        self,
        op_name: str,
        *,
        sampling_backend: str,
        fallback_chain: tuple[str, ...],
    ) -> Callable:
        for family in fallback_chain:
            impl = self.get(family, op_name, sampling_backend)
            if impl is not None:
                return impl
            impl = self.get(family, op_name, None)
            if impl is not None:
                return impl
        raise KeyError(
            f"Sampling op {op_name!r} not found for "
            f"sampling_backend={sampling_backend!r}, fallback_chain={fallback_chain!r}"
        )


sampling_registry = SamplingRegistry()


def register_sampling_op(
    impl_family: str,
    *,
    sampling_backend: str | None = None,
) -> Callable[[F], F]:

    def decorator(fn: F) -> F:
        sampling_registry.register(
            impl_family, 
            fn.__name__, 
            fn, 
            sampling_backend=sampling_backend,
        )
        return fn

    return decorator
