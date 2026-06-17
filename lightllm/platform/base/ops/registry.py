import inspect
from typing import Callable, TypeVar
from lightllm.platform.base.ops.ensure_out import AutoOutSpec, wrap_with_out

F = TypeVar("F", bound=Callable)


# Helper function to validate tensor name in function parameters
def _require_tensor_param(op_name: str, param_name: str, parameters: dict[str, inspect.Parameter]) -> None:
    if param_name not in parameters:
        raise ValueError(
            f"register_op({op_name!r}): tensor param {param_name!r} "
            f"not found in function parameters {list(parameters)}"
        )


def _validate_auto_out_spec(op_name: str, out: AutoOutSpec, sig: inspect.Signature) -> None:
    parameters = sig.parameters
    # if out_shape, out_dtype, and out_device are all specified, then input_name is optional
    fully_specified = (
        out.get("out_shape") is not None
        and out.get("out_dtype") is not None
        and out.get("out_device") is not None
    )
    if not fully_specified:
        if "input_name" not in out:
            raise ValueError(
                f"register_op({op_name!r}): 'input_name' is required unless "
                "out_shape, out_dtype, and out_device are all specified"
            )
        _require_tensor_param(op_name, out["input_name"], parameters)

    out_shape = out.get("out_shape")
    if out_shape is not None:
        if not isinstance(out_shape, tuple) or not out_shape:
            raise ValueError(f"register_op({op_name!r}): 'out_shape' must be a non-empty tuple")
        if not all(isinstance(dim, int) for dim in out_shape):
            for item in out_shape:
                if not isinstance(item, tuple) or len(item) != 2:
                    raise ValueError(
                        f"register_op({op_name!r}): invalid out_shape item {item!r}, "
                        "expected (tensor_name, dim_index)"
                    )
                name, dim = item
                if not isinstance(name, str) or not isinstance(dim, int):
                    raise ValueError(
                        f"register_op({op_name!r}): invalid out_shape item {item!r}, "
                        "expected (tensor_name, dim_index)"
                    )
                _require_tensor_param(op_name, name, parameters)

    for key in ("out_dtype", "out_device"):
        spec = out.get(key)
        if isinstance(spec, str):
            _require_tensor_param(op_name, spec, parameters)


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


op_registry = OpRegistry()


def register_op(
    impl_family: str,
    *,
    name: str | None = None,
    out: AutoOutSpec | None = None,
) -> Callable[[F], F]:

    def decorator(fn: F) -> F:
        op_name = name or fn.__name__
        if out is not None:
            _validate_auto_out_spec(op_name, out, inspect.signature(fn))
            impl: Callable = wrap_with_out(out, fn)
        else:
            impl = fn
        op_registry.register(impl_family, op_name, impl)
        return fn

    return decorator