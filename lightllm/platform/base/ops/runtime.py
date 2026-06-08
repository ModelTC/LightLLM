import inspect
from typing import Callable, TypeVar

from lightllm.platform.base.ops.base import OP_NAMES, OpsProtocol
from lightllm.platform.base.ops.ensure_out import AutoOutSpec, wrap_with_out
from lightllm.platform.base.registry import get_op_modules_for_fallback
from lightllm.platform.plugins import get_ops_plugin_config, resolve_op_fallback

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


op_registry = OpRegistry()


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


def load_ops(fallback_chain: tuple[str, ...], extra_modules: tuple[str, ...] = ()) -> None:
    import importlib

    # Load extra modules first
    for module_name in extra_modules:
        importlib.import_module(module_name)
 
    for module_name in get_op_modules_for_fallback(fallback_chain):
        importlib.import_module(module_name)


class OpsView(OpsProtocol):

    def __init__(self, platform: str, *, fallback_chain: tuple[str, ...]) -> None:
        self._platform = platform

        for op_name in OP_NAMES:
            impl = None
            for impl_family in fallback_chain:
                impl = op_registry.get(impl_family, op_name)
                if impl is not None:
                    break
            if impl is None:
                raise KeyError(
                    f"Op '{op_name}' is not registered for platform '{platform}' "
                    f"via fallback chain {fallback_chain}"
                )
            setattr(self, op_name, impl)

    def __getattr__(self, name: str):
        raise AttributeError(f"Op '{name}' is not registered for platform '{self._platform}'")


def build_ops(platform: str) -> OpsView:
    plugin_config = get_ops_plugin_config()
    fallback_chain = resolve_op_fallback(platform, plugin_config)
    load_ops(fallback_chain, plugin_config.extra_modules)
    _validate_extra_fallback_loaded(platform, plugin_config)

    return OpsView(platform, fallback_chain=fallback_chain)


def _validate_extra_fallback_loaded(platform: str, plugin_config) -> None:
    if not plugin_config.extra_fallback:
        return

    from lightllm.platform.base.registry import get_platform_spec

    missing: list[str] = []
    for family in plugin_config.extra_fallback:
        if op_registry.has_impl_family(family):
            continue
        missing.append(family)

    if not missing:
        return

    hints = _format_extra_fallback_hints(missing, plugin_config)
    raise RuntimeError(
        "External op impl families configured in extra_op_fallback did not register "
        f"any @register_op implementations after loading: {missing}. "
        f"Fallback chain for platform {platform!r} was "
        f"{resolve_op_fallback(platform, plugin_config)!r}, but these families are empty "
        f"so all ops silently fall back to {get_platform_spec(platform).op_fallback!r}. "
        f"{hints}"
    )


def _format_extra_fallback_hints(missing: list[str], plugin_config) -> str:
    hints: list[str] = []
    if plugin_config.extra_modules:
        hints.append(
            "Check that --extra_op_modules / plugin extra_modules import paths are correct "
            "and @register_op impl_family names match extra_op_fallback."
        )
    else:
        hints.append(
            "Add --extra_op_modules <module> (scheme 1) or --extra_op_plugins <name> (scheme 2)."
        )
    try:
        from importlib.metadata import entry_points as eps_fn

        available = sorted(ep.name for ep in eps_fn(group="lightllm.op_plugins"))
    except Exception:
        available = []
    if available:
        hints.append(f"Installed op plugins: {available}.")
    for family in missing:
        if family.endswith("_plugin") or family == "example_plugin":
            hints.append(
                f"Family {family!r} looks like a pip plugin impl name; "
                f"you likely want --extra_op_plugins example_op_plugin, "
                f"not --extra_op_fallback {family}."
            )
    return " ".join(hints)
