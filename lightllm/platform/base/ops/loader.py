from lightllm.platform.base.fallback_loader import FallbackLoaderSpec, make_fallback_loader
from lightllm.platform.base.ops.base import OP_NAMES, OpsProtocol
from lightllm.platform.base.ops.registry import op_registry
from lightllm.platform.plugins import OPS

OP_FAMILY_MODULES_PREFIX = "lightllm.platform.ops."


def _resolve_op(op_name: str, fallback_chain: tuple[str, ...], platform: str):
    for impl_family in fallback_chain:
        impl = op_registry.get(impl_family, op_name)
        if impl is not None:
            return impl
    raise KeyError(
        f"Op '{op_name}' is not registered for platform '{platform}' "
        f"via fallback chain {fallback_chain}"
    )


build_ops = make_fallback_loader(
    plugin=OPS,
    spec=FallbackLoaderSpec(
        module_prefix=OP_FAMILY_MODULES_PREFIX,
        op_names=OP_NAMES,
        register_decorator="@register_op",
        platform_fallback_field="ops_fallback",
        view_label="Op",
        silent_fallback_entity="all ops",
    ),
    registry=op_registry,
    resolve_impl=_resolve_op,
    view_protocol=OpsProtocol,
)
