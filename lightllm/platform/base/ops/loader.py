from lightllm.platform.base.fallback_loader import FallbackLoaderSpec, make_fallback_loader
from lightllm.platform.base.ops.base import OP_NAMES
from lightllm.platform.base.ops.registry import op_registry
from lightllm.platform.plugins import OPS

OP_FAMILY_MODULES_PREFIX = "lightllm.platform.ops."


def _resolve_op(op_name: str, fallback_chain: tuple[str, ...]):
    return op_registry.resolve(op_name, fallback_chain=fallback_chain)


build_ops = make_fallback_loader(
    plugin=OPS,
    spec=FallbackLoaderSpec(
        module_prefix=OP_FAMILY_MODULES_PREFIX,
        op_names=OP_NAMES,
    ),
    registry=op_registry,
    resolve_impl=_resolve_op,
)
