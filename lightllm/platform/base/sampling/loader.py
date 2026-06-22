from lightllm.platform.base.fallback_loader import FallbackLoaderSpec, make_fallback_loader
from lightllm.platform.base.sampling.base import SAMPLING_OP_NAMES, SamplingProtocol
from lightllm.platform.base.sampling.registry import sampling_registry
from lightllm.platform.plugins import SAMPLING
from lightllm.utils.envs_utils import get_env_start_args

SAMPLING_FAMILY_MODULES_PREFIX = "lightllm.platform.sampling."


def _resolve_sampling(op_name: str, fallback_chain: tuple[str, ...], platform: str):
    return sampling_registry.resolve(
        op_name,
        sampling_backend=get_env_start_args().sampling_backend,
        fallback_chain=fallback_chain,
    )


build_sampling = make_fallback_loader(
    plugin=SAMPLING,
    spec=FallbackLoaderSpec(
        module_prefix=SAMPLING_FAMILY_MODULES_PREFIX,
        op_names=SAMPLING_OP_NAMES,
        register_decorator="@register_sampling_op",
        platform_fallback_field="sampling_fallback",
        view_label="Sampling op",
        silent_fallback_entity="sampling ops",
    ),
    registry=sampling_registry,
    resolve_impl=_resolve_sampling,
    view_protocol=SamplingProtocol,
)
