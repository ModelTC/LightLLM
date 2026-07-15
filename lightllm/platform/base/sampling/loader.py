from lightllm.platform.base.fallback_loader import FallbackLoaderSpec, make_fallback_loader
from lightllm.platform.base.sampling.base import SAMPLING_OP_NAMES
from lightllm.platform.base.sampling.registry import sampling_registry
from lightllm.platform.plugins import SAMPLING
from lightllm.utils.envs_utils import get_env_start_args

SAMPLING_FAMILY_MODULES_PREFIX = "lightllm.platform.sampling."


def _resolve_sampling(op_name: str, fallback_chain: tuple[str, ...]):
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
    ),
    registry=sampling_registry,
    resolve_impl=_resolve_sampling,
)
