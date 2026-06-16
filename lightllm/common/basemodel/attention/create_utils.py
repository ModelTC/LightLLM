"""Attention backend selection utilities."""
from lightllm.platform.base.attention import (
    AttCategory,
    att_backend_registry,
    ensure_att_backends_loaded,
)
from lightllm.common.basemodel.attention.base_att import BaseAttBackend
from lightllm.utils.backend_validator import validate
from lightllm.utils.envs_utils import get_env_start_args, get_page_size
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


def _resolve_registry_name(
    name: str,
    *,
    category: AttCategory,
    kv_type: str,
    platform: str | None,
) -> str:
    if (
        name == "fa3"
        and category == "standard"
        and kv_type == "None"
        and get_page_size() > 1
        and att_backend_registry.is_registered(
            category=category,
            name="paged_fa3",
            kv_type=kv_type,
            platform=platform,
        )
    ):
        return "paged_fa3"
    return name


def _get_att_backend_class(
    *,
    category: AttCategory,
    backend_name: str,
    kv_type: str,
    platform: str | None,
) -> type:
    ensure_att_backends_loaded()
    resolved_name = _resolve_registry_name(
        backend_name,
        category=category,
        kv_type=kv_type,
        platform=platform,
    )
    return att_backend_registry.resolve_backend_cls(
        category=category,
        name=resolved_name,
        kv_type=kv_type,
        platform=platform,
    )


def _fallback_att_backend_class(
    *,
    category: AttCategory,
    kv_type: str,
    platform: str | None,
) -> type:
    """Pick a registered backend without validation when auto-selection fails."""
    registered = att_backend_registry.list_names(
        category=category,
        kv_type=kv_type,
        platform=platform,
    )
    if not registered:
        raise ValueError(
            f"No attention backends are registered for "
            f"category={category!r}, kv_type={kv_type!r}, platform={platform!r}"
        )

    fallback_name = "triton" if "triton" in registered else registered[0]
    resolved_name = _resolve_registry_name(
        fallback_name,
        category=category,
        kv_type=kv_type,
        platform=platform,
    )
    logger.warning(
        f"No backend validation succeeded, falling back to {fallback_name!r} "
        f"(resolved as {resolved_name!r})"
    )
    return att_backend_registry.resolve_backend_cls(
        category=category,
        name=resolved_name,
        kv_type=kv_type,
        platform=platform,
    )


def _auto_select_backend(
    *,
    category: AttCategory,
    kv_type: str,
    platform: str | None,
    priority_list: list[str],
) -> type:
    ensure_att_backends_loaded()

    if get_env_start_args().enable_ep_moe:
        logger.info("Expert parallelism with MoE enabled, excluding flashinfer attention backend")
        priority_list = [name for name in priority_list if name != "flashinfer"]

    for backend_name in priority_list:
        resolved_name = _resolve_registry_name(
            backend_name,
            category=category,
            kv_type=kv_type,
            platform=platform,
        )
        if not att_backend_registry.is_registered(
            category=category,
            name=resolved_name,
            kv_type=kv_type,
            platform=platform,
        ):
            continue

        # Get 'AttBackendSpec'
        spec = att_backend_registry.get_spec(
            category=category,
            name=resolved_name,
            kv_type=kv_type,
        )
        validate_name = spec.effective_validate_name() if spec is not None else backend_name
        if validate(validate_name):
            logger.info(f"Auto-selected {backend_name} backend (validated)")
            return att_backend_registry.resolve_backend_cls(
                category=category,
                name=resolved_name,
                kv_type=kv_type,
                platform=platform,
            )


    return _fallback_att_backend_class(
        category=category,
        kv_type=kv_type,
        platform=platform,
    )


def _select_att_backend_class(
    *,
    category: AttCategory,
    backend_str: str,
    priority_list: list[str],
) -> type:
    args = get_env_start_args()
    kv_type = args.llm_kv_type
    platform = args.hardware_platform
    # If backend_str is not "auto", use the specified backend
    if backend_str != "auto":
        return _get_att_backend_class(
            category=category,
            backend_name=backend_str,
            kv_type=kv_type,
            platform=platform,
        )
    # Auto select backend from priority_list
    return _auto_select_backend(
        category=category,
        kv_type=kv_type,
        platform=platform,
        priority_list=priority_list,
    )


def get_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    return _select_att_backend_class(
        category="standard",
        backend_str=args.llm_prefill_att_backend[index],
        priority_list=priority_list,
    )


def get_decode_att_backend_class(index=0, priority_list: list = ["flashinfer", "fa3", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    return _select_att_backend_class(
        category="standard",
        backend_str=args.llm_decode_att_backend[index],
        priority_list=priority_list,
    )


def get_mla_prefill_att_backend_class(index=0, priority_list: list = ["fa3", "flashinfer", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    return _select_att_backend_class(
        category="mla",
        backend_str=args.llm_prefill_att_backend[index],
        priority_list=priority_list,
    )


def get_mla_decode_att_backend_class(index=0, priority_list: list = ["flashinfer", "fa3", "triton"]) -> BaseAttBackend:
    args = get_env_start_args()
    return _select_att_backend_class(
        category="mla",
        backend_str=args.llm_decode_att_backend[index],
        priority_list=priority_list,
    )


def get_nsa_prefill_att_backend_class(index=0, priority_list: list = ["flashmla_sparse"]) -> BaseAttBackend:
    args = get_env_start_args()
    return _select_att_backend_class(
        category="nsa",
        backend_str=args.llm_prefill_att_backend[index],
        priority_list=priority_list,
    )


def get_nsa_decode_att_backend_class(index=0, priority_list: list = ["flashmla_sparse"]) -> BaseAttBackend:
    args = get_env_start_args()
    return _select_att_backend_class(
        category="nsa",
        backend_str=args.llm_decode_att_backend[index],
        priority_list=priority_list,
    )
