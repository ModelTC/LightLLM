from dataclasses import dataclass

from lightllm.platform.plugin.common import is_plugin_registering
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

AttKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class _AttImpl:
    backend_cls: type
    plugin: bool


ATT_IMPLS: dict[AttKey, _AttImpl] = {}


def register_att_backend(
    name: str,
    *,
    category: str,
    kv_type: str = "None",
    platforms: tuple[str, ...] = ("cuda",),
):
    def decorator(backend_cls):
        plugin = is_plugin_registering()
        staged: dict[AttKey, _AttImpl] = {}
        for platform in platforms:
            key = (platform, name, category, kv_type)
            current = staged.get(key, ATT_IMPLS.get(key))
            if current is None:
                staged[key] = _AttImpl(backend_cls, plugin)
                continue
            if plugin and not current.plugin:
                logger.info(
                    "Att backend %s category %s kv type %s on platform %s: plugin replaced builtin implementation.",
                    name,
                    category,
                    kv_type,
                    platform,
                )
                staged[key] = _AttImpl(backend_cls, plugin=True)
                continue
            raise ValueError(f"The att backend {name} is already registered for platform {platform!r}.")
        ATT_IMPLS.update(staged)
        return backend_cls

    return decorator


def get_att_backend_class(
    name: str,
    category: str,
    kv_type: str,
    platform: str,
) -> type:
    key = (platform, name, category, kv_type)
    entry = ATT_IMPLS.get(key)
    if entry is None:
        raise ValueError(
            f"The att backend {name} is not registered for category {category}, "
            f"kv type {kv_type}, and platform {platform}."
        )
    return entry.backend_cls
