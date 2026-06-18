import importlib
import importlib.util

from lightllm.platform.base.registry import get_platform_spec
from lightllm.platform.base.sampling import SAMPLING_OP_NAMES, SamplingProtocol
from lightllm.platform.plugins import (
    SamplingPluginConfig,
    get_sampling_plugin_config,
    resolve_sampling_fallback,
)
from lightllm.platform.plugins.sampling import ENTRY_POINT_GROUP
from lightllm.platform.plugins.common import list_installed_plugin_names
from lightllm.utils.envs_utils import get_env_start_args

SAMPLING_FAMILY_MODULES_PREFIX = "lightllm.platform.sampling."


def has_builtin_sampling_module(family: str) -> bool:
    module_name = f"{SAMPLING_FAMILY_MODULES_PREFIX}{family}"
    return importlib.util.find_spec(module_name) is not None


def get_sampling_family_modules_for_fallback(fallback_chain: tuple[str, ...]) -> tuple[str, ...]:
    modules: list[str] = []
    seen: set[str] = set()
    for family in fallback_chain:
        module_name = f"{SAMPLING_FAMILY_MODULES_PREFIX}{family}"
        if module_name in seen:
            continue
        if not has_builtin_sampling_module(family):
            continue
        modules.append(module_name)
        seen.add(module_name)
    return tuple(modules)


def load_sampling(fallback_chain: tuple[str, ...], extra_modules: tuple[str, ...] = ()) -> None:
    for module_name in extra_modules:
        importlib.import_module(module_name)

    for module_name in get_sampling_family_modules_for_fallback(fallback_chain):
        importlib.import_module(module_name)


class SamplingView(SamplingProtocol):

    def __init__(self, platform: str, *, fallback_chain: tuple[str, ...]) -> None:
        from lightllm.platform.base.sampling import sampling_registry

        self._platform = platform
        sampling_backend = get_env_start_args().sampling_backend

        for op_name in SAMPLING_OP_NAMES:
            impl = sampling_registry.resolve(
                op_name,
                sampling_backend=sampling_backend,
                fallback_chain=fallback_chain,
            )
            setattr(self, op_name, impl)

    def __getattr__(self, name: str):
        raise AttributeError(
            f"Sampling op '{name}' is not registered for platform '{self._platform}'"
        )


def build_sampling(platform: str) -> SamplingView:
    plugin_config = get_sampling_plugin_config()
    fallback_chain = resolve_sampling_fallback(platform, plugin_config)
    load_sampling(fallback_chain, plugin_config.extra_modules)
    _validate_extra_fallback_loaded(platform, plugin_config)

    return SamplingView(platform, fallback_chain=fallback_chain)


def _validate_extra_fallback_loaded(platform: str, plugin_config: SamplingPluginConfig) -> None:
    if not plugin_config.extra_fallback:
        return

    from lightllm.platform.base.sampling import sampling_registry

    missing: list[str] = []
    for family in plugin_config.extra_fallback:
        if sampling_registry.has_impl_family(family):
            continue
        missing.append(family)

    if not missing:
        return

    hints = _format_extra_fallback_hints(missing, plugin_config)
    raise RuntimeError(
        "External sampling impl families configured in extra_sampling_fallback did not register "
        f"any @register_sampling_op implementations after loading: {missing}. "
        f"Fallback chain for platform {platform!r} was "
        f"{resolve_sampling_fallback(platform, plugin_config)!r}, but these families are empty "
        f"so sampling ops silently fall back to {get_platform_spec(platform).sampling_fallback!r}. "
        f"{hints}"
    )


def _format_extra_fallback_hints(missing: list[str], plugin_config: SamplingPluginConfig) -> str:
    hints: list[str] = []
    if plugin_config.extra_modules:
        hints.append(
            "Check that --extra_sampling_modules / plugin extra_modules import paths are correct "
            "and @register_sampling_op impl_family names match extra_sampling_fallback."
        )
    else:
        hints.append(
            "Add --extra_sampling_modules <module> (scheme 1) or "
            "--extra_sampling_plugins <name> (scheme 2)."
        )
    available = list_installed_plugin_names(ENTRY_POINT_GROUP)
    if available:
        hints.append(f"Installed sampling plugins: {available}.")
    for family in missing:
        if family.endswith("_plugin") or family == "example_plugin":
            hints.append(
                f"Family {family!r} looks like a pip plugin impl name; "
                f"you likely want --extra_sampling_plugins example_sampling_plugin, "
                f"not --extra_sampling_fallback {family}."
            )
    return " ".join(hints)
