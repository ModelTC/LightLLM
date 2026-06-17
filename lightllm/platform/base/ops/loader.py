import importlib.util
from lightllm.platform.base.ops import op_registry, OpsProtocol, OP_NAMES
from lightllm.platform.plugins import OpsPluginConfig, get_ops_plugin_config, resolve_ops_fallback
from lightllm.platform.plugins.ops import ENTRY_POINT_GROUP
from lightllm.platform.plugins.common import list_installed_plugin_names

# For internal module registration, we use the prefix "lightllm.platform.ops."
OP_FAMILY_MODULES_PREFIX = "lightllm.platform.ops."


def has_builtin_ops_module(family: str) -> bool:
    module_name = f"{OP_FAMILY_MODULES_PREFIX}{family}"
    return importlib.util.find_spec(module_name) is not None


def get_ops_family_modules_for_fallback(fallback_chain: tuple[str, ...]) -> tuple[str, ...]:
    modules: list[str] = []
    seen: set[str] = set()
    for family in fallback_chain:
        module_name = f"{OP_FAMILY_MODULES_PREFIX}{family}"
        if module_name in seen:
            continue
        # External modules are not required to be imported by lightllm.platform.ops
        if not has_builtin_ops_module(family):
            continue

        modules.append(module_name)
        seen.add(module_name)

    return tuple(modules)


def load_ops(fallback_chain: tuple[str, ...], extra_modules: tuple[str, ...] = ()) -> None:
    import importlib

    # Load extra modules first
    for module_name in extra_modules:
        importlib.import_module(module_name)
 
    for module_name in get_ops_family_modules_for_fallback(fallback_chain):
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
    fallback_chain = resolve_ops_fallback(platform, plugin_config)
    load_ops(fallback_chain, plugin_config.extra_modules)
    _validate_extra_fallback_loaded(platform, plugin_config)

    return OpsView(platform, fallback_chain=fallback_chain)


def _validate_extra_fallback_loaded(platform: str, plugin_config: OpsPluginConfig) -> None:
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
        f"{resolve_ops_fallback(platform, plugin_config)!r}, but these families are empty "
        f"so all ops silently fall back to {get_platform_spec(platform).op_fallback!r}. "
        f"{hints}"
    )


def _format_extra_fallback_hints(missing: list[str], plugin_config: OpsPluginConfig) -> str:
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
    available = list_installed_plugin_names(ENTRY_POINT_GROUP)
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