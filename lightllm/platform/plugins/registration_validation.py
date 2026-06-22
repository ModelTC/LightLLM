from typing import Callable

from lightllm.platform.plugins.common import (
    FallbackPluginConfig,
    PluginKind,
    PluginKindSpec,
)


def format_registration_hints(
    kind_spec: PluginKindSpec,
    *,
    register_decorator: str,
    missing: tuple[str, ...],
    installed_plugins: tuple[str, ...],
    has_extra_modules: bool | None = None,
) -> str:
    """ Build hint text for post-load registration validation failures. """
    modules_flag = kind_spec.cli_flag("modules")
    plugins_flag = kind_spec.cli_flag("plugins")
    fallback_flag = kind_spec.cli_flag("fallback")
    hints: list[str] = []

    if has_extra_modules is None:
        hints.append(
            f"Check that {modules_flag} / plugin extra_modules import paths are correct "
            f"and modules define {register_decorator}."
        )
    elif has_extra_modules:
        hints.append(
            f"Check that {modules_flag} / plugin extra_modules import paths are correct "
            f"and {register_decorator} impl_family names match extra_fallback."
        )
    else:
        hints.append(
            f"External impl families need modules that call {register_decorator}. "
            f"Use {modules_flag} (scheme 1) or {plugins_flag} (scheme 2)."
        )

    if installed_plugins:
        hints.append(f"Installed {kind_spec.kind} plugins: {installed_plugins}.")
    elif has_extra_modules is None:
        hints.append(
            f"No {kind_spec.kind} plugins installed; use {plugins_flag} <name> with a package "
            f"registered under entry point group {kind_spec.entry_point_group!r}."
        )

    if has_extra_modules is not None:
        for family in missing:
            if family.endswith("_plugin") or family == "example_plugin":
                hints.append(
                    f"For family {family!r}, did you mean {plugins_flag} "
                    f"{kind_spec.example_plugin_name} instead of {fallback_flag} {family}?"
                )

    return " ".join(hints)


def validate_fallback_families_registered(
    *,
    plugin: PluginKind[FallbackPluginConfig],
    plugin_config: FallbackPluginConfig,
    register_decorator: str,
    platform_fallback_field: str,
    silent_fallback_entity: str,
    registry_has_impl_family: Callable[[str], bool],
    platform: str,
) -> None:
    """ Raise if configured extra_fallback families did not register after loading. """
    if not plugin_config.extra_fallback:
        return

    missing = tuple(
        family
        for family in plugin_config.extra_fallback
        if not registry_has_impl_family(family)
    )
    if not missing:
        return

    from lightllm.platform.base.registry import get_platform_spec

    platform_fallback = getattr(get_platform_spec(platform), platform_fallback_field)
    fallback_flag = plugin.spec.cli_flag("fallback")
    hints = format_registration_hints(
        plugin.spec,
        register_decorator=register_decorator,
        missing=missing,
        installed_plugins=plugin.installed(),
        has_extra_modules=bool(plugin_config.extra_modules),
    )
    raise RuntimeError(
        f"External {plugin.kind} impl families configured in {fallback_flag} did not register "
        f"any {register_decorator} implementations after loading: {list(missing)}. "
        f"Fallback chain for platform {platform!r} was "
        f"{plugin.resolve_fallback_for(platform, plugin_config)!r}, but these families are empty "
        f"so {silent_fallback_entity} silently fall back to {platform_fallback!r}. "
        f"{hints}"
    )


def validate_extra_modules_registered(
    *,
    kind_spec: PluginKindSpec,
    register_decorator: str,
    extra_modules: tuple[str, ...],
    registered_module_names: set[str],
    installed_plugins: tuple[str, ...],
) -> None:
    """ Raise if configured extra_modules did not register after loading. """
    if not extra_modules:
        return

    missing = tuple(
        module_name
        for module_name in extra_modules
        if module_name not in registered_module_names
    )
    if not missing:
        return

    hints = format_registration_hints(
        kind_spec,
        register_decorator=register_decorator,
        missing=missing,
        installed_plugins=installed_plugins,
        has_extra_modules=None,
    )
    modules_flag = kind_spec.cli_flag("modules")
    raise RuntimeError(
        f"External {kind_spec.kind} modules configured in {modules_flag} / "
        f"{kind_spec.cli_flag('plugins')} did not register any {register_decorator} "
        f"implementations after loading: {list(missing)}. {hints}"
    )
