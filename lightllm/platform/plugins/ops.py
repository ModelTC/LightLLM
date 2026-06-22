from dataclasses import dataclass

from lightllm.platform.plugins.common import (
    FallbackPluginConfig,
    PluginKind,
    make_fallback_plugin_kind,
)


@dataclass(frozen=True)
class OpsPluginConfig(FallbackPluginConfig):
    pass


OPS: PluginKind[OpsPluginConfig] = make_fallback_plugin_kind(
    kind="ops",
    entry_point_group="lightllm.ops_plugins",
    ambiguous_error=(
        "Ops plugin configuration is ambiguous: use either --extra_ops_plugins or "
        "(--extra_ops_fallback / --extra_ops_modules), not both."
    ),
    platform_fallback_field="ops_fallback",
    register_decorator="@register_op",
    config_cls=OpsPluginConfig,
)
