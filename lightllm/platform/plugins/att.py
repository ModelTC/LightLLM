from dataclasses import dataclass

from lightllm.platform.plugins.common import (
    ModulePluginConfig,
    PluginKind,
    make_module_plugin_kind,
)


@dataclass(frozen=True)
class AttPluginConfig(ModulePluginConfig):
    pass


ATT: PluginKind[AttPluginConfig] = make_module_plugin_kind(
    kind="att",
    entry_point_group="lightllm.att_plugins",
    ambiguous_error=(
        "Att plugin configuration is ambiguous: use either --extra_att_plugins or "
        "--extra_att_modules, not both."
    ),
    config_cls=AttPluginConfig,
)
