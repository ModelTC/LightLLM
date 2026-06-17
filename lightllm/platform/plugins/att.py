from dataclasses import dataclass
from typing import Any, Iterable
from lightllm.platform.plugins.common import (
    ModulePluginConfig,
    load_entry_point_plugins,
    merge_config_field,
    parse_plugin_config,
    plugin_config_from_cli,
    plugin_names_from_cli,
)

PLUGIN_KIND = "att"
ENTRY_POINT_GROUP = "lightllm.att_plugins"


@dataclass(frozen=True)
class AttPluginConfig(ModulePluginConfig):
    pass


_att_plugin_config: AttPluginConfig | None = None   


def merge_att_plugin_configs(configs: Iterable[AttPluginConfig]) -> AttPluginConfig:
    return AttPluginConfig(
        extra_modules=merge_config_field(configs, "extra_modules"),
    )


def configure_att_plugins() -> AttPluginConfig:
    global _att_plugin_config

    plugin_names = plugin_names_from_cli(PLUGIN_KIND)

    direct_config = plugin_config_from_cli(
        AttPluginConfig,
        plugin_kind=PLUGIN_KIND,
        fields=("extra_modules",),
    )

    has_plugins = bool(plugin_names)
    has_direct = bool(direct_config.extra_modules)
    if has_plugins and has_direct:
        raise RuntimeError(
            "Att plugin configuration is ambiguous: use either --extra_att_plugins or "
            "--extra_att_modules, not both."
        )

    if has_plugins:
        _att_plugin_config = merge_att_plugin_configs(
            load_entry_point_plugins(
                plugin_names,
                entry_point_group=ENTRY_POINT_GROUP,
                parser=parse_att_plugin_config,
                plugin_kind=PLUGIN_KIND,
            )
        )
    elif has_direct:
        _att_plugin_config = direct_config
    else:
        _att_plugin_config = AttPluginConfig()

    return _att_plugin_config


def parse_att_plugin_config(value: Any) -> AttPluginConfig:
    return parse_plugin_config(
        value,
        AttPluginConfig,
        fields=("extra_modules",),
        plugin_kind=PLUGIN_KIND,
    )


def get_att_plugin_config() -> AttPluginConfig:
    if _att_plugin_config is None:
        return AttPluginConfig()
    return _att_plugin_config
