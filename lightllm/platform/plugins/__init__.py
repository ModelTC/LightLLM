from lightllm.platform.plugins.common import Plugin


OPS = Plugin(
    name="ops",
    entry_point_group="lightllm.ops_plugins",
    register_decorator="@register_op",
    platform_fallback_field="ops_fallback",
)

SAMPLING = Plugin(
    name="sampling",
    entry_point_group="lightllm.sampling_plugins",
    register_decorator="@register_sampling_op",
    platform_fallback_field="sampling_fallback",
)

ATT = Plugin(
    name="att",
    entry_point_group="lightllm.att_plugins",
    register_decorator="@register_att_backend",
)

ALL_PLUGINS = (OPS, SAMPLING, ATT)


def configure_plugins() -> None:
    for plugin in ALL_PLUGINS:
        plugin.configure()


def add_plugin_cli_args(parser) -> None:
    for plugin in ALL_PLUGINS:
        for field in ("plugins",) + plugin.fields:
            parser.add_argument(
                plugin.cli_flag(field),
                type=str,
                default=None,
                help=plugin.cli_help(field),
            )


__all__ = ["OPS", "SAMPLING", "ATT", "ALL_PLUGINS", "configure_plugins", "add_plugin_cli_args"]
