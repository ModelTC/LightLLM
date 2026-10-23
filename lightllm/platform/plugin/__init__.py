from lightllm.platform.plugin.common import Plugin, plugin_registration


OPS = Plugin(name="ops", entry_point_group="lightllm.ops_plugin")
ATT = Plugin(name="att", entry_point_group="lightllm.att_plugin")

_PLUGINS = (OPS, ATT)


def configure_plugins() -> None:
    with plugin_registration():
        for plugin in _PLUGINS:
            plugin.load()
