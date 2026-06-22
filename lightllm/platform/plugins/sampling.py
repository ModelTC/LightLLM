from dataclasses import dataclass

from lightllm.platform.plugins.common import (
    FallbackPluginConfig,
    PluginKind,
    make_fallback_plugin_kind,
)


@dataclass(frozen=True)
class SamplingPluginConfig(FallbackPluginConfig):
    pass


SAMPLING: PluginKind[SamplingPluginConfig] = make_fallback_plugin_kind(
    kind="sampling",
    entry_point_group="lightllm.sampling_plugins",
    ambiguous_error=(
        "Sampling plugin configuration is ambiguous: use either --extra_sampling_plugins or "
        "(--extra_sampling_fallback / --extra_sampling_modules), not both."
    ),
    platform_fallback_field="sampling_fallback",
    register_decorator="@register_sampling_op",
    config_cls=SamplingPluginConfig,
)
