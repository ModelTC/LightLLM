"""Select explicit compatibility classes without importing LightLLM model implementations."""

from importlib import import_module

_LOCAL_CONFIGS = {
    "deepseek_v32": ("composite", "DeepseekV32Config"),
    "internlm": ("internlm", "InternLMConfig"),
    "internlm2": ("internlm2", "InternLM2Config"),
    "minicpm": ("minicpm", "MiniCPMConfig"),
    "qwen": ("qwen", "QWenConfig"),
    "intern_vit_6b": ("intern_vit", "InternVisionConfig"),
    "internvl_chat": ("composite", "InternVLChatConfig"),
}


def get_local_config_class(source):
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    model_type = source.get("model_type")
    if model_type == "qwen3_5" and "dflash_config" in source:
        from lightllm.utils.hf_configs.qwen3_5 import Qwen3_5DraftConfig

        return Qwen3_5DraftConfig
    if model_type == "llava":
        from lightllm.utils.hf_configs.composite import LegacyLlavaConfig, TarsierConfig

        if (source.get("architectures") or [])[:1] == ["TarsierForConditionalGeneration"]:
            return TarsierConfig
        if "text_config" not in source:
            return LegacyLlavaConfig
    # Early MiniCPM checkpoints identify their Config only through auto_map.
    if model_type is None and (source.get("auto_map") or {}).get("AutoConfig") == "configuration_minicpm.MiniCPMConfig":
        model_type = "minicpm"
    if model_type in _LOCAL_CONFIGS:
        if model_type not in CONFIG_MAPPING:
            module, name = _LOCAL_CONFIGS[model_type]
            config_class = getattr(import_module(f"lightllm.utils.hf_configs.{module}"), name)
            AutoConfig.register(model_type, config_class)
        return CONFIG_MAPPING[model_type]
    return None


def config_from_dict(source):
    """Construct a known nested component; unknown model types fail explicitly."""
    from transformers import AutoConfig, PretrainedConfig

    if isinstance(source, PretrainedConfig):
        return source
    config_class = get_local_config_class(source)
    if config_class is not None:
        return config_class.from_dict(source)
    values = dict(source)
    return AutoConfig.for_model(values.pop("model_type"), **values)
