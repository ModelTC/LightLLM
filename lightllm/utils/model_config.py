"""Load HF config objects; adapt only at LightLLM's dictionary boundaries."""

from copy import deepcopy
from typing import TYPE_CHECKING

from lightllm.common.build_utils import repair_config

if TYPE_CHECKING:
    from transformers import PretrainedConfig


def read_model_config(model_path, **kwargs) -> dict:
    """Read source metadata using HF's path, revision and versioned-file resolution."""
    from transformers import PretrainedConfig

    config, _ = PretrainedConfig.get_config_dict(str(model_path), **kwargs)
    return config


def load_model_config(model_path, *, trust_remote_code=False, **kwargs) -> "PretrainedConfig":
    """Return a fresh concrete Config. Custom checkpoint code requires explicit trust."""
    _, config = _load_model_config(model_path, trust_remote_code=trust_remote_code, **kwargs)
    return config


def _load_model_config(model_path, *, trust_remote_code, **kwargs):
    """Keep source identity separate from Config-class parsing, as in vLLM's HF parser."""
    from transformers import AutoConfig
    from lightllm.utils.hf_config import get_local_config_class

    source = read_model_config(model_path, **kwargs)
    custom_config = (source.get("auto_map") or {}).get("AutoConfig")
    if custom_config and trust_remote_code:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # AutoConfig prioritizes registered local classes even with explicit trust.
        # Use HF's dynamic loader so previous compatibility loads cannot hide auto_map.
        kwargs["_from_auto"] = True
        kwargs["name_or_path"] = str(model_path)
        code_revision = kwargs.pop("code_revision", None)
        config_class = get_class_from_dynamic_module(
            custom_config, str(model_path), code_revision=code_revision, **kwargs
        )
        config_class.register_for_auto_class()
        return source, config_class.from_pretrained(str(model_path), **kwargs)
    config_class = get_local_config_class(source)
    if config_class is not None:
        return source, config_class.from_pretrained(str(model_path), **kwargs)
    return source, AutoConfig.from_pretrained(str(model_path), trust_remote_code=trust_remote_code, **kwargs)


def get_text_config(config: "PretrainedConfig") -> "PretrainedConfig":
    """Select the text component of an owned root, including legacy InternVL/Omni."""
    thinker_config = getattr(config, "thinker_config", None)
    if thinker_config is not None:
        config = thinker_config
    llm_config = getattr(config, "llm_config", None)
    if llm_config is not None:
        return llm_config
    text = config.get_text_config()
    return config if text is config else get_text_config(text)


def to_model_config_dict(config: "PretrainedConfig") -> dict:
    """Create an independent runtime dictionary with LightLLM dimension/RoPE aliases."""
    from transformers import PretrainedConfig

    result = deepcopy(config.to_dict())
    for key in result:
        component = getattr(config, key, None)
        if isinstance(component, PretrainedConfig):
            result[key] = to_model_config_dict(component)

    # Tarsier's existing registry/tokenizer contract identifies a Qwen2-VL
    # backbone at text_config; HF 5.8 places its text dimensions one level deeper.
    if (
        config.model_type == "llava"
        and (config.architectures or [])[:1] == ["TarsierForConditionalGeneration"]
        and getattr(config.text_config, "model_type", None) == "qwen2_vl"
    ):
        result["text_config"].update(to_model_config_dict(get_text_config(config.text_config)))
        result["text_config"]["model_type"] = "qwen2_vl"

    for names in (
        ("num_attention_heads", "n_head"),
        ("hidden_size", "n_embd", "n_embed"),
        ("num_hidden_layers", "n_layer"),
    ):
        # Attribute access honors HF's attribute_map (e.g. GPTBigCode.n_embd).
        value = getattr(config, names[0], None)
        if value is not None:
            result[names[0]] = value
            repair_config(result, same_names=names)

    rope = getattr(config, "rope_parameters", None)
    if config.model_type == "gemma3_text" and isinstance(rope, dict):
        rope = rope.get("full_attention")
    if isinstance(rope, dict) and "rope_type" in rope:
        result["rope_scaling"] = deepcopy(rope)
        if rope["rope_type"] == "longrope":
            result["rope_scaling"]["rope_type"] = "su"
        for key in ("rope_theta", "partial_rotary_factor"):
            if key in rope:
                result[key] = rope[key]
    if config.model_type in {"qwen3_next", "qwen3_5_text", "qwen3_5_moe_text"}:
        result["full_attention_interval"] = get_full_attention_interval(config)
    return result


def get_full_attention_interval(config: "PretrainedConfig") -> int:
    """Translate HF's layer list for the engine's periodic hybrid attention layout."""
    # TODO: Migrate model execution and cache layout to consume layer_types directly,
    # then remove full_attention_interval and this periodic-layout adapter.
    layers = config.layer_types
    interval = next((index + 1 for index, kind in enumerate(layers) if kind == "full_attention"), len(layers) + 1)
    expected = ["full_attention" if (index + 1) % interval == 0 else "linear_attention" for index in range(len(layers))]
    if layers != expected:
        raise ValueError("LightLLM requires a periodic linear/full attention layer layout")
    return interval


def load_model_config_dict(model_path, *, trust_remote_code=None, **kwargs) -> dict:
    """Compatibility boundary for existing engine, tokenizer and registry dict consumers."""
    if trust_remote_code is None:
        trust_remote_code = get_config_trust_remote_code()
    source, config = _load_model_config(model_path, trust_remote_code=trust_remote_code, **kwargs)
    result = to_model_config_dict(config)
    # Registry predicates consume checkpoint identity. A custom Config may
    # serialize its class name instead; absent source fields may use HF defaults.
    for key in ("model_type", "architectures"):
        if key in source:
            result[key] = deepcopy(source[key])
    return result


def get_config_trust_remote_code() -> bool:
    """Use the application's explicit flag; standalone callers default to no custom code."""
    import os
    from lightllm.utils.envs_utils import get_env_start_args

    return getattr(get_env_start_args(), "trust_remote_code", False) if "LIGHTLLM_START_ARGS" in os.environ else False
