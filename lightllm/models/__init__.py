from importlib import import_module

from .registry import get_model as _registry_get_model
from .registry import get_model_class as _registry_get_model_class


_MODEL_EXPORTS = {
    "MixtralTpPartModel": ("lightllm.models.mixtral.model", "MixtralTpPartModel"),
    "BloomTpPartModel": ("lightllm.models.bloom.model", "BloomTpPartModel"),
    "LlamaTpPartModel": ("lightllm.models.llama.model", "LlamaTpPartModel"),
    "StarcoderTpPartModel": ("lightllm.models.starcoder.model", "StarcoderTpPartModel"),
    "Starcoder2TpPartModel": ("lightllm.models.starcoder2.model", "Starcoder2TpPartModel"),
    "QWenTpPartModel": ("lightllm.models.qwen.model", "QWenTpPartModel"),
    "Qwen2TpPartModel": ("lightllm.models.qwen2.model", "Qwen2TpPartModel"),
    "Qwen3TpPartModel": ("lightllm.models.qwen3.model", "Qwen3TpPartModel"),
    "Qwen3MOEModel": ("lightllm.models.qwen3_moe.model", "Qwen3MOEModel"),
    "Qwen3NextTpPartModel": ("lightllm.models.qwen3next.model", "Qwen3NextTpPartModel"),
    "InternlmTpPartModel": ("lightllm.models.internlm.model", "InternlmTpPartModel"),
    "StablelmTpPartModel": ("lightllm.models.stablelm.model", "StablelmTpPartModel"),
    "Internlm2TpPartModel": ("lightllm.models.internlm2.model", "Internlm2TpPartModel"),
    "Internlm2RewardTpPartModel": (
        "lightllm.models.internlm2_reward.model",
        "Internlm2RewardTpPartModel",
    ),
    "MistralTpPartModel": ("lightllm.models.mistral.model", "MistralTpPartModel"),
    "MiniCPMTpPartModel": ("lightllm.models.minicpm.model", "MiniCPMTpPartModel"),
    "LlavaTpPartModel": ("lightllm.models.llava.model", "LlavaTpPartModel"),
    "QWenVLTpPartModel": ("lightllm.models.qwen_vl.model", "QWenVLTpPartModel"),
    "Gemma_2bTpPartModel": ("lightllm.models.gemma_2b.model", "Gemma_2bTpPartModel"),
    "Phi3TpPartModel": ("lightllm.models.phi3.model", "Phi3TpPartModel"),
    "Deepseek2TpPartModel": ("lightllm.models.deepseek2.model", "Deepseek2TpPartModel"),
    "Deepseek3_2TpPartModel": ("lightllm.models.deepseek3_2.model", "Deepseek3_2TpPartModel"),
    "Glm4MoeLiteTpPartModel": (
        "lightllm.models.glm4_moe_lite.model",
        "Glm4MoeLiteTpPartModel",
    ),
    "InternVLLlamaTpPartModel": ("lightllm.models.internvl.model", "InternVLLlamaTpPartModel"),
    "InternVLPhi3TpPartModel": ("lightllm.models.internvl.model", "InternVLPhi3TpPartModel"),
    "InternVLQwen2TpPartModel": ("lightllm.models.internvl.model", "InternVLQwen2TpPartModel"),
    "InternVLDeepSeek2TpPartModel": (
        "lightllm.models.internvl.model",
        "InternVLDeepSeek2TpPartModel",
    ),
    "InternVLInternlm2TpPartModel": (
        "lightllm.models.internvl.model",
        "InternVLInternlm2TpPartModel",
    ),
    "Qwen2VLTpPartModel": ("lightllm.models.qwen2_vl.model", "Qwen2VLTpPartModel"),
    "Qwen2RewardTpPartModel": ("lightllm.models.qwen2_reward.model", "Qwen2RewardTpPartModel"),
    "Qwen3VLTpPartModel": ("lightllm.models.qwen3_vl.model", "Qwen3VLTpPartModel"),
    "Qwen3VLMOETpPartModel": ("lightllm.models.qwen3_vl_moe.model", "Qwen3VLMOETpPartModel"),
    "Gemma3TpPartModel": ("lightllm.models.gemma3.model", "Gemma3TpPartModel"),
    "Gemma4TpPartModel": ("lightllm.models.gemma4.model", "Gemma4TpPartModel"),
    "Tarsier2Qwen2TpPartModel": ("lightllm.models.tarsier2.model", "Tarsier2Qwen2TpPartModel"),
    "Tarsier2Qwen2VLTpPartModel": (
        "lightllm.models.tarsier2.model",
        "Tarsier2Qwen2VLTpPartModel",
    ),
    "Tarsier2LlamaTpPartModel": ("lightllm.models.tarsier2.model", "Tarsier2LlamaTpPartModel"),
    "GptOssTpPartModel": ("lightllm.models.gpt_oss.model", "GptOssTpPartModel"),
    "Qwen3OmniMOETpPartModel": (
        "lightllm.models.qwen3_omni_moe_thinker.model",
        "Qwen3OmniMOETpPartModel",
    ),
    "Qwen3_5TpPartModel": ("lightllm.models.qwen3_5.model", "Qwen3_5TpPartModel"),
    "Qwen3_5MOETpPartModel": ("lightllm.models.qwen3_5_moe.model", "Qwen3_5MOETpPartModel"),
}

_MODEL_TYPE_REGISTRY_MODULES = {
    "starcoder2": ("lightllm.models.starcoder2.model",),
    "internlm2": ("lightllm.models.internlm2.model",),
    "llava": ("lightllm.models.llava.model",),
    "qwen": ("lightllm.models.qwen.model",),
    "qwen2": ("lightllm.models.qwen2.model",),
    "qwen2_vl": ("lightllm.models.qwen2_vl.model",),
    "qwen2_5_vl": ("lightllm.models.qwen2_vl.model",),
    "qwen3": ("lightllm.models.qwen3.model",),
    "qwen3_moe": ("lightllm.models.qwen3_moe.model",),
    "qwen3_next": ("lightllm.models.qwen3next.model",),
    "qwen3_vl": ("lightllm.models.qwen3_vl.model",),
    "qwen3_vl_moe": ("lightllm.models.qwen3_vl_moe.model",),
    "qwen3_omni_moe": ("lightllm.models.qwen3_omni_moe_thinker.model",),
    "qwen3_5": ("lightllm.models.qwen3_5.model",),
    "qwen3_5_moe": ("lightllm.models.qwen3_5_moe.model",),
    "deepseek_v2": ("lightllm.models.deepseek2.model",),
    "deepseek_v3": ("lightllm.models.deepseek2.model",),
    "deepseek_v32": ("lightllm.models.deepseek3_2.model",),
    "glm4_moe_lite": ("lightllm.models.glm4_moe_lite.model",),
    "bloom": ("lightllm.models.bloom.model",),
    "gpt_bigcode": ("lightllm.models.starcoder.model",),
    "minicpm": ("lightllm.models.minicpm.model",),
    "gemma3": ("lightllm.models.gemma3.model",),
    "gemma": ("lightllm.models.gemma_2b.model",),
    "gemma4": ("lightllm.models.gemma4.model",),
    "internlm": ("lightllm.models.internlm.model",),
    "stablelm": ("lightllm.models.stablelm.model",),
    "mistral": ("lightllm.models.mistral.model",),
    "gpt_oss": ("lightllm.models.gpt_oss.model",),
    "phi3": ("lightllm.models.phi3.model",),
    "llama": ("lightllm.models.llama.model",),
    "mixtral": ("lightllm.models.mixtral.model",),
    "internvl_chat": ("lightllm.models.internvl.model",),
}
_bootstrapped_registry_modules = set()


def _load_model_attr(name):
    module_name, attr_name = _MODEL_EXPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def _has_architecture(model_cfg: dict, name: str) -> bool:
    return any(name in architecture for architecture in model_cfg.get("architectures", []))


def _llava_text_model_type(model_cfg: dict) -> str:
    return model_cfg.get("llm_config", {}).get("model_type", "") or model_cfg.get("text_config", {}).get(
        "model_type", ""
    )


def _registry_modules_for_model_cfg(model_cfg: dict):
    model_type = str(model_cfg.get("model_type", ""))

    module_names = _MODEL_TYPE_REGISTRY_MODULES.get(model_type)
    if module_names is None:
        # Leave already-registered plugin/custom models available, but avoid
        # importing every built-in module just to produce an unsupported-model
        # error. Some built-ins have optional multimodal dependencies.
        return ()

    if model_type == "qwen" and "visual" in model_cfg:
        module_names = module_names + ("lightllm.models.qwen_vl.model",)
    elif model_type == "qwen2" and _has_architecture(model_cfg, "RewardModel"):
        module_names = module_names + ("lightllm.models.qwen2_reward.model",)
    elif model_type == "internlm2" and _has_architecture(model_cfg, "RewardModel"):
        module_names = module_names + ("lightllm.models.internlm2_reward.model",)
    elif model_type == "llava" and _llava_text_model_type(model_cfg) in {"qwen2", "qwen2_vl", "llama"}:
        module_names = module_names + ("lightllm.models.tarsier2.model",)

    return module_names


def _ensure_model_registry_bootstrapped(model_cfg: dict) -> None:
    module_names = _registry_modules_for_model_cfg(model_cfg)

    for module_name in module_names:
        if module_name in _bootstrapped_registry_modules:
            continue
        import_module(module_name)
        _bootstrapped_registry_modules.add(module_name)
    return


def get_model(model_cfg: dict, model_kvargs: dict):
    _ensure_model_registry_bootstrapped(model_cfg)
    return _registry_get_model(model_cfg, model_kvargs)


def get_model_class(model_cfg: dict):
    _ensure_model_registry_bootstrapped(model_cfg)
    return _registry_get_model_class(model_cfg)


def __getattr__(name):
    if name in _MODEL_EXPORTS:
        return _load_model_attr(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_model", "get_model_class"] + list(_MODEL_EXPORTS)
