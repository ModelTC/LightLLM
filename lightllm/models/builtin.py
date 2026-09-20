"""Built-in model registrations; keep implementation imports out of this file."""

from .registry import ModelRegistry, is_reward_model, llm_model_type_is
from .draft_registry import DraftModelRegistry


def _is_tarsier(model_cfg):
    # Match the primary architecture used by the tokenizer and vision server.
    return (model_cfg.get("architectures") or [])[:1] == ["TarsierForConditionalGeneration"]


def _tarsier_text_model_is(model_type):
    return lambda cfg: (_is_tarsier(cfg) and (cfg.get("text_config") or {}).get("model_type") == model_type)


# Target models. Conditional registrations override the default for a model type.
ModelRegistry.register("bloom", "lightllm.models.bloom.model:BloomTpPartModel")
ModelRegistry.register(["deepseek_v2", "deepseek_v3"], "lightllm.models.deepseek2.model:Deepseek2TpPartModel")
ModelRegistry.register(["deepseek_v32"], "lightllm.models.deepseek3_2.model:Deepseek3_2TpPartModel")
ModelRegistry.register("gemma3", "lightllm.models.gemma3.model:Gemma3TpPartModel")
ModelRegistry.register("gemma4", "lightllm.models.gemma4.model:Gemma4TpPartModel", is_multimodal=True)
ModelRegistry.register("gemma", "lightllm.models.gemma_2b.model:Gemma_2bTpPartModel")
ModelRegistry.register("glm4_moe_lite", "lightllm.models.glm4_moe_lite.model:Glm4MoeLiteTpPartModel")
ModelRegistry.register("gpt_oss", "lightllm.models.gpt_oss.model:GptOssTpPartModel")
ModelRegistry.register("internlm", "lightllm.models.internlm.model:InternlmTpPartModel")
ModelRegistry.register("internlm2", "lightllm.models.internlm2.model:Internlm2TpPartModel")
ModelRegistry.register(
    "internlm2",
    "lightllm.models.internlm2_reward.model:Internlm2RewardTpPartModel",
    condition=is_reward_model(),
)
ModelRegistry.register(
    ["internvl_chat"],
    "lightllm.models.internvl.model:InternVLPhi3TpPartModel",
    is_multimodal=True,
    condition=llm_model_type_is("phi3"),
)
ModelRegistry.register(
    ["internvl_chat"],
    "lightllm.models.internvl.model:InternVLInternlm2TpPartModel",
    is_multimodal=True,
    condition=llm_model_type_is("internlm2"),
)
ModelRegistry.register(
    ["internvl_chat"],
    "lightllm.models.internvl.model:InternVLLlamaTpPartModel",
    is_multimodal=True,
    condition=llm_model_type_is("llama"),
)
ModelRegistry.register(
    ["internvl_chat"],
    "lightllm.models.internvl.model:InternVLQwen2TpPartModel",
    is_multimodal=True,
    condition=llm_model_type_is("qwen2"),
)
ModelRegistry.register(
    ["internvl_chat"],
    "lightllm.models.internvl.model:InternVLDeepSeek2TpPartModel",
    is_multimodal=True,
    condition=llm_model_type_is(["deepseek_v2", "deepseek_v3"]),
)
ModelRegistry.register(
    ["internvl_chat"],
    "lightllm.models.internvl.model:InternVLQwen3TpPartModel",
    is_multimodal=True,
    condition=llm_model_type_is("qwen3"),
)
ModelRegistry.register(
    ["internvl_chat"],
    "lightllm.models.internvl.model:InternVLQwen3MOETpPartModel",
    is_multimodal=True,
    condition=llm_model_type_is("qwen3_moe"),
)
ModelRegistry.register("llama", "lightllm.models.llama.model:LlamaTpPartModel")
ModelRegistry.register(
    "llava",
    "lightllm.models.llava.model:LlavaTpPartModel",
    is_multimodal=True,
    condition=lambda cfg: not _is_tarsier(cfg),
    is_fallback=True,
)
ModelRegistry.register("minicpm", "lightllm.models.minicpm.model:MiniCPMTpPartModel")
ModelRegistry.register("mistral", "lightllm.models.mistral.model:MistralTpPartModel")
ModelRegistry.register("mixtral", "lightllm.models.mixtral.model:MixtralTpPartModel")
ModelRegistry.register(
    "neo_chat",
    "lightllm.models.neo_chat.model:NeoTpPartModel",
    is_multimodal=True,
    condition=llm_model_type_is("qwen3"),
)
ModelRegistry.register(
    "neo_chat",
    "lightllm.models.neo_chat_moe.model:NeoTpMOEPartModel",
    is_multimodal=True,
    condition=llm_model_type_is("qwen3_moe"),
)
ModelRegistry.register("phi3", "lightllm.models.phi3.model:Phi3TpPartModel")
ModelRegistry.register("qwen", "lightllm.models.qwen.model:QWenTpPartModel")
ModelRegistry.register("qwen2", "lightllm.models.qwen2.model:Qwen2TpPartModel")
ModelRegistry.register(
    "qwen2", "lightllm.models.qwen2_reward.model:Qwen2RewardTpPartModel", condition=is_reward_model()
)
ModelRegistry.register(
    ["qwen2_vl", "qwen2_5_vl"], "lightllm.models.qwen2_vl.model:Qwen2VLTpPartModel", is_multimodal=True
)
ModelRegistry.register("qwen3", "lightllm.models.qwen3.model:Qwen3TpPartModel")
ModelRegistry.register(["qwen3_5"], "lightllm.models.qwen3_5.model:Qwen3_5TpPartModel", is_multimodal=True)
ModelRegistry.register("qwen3_5_moe", "lightllm.models.qwen3_5_moe.model:Qwen3_5MOETpPartModel", is_multimodal=True)
ModelRegistry.register("qwen3_moe", "lightllm.models.qwen3_moe.model:Qwen3MOEModel")
ModelRegistry.register(
    ["qwen3_omni_moe"],
    "lightllm.models.qwen3_omni_moe_thinker.model:Qwen3OmniMOETpPartModel",
    is_multimodal=True,
)
ModelRegistry.register(["qwen3_vl"], "lightllm.models.qwen3_vl.model:Qwen3VLTpPartModel", is_multimodal=True)
ModelRegistry.register(["qwen3_vl_moe"], "lightllm.models.qwen3_vl_moe.model:Qwen3VLMOETpPartModel", is_multimodal=True)
ModelRegistry.register("qwen3_next", "lightllm.models.qwen3next.model:Qwen3NextTpPartModel")
ModelRegistry.register(
    "qwen",
    "lightllm.models.qwen_vl.model:QWenVLTpPartModel",
    is_multimodal=True,
    condition=lambda cfg: "visual" in cfg,
)
ModelRegistry.register("stablelm", "lightllm.models.stablelm.model:StablelmTpPartModel")
ModelRegistry.register("gpt_bigcode", "lightllm.models.starcoder.model:StarcoderTpPartModel")
ModelRegistry.register("starcoder2", "lightllm.models.starcoder2.model:Starcoder2TpPartModel")
ModelRegistry.register(
    "llava",
    "lightllm.models.tarsier2.model:Tarsier2Qwen2TpPartModel",
    is_multimodal=True,
    condition=_tarsier_text_model_is("qwen2"),
)
ModelRegistry.register(
    "llava",
    "lightllm.models.tarsier2.model:Tarsier2Qwen2VLTpPartModel",
    is_multimodal=True,
    condition=_tarsier_text_model_is("qwen2_vl"),
)
ModelRegistry.register(
    "llava",
    "lightllm.models.tarsier2.model:Tarsier2LlamaTpPartModel",
    is_multimodal=True,
    condition=_tarsier_text_model_is("llama"),
)


# Draft keys use the draft checkpoint model_type and the speculative mode.
DraftModelRegistry.register(
    "deepseek_v3",
    ("vanilla_with_att", "eagle_with_att"),
    "lightllm.models.deepseek_mtp.model:Deepseek3MTPModel",
)
DraftModelRegistry.register(
    "glm4_moe_lite",
    ("vanilla_with_att", "eagle_with_att"),
    "lightllm.models.glm4_moe_lite_mtp.model:Glm4MoeLiteMTPModel",
)
DraftModelRegistry.register(
    "mistral", ("vanilla_no_att", "eagle_no_att"), "lightllm.models.mistral_mtp.model:MistralMTPModel"
)
DraftModelRegistry.register(
    ("qwen3_5", "qwen3_5_text"), "dflash", "lightllm.models.qwen3_5_dflash.model:Qwen3_5DFlashModel"
)
DraftModelRegistry.register(
    ("qwen3_5", "qwen3_5_text"), "dspark", "lightllm.models.qwen3_5_dspark.model:Qwen3_5DSparkModel"
)
DraftModelRegistry.register(
    ("qwen3_5_moe", "qwen3_5_moe_text"),
    ("vanilla_with_att", "eagle_with_att"),
    "lightllm.models.qwen3_5_moe_mtp.model:Qwen3_5MoeMTPModel",
)
DraftModelRegistry.register(
    ("qwen3_5", "qwen3_5_text"),
    ("vanilla_with_att", "eagle_with_att"),
    "lightllm.models.qwen3_5_mtp.model:Qwen3_5MTPModel",
)
DraftModelRegistry.register("qwen3", "dflash", "lightllm.models.qwen3_dflash.model:Qwen3DFlashModel")
DraftModelRegistry.register("qwen3", "dspark", "lightllm.models.qwen3_dspark.model:Qwen3DSparkModel")
DraftModelRegistry.register("qwen3", "eagle3", "lightllm.models.qwen3_eagle.model:Qwen3EagleModel")
DraftModelRegistry.register(
    "qwen3_moe", ("vanilla_no_att", "eagle_no_att"), "lightllm.models.qwen3_moe_mtp.model:Qwen3MOEMTPModel"
)


# Derive historical package-level class exports from the same registrations.
MODEL_CLASS_PATHS = {
    config.model_class.rsplit(":", 1)[1]: config.model_class
    for configs in ModelRegistry._registry.values()
    for config in configs
}
MODEL_CLASS_PATHS.update({path.rsplit(":", 1)[1]: path for path in DraftModelRegistry._registry.values()})
