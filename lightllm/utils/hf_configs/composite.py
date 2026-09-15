"""Compatibility for existing LightLLM checkpoint layouts absent from HF AutoConfig."""

from transformers import LlamaConfig, PretrainedConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from lightllm.utils.hf_config import config_from_dict
from lightllm.utils.hf_configs.intern_vit import InternVisionConfig


class DeepseekV32Config(DeepseekV3Config):
    model_type = "deepseek_v32"


class LegacyLlavaConfig(LlamaConfig):
    # Original LLaVA checkpoints store the Llama backbone at the root.
    model_type = "llava"


class TarsierConfig(PretrainedConfig):
    # Config contract: bytedance/tarsier@63de87a760c2aec6b31470a08046969a13787126,
    # models/modeling_tarsier.py:LlavaConfig. Keep components separate from HF Llava.
    model_type = "llava"
    sub_configs = {"vision_config": PretrainedConfig, "text_config": PretrainedConfig}

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        image_newline_idx=32002,
        image_new_idx=32003,
        projection_head="MLP",
        **kwargs
    ):
        super().__init__(**kwargs)
        if isinstance(vision_config, dict):
            vision_config = {"model_type": "clip_vision_model", **vision_config}
            # Original Tarsier2 uses qwen2_vl for both text and vision Configs.
            # HF 5.8 gives the vision component its own model_type.
            if vision_config["model_type"] == "qwen2_vl":
                vision_config["model_type"] = "qwen2_vl_vision"
        if isinstance(text_config, dict):
            text_config = {"model_type": "llama", **text_config}
        self.vision_config = config_from_dict(vision_config) if vision_config is not None else None
        self.text_config = config_from_dict(text_config) if text_config is not None else None
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.image_newline_idx = image_newline_idx
        self.image_new_idx = image_new_idx
        self.projection_head = projection_head


class InternVLChatConfig(PretrainedConfig):
    # Component/parameter contract: OpenGVLab/InternVL3-8B,
    # revision 853e3a797a661694b1b8ece0cb72dc2b23e3dac9/configuration_internvl_chat.py.
    # Resolve all registered text backbones, including LightLLM's InternLM variants.
    model_type = "internvl_chat"
    sub_configs = {"vision_config": InternVisionConfig, "llm_config": PretrainedConfig}

    def __init__(
        self,
        vision_config=None,
        llm_config=None,
        text_config=None,
        use_backbone_lora=0,
        use_llm_lora=0,
        select_layer=-1,
        force_image_size=None,
        downsample_ratio=0.5,
        template=None,
        dynamic_image_size=False,
        use_thumbnail=False,
        ps_version="v1",
        min_dynamic_patch=1,
        max_dynamic_patch=6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vision_config = (
            vision_config
            if isinstance(vision_config, PretrainedConfig)
            else InternVisionConfig(**(vision_config or {}))
        )
        if llm_config is None:
            llm_config = text_config
        text = (
            dict(llm_config or {"model_type": "qwen2"}) if not isinstance(llm_config, PretrainedConfig) else llm_config
        )
        if isinstance(text, dict) and not text.get("model_type"):
            architectures = text.get("architectures") or []
            families = {
                "LlamaForCausalLM": "llama",
                "Qwen2ForCausalLM": "qwen2",
                "InternLMForCausalLM": "internlm",
                "InternLM2ForCausalLM": "internlm2",
            }
            if not architectures or architectures[0] not in families:
                raise ValueError("InternVL llm_config needs a supported model_type or architecture")
            text["model_type"] = families[architectures[0]]
        self.llm_config = config_from_dict(text)
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.tie_word_embeddings = self.llm_config.tie_word_embeddings
