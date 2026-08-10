from lightllm.models.mixtral.model import MixtralTpPartModel
from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.starcoder.model import StarcoderTpPartModel
from lightllm.models.starcoder2.model import Starcoder2TpPartModel
from lightllm.models.qwen.model import QWenTpPartModel
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.models.qwen3_moe.model import Qwen3MOEModel
from lightllm.models.qwen3next.model import Qwen3NextTpPartModel
from lightllm.models.internlm.model import InternlmTpPartModel
from lightllm.models.stablelm.model import StablelmTpPartModel
from lightllm.models.internlm2.model import Internlm2TpPartModel
from lightllm.models.internlm2_reward.model import Internlm2RewardTpPartModel
from lightllm.models.mistral.model import MistralTpPartModel
from lightllm.models.minicpm.model import MiniCPMTpPartModel
from lightllm.models.llava.model import LlavaTpPartModel
from lightllm.models.qwen_vl.model import QWenVLTpPartModel
from lightllm.models.gemma_2b.model import Gemma_2bTpPartModel
from lightllm.models.phi3.model import Phi3TpPartModel
from lightllm.models.deepseek2.model import Deepseek2TpPartModel
from lightllm.models.deepseek3_2.model import Deepseek3_2TpPartModel
from lightllm.models.glm4_moe_lite.model import Glm4MoeLiteTpPartModel
from lightllm.models.internvl.model import (
    InternVLLlamaTpPartModel,
    InternVLPhi3TpPartModel,
    InternVLQwen2TpPartModel,
    InternVLDeepSeek2TpPartModel,
)
from lightllm.models.internvl.model import InternVLInternlm2TpPartModel
from lightllm.models.qwen2_vl.model import Qwen2VLTpPartModel
from lightllm.models.qwen2_reward.model import Qwen2RewardTpPartModel
from lightllm.models.qwen3_vl.model import Qwen3VLTpPartModel
from lightllm.models.qwen3_vl_moe.model import Qwen3VLMOETpPartModel
from lightllm.models.gemma3.model import Gemma3TpPartModel
from lightllm.models.gemma4.model import Gemma4TpPartModel
from lightllm.models.tarsier2.model import (
    Tarsier2Qwen2TpPartModel,
    Tarsier2Qwen2VLTpPartModel,
    Tarsier2LlamaTpPartModel,
)
from lightllm.models.gpt_oss.model import GptOssTpPartModel
from lightllm.models.qwen3_omni_moe_thinker.model import Qwen3OmniMOETpPartModel
from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
from lightllm.models.qwen3_5_moe.model import Qwen3_5MOETpPartModel
from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.models.glm4_moe_lite_mtp.model import Glm4MoeLiteMTPModel
from lightllm.models.mistral_mtp.model import MistralMTPModel
from lightllm.models.qwen3_5_dflash.model import Qwen3_5DFlashModel
from lightllm.models.qwen3_5_dspark.model import Qwen3_5DSparkModel
from lightllm.models.qwen3_5_moe_mtp.model import Qwen3_5MoeMTPModel
from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_dspark.model import Qwen3DSparkModel
from lightllm.models.qwen3_eagle.model import Qwen3EagleModel
from lightllm.models.qwen3_moe_mtp.model import Qwen3MOEMTPModel
from .registry import get_model, get_model_class


_ATTENTION_DRAFT_MODELS = {
    "deepseek_v3": Deepseek3MTPModel,
    "glm4_moe_lite": Glm4MoeLiteMTPModel,
    "qwen3_5": Qwen3_5MTPModel,
    "qwen3_5_text": Qwen3_5MTPModel,
    "qwen3_5_moe": Qwen3_5MoeMTPModel,
    "qwen3_5_moe_text": Qwen3_5MoeMTPModel,
}

_NO_ATTENTION_DRAFT_MODELS = {
    "mistral": MistralMTPModel,
    "qwen3_moe": Qwen3MOEMTPModel,
}


def get_draft_model_class(model_cfg, spec_mode, is_linear_att_mixed_model=None):
    architectures = set(model_cfg.get("architectures", ()))

    if spec_mode == "eagle3" and "Qwen3Eagle3Model" in architectures:
        return Qwen3EagleModel

    if spec_mode == "dflash":
        if "Qwen3_5DFlashModel" in architectures:
            if is_linear_att_mixed_model is False:
                raise ValueError("Qwen3_5DFlashModel requires a linear-attention mixed target")
            return Qwen3_5DFlashModel
        if architectures.intersection(("Qwen3DFlashModel", "Qwen3DSparkModel")):
            if is_linear_att_mixed_model is True:
                raise ValueError("linear-attention mixed targets require a Qwen3_5DFlashModel checkpoint")
            return Qwen3DFlashModel

    if spec_mode == "dspark" and "Qwen3DSparkModel" in architectures:
        if is_linear_att_mixed_model:
            return Qwen3_5DSparkModel
        return Qwen3DSparkModel

    model_type = model_cfg.get("model_type", "")
    if model_type in _ATTENTION_DRAFT_MODELS:
        if spec_mode not in ("vanilla_with_att", "eagle_with_att", "eagle3", "dspark", "dflash"):
            raise ValueError(f"{model_type} requires an attention draft mode, got {spec_mode}")
        return _ATTENTION_DRAFT_MODELS[model_type]

    if model_type in _NO_ATTENTION_DRAFT_MODELS:
        if spec_mode not in ("vanilla_no_att", "eagle_no_att", "qwen3next_vanilla", "qwen3next_eagle"):
            raise ValueError(f"{model_type} requires a no-attention draft mode, got {spec_mode}")
        return _NO_ATTENTION_DRAFT_MODELS[model_type]

    raise ValueError(
        f"Unsupported speculative draft model: mode={spec_mode}, "
        f"model_type={model_cfg.get('model_type')}, architectures={sorted(architectures)}"
    )
