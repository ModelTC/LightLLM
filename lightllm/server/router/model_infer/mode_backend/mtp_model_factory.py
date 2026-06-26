from lightllm.models.deepseek_mtp.model import Deepseek3MTPModel
from lightllm.models.qwen3_moe_mtp.model import Qwen3MOEMTPModel
from lightllm.models.mistral_mtp.model import MistralMTPModel
from lightllm.models.glm4_moe_lite_mtp.model import Glm4MoeLiteMTPModel


def create_mtp_draft_model(model_type: str, mtp_mode: str, mtp_model_kvargs: dict):
    """Single source of truth for (model_type, mtp_mode) -> MTP draft model (#10).
    Shared by base_backend and the static MTP benchmark."""
    if model_type == "deepseek_v3":
        assert mtp_mode in ["vanilla_with_att", "eagle_with_att"]
        return Deepseek3MTPModel(mtp_model_kvargs)
    elif model_type == "qwen3_moe":
        assert mtp_mode in ["vanilla_no_att", "eagle_no_att"]
        return Qwen3MOEMTPModel(mtp_model_kvargs)
    elif model_type == "mistral":
        assert mtp_mode in ["vanilla_no_att", "eagle_no_att"]
        return MistralMTPModel(mtp_model_kvargs)
    elif model_type == "glm4_moe_lite":
        assert mtp_mode in ["vanilla_with_att", "eagle_with_att"]
        return Glm4MoeLiteMTPModel(mtp_model_kvargs)
    elif model_type in ("qwen3_5", "qwen3_5_text"):
        assert mtp_mode in ["vanilla_with_att", "eagle_with_att"]
        from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel

        return Qwen3_5MTPModel(mtp_model_kvargs)
    elif model_type in ("qwen3_5_moe", "qwen3_5_moe_text"):
        assert mtp_mode in ["vanilla_with_att", "eagle_with_att"]
        from lightllm.models.qwen3_5_moe_mtp.model import Qwen3_5MoeMTPModel

        return Qwen3_5MoeMTPModel(mtp_model_kvargs)
    else:
        raise ValueError(f"Unsupported MTP model type: {model_type}")
