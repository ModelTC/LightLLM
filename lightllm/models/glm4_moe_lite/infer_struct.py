from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo


class Glm4MoeLiteInferStateInfo(Deepseek2InferStateInfo):
    """Inference state for GLM-4.7-Flash (glm4_moe_lite architecture).

    Inherits from Deepseek2InferStateInfo as GLM-4.7-Flash uses the same
    MLA (Multi-Head Latent Attention) mechanism as DeepSeek-V2/V3.
    """

    def __init__(self):
        super().__init__()
