from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer


class Glm5NextMTPPreLayerInfer(Deepseek3MTPPreLayerInfer, LlamaMultimodalPreLayerInfer):
    """Resolve shifted image tokens before the standard NextN embedding/hidden fusion."""
