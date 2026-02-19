from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import Qwen3NextFullAttentionBaseLayerInfer
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class Qwen3NextMTPTransformerLayerInfer(Qwen3NextFullAttentionBaseLayerInfer):
    """
    Qwen3Next MTP Transformer Layer Inference.
    MTP layers use full attention (not linear attention) with MoE FFN and shared expert.
    Inherits shared methods from Qwen3NextFullAttentionBaseLayerInfer.
    """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.tp_k_head_num_ = max(self.tp_k_head_num_, 1)
        self.tp_v_head_num_ = max(self.tp_v_head_num_, 1)
        return

    def _bind_ffn(self):
        """MTP always uses shared expert + MoE"""
        from functools import partial
        import os

        moe_mode = os.environ.get("MOE_MODE", "TP")
        if moe_mode == "EP":
            self._ffn = partial(Qwen3NextFullAttentionBaseLayerInfer._ffn_with_shared_expert_ep, self)
        else:
            self._ffn = partial(Qwen3NextFullAttentionBaseLayerInfer._ffn_with_shared_expert_tp, self)
        return
