import os
from functools import partial
from torch.distributed import ReduceOp

from lightllm.models.qwen3next_mtp.layer_weights.transformer_layer_weight import Qwen3NextMTPTransformerLayerWeight
from lightllm.models.qwen3next.layer_infer.transformer_layer_infer import (
    Qwen3NextFullAttentionTransformerLayerInfer,
    Qwen3NextFFNMixin,
)
from lightllm.models.qwen3next.infer_struct import Qwen3NextInferStateInfo
from lightllm.distributed import all_reduce
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)

# Module-level constant for MoE mode
MOE_MODE = os.environ.get("MOE_MODE", "TP")


class Qwen3NextMTPTransformerLayerInfer(Qwen3NextFullAttentionTransformerLayerInfer):
    """
    Qwen3Next MTP Transformer Layer Inference.
    MTP layers use full attention (not linear attention) with MoE FFN and shared expert.
    Inherits shared methods from Qwen3NextFullAttentionTransformerLayerInfer.
    """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.tp_k_head_num_ = max(self.tp_k_head_num_, 1)
        self.tp_v_head_num_ = max(self.tp_v_head_num_, 1)

    def _bind_ffn(self):
        """MTP always uses shared expert + MoE."""
        if MOE_MODE == "EP":
            self._ffn = partial(Qwen3NextFFNMixin._ffn_with_shared_expert_ep, self)
        else:
            self._ffn = partial(Qwen3NextFFNMixin._ffn_with_shared_expert_tp, self)

    def context_forward(
        self, input_embdings, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextMTPTransformerLayerWeight
    ):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_kernel(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings

    def token_forward(
        self, input_embdings, infer_state: Qwen3NextInferStateInfo, layer_weight: Qwen3NextMTPTransformerLayerWeight
    ):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._token_attention_kernel(q, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        if self.tp_world_size_ > 1:
            all_reduce(o, op=ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        if self.tp_world_size_ > 1:
            all_reduce(ffn_out, op=ReduceOp.SUM, group=infer_state.dist_group, async_op=False)
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))
        return input_embdings
