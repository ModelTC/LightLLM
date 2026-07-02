import torch
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen.layer_weights.transformer_layer_weight import QwenTransformerLayerWeight
from lightllm.models.qwen.infer_struct import QwenInferStateInfo


class QwenTransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        return

    def _get_qkv(self, input_emb, infer_state: QwenInferStateInfo, layer_weight: QwenTransformerLayerWeight):
        input_emb = self._tpsp_allgather(input_emb, infer_state)
        q = layer_weight.q_proj.mm(input_emb)
        cache_kv = layer_weight.kv_proj.mm(input_emb).view(
            -1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_
        )

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
        )
        if infer_state.logn_values is not None:
            q.mul_(infer_state.logn_values.view(-1, 1))
        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)
        return q, cache_kv
