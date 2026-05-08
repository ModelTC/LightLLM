from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.phi3.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.phi3.layer_weights.transformer_layer_weight import Phi3TransformerLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Phi3TransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        return

    def _get_qkv(self, input_emb, infer_state: LlamaInferStateInfo, layer_weight: Phi3TransformerLayerWeight):
        input_emb = self._tpsp_allgather(input=input_emb.view(-1, self.embed_dim_), infer_state=infer_state)
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
        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)
        return q, cache_kv
