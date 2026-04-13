import torch
from functools import partial
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.stablelm.layer_weights.transformer_layer_weight import StablelmTransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class StablelmTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.partial_rotary_factor = self.network_config_.get("partial_rotary_factor", 1)
        return

    def _bind_norm(self):
        self._att_norm = partial(StablelmTransformerLayerInfer._att_norm, self)
        self._ffn_norm = partial(StablelmTransformerLayerInfer._ffn_norm, self)
        return

    def _get_qkv(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: StablelmTransformerLayerWeight
    ) -> torch.Tensor:
        input = self._tpsp_allgather(input, infer_state)
        q = layer_weight.q_proj.mm(input.view(-1, self.embed_dim_))
        cache_kv = layer_weight.kv_proj.mm(
            input.view(-1, self.embed_dim_),
        ).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, 0 : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            self.partial_rotary_factor,
        )
        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)
        return q, cache_kv

    def _get_o(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: StablelmTransformerLayerWeight
    ) -> torch.Tensor:
        if infer_state.need_dp_prefill_balance:
            input = infer_state._all_to_all_balance_get(data=input)
        o_tensor = layer_weight.o_proj.mm(
            input.view(-1, self.tp_o_head_num_ * self.head_dim_),
        )
        o_tensor = self._tpsp_reduce(input=o_tensor, infer_state=infer_state)
        return o_tensor

    def _att_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: StablelmTransformerLayerWeight
    ) -> torch.Tensor:
        return layer_weight.att_norm_weight_(
            input=input.view(-1, self.embed_dim_), eps=self.eps_, alloc_func=self.alloc_tensor
        )

    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: StablelmTransformerLayerWeight
    ) -> torch.Tensor:
        return layer_weight.ffn_norm_weight_(
            input=input.view(-1, self.embed_dim_), eps=self.eps_, alloc_func=self.alloc_tensor
        )
