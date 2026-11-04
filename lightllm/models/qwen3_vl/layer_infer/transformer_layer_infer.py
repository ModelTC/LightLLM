import torch
from typing import Tuple
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.qwen3.layer_weights.transformer_layer_weight import Qwen3TransformerLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.qwen3_vl.triton_kernel.deepstack_multimodal_emb import apply_deepstack_features
from lightllm.models.qwen2_vl.layer_infer.transformer_layer_infer import Qwen2VLTransformerLayerInfer


class Qwen3VLTransformerLayerInfer(Qwen2VLTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.head_dim_ = network_config["head_dim"]
        self.mrope_section = torch.tensor(
            network_config["rope_scaling"]["mrope_section"], dtype=torch.int32, device="cuda"
        )

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3TransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input)
        layer_weight.qk_norm_weight_(
            q,
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            eps=self.eps_,
        )
        cache_kv = cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        mrope_triton_fused(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            self.mrope_section,
            is_interleaved=True,
        )
        if infer_state.need_dp_prefill_balance:
            q = infer_state._all_to_all_unbalance_get(data=q)
            cache_kv = infer_state._all_to_all_unbalance_get(data=cache_kv)
        return q, cache_kv

    def context_forward(self, input_embdings, infer_state: Qwen3VLInferStateInfo, layer_weight):
        input1 = self._att_norm(input_embdings, infer_state, layer_weight)
        q, cache_kv = self._get_qkv(input1, infer_state, layer_weight)
        input1 = None
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        o = self._context_attention_wrapper_run(q, cache_kv, infer_state, layer_weight)
        q = None
        o = self._get_o(o, infer_state, layer_weight)
        input_embdings.add_(o.view(-1, self.embed_dim_))
        o = None

        input1 = self._ffn_norm(input_embdings, infer_state, layer_weight)
        ffn_out = self._ffn(input1, infer_state, layer_weight)
        input1 = None
        input_embdings.add_(ffn_out.view(-1, self.embed_dim_))

        input_embdings = self._tpsp_allgather(input=input_embdings, infer_state=infer_state)
        self._apply_deepstack_features_wrapper_run(
            input_embeddings=input_embdings,
            infer_state=infer_state,
            layer_num=self.layer_num_,
        )
        input_embdings = self._tpsp_sp_split(input=input_embdings, infer_state=infer_state)

        return input_embdings

    def _apply_deepstack_features_wrapper_run(self, input_embeddings, infer_state, layer_num):
        def apply(tensors, batch):
            apply_deepstack_features(input_embeddings=tensors[0], infer_state=batch, layer_num=layer_num)

        program = infer_state.prefill_program
        if program is not None and program.is_capturing:
            program.defer(apply, (input_embeddings,))
        else:
            apply((input_embeddings,), infer_state)
