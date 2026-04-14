import torch
from typing import Tuple
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.qwen3_moe.layer_infer.transformer_layer_infer import Qwen3MOETransformerLayerInfer
from lightllm.models.qwen3_moe.layer_weights.transformer_layer_weight import Qwen3MOETransformerLayerWeight
from lightllm.models.qwen3_vl.infer_struct import Qwen3VLInferStateInfo
from lightllm.models.qwen3_vl.triton_kernel.deepstack_multimodal_emb import apply_deepstack_features
from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor


class Qwen3VLMOETransformerLayerInfer(Qwen3MOETransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.mrope_section = torch.tensor(
            network_config["rope_scaling"]["mrope_section"], dtype=torch.int32, device="cuda"
        )

    def _get_qkv(
        self,
        input: torch.Tensor,
        infer_state: Qwen3VLInferStateInfo,
        layer_weight: Qwen3MOETransformerLayerWeight,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input = input.view(-1, self.embed_dim_)
        input = self._tpsp_allgather(input=input, infer_state=infer_state)
        qkv = layer_weight.qkv_proj.mm(input)
        q, cache_kv = qkv.split(
            [self.tp_q_head_num_ * self.head_dim_, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_], dim=-1
        )
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

    def _apply_deepstack_features_wrapper_run(
        self,
        input_embeddings: torch.Tensor,
        infer_state: InferStateInfo,
        layer_num: int,
    ):
        if torch.cuda.is_current_stream_capturing():
            input_embeddings = input_embeddings.contiguous()
            _input_embeddings = tensor_to_no_ref_tensor(input_embeddings)
            pre_capture_graph = infer_state.prefill_cuda_graph_get_current_capture_graph()
            pre_capture_graph.__exit__(None, None, None)

            infer_state.prefill_cuda_graph_create_graph_obj()
            infer_state.prefill_cuda_graph_get_current_capture_graph().__enter__()

            def apply_func(new_infer_state: InferStateInfo):
                apply_deepstack_features(
                    input_embeddings=_input_embeddings,
                    infer_state=new_infer_state,
                    layer_num=layer_num,
                )
                return

            infer_state.prefill_cuda_graph_add_cpu_runnning_func(func=apply_func, after_graph=pre_capture_graph)
        else:
            apply_deepstack_features(
                input_embeddings=input_embeddings,
                infer_state=infer_state,
                layer_num=layer_num,
            )

        return
