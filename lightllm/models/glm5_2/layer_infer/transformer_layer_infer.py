import torch

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.models.deepseek2.layer_infer.transformer_layer_infer import Deepseek2TransformerLayerInfer
from lightllm.models.deepseek3_2.layer_infer.transformer_layer_infer import Deepseek3_2TransformerLayerInfer, NsaInfer
from lightllm.models.glm5_2.indexshare import owns_indexer_layer


class Glm5_2TransformerLayerInfer(Deepseek3_2TransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        self.has_indexer = owns_indexer_layer(layer_num, network_config)
        Deepseek2TransformerLayerInfer.__init__(self, layer_num, network_config)
        self.indexer = (
            NsaInfer(layer_idx=self.layer_num_, network_config=self.network_config_, tp_world_size=self.tp_world_size_)
            if self.has_indexer
            else None
        )

    def _get_or_reuse_topk_indices(self, infer_state, att_state, layer_weight):
        if getattr(infer_state, "glm5_2_reuse_mtp_topk_indices", False):
            model_input = getattr(infer_state, "glm5_2_model_input", None)
            cached_topk = getattr(model_input, "glm5_2_mtp_topk_cache", None)
            if cached_topk is not None:
                infer_state.glm5_2_indexshare_topk_cache = cached_topk
                return cached_topk

        if self.indexer is not None:
            topk_mem_indices, topk_indices = self.indexer._get_indices(
                hidden_states=infer_state.get_topk_indices_params["hidden_states"],
                q_lora=infer_state.get_topk_indices_params["q_lora"],
                infer_state=infer_state,
                att_state=att_state,
                layer_weight=layer_weight,
            )
            infer_state.glm5_2_indexshare_topk_cache = (topk_mem_indices, topk_indices)
            if getattr(infer_state, "glm5_2_reuse_mtp_topk_indices", False):
                model_input = getattr(infer_state, "glm5_2_model_input", None)
                if model_input is not None:
                    model_input.glm5_2_mtp_topk_cache = infer_state.glm5_2_indexshare_topk_cache
            return topk_mem_indices, topk_indices

        if not hasattr(infer_state, "glm5_2_indexshare_topk_cache"):
            raise RuntimeError(
                f"GLM-5.2 layer {self.layer_num_} needs cached IndexShare top-k indices, "
                "but no previous indexer layer has produced them."
            )
        return infer_state.glm5_2_indexshare_topk_cache

    def _context_attention_kernel(self, q, kv, infer_state, layer_weight, out=None):
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)
        q_all = q_nope if self.qk_rope_head_dim == 0 else torch.cat([q_nope, q_rope], dim=-1)

        att_state = infer_state.prefill_att_state
        topk_mem_indices, topk_indices = self._get_or_reuse_topk_indices(infer_state, att_state, layer_weight)
        del infer_state.get_topk_indices_params

        att_control = AttControl(
            nsa_prefill=True,
            nsa_prefill_dict={
                "topk_mem_indices": topk_mem_indices,
                "topk_indices": topk_indices,
                "prefill_cache_kv": kv,
                "softmax_scale": self.softmax_scale,
                "kv_lora_rank": self.kv_lora_rank,
            },
        )

        return infer_state.prefill_att_state.prefill_att(
            q=q_all,
            k=infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_),
            v=None,
            att_control=att_control,
        )

    def _token_attention_kernel(self, q, infer_state, layer_weight, out=None):
        q_nope, q_rope = q[:, :, : -self.qk_rope_head_dim], q[:, :, -self.qk_rope_head_dim :]
        q_nope = layer_weight.k_b_proj_.bmm(q_nope.transpose(0, 1)).transpose(0, 1)

        att_state = infer_state.decode_att_state
        topk_mem_indices, topk_indices = self._get_or_reuse_topk_indices(infer_state, att_state, layer_weight)
        del infer_state.get_topk_indices_params

        att_control = AttControl(
            nsa_decode=True,
            nsa_decode_dict={
                "layer_index": self.layer_num_,
                "topk_mem_indices": topk_mem_indices,
                "topk_indices": topk_indices,
                "softmax_scale": self.softmax_scale,
                "kv_lora_rank": self.kv_lora_rank,
                "qk_rope_head_dim": self.qk_rope_head_dim,
            },
        )

        return infer_state.decode_att_state.decode_att(
            q=(q_nope, q_rope),
            k=infer_state.mem_manager.get_att_input_params(layer_index=self.layer_num_),
            v=None,
            att_control=att_control,
        )
