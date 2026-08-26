import torch

from lightllm.common.basemodel.triton_kernel.norm.qk_norm import qk_rmsnorm_forward
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.qwen3_dflash.infer_struct import Qwen3DFlashInferStateInfo
from lightllm.models.qwen3_dflash.layer_weights.transformer_layer_weight import Qwen3DFlashTransformerLayerWeight


class Qwen3DFlashTransformerLayerInfer(LlamaTransformerLayerInfer):
    """DFlash layer inference.

    The model path is built from two explicit layer primitives:
    - commit accepted target hidden rows into draft KV
    - run one non-causal draft block over prefix KV + scratch KV
    """

    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)
        self.head_dim_ = network_config["head_dim"]
        self.partial_rotary_factor = network_config.get("partial_rotary_factor", 1.0)

    def context_forward(
        self,
        input_embdings: torch.Tensor,
        infer_state: Qwen3DFlashInferStateInfo,
        layer_weight: Qwen3DFlashTransformerLayerWeight,
    ) -> torch.Tensor:
        token_num, _ = input_embdings.shape
        cache_kv = layer_weight.kv_proj.mm(input_embdings, use_custom_tensor_mananger=False)
        qk_rmsnorm_forward(
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            layer_weight.qk_norm_weight_.k_weight,
            self.eps_,
        )
        cache_kv = cache_kv.view(token_num, self.tp_k_head_num_ + self.tp_v_head_num_, self.head_dim_)
        rotary_emb_fwd(
            cache_kv[:, : self.tp_k_head_num_, :],
            None,
            infer_state.position_cos,
            infer_state.position_sin,
            partial_rotary_factor=self.partial_rotary_factor,
        )
        self._post_cache_kv(cache_kv, infer_state, layer_weight)
        return input_embdings

    def _get_qkv(self, input, infer_state: Qwen3DFlashInferStateInfo, layer_weight: Qwen3DFlashTransformerLayerWeight):
        q = layer_weight.q_proj.mm(input, use_custom_tensor_mananger=False)
        cache_kv = layer_weight.kv_proj.mm(input, use_custom_tensor_mananger=False)

        layer_weight.qk_norm_weight_(
            q,
            cache_kv[:, : self.tp_k_head_num_ * self.head_dim_],
            eps=self.eps_,
        )
        cache_kv = cache_kv.view(
            -1,
            self.tp_k_head_num_ + self.tp_v_head_num_,
            self.head_dim_,
        )

        rotary_emb_fwd(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            partial_rotary_factor=self.partial_rotary_factor,
        )
        return q, cache_kv
