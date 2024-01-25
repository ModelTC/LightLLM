import torch
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.triton_kernel.rotary_emb import rotary_emb_fwd
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_weights.transformer_layer_weight import LlamaTransformerLayerWeight


class Baichuan2_7bTransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, tp_rank, world_size, network_config, mode=[]):
        super().__init__(layer_num, tp_rank, world_size, network_config, mode)
        return

    def _get_qkv(
        self, input, cache_kv: torch.Tensor, infer_state: LlamaInferStateInfo, layer_weight: LlamaTransformerLayerWeight
    ) -> torch.Tensor:
        q = torch.mm(input.view(-1, self.embed_dim_), layer_weight.q_weight_).view(
            -1, self.tp_q_head_num_, self.head_dim_
        )
        torch.mm(
            input.view(-1, self.embed_dim_),
            layer_weight.kv_weight_,
            out=cache_kv.view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_) * self.head_dim_),
        )
        q_ = q.float()
        cache_k_ = cache_kv[:, 0 : self.tp_k_head_num_, :].float()
        rotary_emb_fwd(q_, cache_k_, infer_state.position_cos, infer_state.position_sin)
        cache_kv[:, 0 : self.tp_k_head_num_, :].copy_(cache_k_)
        q.copy_(q_)
        return q, cache_kv
