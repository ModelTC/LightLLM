import torch
from typing import Tuple
from lightllm.models.qwen2_vl.triton_kernel.mrope import mrope_triton_fused
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer


class Qwen2VLTransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        mrope_section = network_config["rope_scaling"]["mrope_section"]
        self.mrope_section = torch.tensor(mrope_section, dtype=torch.int32, device="cuda")

    def _get_qkv(self, input, infer_state, layer_weight):
        q = layer_weight.q_proj.mm(input)
        cache_kv = layer_weight.kv_proj.mm(input).view(-1, (self.tp_k_head_num_ + self.tp_v_head_num_), self.head_dim_)
        mrope_triton_fused(
            q.view(-1, self.tp_q_head_num_, self.head_dim_),
            cache_kv[:, : self.tp_k_head_num_, :],
            infer_state.position_cos,
            infer_state.position_sin,
            self.mrope_section,
            is_interleaved=False,
        )
        return q, cache_kv

    def _tpsp_get_qkv(self, input, infer_state, layer_weight) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO
        raise Exception("not impl")
