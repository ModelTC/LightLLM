import torch
from lightllm.models.starcoder2.layer_weights.transformer_layer_weight import Starcoder2TransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Starcoder2TransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config):
        super().__init__(layer_num, network_config)

    def _att_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Starcoder2TransformerLayerWeight
    ) -> torch.Tensor:
        return layer_weight.att_norm_weight_.layernorm_forward(
            input=input.view(-1, self.embed_dim_),
            eps=self.eps_,
            alloc_func=self.alloc_tensor,
        )

    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Starcoder2TransformerLayerWeight
    ) -> torch.Tensor:
        return layer_weight.ffn_norm_weight_.layernorm_forward(
            input=input.view(-1, self.embed_dim_),
            eps=self.eps_,
            alloc_func=self.alloc_tensor,
        )

    def _ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Starcoder2TransformerLayerWeight
    ) -> torch.Tensor:
        ffn1_out = layer_weight.up_proj.mm(input.view(-1, self.embed_dim_))
        input = None
        gelu_out = torch.nn.functional.gelu(ffn1_out, approximate="tanh")
        ffn1_out = None
        ffn2_out = layer_weight.down_proj.mm(gelu_out)
        gelu_out = None
        return ffn2_out
