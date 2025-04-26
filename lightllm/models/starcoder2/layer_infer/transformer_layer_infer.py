import torch
from lightllm.models.bloom.triton_kernel.layernorm import layernorm_forward
from lightllm.models.starcoder2.layer_weights.transformer_layer_weight import Starcoder2TransformerLayerWeight
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Starcoder2TransformerLayerInfer(LlamaTransformerLayerInfer):
    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)

    def _att_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Starcoder2TransformerLayerWeight
    ) -> torch.Tensor:
        return layernorm_forward(
            input.view(-1, self.embed_dim_),
            weight=layer_weight.att_norm_weight_.weight,
            bias=layer_weight.att_norm_weight_.bias,
            eps=self.eps_,
        )

    def _ffn_norm(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: Starcoder2TransformerLayerWeight
    ) -> torch.Tensor:
        return layernorm_forward(
            input.view(-1, self.embed_dim_),
            weight=layer_weight.ffn_norm_weight_.weight,
            bias=layer_weight.ffn_norm_weight_.bias,
            eps=self.eps_,
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
