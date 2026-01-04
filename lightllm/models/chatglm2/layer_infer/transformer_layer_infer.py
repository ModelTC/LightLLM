import torch
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.transformer_layer_infer import LlamaTransformerLayerInfer
from lightllm.models.chatglm2.layer_weights.transformer_layer_weight import ChatGLM2TransformerLayerWeight


class ChatGLM2TransformerLayerInfer(LlamaTransformerLayerInfer):
    """ """

    def __init__(self, layer_num, network_config, mode=[]):
        super().__init__(layer_num, network_config, mode)
        return

    def swiglu(self, x):
        x = torch.chunk(x, 2, dim=-1)
        return torch.nn.functional.silu(x[0]) * x[1]

    def _ffn(
        self, input, infer_state: LlamaInferStateInfo, layer_weight: ChatGLM2TransformerLayerWeight
    ) -> torch.Tensor:
        input = input.view(-1, self.embed_dim_)
        up_gate_out = layer_weight.gate_up_proj.mm(input)
        input = None
        ffn1_out = self.swiglu(up_gate_out)
        up_gate_out = None
        ffn2_out = layer_weight.down_proj.mm(ffn1_out)
        ffn1_out = None
        return ffn2_out
