import torch

from lightllm.models.deepseek_v4.infer_struct import DeepseekV4InferStateInfo
from lightllm.models.deepseek_v4_mtp.layer_weights.pre_and_post_layer_weight import (
    DeepseekV4MTPPreAndPostLayerWeight,
)
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer


class DeepseekV4MTPPreLayerInfer(LlamaPreLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = network_config["rms_norm_eps"]
        self.hidden_size = network_config["hidden_size"]
        self.hc_mult = network_config["hc_mult"]
        return

    def _mtp_forward(
        self,
        input_embdings: torch.Tensor,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4MTPPreAndPostLayerWeight,
    ):
        input_embdings = input_embdings.masked_fill(infer_state.position_ids.eq(0).view(-1, 1), 0)
        target = infer_state.mtp_draft_input_hiddens

        layer_weight.enorm_weight_(input=input_embdings, eps=self.eps_, out=input_embdings)
        e_proj = layer_weight.e_proj_weight_.mm(input_embdings)

        target = target.view(-1, self.hc_mult, self.hidden_size).contiguous()
        target = layer_weight.hnorm_weight_(input=target, eps=self.eps_, alloc_func=self.alloc_tensor)
        h_proj = layer_weight.h_proj_weight_.mm(target.reshape(-1, self.hidden_size))
        h_proj = h_proj.view(-1, self.hc_mult, self.hidden_size)

        output = h_proj + e_proj.unsqueeze(1)
        return output.reshape(output.shape[0], self.hc_mult * self.hidden_size)

    def context_forward(
        self,
        input_ids,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4MTPPreAndPostLayerWeight,
    ):
        input_embdings = super().context_forward(input_ids, infer_state, layer_weight)
        return self._mtp_forward(input_embdings, infer_state, layer_weight)

    def token_forward(
        self,
        input_ids,
        infer_state: DeepseekV4InferStateInfo,
        layer_weight: DeepseekV4MTPPreAndPostLayerWeight,
    ):
        input_embdings = super().token_forward(input_ids, infer_state, layer_weight)
        return self._mtp_forward(input_embdings, infer_state, layer_weight)
