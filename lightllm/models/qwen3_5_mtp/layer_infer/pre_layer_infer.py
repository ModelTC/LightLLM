import torch

from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
from lightllm.models.qwen3_5_mtp.layer_weights.pre_and_post_layer_weight import Qwen3_5MTPPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo


class Qwen3_5MTPPreLayerInfer(Qwen3VLMultimodalPreLayerInfer):
    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = network_config["rms_norm_eps"]
        self.hidden_size = network_config["hidden_size"]
        return

    def _mtp_fuse(
        self,
        input_embdings: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3_5MTPPreAndPostLayerWeight,
    ) -> torch.Tensor:
        tgt_embdings = infer_state.mtp_draft_input_hiddens
        assert (
            input_embdings.shape[0] == tgt_embdings.shape[0]
        ), f"shape {input_embdings.shape} != shape {tgt_embdings.shape}"

        layer_weight.enorm_weight_(input=input_embdings, eps=self.eps_, out=input_embdings)
        layer_weight.hnorm_weight_(input=tgt_embdings, eps=self.eps_, out=tgt_embdings)
        cat_embdings = torch.cat((input_embdings, tgt_embdings), dim=-1)

        return layer_weight.eh_proj_weight_.mm(cat_embdings)

    def context_forward(
        self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: Qwen3_5MTPPreAndPostLayerWeight
    ):
        input_embdings = super().context_forward(input_ids, infer_state, layer_weight)
        return self._mtp_fuse(input_embdings, infer_state, layer_weight)

    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: Qwen3_5MTPPreAndPostLayerWeight):
        input_embdings = super().token_forward(input_ids, infer_state, layer_weight)
        return self._mtp_fuse(input_embdings, infer_state, layer_weight)
