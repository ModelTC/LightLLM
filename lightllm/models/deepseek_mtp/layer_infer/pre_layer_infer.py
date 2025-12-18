import torch

from lightllm.models.deepseek_mtp.layer_weights.pre_and_post_layer_weight import Deepseek3MTPPreAndPostLayerWeight
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward


class Deepseek3MTPPreLayerInfer(LlamaPreLayerInfer):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        self.eps_ = network_config["rms_norm_eps"]
        self.hidden_size = network_config["hidden_size"]
        return

    def _mtp_context_forward(
        self, input_embdings, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek3MTPPreAndPostLayerWeight
    ):
        tgt_embdings = infer_state.deepseekv3_mtp_draft_input_hiddens
        assert (
            input_embdings.shape[0] == tgt_embdings.shape[0]
        ), f"shape {input_embdings.shape} != shape {tgt_embdings.shape}"
        rmsnorm_forward(input_embdings, weight=layer_weight.enorm_weight_, eps=self.eps_, out=input_embdings)
        rmsnorm_forward(tgt_embdings, weight=layer_weight.hnorm_weight_, eps=self.eps_, out=tgt_embdings)

        cat_embdings = torch.cat((input_embdings, tgt_embdings), dim=-1)

        ans_logics = self.alloc_tensor(
            (cat_embdings.shape[0], layer_weight.eh_proj_weight_.shape[1]), dtype=input_embdings.dtype
        )
        torch.mm(cat_embdings, layer_weight.eh_proj_weight_, out=ans_logics)
        return ans_logics

    def _mtp_token_forward(
        self, input_embdings, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek3MTPPreAndPostLayerWeight
    ):
        tgt_embdings = infer_state.deepseekv3_mtp_draft_input_hiddens
        assert input_embdings.shape[0] == tgt_embdings.shape[0]
        rmsnorm_forward(input_embdings, weight=layer_weight.enorm_weight_, eps=self.eps_, out=input_embdings)
        rmsnorm_forward(tgt_embdings, weight=layer_weight.hnorm_weight_, eps=self.eps_, out=tgt_embdings)

        cat_embdings = torch.cat((input_embdings, tgt_embdings), dim=-1)

        ans_logics = self.alloc_tensor(
            (cat_embdings.shape[0], layer_weight.eh_proj_weight_.shape[1]), dtype=input_embdings.dtype
        )
        torch.mm(cat_embdings, layer_weight.eh_proj_weight_, out=ans_logics)
        return ans_logics

    def context_forward(
        self, input_ids, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek3MTPPreAndPostLayerWeight
    ):
        input_embdings = super().context_forward(input_ids, infer_state, layer_weight)
        return self._mtp_context_forward(input_embdings, infer_state, layer_weight)

    def token_forward(
        self, input_ids, infer_state: Deepseek2InferStateInfo, layer_weight: Deepseek3MTPPreAndPostLayerWeight
    ):
        input_embdings = super().token_forward(input_ids, infer_state, layer_weight)
        return self._mtp_token_forward(input_embdings, infer_state, layer_weight)
