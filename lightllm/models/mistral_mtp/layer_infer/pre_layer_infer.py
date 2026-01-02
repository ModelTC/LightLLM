import torch
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.mistral_mtp.layer_weights.pre_and_post_layer_weight import MistralMTPPreAndPostLayerWeight


class MistralMTPPreLayerInfer(LlamaPreLayerInfer):
    """ """

    def __init__(self, network_config, mode):
        super().__init__(network_config, mode)
        return

    def _mtp_context_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: MistralMTPPreAndPostLayerWeight
    ):
        tgt_embdings = infer_state.mtp_draft_input_hiddens
        assert (
            input_embdings.shape[0] == tgt_embdings.shape[0]
        ), f"shape {input_embdings.shape} != shape {tgt_embdings.shape}"

        layer_weight.enorm_weight_.rmsnorm_forward(
            input=input_embdings,
            eps=self.eps_,
            out=input_embdings,
        )

        tgt_embdings = layer_weight.final_norm_weight_.rmsnorm_forward(
            input=tgt_embdings,
            eps=self.eps_,
            alloc_func=self.alloc_tensor,
        )
        layer_weight.hnorm_weight_.rmsnorm_forward(
            input=tgt_embdings,
            eps=self.eps_,
            out=tgt_embdings,
        )

        cat_embdings = torch.cat((input_embdings, tgt_embdings), dim=-1)

        ans_logics = layer_weight.eh_proj_weight_.mm(cat_embdings)
        return ans_logics

    def _mtp_token_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: MistralMTPPreAndPostLayerWeight
    ):
        tgt_embdings = infer_state.mtp_draft_input_hiddens
        assert input_embdings.shape[0] == tgt_embdings.shape[0]

        layer_weight.enorm_weight_.rmsnorm_forward(
            input=input_embdings,
            eps=self.eps_,
            out=input_embdings,
        )

        tgt_embdings = layer_weight.final_norm_weight_.rmsnorm_forward(
            input=tgt_embdings,
            eps=self.eps_,
            alloc_func=self.alloc_tensor,
        )
        layer_weight.hnorm_weight_.rmsnorm_forward(
            input=tgt_embdings,
            eps=self.eps_,
            out=tgt_embdings,
        )

        cat_embdings = torch.cat((input_embdings, tgt_embdings), dim=-1)

        ans_logics = layer_weight.eh_proj_weight_.mm(cat_embdings)
        return ans_logics

    def context_forward(
        self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: MistralMTPPreAndPostLayerWeight
    ):
        input_embdings = super().context_forward(input_ids, infer_state, layer_weight)
        return self._mtp_context_forward(input_embdings, infer_state, layer_weight)

    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: MistralMTPPreAndPostLayerWeight):
        input_embdings = super().token_forward(input_ids, infer_state, layer_weight)
        return self._mtp_token_forward(input_embdings, infer_state, layer_weight)
