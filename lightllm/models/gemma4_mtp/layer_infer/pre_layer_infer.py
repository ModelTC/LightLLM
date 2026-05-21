import torch
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.gemma4_mtp.layer_weights.pre_and_post_layer_weight import Gemma4MTPPreAndPostLayerWeight


class Gemma4MTPPreLayerInfer(LlamaPreLayerInfer):
    """
    Gemma-4 assistant input/output glue.

    Input fusion (context_forward / token_forward):
        embed = target_wte(input_ids) * sqrt(backbone_hidden)   # backbone width
        prev  = mtp_draft_input_hiddens                         # backbone width
        fused = pre_projection(concat[embed, prev])             # -> draft width

    Output projection is handled by Gemma4MTPPostLayerInfer, which returns
    post_projection(norm(draft_hidden)) as mtp_main_output_hiddens.
    """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.draft_hidden_ = network_config["hidden_size"]
        self.backbone_hidden_ = network_config["backbone_hidden_size"]
        self.embed_scale_ = float(self.backbone_hidden_) ** 0.5

    def _mtp_fuse(self, input_embdings, infer_state, layer_weight: Gemma4MTPPreAndPostLayerWeight):
        prev = infer_state.mtp_draft_input_hiddens
        assert (
            input_embdings.shape[0] == prev.shape[0]
        ), f"token count mismatch: embed {input_embdings.shape} vs prev_hidden {prev.shape}"
        # The target embedding is backbone width; scale like the target model does.
        input_embdings = input_embdings * self.embed_scale_
        cat = torch.cat((input_embdings, prev), dim=-1)
        return layer_weight.pre_projection_weight_.mm(cat)

    def context_forward(self, input_ids, infer_state, layer_weight):
        input_embdings = super().context_forward(input_ids, infer_state, layer_weight)
        return self._mtp_fuse(input_embdings, infer_state, layer_weight)

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embdings = super().token_forward(input_ids, infer_state, layer_weight)
        return self._mtp_fuse(input_embdings, infer_state, layer_weight)
