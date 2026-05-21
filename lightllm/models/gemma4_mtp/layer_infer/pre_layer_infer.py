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

    Output projection (_tpsp_allgather override): the base model writes
    `mtp_main_output_hiddens` as `pre_infer._tpsp_allgather(trunk_output)`. The
    assistant trunk runs in draft width, but the recurrent hidden must be backbone
    width - both so it lines up with the target model's hidden state at the first
    draft step and so every draft invocation sees a single fixed input width
    (cudagraph-friendly). So post_projection (draft -> backbone) is applied here,
    which is exactly the op HF runs at the end of each draft step.
    """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.draft_hidden_ = network_config["hidden_size"]
        self.backbone_hidden_ = network_config["backbone_hidden_size"]
        self.embed_scale_ = float(self.backbone_hidden_) ** 0.5
        # Set by Gemma4MTPModel._init_infer_layer (the post_projection weight lives
        # in pre_post_weight, which _tpsp_allgather does not otherwise receive).
        self._post_projection_weight_ = None

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

    def _tpsp_allgather(self, input: torch.Tensor, infer_state):
        # Called by the base model only to materialize mtp_main_output_hiddens from
        # the draft trunk output. Lift draft width -> backbone width here.
        gathered = super()._tpsp_allgather(input=input, infer_state=infer_state)
        return self._post_projection_weight_.mm(gathered)
