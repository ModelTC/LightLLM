import torch

from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import LlamaMultimodalPreLayerInfer


class Glm5NextMTPPreLayerInfer(LlamaMultimodalPreLayerInfer):
    """Resolve shifted image tokens before the standard NextN embedding/hidden fusion."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = network_config["rms_norm_eps"]

    def _fuse_hidden(self, input_embeddings, infer_state, layer_weight):
        previous_hidden = infer_state.mtp_draft_input_hiddens
        assert input_embeddings.shape[0] == previous_hidden.shape[0]
        layer_weight.enorm_weight_(input=input_embeddings, eps=self.eps_, out=input_embeddings)
        layer_weight.hnorm_weight_(input=previous_hidden, eps=self.eps_, out=previous_hidden)
        return layer_weight.eh_proj_weight_.mm(torch.cat((input_embeddings, previous_hidden), dim=-1))

    def context_forward(self, input_ids, infer_state, layer_weight):
        input_embeddings = super().context_forward(input_ids, infer_state, layer_weight)
        return self._fuse_hidden(input_embeddings, infer_state, layer_weight)

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embeddings = super().token_forward(input_ids, infer_state, layer_weight)
        return self._fuse_hidden(input_embeddings, infer_state, layer_weight)
