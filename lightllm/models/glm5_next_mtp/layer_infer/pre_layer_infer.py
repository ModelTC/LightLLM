import torch

from lightllm.common.basemodel.triton_kernel.norm.rmsnorm import rmsnorm_num_warps
from lightllm.models.qwen_vl.layer_infer.pre_layer_infer import (
    LlamaMultimodalPreLayerInfer,
)


class Glm5NextMTPPreLayerInfer(LlamaMultimodalPreLayerInfer):
    """Resolve shifted image tokens before the standard NextN embedding/hidden fusion."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = network_config["rms_norm_eps"]

    def _fuse_hidden(self, input_embeddings, infer_state, layer_weight):
        previous_hidden = infer_state.mtp_draft_input_hiddens
        assert input_embeddings.shape[0] == previous_hidden.shape[0]
        # Exact rounding was validated against the standard eight-warp norms.
        # Respect custom RMSNORM_WARPS by retaining their established path.
        if (
            rmsnorm_num_warps == 8
            and input_embeddings.is_cuda
            and input_embeddings.shape[-1] == 4096
            and input_embeddings.dtype in (torch.bfloat16, torch.float16)
        ):
            from lightllm.models.glm5_next_mtp.triton_kernel.fused_eh_norm import (
                fused_mtp_norm,
            )

            normed = fused_mtp_norm(
                input_embeddings,
                previous_hidden,
                layer_weight.main_norm_weight_.weight,
                layer_weight.enorm_weight_.weight,
                layer_weight.hnorm_weight_.weight,
                self.eps_,
            )
            return layer_weight.eh_proj_weight_.mm(normed)
        layer_weight.main_norm_weight_(input=previous_hidden, eps=self.eps_, out=previous_hidden)
        layer_weight.enorm_weight_(input=input_embeddings, eps=self.eps_, out=input_embeddings)
        layer_weight.hnorm_weight_(input=previous_hidden, eps=self.eps_, out=previous_hidden)
        return layer_weight.eh_proj_weight_.mm(torch.cat((input_embeddings, previous_hidden), dim=-1))

    def context_forward(self, input_ids, infer_state, layer_weight):
        input_embeddings = super().context_forward(input_ids, infer_state, layer_weight)
        return self._fuse_hidden(input_embeddings, infer_state, layer_weight)

    def token_forward(self, input_ids, infer_state, layer_weight):
        input_embeddings = super().token_forward(input_ids, infer_state, layer_weight)
        return self._fuse_hidden(input_embeddings, infer_state, layer_weight)
