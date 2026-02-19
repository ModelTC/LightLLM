import torch

from lightllm.models.qwen3next_mtp.layer_weights.pre_and_post_layer_weight import Qwen3NextMTPPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.layer_infer.pre_layer_infer import LlamaPreLayerInfer
from lightllm.models.qwen3next.triton_kernel.gemma_rmsnorm import gemma_rmsnorm_forward


class Qwen3NextMTPPreLayerInfer(LlamaPreLayerInfer):
    """
    Qwen3Next MTP Pre-Layer Inference.
    Similar to DeepSeek MTP but with different weight structure.

    MTP forward flow:
    1. Get embedding from input_ids
    2. Get hidden state from main model (passed via infer_state)
    3. Normalize embedding with pre_fc_norm_embedding
    4. Normalize hidden with pre_fc_norm_hidden
    5. Concat normalized embedding and hidden
    6. Project through fc to get hidden_dim output
    """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = network_config["rms_norm_eps"]
        self.hidden_size = network_config["hidden_size"]
        return

    def _mtp_forward(
        self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextMTPPreAndPostLayerWeight
    ):
        tgt_embdings = infer_state.mtp_draft_input_hiddens
        assert input_embdings.shape[0] == tgt_embdings.shape[0]

        # Normalize embedding
        input_embdings_normed = self.alloc_tensor(input_embdings.shape, input_embdings.dtype)
        gemma_rmsnorm_forward(
            input_embdings, layer_weight.pre_fc_norm_embedding_weight_.weight, self.eps_, out=input_embdings_normed
        )

        # Normalize hidden state
        tgt_embdings_normed = self.alloc_tensor(tgt_embdings.shape, tgt_embdings.dtype)
        gemma_rmsnorm_forward(
            tgt_embdings, layer_weight.pre_fc_norm_hidden_weight_.weight, self.eps_, out=tgt_embdings_normed
        )

        # Concat normalized embedding and hidden
        cat_embdings = torch.cat((input_embdings_normed, tgt_embdings_normed), dim=-1)

        # Project to hidden_size
        ans_logics = self.alloc_tensor(
            (cat_embdings.shape[0], layer_weight.fc_weight_.shape[1]), dtype=cat_embdings.dtype
        )
        torch.mm(cat_embdings, layer_weight.fc_weight_, out=ans_logics)

        return ans_logics

    def context_forward(
        self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextMTPPreAndPostLayerWeight
    ):
        input_embdings = super().context_forward(input_ids, infer_state, layer_weight)
        return self._mtp_forward(input_embdings, infer_state, layer_weight)

    def token_forward(
        self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: Qwen3NextMTPPreAndPostLayerWeight
    ):
        input_embdings = super().token_forward(input_ids, infer_state, layer_weight)
        return self._mtp_forward(input_embdings, infer_state, layer_weight)
