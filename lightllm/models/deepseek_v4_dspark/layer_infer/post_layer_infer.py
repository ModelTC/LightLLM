import torch

from lightllm.common.basemodel.batch_objs import PostLayerOutput
from lightllm.common.basemodel.triton_kernel.mhc import hc_head, hc_post
from lightllm.common.basemodel.triton_kernel.mtp_utils import pack_dspark_confidence_features
from lightllm.models.deepseek_v4_dspark.layer_weights.pre_and_post_layer_weight import (
    DeepseekV4DSparkPreAndPostLayerWeight,
)
from lightllm.models.qwen3_dspark.layer_infer.post_layer_infer import (
    Qwen3DSparkPostLayerInfer,
)


class DeepseekV4DSparkPostLayerInfer(Qwen3DSparkPostLayerInfer):
    """Collapse mHC streams, then run the shared head plus DSpark Markov/confidence heads."""

    @torch.no_grad()
    def predict_confidence_logits(
        self,
        block_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        sampled_tokens: torch.Tensor,
        layer_weight: DeepseekV4DSparkPreAndPostLayerWeight,
    ):
        features = pack_dspark_confidence_features(
            block_hidden, anchor_token_ids, sampled_tokens, layer_weight.markov_w1_weight_.weight
        )
        logits = layer_weight.confidence_head_weight_.mm(features)
        return logits.view(block_hidden.shape[:-1])

    def _mix_hc_streams(self, input_embdings, layer_weight):
        cfg = layer_weight.network_config_
        if isinstance(input_embdings, tuple):
            input_embdings = hc_post(*input_embdings, streams=cfg["hc_mult"])
        out = self.alloc_tensor(
            (input_embdings.shape[0], cfg["hidden_size"]), dtype=input_embdings.dtype, device=input_embdings.device
        )
        return hc_head(
            x=input_embdings,
            fn=layer_weight.hc_head_fn_.weight,
            scale=layer_weight.hc_head_scale_.weight,
            base=layer_weight.hc_head_base_.weight,
            streams=cfg["hc_mult"],
            rms_eps=cfg["rms_norm_eps"],
            hc_eps=cfg.get("hc_eps", 1e-6),
            out=out,
        )

    def token_forward(
        self,
        input_embdings: torch.Tensor,
        infer_state,
        layer_weight: DeepseekV4DSparkPreAndPostLayerWeight,
    ):
        collapsed = self._mix_hc_streams(input_embdings, layer_weight)
        last_input, token_num = self._slice_get_last_input(collapsed, infer_state)
        num_reqs = token_num // self.block_size_
        block_hidden = last_input.reshape(num_reqs, self.block_size_, -1)
        anchor_token_ids = infer_state.input_ids.reshape(num_reqs, self.block_size_)[:, 0]

        normed_input = self._norm(last_input, infer_state, layer_weight)
        lm_head_input = normed_input.permute(1, 0).reshape(-1, token_num)
        local_logits = layer_weight.lm_head_weight_(input=lm_head_input, alloc_func=self.alloc_tensor)
        sampled_tokens = self._sample_markov(
            local_logits,
            block_hidden=block_hidden,
            infer_state=infer_state,
            anchor_token_ids=anchor_token_ids,
            layer_weight=layer_weight,
        )
        confidence_logits = self.predict_confidence_logits(
            block_hidden,
            anchor_token_ids=anchor_token_ids,
            sampled_tokens=sampled_tokens,
            layer_weight=layer_weight,
        )
        infer_state.hidden_collector.add_mtp_outputs(
            draft_token_ids=sampled_tokens.reshape(-1),
            confidence_logits=confidence_logits,
        )
        # The proposer consumes token ids directly; keep only the row dimension for graph unpadding.
        return PostLayerOutput(logits=local_logits.new_empty((token_num, 1)))
