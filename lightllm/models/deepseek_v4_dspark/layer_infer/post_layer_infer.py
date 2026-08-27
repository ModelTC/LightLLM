import torch

from lightllm.models.deepseek_v4.layer_infer.hyper_connection import hc_head, hc_post
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
        prev_token_ids = torch.cat(
            [anchor_token_ids.view(-1, 1), sampled_tokens[:, :-1]],
            dim=1,
        )
        prev_embeddings = self._markov_prev_embeddings(prev_token_ids, layer_weight)
        features = torch.cat(
            [block_hidden, prev_embeddings.to(dtype=block_hidden.dtype)], dim=-1
        )
        logits = layer_weight.confidence_head_weight_.mm(
            features.flatten(0, -2).float()
        )
        return logits.view(features.shape[:-1])

    def token_forward(
        self,
        input_embdings: torch.Tensor,
        infer_state,
        layer_weight: DeepseekV4DSparkPreAndPostLayerWeight,
    ):
        if infer_state.is_prefill:
            return input_embdings.new_empty((0,))

        if isinstance(input_embdings, tuple):
            streams = hc_post(*input_embdings)
            input_embdings = streams.reshape(streams.shape[0], -1)
        collapsed = hc_head(
            input_embdings,
            layer_weight.hc_head_fn_.weight,
            layer_weight.hc_head_scale_.weight,
            layer_weight.hc_head_base_.weight,
            layer_weight.network_config_["hc_mult"],
            layer_weight.network_config_["hidden_size"],
            layer_weight.network_config_["rms_norm_eps"],
            layer_weight.network_config_.get("hc_eps", 1e-6),
            self.alloc_tensor,
        )

        last_input, token_num = self._slice_get_last_input(collapsed, infer_state)
        num_reqs = token_num // self.block_size_
        block_hidden = last_input.reshape(num_reqs, self.block_size_, -1)
        anchor_token_ids = infer_state.input_ids.reshape(num_reqs, self.block_size_)[
            :, 0
        ]

        normed_input = self._norm(last_input, infer_state, layer_weight)
        lm_head_input = normed_input.permute(1, 0).reshape(-1, token_num)
        local_logits = layer_weight.lm_head_weight_(
            input=lm_head_input, alloc_func=self.alloc_tensor
        )
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
        return local_logits.new_empty((token_num, 1))
