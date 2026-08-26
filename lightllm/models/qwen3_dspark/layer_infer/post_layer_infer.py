import torch

from lightllm.distributed.communication_op import all_gather_into_tensor
from lightllm.models.qwen3_dflash.infer_struct import Qwen3DFlashInferStateInfo
from lightllm.models.qwen3_dflash.layer_infer.post_layer_infer import Qwen3DFlashPostLayerInfer
from lightllm.models.qwen3_dspark.layer_weights.pre_and_post_layer_weight import (
    Qwen3DSparkPreAndPostLayerWeight,
)


class Qwen3DSparkPostLayerInfer(Qwen3DFlashPostLayerInfer):
    """DSpark post layer.

    The block backbone produces one flat logits row per block position. This
    post layer applies DSpark's sequential Markov correction before returning
    logits, and stores raw confidence logits for dynamic verify scheduling.
    """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.block_size_ = network_config["block_size"]
        self.markov_rank_ = network_config.get("markov_rank", 0)
        self.markov_head_type_ = (network_config.get("markov_head_type") or "").lower()
        self.enable_confidence_head_ = network_config.get("enable_confidence_head", False)
        self.confidence_head_with_markov_ = network_config.get("confidence_head_with_markov", False)

    def _markov_prev_embeddings(
        self,
        token_ids: torch.Tensor,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ) -> torch.Tensor:
        return torch.nn.functional.embedding(token_ids, layer_weight.markov_w1_weight_.weight)

    def _markov_step_latent(
        self,
        prev_embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ):
        if self.markov_head_type_ == "vanilla":
            return state, prev_embeddings

        hidden_states = hidden_states.to(dtype=prev_embeddings.dtype)
        if self.markov_head_type_ == "gated":
            gate_input = torch.cat([hidden_states, prev_embeddings], dim=-1)
            gate = torch.sigmoid(layer_weight.markov_gate_proj_weight_.mm(gate_input))
            return state, gate * prev_embeddings

        if state is None:
            state = torch.zeros_like(prev_embeddings)
        joint_input = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        joint = layer_weight.markov_joint_proj_weight_.mm(joint_input)
        gate_raw, candidate_raw, output_raw = joint.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        state = gate * state + (1.0 - gate) * candidate
        return state, torch.tanh(output_raw)

    @torch.no_grad()
    def predict_confidence_logits(
        self,
        block_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        sampled_tokens: torch.Tensor,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ):
        if not self.enable_confidence_head_:
            return None

        features = block_hidden
        if self.confidence_head_with_markov_:
            prev_token_ids = torch.cat(
                [anchor_token_ids.view(-1, 1), sampled_tokens[:, :-1]],
                dim=1,
            )
            prev_embeddings = self._markov_prev_embeddings(prev_token_ids, layer_weight).to(dtype=block_hidden.dtype)
            features = torch.cat([block_hidden, prev_embeddings], dim=-1)

        logits = layer_weight.confidence_head_weight_.mm(features.flatten(0, -2))
        return logits.float().view(features.shape[:-1])

    def _sample_markov(
        self,
        local_logits: torch.Tensor,
        block_hidden: torch.Tensor,
        infer_state: Qwen3DFlashInferStateInfo,
        anchor_token_ids: torch.Tensor,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ) -> torch.Tensor:
        """Run sequential Markov decoding over TP-local vocabulary logits."""

        num_reqs = anchor_token_ids.shape[0]
        local_start = layer_weight.markov_w2_weight_.tp_vocab_start_id
        prev_token_ids = anchor_token_ids
        state = None
        sampled_tokens = []
        req_rows = torch.arange(num_reqs, dtype=torch.long, device=local_logits.device)
        for step_idx in range(self.block_size_):
            prev_embeddings = self._markov_prev_embeddings(prev_token_ids, layer_weight)
            state, markov_latent = self._markov_step_latent(
                prev_embeddings=prev_embeddings,
                hidden_states=block_hidden[:, step_idx, :],
                state=state,
                layer_weight=layer_weight,
            )
            markov_w2 = layer_weight.markov_w2_weight_.weight
            local_markov_bias = torch.mm(markov_latent.to(dtype=markov_w2.dtype), markov_w2.t())
            local_base_logits = local_logits[:, step_idx :: self.block_size_].permute(1, 0).float()
            local_scores = local_base_logits + local_markov_bias
            local_max_values, local_max_indexes = torch.max(local_scores, dim=-1)
            local_token_ids = local_max_indexes + local_start

            if self.tp_world_size_ == 1:
                next_token_ids = local_token_ids
            else:
                local_winners = torch.stack(
                    [local_max_values, local_token_ids.to(dtype=torch.float32)],
                    dim=-1,
                ).contiguous()
                gathered_winners = self.alloc_tensor(
                    (self.tp_world_size_ * num_reqs, 2),
                    dtype=torch.float32,
                )
                all_gather_into_tensor(
                    gathered_winners,
                    local_winners,
                    group=infer_state.dist_group,
                    async_op=False,
                )
                gathered_winners = gathered_winners.view(self.tp_world_size_, num_reqs, 2)
                winning_ranks = torch.argmax(gathered_winners[:, :, 0], dim=0)
                next_token_ids = gathered_winners[winning_ranks, req_rows, 1].long()

            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids

        return torch.stack(sampled_tokens, dim=1)

    def token_forward(
        self,
        input_embdings: torch.Tensor,
        infer_state: Qwen3DFlashInferStateInfo,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ):
        if infer_state.is_prefill:
            return super().token_forward(
                input_embdings=input_embdings,
                infer_state=infer_state,
                layer_weight=layer_weight,
            )

        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        num_reqs = token_num // self.block_size_
        block_hidden = last_input.reshape(num_reqs, self.block_size_, -1)
        anchor_token_ids = infer_state.input_ids.reshape(num_reqs, self.block_size_)[:, 0]

        if self.markov_rank_ > 0:
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
            # Graph unpadding still uses the leading logits dimension when token ids are returned directly.
            return local_logits.new_empty((token_num, 1))

        logits = self._lm_head_and_gather(last_input, token_num, layer_weight, infer_state)
        block_logits = logits.reshape(num_reqs, self.block_size_, -1)
        sampled_tokens = torch.argmax(block_logits, dim=-1)
        confidence_logits = self.predict_confidence_logits(
            block_hidden,
            anchor_token_ids=anchor_token_ids,
            sampled_tokens=sampled_tokens,
            layer_weight=layer_weight,
        )
        infer_state.hidden_collector.add_mtp_outputs(
            draft_token_ids=None,
            confidence_logits=confidence_logits,
        )
        return logits
