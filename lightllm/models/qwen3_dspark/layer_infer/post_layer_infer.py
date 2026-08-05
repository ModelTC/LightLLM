import numpy as np
import torch
import torch.nn.functional as F

from lightllm.distributed.communication_op import all_gather, all_gather_into_tensor
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
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
        self.block_size_ = int(network_config["block_size"])
        self.markov_rank_ = int(network_config.get("markov_rank", 0))
        self.markov_head_type_ = str(network_config.get("markov_head_type", "")).lower()
        self.enable_confidence_head_ = bool(network_config.get("enable_confidence_head", False))
        self.confidence_head_with_markov_ = bool(network_config.get("confidence_head_with_markov", False))
        self.mtp_draft_confidence_logits = None
        self.mtp_draft_token_ids = None

    def pop_mtp_draft_confidence_logits(self):
        logits = self.mtp_draft_confidence_logits
        self.mtp_draft_confidence_logits = None
        return logits

    def pop_mtp_draft_token_ids(self):
        token_ids = self.mtp_draft_token_ids
        self.mtp_draft_token_ids = None
        return token_ids

    def has_markov_head(self) -> bool:
        return self.markov_rank_ > 0

    def has_confidence_head(self) -> bool:
        return self.enable_confidence_head_

    def _linear_parameter(self, input_tensor: torch.Tensor, parameter_weight) -> torch.Tensor:
        assert parameter_weight is not None
        weight = parameter_weight.weight
        bias = parameter_weight.bias
        return F.linear(input_tensor.to(dtype=weight.dtype), weight, bias)

    def _markov_prev_embeddings(
        self,
        token_ids: torch.Tensor,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ) -> torch.Tensor:
        assert layer_weight.markov_w1_weight_ is not None
        return F.embedding(token_ids.long(), layer_weight.markov_w1_weight_.weight)

    def _markov_project_bias(
        self,
        latent_states: torch.Tensor,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ) -> torch.Tensor:
        assert layer_weight.markov_w2_weight_ is not None
        weight = layer_weight.markov_w2_weight_.weight
        return F.linear(latent_states.to(dtype=weight.dtype), weight)

    def _markov_step_bias(
        self,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        state: torch.Tensor,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ):
        prev_embeddings = self._markov_prev_embeddings(prev_token_ids, layer_weight)
        if self.markov_head_type_ == "vanilla":
            return state, self._markov_project_bias(prev_embeddings, layer_weight)

        hidden_states = hidden_states.to(dtype=prev_embeddings.dtype)
        if self.markov_head_type_ == "gated":
            gate_input = torch.cat([hidden_states, prev_embeddings], dim=-1)
            gate = torch.sigmoid(self._linear_parameter(gate_input, layer_weight.markov_gate_proj_weight_))
            return state, self._markov_project_bias(gate * prev_embeddings, layer_weight)

        assert self.markov_head_type_ == "rnn"
        if state is None:
            state = torch.zeros_like(prev_embeddings)
        joint_input = torch.cat([state, prev_embeddings, hidden_states], dim=-1)
        joint = self._linear_parameter(joint_input, layer_weight.markov_joint_proj_weight_)
        gate_raw, candidate_raw, output_raw = joint.chunk(3, dim=-1)
        gate = torch.sigmoid(gate_raw)
        candidate = torch.tanh(candidate_raw)
        state = gate * state + (1.0 - gate) * candidate
        return state, self._markov_project_bias(torch.tanh(output_raw), layer_weight)

    @torch.no_grad()
    def apply_markov_logits(
        self,
        base_logits: torch.Tensor,
        *,
        block_hidden: torch.Tensor,
        anchor_token_ids: torch.Tensor,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ):
        if not self.has_markov_head():
            return base_logits, torch.argmax(base_logits, dim=-1)

        sampled_tokens = []
        corrected_logits = []
        prev_token_ids = anchor_token_ids.long()
        state = None
        for step_idx in range(base_logits.shape[1]):
            state, markov_bias = self._markov_step_bias(
                prev_token_ids=prev_token_ids,
                hidden_states=block_hidden[:, step_idx, :],
                state=state,
                layer_weight=layer_weight,
            )
            step_logits = base_logits[:, step_idx, :] + markov_bias
            next_token_ids = torch.argmax(step_logits, dim=-1)
            sampled_tokens.append(next_token_ids)
            corrected_logits.append(step_logits.unsqueeze(1))
            prev_token_ids = next_token_ids

        return torch.cat(corrected_logits, dim=1), torch.stack(sampled_tokens, dim=1)

    @torch.no_grad()
    def predict_confidence_logits(
        self,
        block_hidden: torch.Tensor,
        *,
        anchor_token_ids: torch.Tensor,
        sampled_tokens: torch.Tensor,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ):
        if not self.has_confidence_head():
            return None

        features = block_hidden
        if self.confidence_head_with_markov_:
            prev_token_ids = torch.cat(
                [anchor_token_ids.view(-1, 1), sampled_tokens[:, :-1]],
                dim=1,
            )
            prev_embeddings = self._markov_prev_embeddings(prev_token_ids, layer_weight).to(dtype=block_hidden.dtype)
            features = torch.cat([block_hidden, prev_embeddings], dim=-1)

        logits = self._linear_parameter(features, layer_weight.confidence_head_weight_)
        return logits.float().squeeze(-1)

    def _token_forward_with_hidden(
        self,
        input_embdings: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings_dtype = input_embdings.dtype
        head_hidden = last_input
        normed_input = self._norm(last_input, infer_state, layer_weight)
        lm_head_input = normed_input.permute(1, 0).reshape(-1, token_num)
        logic_batch = layer_weight.lm_head_weight_(input=lm_head_input, alloc_func=self.alloc_tensor)
        normed_input = None
        lm_head_input = None
        vocab_size = layer_weight.lm_head_weight_.vocab_size
        if self.tp_world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = self.alloc_tensor((vocab_size, token_num), dtype=input_embdings_dtype)
            split_indexes = np.linspace(0, vocab_size, self.tp_world_size_ + 1, dtype=np.int64)
            all_gather(
                [gather_data[split_indexes[i] : split_indexes[i + 1], :] for i in range(self.tp_world_size_)],
                logic_batch,
                group=infer_state.dist_group,
                async_op=False,
            )
        logic_batch = None
        logits = self.alloc_tensor(
            (token_num, vocab_size),
            dtype=torch.float32,
        )
        logits[:, :] = gather_data.permute(1, 0)
        gather_data = None
        return logits, head_hidden

    def _token_forward_with_local_logits_and_hidden(
        self,
        input_embdings: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ):
        """Project the LM head but keep its vocabulary shard local.

        Vanilla Markov decoding only needs the global maximum token at each
        block position. Keeping logits sharded avoids a full-vocabulary TP
        all-gather and prevents every rank from repeating the same Markov
        projection over the complete vocabulary.
        """

        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        head_hidden = last_input
        normed_input = self._norm(last_input, infer_state, layer_weight)
        lm_head_input = normed_input.permute(1, 0).reshape(-1, token_num)
        local_logits = layer_weight.lm_head_weight_(input=lm_head_input, alloc_func=self.alloc_tensor)
        return local_logits, head_hidden

    def _sample_tp_sharded_vanilla_markov(
        self,
        local_logits: torch.Tensor,
        *,
        infer_state: LlamaInferStateInfo,
        anchor_token_ids: torch.Tensor,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ) -> torch.Tensor:
        """Run exact greedy Markov decoding on TP vocabulary shards."""

        assert self.markov_head_type_ == "vanilla"
        assert layer_weight.markov_w2_weight_ is not None
        token_num = local_logits.shape[1]
        assert token_num % self.block_size_ == 0
        num_reqs = token_num // self.block_size_

        vocab_size = int(layer_weight.lm_head_weight_.vocab_size)
        split_indexes = np.linspace(0, vocab_size, self.tp_world_size_ + 1, dtype=np.int64)
        local_start = int(split_indexes[self.tp_rank_])
        local_end = int(split_indexes[self.tp_rank_ + 1])
        assert local_logits.shape[0] == local_end - local_start, (
            f"local LM head rows must match TP vocabulary shard [{local_start}, {local_end}), "
            f"got {local_logits.shape[0]}"
        )
        local_markov_w2 = layer_weight.markov_w2_weight_.weight[local_start:local_end, :]

        prev_token_ids = anchor_token_ids.long()
        sampled_tokens = []
        req_rows = torch.arange(num_reqs, dtype=torch.long, device=local_logits.device)
        for step_idx in range(self.block_size_):
            prev_embeddings = self._markov_prev_embeddings(prev_token_ids, layer_weight)
            local_markov_bias = F.linear(prev_embeddings.to(dtype=local_markov_w2.dtype), local_markov_w2)
            local_base_logits = local_logits[:, step_idx::self.block_size_].permute(1, 0).float()
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
        infer_state: LlamaInferStateInfo,
        layer_weight: Qwen3DSparkPreAndPostLayerWeight,
    ):
        self.mtp_draft_confidence_logits = None
        self.mtp_draft_token_ids = None
        if self._is_commit_prefill(infer_state):
            return super().token_forward(
                input_embdings=input_embdings,
                infer_state=infer_state,
                layer_weight=layer_weight,
            )

        if infer_state.is_prefill:
            logits, _ = self._token_forward_with_hidden(
                input_embdings=input_embdings,
                infer_state=infer_state,
                layer_weight=layer_weight,
            )
            return logits

        use_tp_sharded_markov = (
            self.tp_world_size_ > 1
            and self.has_markov_head()
            and self.markov_head_type_ == "vanilla"
        )
        if use_tp_sharded_markov:
            local_logits, head_hidden = self._token_forward_with_local_logits_and_hidden(
                input_embdings=input_embdings,
                infer_state=infer_state,
                layer_weight=layer_weight,
            )
            token_num = local_logits.shape[1]
            assert token_num % self.block_size_ == 0
            num_reqs = token_num // self.block_size_
            block_hidden = head_hidden.reshape(num_reqs, self.block_size_, -1)
            anchor_token_ids = infer_state.input_ids.reshape(num_reqs, self.block_size_)[:, 0]
            sampled_tokens = self._sample_tp_sharded_vanilla_markov(
                local_logits,
                infer_state=infer_state,
                anchor_token_ids=anchor_token_ids,
                layer_weight=layer_weight,
            )
            self.mtp_draft_token_ids = sampled_tokens.reshape(-1)
            self.mtp_draft_confidence_logits = self.predict_confidence_logits(
                block_hidden,
                anchor_token_ids=anchor_token_ids,
                sampled_tokens=sampled_tokens,
                layer_weight=layer_weight,
            )
            # The proposer consumes mtp_draft_token_ids directly. Keep the
            # leading row dimension for generic graph padding/unpadding while
            # avoiding an otherwise unused [rows, vocab] tensor. A single
            # placeholder column is required because CUDA graph's no-ref
            # tensor wrapper cannot represent a zero-byte allocation.
            return local_logits.new_empty((token_num, 1))

        logits, head_hidden = self._token_forward_with_hidden(
            input_embdings=input_embdings,
            infer_state=infer_state,
            layer_weight=layer_weight,
        )

        assert (
            logits.shape[0] % self.block_size_ == 0
        ), f"DSpark draft logits rows must be a multiple of block_size={self.block_size_}, got {logits.shape[0]}"
        num_reqs = logits.shape[0] // self.block_size_
        block_logits = logits.reshape(num_reqs, self.block_size_, -1)
        block_hidden = head_hidden.reshape(num_reqs, self.block_size_, -1)
        anchor_token_ids = infer_state.input_ids.reshape(num_reqs, self.block_size_)[:, 0]

        corrected_logits, sampled_tokens = self.apply_markov_logits(
            block_logits,
            block_hidden=block_hidden,
            anchor_token_ids=anchor_token_ids,
            layer_weight=layer_weight,
        )
        self.mtp_draft_confidence_logits = self.predict_confidence_logits(
            block_hidden,
            anchor_token_ids=anchor_token_ids,
            sampled_tokens=sampled_tokens,
            layer_weight=layer_weight,
        )
        return corrected_logits.reshape(logits.shape[0], -1)
