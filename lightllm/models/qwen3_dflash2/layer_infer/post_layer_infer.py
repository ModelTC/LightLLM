import torch
import torch.nn.functional as F

from lightllm.distributed.communication_op import all_gather_into_tensor
from lightllm.models.qwen3_dflash.layer_infer.post_layer_infer import Qwen3DFlashPostLayerInfer
from lightllm.models.qwen3_dflash2.triton_kernel.selector_walk import selector_walk


class Qwen3DFlash2PostLayerInfer(Qwen3DFlashPostLayerInfer):
    """Run DFlash2's top-k pair selector over a parallel draft block."""

    def __init__(self, network_config):
        super().__init__(network_config)
        self.block_size_ = int(network_config["block_size"])
        self.selector_top_k_ = int(network_config["selector_top_k"])
        self.output_multiplier_ = float(network_config.get("output_multiplier", 1.0))
        softcap = network_config.get("final_logit_softcapping")
        self.final_logit_softcapping_ = None if softcap is None else float(softcap)
        if self.final_logit_softcapping_ is not None and self.final_logit_softcapping_ <= 0:
            raise ValueError("final_logit_softcapping must be greater than 0")

    def _transform_unary_logits(self, logits):
        logits = logits.float()
        if self.output_multiplier_ != 1.0:
            logits.mul_(self.output_multiplier_)
        if self.final_logit_softcapping_ is not None:
            logits.div_(self.final_logit_softcapping_).tanh_().mul_(self.final_logit_softcapping_)
        return logits

    def _compute_candidates(self, hidden, infer_state, layer_weight):
        """Select global top-k candidates without gathering full-vocabulary logits."""

        token_num = hidden.shape[0]
        lm_head = layer_weight.lm_head_weight_
        local_vocab_size = lm_head.weight.shape[0]
        if local_vocab_size < self.selector_top_k_:
            raise ValueError(
                "DFlash2 selector_top_k exceeds the TP-local vocabulary size: "
                f"top_k={self.selector_top_k_}, local_vocab_size={local_vocab_size}"
            )

        lm_head_input = hidden.permute(1, 0).contiguous()
        local_logits = lm_head(input=lm_head_input, alloc_func=self.alloc_tensor)
        local_logits = local_logits.permute(1, 0).contiguous()
        local_values, local_ids = torch.topk(local_logits, k=self.selector_top_k_, dim=-1)
        global_ids = local_ids.long().add_(lm_head.tp_vocab_start_id)

        if self.tp_world_size_ == 1:
            return global_ids, self._transform_unary_logits(local_values)

        gathered_values = self.alloc_tensor(
            (self.tp_world_size_ * token_num, self.selector_top_k_),
            dtype=torch.float32,
        )
        all_gather_into_tensor(
            gathered_values,
            local_values.float().contiguous(),
            group=infer_state.dist_group,
            async_op=False,
        )
        gathered_ids = self.alloc_tensor(
            (self.tp_world_size_ * token_num, self.selector_top_k_),
            dtype=torch.int64,
        )
        all_gather_into_tensor(
            gathered_ids,
            global_ids.contiguous(),
            group=infer_state.dist_group,
            async_op=False,
        )

        gathered_values = (
            gathered_values.view(self.tp_world_size_, token_num, self.selector_top_k_)
            .permute(1, 0, 2)
            .reshape(token_num, self.tp_world_size_ * self.selector_top_k_)
        )
        gathered_ids = (
            gathered_ids.view(self.tp_world_size_, token_num, self.selector_top_k_)
            .permute(1, 0, 2)
            .reshape(token_num, self.tp_world_size_ * self.selector_top_k_)
        )
        top_values, top_indexes = torch.topk(gathered_values, k=self.selector_top_k_, dim=-1)
        candidate_ids = torch.gather(gathered_ids, dim=-1, index=top_indexes)
        return candidate_ids, self._transform_unary_logits(top_values)

    def _select_path(self, hidden, candidate_ids, unary, anchor_token_ids, infer_state, layer_weight):
        req_num, draft_width, _ = candidate_ids.shape
        assert draft_width > 0
        assert hidden.shape[:2] == (req_num, draft_width)
        assert unary.shape == candidate_ids.shape

        candidate_hidden = hidden.reshape(req_num * draft_width, -1)
        gate = layer_weight.selector_hidden_projection_weight_.mm(candidate_hidden)
        gate = gate.view(req_num, draft_width, -1)

        predecessor_codebook = layer_weight.selector_predecessor_codebook_weight_.weight
        successor_codebook = layer_weight.selector_successor_codebook_weight_.weight
        successor = F.embedding(candidate_ids, successor_codebook)

        anchor = F.embedding(anchor_token_ids, predecessor_codebook)
        first_scores = unary[:, 0, :] + torch.sum(
            anchor[:, None, :] * gate[:, 0, None, :] * successor[:, 0, :, :],
            dim=-1,
        )

        pair_scores = None
        if draft_width > 1:
            predecessor = F.embedding(candidate_ids[:, :-1, :], predecessor_codebook)
            conditioned_predecessor = predecessor * gate[:, 1:, None, :]
            transitions = torch.matmul(conditioned_predecessor, successor[:, 1:, :, :].transpose(-1, -2))
            pair_scores = transitions + unary[:, 1:, None, :]

        score_lattice = first_scores[:, None, None, :].expand(-1, 1, self.selector_top_k_, -1)
        if pair_scores is not None:
            score_lattice = torch.cat((score_lattice, pair_scores), dim=1)

        request_ids = infer_state.b_req_idx.view(req_num, self.block_size_)[:, 0].long()
        sampling_manager = infer_state.req_manager.req_sampling_params_manager
        temperatures = sampling_manager.req_to_temperature.index_select(0, request_ids).clamp_min_(1e-5)
        greedy_mask = sampling_manager.req_to_top_k.index_select(0, request_ids).eq(1)
        # DFlash2 currently does not guarantee request-level determinism: draft selection uses the global RNG,
        # while request seeds only control target verification.
        uniforms = torch.rand(
            (req_num, draft_width),
            dtype=torch.float32,
            device=candidate_ids.device,
        )
        selected_ids, q_rows, _ = selector_walk(
            scores=score_lattice,
            candidate_ids=candidate_ids,
            uniforms=uniforms,
            temperatures=temperatures,
            greedy_mask=greedy_mask,
        )
        return selected_ids, q_rows

    def token_forward(self, input_embdings, infer_state, layer_weight):
        if infer_state.is_prefill:
            return super().token_forward(input_embdings, infer_state, layer_weight)

        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        assert token_num % self.block_size_ == 0
        req_num = token_num // self.block_size_
        normed_hidden = self._norm(last_input, infer_state, layer_weight)
        block_hidden = normed_hidden.view(req_num, self.block_size_, -1)
        candidate_hidden = block_hidden[:, 1:, :].reshape(req_num * (self.block_size_ - 1), -1)
        candidate_ids, unary = self._compute_candidates(
            hidden=candidate_hidden,
            infer_state=infer_state,
            layer_weight=layer_weight,
        )
        candidate_ids = candidate_ids.view(req_num, self.block_size_ - 1, self.selector_top_k_)
        unary = unary.view(req_num, self.block_size_ - 1, self.selector_top_k_)
        anchor_token_ids = infer_state.input_ids.view(req_num, self.block_size_)[:, 0]
        draft_token_ids, draft_candidate_probs = self._select_path(
            hidden=block_hidden[:, 1:, :],
            candidate_ids=candidate_ids,
            unary=unary,
            anchor_token_ids=anchor_token_ids,
            infer_state=infer_state,
            layer_weight=layer_weight,
        )
        infer_state.hidden_collector.add_mtp_outputs(
            draft_token_ids=draft_token_ids,
            draft_candidate_ids=candidate_ids,
            draft_candidate_probs=draft_candidate_probs,
            confidence_logits=None,
        )
        # The proposer consumes selector outputs directly. Keep only a graph-
        # compatible leading dimension instead of retaining full-vocabulary logits.
        return unary.new_empty((token_num, 1))
