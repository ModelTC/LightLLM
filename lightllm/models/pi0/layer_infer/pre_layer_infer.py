from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lightllm.common.basemodel import PreLayerInferTpl
from lightllm.common.basemodel.triton_kernel.multimodal_emb import multimodal_emb
from lightllm.distributed.communication_op import all_reduce
from lightllm.models.gemma_2b.layer_infer.pre_layer_infer import (
    Gemma_2bPreLayerInfer,
)
from lightllm.models.pi0.layer_weights.pre_and_post_layer_weight import (
    Pi0ActionPreAndPostLayerWeight,
    Pi0VLMPreAndPostLayerWeight,
)
from lightllm.models.pi0.math_utils import create_sinusoidal_pos_embedding


class Pi0VLMPreLayerInfer(Gemma_2bPreLayerInfer):
    """Standard Gemma embedding stage with in-request image replacement."""

    def context_forward(self, input_ids, infer_state, layer_weight: Pi0VLMPreAndPostLayerWeight):
        has_direct_image_embeds = any(
            params.get("pi0_image_embeds") is not None for params in infer_state.multimodal_params
        )
        if not has_direct_image_embeds:
            if any(params.get("images") for params in infer_state.multimodal_params):
                return self._context_forward_from_embed_cache(input_ids, infer_state, layer_weight)
            # BaseModel warmup, padded requests, and future text-only VLA
            # traffic contain no image metadata. They should keep Gemma's
            # ordinary embedding path unchanged.
            return super().context_forward(input_ids, infer_state, layer_weight)

        # Direct embeddings remain as a model-level escape hatch for parity
        # tests. Production requests arrive through visualserver's embed cache.
        hidden_states = super().context_forward(input_ids, infer_state, layer_weight)

        sequence_offset = 0
        sequence_lengths = infer_state.b_q_seq_len.tolist()
        for sequence_length, params in zip(sequence_lengths, infer_state.multimodal_params, strict=True):
            image_embeds = params.get("pi0_image_embeds")
            if image_embeds is not None:
                image_embeds = image_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
                image_token_num = image_embeds.shape[0]
                if image_token_num > sequence_length:
                    raise ValueError("pi0 image embeddings exceed packed sequence")
                hidden_states[sequence_offset : sequence_offset + image_token_num].copy_(image_embeds)
            sequence_offset += sequence_length
        return hidden_states

    def _context_forward_from_embed_cache(self, input_ids, infer_state, layer_weight: Pi0VLMPreAndPostLayerWeight):
        image_start_token_ids = []
        image_token_lens = []
        image_cache_locs = []
        seen_token_ids = set()
        for params in infer_state.multimodal_params:
            for image in params.get("images", []):
                token_id = image["token_id"]
                if token_id in seen_token_ids:
                    continue
                seen_token_ids.add(token_id)
                image_start_token_ids.append(token_id)
                image_token_lens.append(image["token_num"])
                image_cache_locs.append(image["start_index_in_embed_cache"])

        device = layer_weight.wte_weight_.weight.device
        dtype = layer_weight.wte_weight_.weight.dtype
        hidden_size = layer_weight.wte_weight_.weight.shape[1]
        output = torch.zeros((input_ids.numel(), hidden_size), dtype=dtype, device=device)

        from lightllm.server.router.model_infer.infer_batch import g_infer_context

        cache_client = g_infer_context.cpu_embed_cache_client
        if cache_client is None:
            raise RuntimeError("pi0 prefill requires the visualserver embed cache")
        embed_cache = cache_client.cpu_embed_cache_tensor
        if embed_cache.shape[2] != hidden_size:
            raise ValueError(f"pi0 image hidden size {embed_cache.shape[2]} != text hidden size {hidden_size}")

        def to_cuda(values):
            return torch.tensor(values, dtype=torch.long, device="cpu", pin_memory=True).to(
                device=device, non_blocking=True
            )

        multimodal_emb(
            out=output,
            prompt_ids=input_ids,
            text_weight_embs=layer_weight.wte_weight_.weight,
            embed_cache=embed_cache,
            img_token_lens=to_cuda(image_token_lens),
            img_start_token_ids=to_cuda(image_start_token_ids),
            img_start_locs_in_cache=to_cuda(image_cache_locs),
            tp_text_start_token_id=layer_weight.wte_weight_.tp_vocab_start_id,
            tp_text_end_token_id=layer_weight.wte_weight_.tp_vocab_end_id,
            tp_world_size=self.tp_world_size_,
            text_embed_scale=self.normfactor,
        )
        if self.tp_world_size_ > 1:
            all_reduce(
                output,
                group=infer_state.dist_group,
                op=dist.ReduceOp.SUM,
                async_op=False,
            )
        return output


class Pi0ActionPreLayerInfer(PreLayerInferTpl):
    def __init__(self, network_config: dict):
        super().__init__(network_config)
        self.config = network_config

    @staticmethod
    def _pad_last_dim(value: torch.Tensor, target: int) -> torch.Tensor:
        if value.shape[-1] > target:
            raise ValueError(f"input dimension {value.shape[-1]} exceeds checkpoint dimension {target}")
        if value.shape[-1] == target:
            return value
        return F.pad(value, (0, target - value.shape[-1]))

    def context_forward(
        self,
        state: torch.Tensor | None,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        weight: Pi0ActionPreAndPostLayerWeight,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        noisy_actions = self._pad_last_dim(noisy_actions, self.config["max_action_dim"]).to(weight.data_type_)
        batch_size, action_horizon = noisy_actions.shape[:2]
        action_embedding = weight.action_in_proj.mm(noisy_actions.reshape(-1, noisy_actions.shape[-1])).view(
            batch_size, action_horizon, -1
        )
        time_embedding = create_sinusoidal_pos_embedding(
            timestep,
            self.config["hidden_size"],
            min_period=self.config["min_period"],
            max_period=self.config["max_period"],
        ).to(dtype=weight.data_type_)

        embeddings = []
        if self.config["is_pi05"]:
            condition = F.silu(weight.time_mlp_in.mm(time_embedding))
            condition = F.silu(weight.time_mlp_out.mm(condition))
            action_tokens = action_embedding
        else:
            if state is None:
                raise ValueError("pi0 action suffix requires continuous state")
            state = self._pad_last_dim(state, self.config["max_state_dim"]).to(weight.data_type_)
            embeddings.append(weight.state_proj.mm(state)[:, None, :])
            expanded_time = time_embedding[:, None, :].expand_as(action_embedding)
            action_tokens = torch.cat([action_embedding, expanded_time], dim=-1)
            action_tokens = F.silu(weight.time_mlp_in.mm(action_tokens.reshape(-1, action_tokens.shape[-1])))
            action_tokens = weight.time_mlp_out.mm(action_tokens).view(batch_size, action_horizon, -1)
            condition = None

        embeddings.append(action_tokens)
        return torch.cat(embeddings, dim=1), condition
