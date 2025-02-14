import torch
import torch.distributed as dist
import numpy as np

from lightllm.models.cohere.infer_struct import CohereInferStateInfo
from lightllm.models.cohere.layer_weights.pre_and_post_layer_weight import CoherePreAndPostLayerWeight
from lightllm.models.cohere.triton_kernels.layernorm import layernorm_forward
from lightllm.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight

from einops import rearrange
from lightllm.common.basemodel import PostLayerInferTpl


class CoherePostLayerInfer(PostLayerInferTpl):
    def __init__(self, tp_rank, world_size, network_config, mode):
        super().__init__(tp_rank, world_size, network_config, mode)
        self.eps_ = network_config["layer_norm_eps"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
        self.logits_scale = network_config["logit_scale"]
        return

    def _norm(self, input, infer_state, layer_weight: CoherePreAndPostLayerWeight) -> torch.Tensor:
        return layernorm_forward(
            input.unsqueeze(1), layer_weight.final_norm_weight_.unsqueeze(0), eps=self.eps_
        ).squeeze(1)

    def _slice_get_last_input(self, input_embdings, infer_state: CohereInferStateInfo):

        if infer_state.is_prefill and infer_state.is_token_healing:
            batch_size = infer_state.batch_size
            b_seq_len_numpy = (infer_state.b_seq_len - infer_state.b_ready_cache_len).detach().cpu().numpy()
            select_index = []
            start_index = 0
            select_token_num = 0
            for cur_len in b_seq_len_numpy:

                select_index.append(start_index + cur_len - 1)
                start_index += cur_len
                select_token_num += 1

            last_index = torch.tensor(select_index, dtype=torch.long, device=input_embdings.device)
            last_input = self.alloc_tensor(
                (select_token_num, self.embed_dim_), device=input_embdings.device, dtype=input_embdings.dtype
            )

            last_input[:, :] = input_embdings[last_index, :]
            return last_input, select_token_num

        if infer_state.is_prefill and not infer_state.return_all_prompt_logics:
            batch_size = infer_state.batch_size
            last_input = self.alloc_tensor(
                (batch_size, self.embed_dim_), device=input_embdings.device, dtype=input_embdings.dtype
            )
            last_index = (
                torch.cumsum(infer_state.b_seq_len - infer_state.b_ready_cache_len, dim=0, dtype=torch.long) - 1
            )
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, batch_size

        if infer_state.is_prefill and infer_state.return_all_prompt_logics:
            total_tokens = infer_state.total_token_num
            return input_embdings, total_tokens

        if not infer_state.is_prefill:
            batch_size = infer_state.batch_size
            return input_embdings[-batch_size:, :], batch_size

        assert False, "Error State"

    def token_forward(
        self, input_embdings, infer_state: CohereInferStateInfo, layer_weight: CoherePreAndPostLayerWeight
    ):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings_dtype = input_embdings.dtype
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, token_num)
        logic_batch = torch.mm(layer_weight.lm_head_weight_, last_input)

        last_input = None
        if self.world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = self.alloc_tensor(
                (self.vocab_size_, token_num), device=logic_batch.device, dtype=input_embdings_dtype
            )
            split_indexes = np.linspace(0, self.vocab_size_, self.world_size_ + 1, dtype=np.int64)
            dist.all_gather(
                [gather_data[split_indexes[i] : split_indexes[i + 1], :] for i in range(self.world_size_)],
                logic_batch,
                group=None,
                async_op=False,
            )
        gather_data = gather_data * self.logits_scale
        logic_batch = None

        ans_logics = gather_data.permute(1, 0).float()
        gather_data = None
        return ans_logics
