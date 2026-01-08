import os
import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from lightllm.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import PostLayerInferTpl
from lightllm.distributed.communication_op import all_gather


class LlamaPostLayerInfer(PostLayerInferTpl):
    """ """

    def __init__(self, network_config):
        super().__init__(network_config)
        self.eps_ = network_config["rms_norm_eps"]
        return

    def _norm(self, input, infer_state, layer_weight: LlamaPreAndPostLayerWeight) -> torch.Tensor:
        return layer_weight.final_norm_weight_.rmsnorm_forward(input=input, eps=self.eps_, alloc_func=self.alloc_tensor)

    def _slice_get_last_input(self, input_embdings: torch.Tensor, infer_state: LlamaInferStateInfo):
        embed_dim_ = input_embdings.shape[1]
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
            last_input = self.alloc_tensor((select_token_num, embed_dim_), dtype=input_embdings.dtype)
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, select_token_num

        if infer_state.is_prefill and not infer_state.return_all_prompt_logics:
            batch_size = infer_state.batch_size
            last_input = self.alloc_tensor((batch_size, embed_dim_), dtype=input_embdings.dtype)
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
        self, input_embdings: torch.Tensor, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight
    ):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings_dtype = input_embdings.dtype
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = last_input.permute(1, 0).view(-1, token_num)
        logic_batch = layer_weight.lm_head_weight_.lm_head(input=last_input, alloc_func=self.alloc_tensor)
        last_input = None
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
        ans_logics = self.alloc_tensor(
            (token_num, vocab_size),
            dtype=torch.float32,
        )
        ans_logics[:, :] = gather_data.permute(1, 0)
        gather_data = None
        return ans_logics

    def tpsp_token_forward(
        self, input_embdings: torch.Tensor, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight
    ):
        if self.tp_world_size_ > 1:
            assert len(input_embdings.shape) == 2
            token_num, hidden_dim = input_embdings.shape
            gather_data = torch.empty(
                (self.tp_world_size_ * token_num, hidden_dim), device=input_embdings.device, dtype=input_embdings.dtype
            )
            all_gather(
                [gather_data[i * token_num : (i + 1) * token_num, :] for i in range(self.tp_world_size_)],
                input_embdings,
                group=infer_state.dist_group,
                async_op=False,
            )
            # len(infer_state.input_ids) 获取真实输入长度
            input_embdings = gather_data[0 : len(infer_state.input_ids)]

        if infer_state.need_dp_prefill_balance:
            input_embdings = infer_state._all_to_all_unbalance_get(data=input_embdings)

        return self.token_forward(input_embdings=input_embdings, infer_state=infer_state, layer_weight=layer_weight)

    def overlap_tpsp_token_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: BaseLayerWeight,
    ):
        if getattr(infer_state, "hook", None) is not None:
            infer_state.hook()
            infer_state.hook = None

        logics = self.tpsp_token_forward(input_embdings, infer_state, layer_weight=layer_weight)

        if getattr(infer_state1, "hook", None) is not None:
            infer_state1.hook()
            infer_state1.hook = None

        logics1 = self.tpsp_token_forward(input_embdings1, infer_state1, layer_weight=layer_weight)

        return logics, logics1
