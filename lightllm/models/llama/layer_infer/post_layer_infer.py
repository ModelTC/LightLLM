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
        return layer_weight.final_norm_weight_(input=input, eps=self.eps_, alloc_func=self.alloc_tensor)

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

            last_index = torch.tensor(select_index, dtype=torch.long, device="cpu").cuda(non_blocking=True)
            last_input = self.alloc_tensor((select_token_num, embed_dim_), dtype=input_embdings.dtype, device="cuda")
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, select_token_num

        if infer_state.is_prefill:
            # logits 始终只取每个请求最后一个位置的 hidden state，用于正常采样。
            batch_size = infer_state.batch_size
            last_input = self.alloc_tensor((batch_size, embed_dim_), dtype=input_embdings.dtype)
            last_index = (
                torch.cumsum(infer_state.b_seq_len - infer_state.b_ready_cache_len, dim=0, dtype=torch.long) - 1
            )
            last_input[:, :] = input_embdings[last_index, :]

            # 在开启 return_all_prompt_logics 模式时，额外保存整个 prefill 阶段
            # 每一个 token 位置对应的 hidden state，用于后续输出 prompt logprobs。
            # input_embdings 本身已经是本次新增的 token（不含已缓存前缀），
            # 仅在 chunked prefill 的 padding 场景下会多出行，padding 部分会在
            # basemodel._create_unpad_prefill_model_output 中按实际 token 数量裁剪掉。
            if infer_state.return_all_prompt_logics:
                infer_state.prompt_logics = input_embdings
            return last_input, batch_size

        if not infer_state.is_prefill:
            batch_size = infer_state.batch_size
            return input_embdings[-batch_size:, :], batch_size

        assert False, "Error State"

    def _token_forward(
        self, input_embdings: torch.Tensor, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight
    ):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings = None

        # 正常采样使用的 logits，始终只对应每个请求最后一个位置。
        ans_logics = self._lm_head_and_gather(last_input, token_num, layer_weight, infer_state)
        # 在 return_all_prompt_logics 模式下，prompt_logics 保存的是完整 prefill
        # 的 hidden state，需要在 norm/lm_head 之前取出来，避免被 input_embdings 置空。
        prompt_logics_hiddens = infer_state.prompt_logics
        infer_state.prompt_logics = None
        # 在 return_all_prompt_logics 模式下，额外计算整个 prefill 阶段所有位置的 logits，
        # 存入返回的 prompt_logics 中，原来的 ans_logics 仅保留最后一个位置的 logits。
        if prompt_logics_hiddens is not None:
            prompt_token_num = prompt_logics_hiddens.shape[0]
            infer_state.prompt_logics = self._lm_head_and_gather(
                prompt_logics_hiddens, prompt_token_num, layer_weight, infer_state
            )

        return ans_logics

    def _lm_head_and_gather(
        self,
        hidden: torch.Tensor,
        token_num: int,
        layer_weight: LlamaPreAndPostLayerWeight,
        infer_state: LlamaInferStateInfo,
    ) -> torch.Tensor:
        normed = self._norm(hidden, infer_state, layer_weight)
        normed = normed.permute(1, 0).view(-1, token_num)
        logic_batch = layer_weight.lm_head_weight_(input=normed, alloc_func=self.alloc_tensor)
        normed = None

        vocab_size = layer_weight.lm_head_weight_.vocab_size
        if self.tp_world_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = self.alloc_tensor((vocab_size, token_num), dtype=hidden.dtype)
            split_indexes = np.linspace(0, vocab_size, self.tp_world_size_ + 1, dtype=np.int64)
            all_gather(
                [gather_data[split_indexes[i] : split_indexes[i + 1], :] for i in range(self.tp_world_size_)],
                logic_batch,
                group=infer_state.dist_group,
                async_op=False,
            )
        logic_batch = None

        ans_logics = self.alloc_tensor((token_num, vocab_size), dtype=torch.float32)
        ans_logics[:, :] = gather_data.permute(1, 0)
        gather_data = None
        return ans_logics

    def token_forward(
        self, input_embdings: torch.Tensor, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight
    ):

        return self._token_forward(input_embdings=input_embdings, infer_state=infer_state, layer_weight=layer_weight)

    def overlap_tpsp_token_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: BaseLayerWeight,
    ):

        logics = self.token_forward(input_embdings, infer_state, layer_weight=layer_weight)

        logics1 = self.token_forward(input_embdings1, infer_state1, layer_weight=layer_weight)

        return logics, logics1
