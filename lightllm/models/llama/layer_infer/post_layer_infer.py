import os
import torch
from lightllm.common.basemodel.batch_objs import PostLayerOutput
import torch.functional as F
import torch.distributed as dist
import numpy as np
from lightllm.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import PostLayerInferTpl
from lightllm.distributed.communication_op import all_gather, all_gather_into_tensor
from lightllm.common.basemodel.triton_kernel.local_vocab_topk import local_vocab_topk
from lightllm.common.basemodel.triton_kernel.pack_vocab_parallel_topk import (
    pack_vocab_parallel_topk,
    unpack_vocab_parallel_topk,
)
from lightllm.utils.envs_utils import get_env_start_args

_MASKED_LOGIT_VALUE = -10000000.0


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
    ) -> PostLayerOutput:
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings = None

        # 正常采样使用的 logits，始终只对应每个请求最后一个位置。
        if infer_state.is_draft_model:
            post_output = self._draft_lm_head_and_gather(last_input, token_num, layer_weight, infer_state)
        else:
            post_output = self._target_lm_head_and_gather(last_input, token_num, layer_weight, infer_state)
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
            ).logits

        return post_output

    def _project_local_logits(
        self,
        hidden: torch.Tensor,
        token_num: int,
        layer_weight: LlamaPreAndPostLayerWeight,
        infer_state: LlamaInferStateInfo,
    ) -> torch.Tensor:
        normed = self._norm(hidden, infer_state, layer_weight)
        normed = normed.permute(1, 0).view(-1, token_num)
        local_logits = layer_weight.lm_head_weight_(input=normed, alloc_func=self.alloc_tensor)
        return local_logits

    def _lm_head_and_gather(
        self,
        hidden: torch.Tensor,
        token_num: int,
        layer_weight: LlamaPreAndPostLayerWeight,
        infer_state: LlamaInferStateInfo,
    ) -> PostLayerOutput:
        """执行原始 lm-head，并收集完整词表 logits。"""

        local_logits = self._project_local_logits(hidden, token_num, layer_weight, infer_state)

        vocab_size = layer_weight.lm_head_weight_.vocab_size
        if self.tp_world_size_ == 1:
            gather_data = local_logits
        else:
            gather_data = self.alloc_tensor((vocab_size, token_num), dtype=hidden.dtype)
            split_indexes = np.linspace(0, vocab_size, self.tp_world_size_ + 1, dtype=np.int64)
            all_gather(
                [gather_data[split_indexes[i] : split_indexes[i + 1], :] for i in range(self.tp_world_size_)],
                local_logits,
                group=infer_state.dist_group,
                async_op=False,
            )
        local_logits = None

        ans_logics = self.alloc_tensor((token_num, vocab_size), dtype=torch.float32)
        ans_logics[:, :] = gather_data.permute(1, 0)
        gather_data = None
        return PostLayerOutput(logits=ans_logics)

    def _vocab_parallel_topk(
        self,
        hidden: torch.Tensor,
        token_num: int,
        layer_weight: LlamaPreAndPostLayerWeight,
        infer_state: LlamaInferStateInfo,
        top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """使用 PyTorch 在各 TP 词表分片上选取并合并全局 top-k。"""

        local_logits = self._project_local_logits(hidden, token_num, layer_weight, infer_state)
        vocab_size = layer_weight.lm_head_weight_.vocab_size
        candidate_count = min(top_k, vocab_size)
        local_vocab_size = local_logits.shape[0]
        assert vocab_size <= torch.iinfo(torch.int32).max, f"vocabulary size {vocab_size} exceeds int32 token ID range"
        assert (
            local_vocab_size >= candidate_count
        ), f"local vocabulary size {local_vocab_size} must be at least the candidate count {candidate_count}"

        # 在本地词表上筛选候选；大批量 BF16 输入先分块转置，提高词表扫描效率。
        local_values, local_token_ids = local_vocab_topk(
            local_logits,
            top_k=candidate_count,
            alloc_func=self.alloc_tensor,
        )
        if self.tp_world_size_ == 1:
            # 分数转为 FP32；将回退路径的转置视图连续化，保证 token ID 的 [B, K] 输出连续。
            return local_values.float(), local_token_ids.contiguous()

        packed_candidates = self.alloc_tensor(
            (token_num, candidate_count * 2),
            dtype=torch.float32,
        )
        pack_vocab_parallel_topk(
            local_values,
            local_token_ids,
            layer_weight.lm_head_weight_.tp_vocab_start_id,
            packed_candidates,
        )
        local_values = None
        local_token_ids = None
        local_logits = None

        gathered_candidates = self.alloc_tensor(
            (self.tp_world_size_ * token_num, candidate_count * 2),
            dtype=torch.float32,
        )
        all_gather_into_tensor(gathered_candidates, packed_candidates, group=infer_state.dist_group)
        packed_candidates = None

        gathered_values = self.alloc_tensor(
            (token_num, self.tp_world_size_ * candidate_count),
            dtype=torch.float32,
        )
        gathered_token_ids = self.alloc_tensor(
            (token_num, self.tp_world_size_ * candidate_count),
            dtype=torch.int64,
        )
        unpack_vocab_parallel_topk(
            gathered_candidates,
            gathered_values,
            gathered_token_ids,
            candidate_count,
            self.tp_world_size_,
        )
        gathered_candidates = None
        return gathered_values, gathered_token_ids

    def _target_lm_head_and_gather(
        self,
        hidden: torch.Tensor,
        token_num: int,
        layer_weight: LlamaPreAndPostLayerWeight,
        infer_state: LlamaInferStateInfo,
    ) -> PostLayerOutput:
        """按 target 配置减少词表通信，并重建完整词表 logits。"""

        top_k = get_env_start_args().target_vocab_topk_sampling
        if top_k is None:
            return self._lm_head_and_gather(hidden, token_num, layer_weight, infer_state)

        candidate_logits, candidate_token_ids = self._vocab_parallel_topk(
            hidden, token_num, layer_weight, infer_state, top_k
        )
        logits = self.alloc_tensor(
            (token_num, layer_weight.lm_head_weight_.vocab_size),
            dtype=torch.float32,
        )
        # 非候选分数为 -1e7，而 Outlines、XGrammar 和首 token 约束将禁止的 token 设为 -1e6。
        # 若合法集合与候选集合没有交集，禁止的 token 反而得分更高；启动入口已断言禁止该组合。
        # TODO: 支持先应用输出约束再筛选候选，或为受约束请求回退到完整词表。
        # 仅修改占位分数无法恢复被裁掉的合法 token 原始分数，也必须处理合法候选为空的情况。
        logits.fill_(_MASKED_LOGIT_VALUE)
        logits.scatter_(dim=1, index=candidate_token_ids, src=candidate_logits)
        candidate_logits = None
        candidate_token_ids = None
        return PostLayerOutput(logits=logits)

    def _draft_lm_head_and_gather(
        self,
        hidden: torch.Tensor,
        token_num: int,
        layer_weight: LlamaPreAndPostLayerWeight,
        infer_state: LlamaInferStateInfo,
    ) -> PostLayerOutput:
        """按 draft 配置返回候选 logits，未开启时回退到完整词表。"""

        top_k = get_env_start_args().draft_vocab_topk_sampling
        if top_k is None:
            return self._lm_head_and_gather(hidden, token_num, layer_weight, infer_state)

        logits, token_ids = self._vocab_parallel_topk(hidden, token_num, layer_weight, infer_state, top_k)
        return PostLayerOutput(logits=logits, logits_token_ids=token_ids)

    def token_forward(
        self, input_embdings: torch.Tensor, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight
    ) -> PostLayerOutput:

        return self._token_forward(input_embdings=input_embdings, infer_state=infer_state, layer_weight=layer_weight)

    def overlap_tpsp_token_forward(
        self,
        input_embdings: torch.Tensor,
        input_embdings1: torch.Tensor,
        infer_state: LlamaInferStateInfo,
        infer_state1: LlamaInferStateInfo,
        layer_weight: BaseLayerWeight,
    ) -> tuple[PostLayerOutput, PostLayerOutput]:

        output = self.token_forward(input_embdings, infer_state, layer_weight=layer_weight)

        output1 = self.token_forward(input_embdings1, infer_state1, layer_weight=layer_weight)

        return output, output1
