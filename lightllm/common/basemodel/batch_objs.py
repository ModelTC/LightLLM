import copy
from dataclasses import dataclass
from typing import List, Optional

import torch

from lightllm.utils.tensor_utils import tensor_to_no_ref_tensor


@dataclass
class ModelInput:
    # 通用变量
    batch_size: int
    total_token_num: int
    # 在 decode 阶段， max_q_seq_len 必定是 1，
    max_q_seq_len: int
    max_kv_seq_len: int
    max_cache_len: int = None
    input_ids: torch.Tensor = None
    b_req_idx: torch.Tensor = None
    b_mtp_index: torch.Tensor = None
    b_seq_len: torch.Tensor = None
    # Original prompt length per request/row.  This remains on CPU and is used
    # only for one-time model-specific fast-path eligibility decisions.
    b_input_len: torch.Tensor = None
    # 在 prefill 阶段，用于在 enable_prefill_decode_mixed 开启下，
    # 用于标识请求是否为 decode 请求混合在 prefill 请求中。
    # 其对应的 input_ids 需要特殊处理, 从 req_to_next_token_ids 中获取。
    # 该字段是 prefill 必填参数；普通 prefill 和 draft prompt prefill 填全 False。
    b_is_decode_req: torch.Tensor = None

    # Decode 逐行携带的 radix cache 共享长度。普通 attention backend 不使用该
    # 信息，diverse attention backend 会结合 b_shared_radix_node_id 构建共享组。
    b_shared_seq_len: torch.Tensor = None
    # Decode 逐行携带的 radix node 标识。相同 id 表示请求引用同一个共享
    # radix node；该 id 只用于重建 diverse attention 的 b_mark_shared_group。
    b_shared_radix_node_id: torch.Tensor = None
    mem_indexes: torch.Tensor = None
    is_prefill: bool = False
    b_ready_cache_len: torch.Tensor = None
    # Request/row-aligned MRoPE position offset. It is decode-only; prefill
    # builds positions directly from the complete prompt layout.
    # Row-aligned decode input transforms must preserve the tensor unchanged.
    b_position_delta: torch.Tensor = None
    b_prefill_start_loc: torch.Tensor = None
    multimodal_params: list = None
    # cpu 变量
    mem_indexes_cpu: torch.Tensor = None
    # prefill 阶段使用的参数，但是不是推理过程使用的参数，是推理外部进行资源管理
    # 的一些变量
    # 标记 prefill 请求是否会在本轮产生输出。Prefill 必填（空 batch 使用空 list），decode 不使用。
    b_prefill_has_output_cpu: List[bool] = None

    # 专有变量，用于一些特殊的模型，特殊的模式下, 传递一些特殊
    # 的输入变量。只在特殊的模型模式下才会具体使用和生效。

    # mtp_draft_input_hiddens 用于模型 mtp 模式下
    # 的 draft 模型的输入
    mtp_draft_input_hiddens: Optional[torch.Tensor] = None

    # Whether every request in this prefill batch starts and ends on an
    # index-kpool boundary.  BaseModel computes this once while the length
    # tensors are still on CPU; NSA layers reuse it without a per-layer GPU
    # synchronization.
    kpool_prefill_aligned: bool = False
    kpool_decode_aligned: bool = False

    # The router enables sparse vocabulary output only when target sampling is
    # exact, unmodified greedy. Draft models always consume greedy proposals.
    use_vocab_parallel_greedy: bool = False

    def to_cuda(self):
        self.check_input()

        # Prefill 和 decode 都必须提供的公共张量。
        if self.mem_indexes is None:
            self.mem_indexes = self.mem_indexes_cpu.cuda(non_blocking=True)
        self.b_req_idx = self.b_req_idx.cuda(non_blocking=True)
        self.b_seq_len = self.b_seq_len.cuda(non_blocking=True)
        self.b_mtp_index = self.b_mtp_index.cuda(non_blocking=True)

        if self.is_prefill:
            # Prefill 必须提供的张量。
            self.input_ids = self.input_ids.cuda(non_blocking=True)
            self.b_ready_cache_len = self.b_ready_cache_len.cuda(non_blocking=True)
            self.b_prefill_start_loc = self.b_prefill_start_loc.cuda(non_blocking=True)
            self.b_is_decode_req = self.b_is_decode_req.cuda(non_blocking=True)
        else:
            # Decode 必须提供的张量。
            self.b_position_delta = self.b_position_delta.cuda(non_blocking=True)
            self.b_shared_seq_len = self.b_shared_seq_len.cuda(non_blocking=True)
            self.b_shared_radix_node_id = self.b_shared_radix_node_id.cuda(non_blocking=True)

            # Decode 可以显式提供 input_ids；未提供时会在模型内部按请求索引收集。
            if self.input_ids is not None:
                self.input_ids = self.input_ids.cuda(non_blocking=True)

    def __post_init__(self):
        self.check_input()

    def check_input(self):
        if self.b_input_len is None:
            # Internal warmup/padding inputs predate this metadata.  Their
            # synthetic prompt occupies the whole sequence, so b_seq_len is
            # the correct conservative default.
            self.b_input_len = self.b_seq_len
        if self.input_ids is not None:
            assert (
                self.input_ids.dtype == torch.int64
            ), f"model input_ids must use torch.int64, got {self.input_ids.dtype}"

        assert self.b_req_idx is not None
        assert self.b_mtp_index is not None
        assert self.b_seq_len is not None
        assert self.b_input_len is not None
        assert self.multimodal_params is not None
        assert self.mem_indexes is not None or self.mem_indexes_cpu is not None

        assert self.b_req_idx.shape == (self.batch_size,)
        assert self.b_mtp_index.shape == self.b_req_idx.shape
        assert self.b_seq_len.shape == self.b_req_idx.shape
        assert self.b_input_len.shape == self.b_req_idx.shape
        assert len(self.multimodal_params) == self.batch_size

        if self.is_prefill:
            assert self.input_ids is not None
            assert self.max_cache_len is not None
            assert self.b_ready_cache_len is not None
            assert self.b_prefill_start_loc is not None
            assert self.b_is_decode_req is not None
            assert self.b_ready_cache_len.shape == self.b_req_idx.shape
            assert self.b_prefill_start_loc.shape == self.b_req_idx.shape
            assert self.b_is_decode_req.shape == self.b_req_idx.shape
            assert self.b_is_decode_req.dtype == torch.bool
            assert self.b_position_delta is None, "prefill must not provide b_position_delta"
            assert self.b_prefill_has_output_cpu is not None, "prefill must provide b_prefill_has_output_cpu"
            assert len(self.b_prefill_has_output_cpu) == self.batch_size
        else:
            assert self.max_q_seq_len == 1
            assert self.b_position_delta is not None
            assert self.b_shared_seq_len is not None
            assert self.b_shared_radix_node_id is not None
            assert self.b_position_delta.shape == self.b_req_idx.shape
            assert self.b_shared_seq_len.shape == self.b_req_idx.shape
            assert self.b_shared_radix_node_id.shape == self.b_req_idx.shape

        mem_indexes = self.mem_indexes if self.mem_indexes is not None else self.mem_indexes_cpu
        assert mem_indexes.ndim == 1


@dataclass
class ModelMtpOutputCollector:
    """保存一次模型 forward 为 MTP 推理产生的可选输出。"""

    # MTP drafter 使用的 hidden 特征。
    # - Vanilla MTP、EAGLE 主模型收集最终层 hidden；对应 draft 模型也会返回最终层 hidden，
    #   供串行的下一层 MTP 模块或下一步自回归 draft 使用。
    # - EAGLE3、DFlash、DSpark 主模型收集 checkpoint 配置指定的若干 target layer hidden，
    #   拼接后交给 draft 模型；EAGLE3 draft 模型还会返回最终层 hidden。
    # - DFlash、DSpark 的 block draft 模型不需要返回 hidden，因此该字段为 None。
    # - 未启用 MTP 时不收集投机特征，该字段同样为 None。
    spec_hidden: Optional[torch.Tensor] = None

    # Draft head 直接生成的 token id。Block drafter 通常返回
    # [request_count * block_size]，autoregressive drafter 通常返回
    # [request_count]。未提供时，调用方从普通 logits 执行 argmax。
    draft_token_ids: Optional[torch.Tensor] = None

    # 与 draft_token_ids 一一对应的精确最大 token 概率。Vocab-parallel
    # draft head 可直接返回该值，避免为了动态 verify 聚合完整词表。
    draft_token_probs: Optional[torch.Tensor] = None

    # DSpark confidence head 输出的原始置信度 logits，形状通常为
    # [request_count, block_size]，供动态 MTP verify 计算各 draft 位置的调度分数。
    # - 仅 DSpark checkpoint 启用 confidence head 时返回；动态 verify 模式要求该字段存在。
    # - 固定 verify 且未启用 confidence head 的 DSpark 模型可以返回 None。
    # - Vanilla MTP、EAGLE、EAGLE3、DFlash 以及未启用 MTP 的模型均不使用该字段。
    confidence_logits: Optional[torch.Tensor] = None

    def to_no_ref_tensor(self) -> None:
        if self.spec_hidden is not None:
            self.spec_hidden = tensor_to_no_ref_tensor(self.spec_hidden)
        if self.draft_token_ids is not None:
            self.draft_token_ids = tensor_to_no_ref_tensor(self.draft_token_ids)
        if self.draft_token_probs is not None:
            self.draft_token_probs = tensor_to_no_ref_tensor(self.draft_token_probs)
        if self.confidence_logits is not None:
            self.confidence_logits = tensor_to_no_ref_tensor(self.confidence_logits)

    def unpad_decode(self, padded_batch_size: int, origin_batch_size: int) -> "ModelMtpOutputCollector":
        collector = copy.copy(self)
        if collector.spec_hidden is not None:
            collector.spec_hidden = collector.spec_hidden[:origin_batch_size]
        if collector.draft_token_ids is not None:
            collector.draft_token_ids = collector.draft_token_ids[:origin_batch_size]
        if collector.draft_token_probs is not None:
            collector.draft_token_probs = collector.draft_token_probs[:origin_batch_size]
        if collector.confidence_logits is not None:
            confidence_row_count = collector.confidence_logits.shape[0]
            assert confidence_row_count > 0 and padded_batch_size % confidence_row_count == 0
            rows_per_confidence = padded_batch_size // confidence_row_count
            assert origin_batch_size % rows_per_confidence == 0
            collector.confidence_logits = collector.confidence_logits[: origin_batch_size // rows_per_confidence]
        return collector

    def unpad_prefill(self, origin_handle_token_num: int) -> "ModelMtpOutputCollector":
        collector = copy.copy(self)
        if collector.spec_hidden is not None:
            collector.spec_hidden = collector.spec_hidden[:origin_handle_token_num]
        return collector


@dataclass
class ModelOutput:
    # 通用变量
    logits: torch.Tensor
    # MTP collector is finalized by HiddenCollector.finish_output before being
    # attached here. ModelOutput therefore owns a stable output view instead
    # of the mutable collector used while the forward is still running.
    mtp_collector: Optional[ModelMtpOutputCollector] = None
    # 用于判断 mem_indexes 是否成功写入 req manager 中的事件对象。
    prefill_mem_indexes_ready_event: torch.Event = None

    # prompt_logics 用于在开启 return_all_prompt_logics 模式（如 enable_prompt_logprobs）时，
    # 保存整个 prefill 阶段每一个 token 位置对应的 logits（而非仅最后一个位置的 logits）。
    # 此时 logits 依然只保存每个请求最后一个位置的 logits，prompt_logics 为可选项，仅在
    # 需要返回 prompt logprobs 信息时才会非空。
    prompt_logics: Optional[torch.Tensor] = None

    # Vocab-parallel outputs keep logits as logits while mapping each sparse
    # column back to its global token id. logits_logsumexp is computed over the
    # complete vocabulary, so sparse argmax probabilities remain exact.
    # Both fields are None for historical dense logits.
    logits_token_ids: Optional[torch.Tensor] = None
    logits_logsumexp: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if self.mtp_collector is None:
            self.mtp_collector = ModelMtpOutputCollector()
        assert (self.logits_token_ids is None) == (self.logits_logsumexp is None)
        if self.logits_token_ids is not None:
            assert self.logits.ndim == 2
            assert self.logits_token_ids.shape == self.logits.shape
            assert self.logits_token_ids.dtype in (torch.int32, torch.int64)
            assert self.logits_token_ids.device == self.logits.device
            assert self.logits_logsumexp.shape == (self.logits.shape[0],)
            assert self.logits_logsumexp.dtype == torch.float32
            assert self.logits_logsumexp.device == self.logits.device

    def to_no_ref_tensor(self):
        self.logits = tensor_to_no_ref_tensor(self.logits)
        if self.logits_token_ids is not None:
            self.logits_token_ids = tensor_to_no_ref_tensor(self.logits_token_ids)
            self.logits_logsumexp = tensor_to_no_ref_tensor(self.logits_logsumexp)
        self.mtp_collector.to_no_ref_tensor()

    @property
    def has_vocab_parallel_logits(self) -> bool:
        return self.logits_token_ids is not None

    def index_select_logits_rows(self, index: torch.Tensor) -> "ModelOutput":
        """Select logit rows without dropping their vocabulary metadata."""

        return ModelOutput(
            logits=self.logits.index_select(0, index),
            logits_token_ids=(
                self.logits_token_ids.index_select(0, index) if self.logits_token_ids is not None else None
            ),
            logits_logsumexp=(
                self.logits_logsumexp.index_select(0, index) if self.logits_logsumexp is not None else None
            ),
        )

    @classmethod
    def concat_logits_rows(cls, outputs: List["ModelOutput"]) -> "ModelOutput":
        """Concatenate compatible dense or sparse-vocabulary logit rows."""

        assert outputs
        sparse = outputs[0].has_vocab_parallel_logits
        assert all(output.has_vocab_parallel_logits == sparse for output in outputs)
        return cls(
            logits=torch.cat([output.logits for output in outputs], dim=0),
            logits_token_ids=(torch.cat([output.logits_token_ids for output in outputs], dim=0) if sparse else None),
            logits_logsumexp=(torch.cat([output.logits_logsumexp for output in outputs], dim=0) if sparse else None),
        )
