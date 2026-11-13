from dataclasses import dataclass, field, fields
from typing import Optional

import torch

from .spec import AttentionPhase


@dataclass(frozen=True)
class AttentionBatch:
    """一次执行的批次描述，tensor 借用自 InferStateInfo / ReqManager。

    b_* 的首维为 batch_size，b1_cu_* 是带起始 0 的 batch_size + 1 个累积偏移。
    token_num 是当前 query token 数；total_token_num 包含历史 KV 的逻辑 token 数，
    捕获 decode Graph 时后者可替换为容量上界。frozen 只固定字段引用，tensor 内容仍可更新。
    """

    phase: AttentionPhase
    batch_size: int
    token_num: int
    total_token_num: int
    max_q_seq_len: int
    max_kv_seq_len: int
    max_cache_len: int
    b_req_idx: torch.Tensor
    # KV 长度包含当前输入；query 长度只包含本次实际计算的 token。
    b_seq_len: torch.Tensor
    b_q_seq_len: torch.Tensor
    b1_cu_q_seq_len: torch.Tensor
    b1_cu_kv_seq_len: torch.Tensor
    req_to_token_indexs: torch.Tensor = field(metadata={"shared_storage": True})
    hold_req_id: int
    b_ready_cache_len: Optional[torch.Tensor] = None
    b_mtp_index: Optional[torch.Tensor] = None
    b_shared_seq_len: Optional[torch.Tensor] = None
    b_shared_radix_node_id: Optional[torch.Tensor] = None
    mem_index: Optional[torch.Tensor] = None
    # Absolute positions of the query rows in the request KV sequence.  Sparse
    # MLA indexers use this to expand per-request metadata without reaching
    # back into InferStateInfo.
    position_ids: Optional[torch.Tensor] = None
    b_image_token_end: Optional[torch.Tensor] = None

    @property
    def device(self):
        return self.b_req_idx.device

    @property
    def is_prefill(self):
        return self.phase is AttentionPhase.PREFILL

    @property
    def b_q_start_loc(self):
        return self.b1_cu_q_seq_len[:-1]

    @property
    def b_kv_start_loc(self):
        return self.b1_cu_kv_seq_len[:-1]

    def validate(self):
        if self.batch_size < 0 or self.token_num < 0:
            raise ValueError("attention batch sizes must be nonnegative")
        for name in ("b_req_idx", "b_seq_len", "b_q_seq_len"):
            tensor = getattr(self, name)
            if tensor.ndim != 1 or tensor.numel() != self.batch_size:
                raise ValueError(f"{name} must have batch_size elements")
        if self.b1_cu_q_seq_len.numel() != self.batch_size + 1:
            raise ValueError("query offsets must have batch_size + 1 elements")

    def copy_inputs(self, source: "AttentionBatch") -> None:
        """向捕获时的 tensor 原址拷入新批次，标量字段保留捕获配置。

        先校验所有 tensor 再写入，避免校验失败留下半更新的输入。
        request table 是共享存储，必须保持原地址，不能通过整表拷贝替换。
        """
        for dst, src in self.input_copies(source):
            dst.copy_(src, non_blocking=True)

    def input_copies(self, source):
        if (self.phase, self.batch_size, self.token_num) != (source.phase, source.batch_size, source.token_num):
            raise ValueError("attention batch does not match the captured shape")
        copies = []
        for item in fields(self):
            dst, src = getattr(self, item.name), getattr(source, item.name)
            if dst is src:
                continue
            if isinstance(dst, torch.Tensor) or isinstance(src, torch.Tensor):
                if not isinstance(dst, torch.Tensor) or not isinstance(src, torch.Tensor):
                    raise ValueError(f"captured input {item.name} changed presence")
                if (dst.shape, dst.dtype, dst.device) != (src.shape, src.dtype, src.device):
                    raise ValueError(f"captured input {item.name} changed shape, dtype or device")
                if item.metadata.get("shared_storage") and dst.data_ptr() != src.data_ptr():
                    raise ValueError("request table storage changed; recapture attention")
                if dst.data_ptr() != src.data_ptr():
                    copies.append((dst, src))
        return copies


@dataclass(frozen=True, kw_only=True)
class ContextCapacity:
    """context 的形状契约：batch_size / token_num 要与输入批次逐项一致，序列长度只检查上界。

    runner 负责选择 Graph bucket 和 padding；backend 不自行改变批次形状。
    """

    phase: AttentionPhase
    batch_size: int
    token_num: int
    max_kv_seq_len: int
    max_q_seq_len: int

    def check(self, batch: AttentionBatch):
        if self.phase is not batch.phase or self.batch_size != batch.batch_size or self.token_num != batch.token_num:
            raise ValueError("attention context capacity does not match the batch shape")
        if batch.max_kv_seq_len > self.max_kv_seq_len or batch.max_q_seq_len > self.max_q_seq_len:
            raise ValueError("attention batch exceeds context sequence capacity")
