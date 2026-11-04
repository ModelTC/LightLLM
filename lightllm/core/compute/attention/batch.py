from dataclasses import dataclass, fields, replace
import torch
from lightllm.core.contracts.device import DeviceEvent


@dataclass
class AttentionBatch:
    batch_size: int
    max_q_len: int
    max_kv_len: int
    microbatch_index: int
    req_to_tokens: torch.Tensor
    req_indices: torch.Tensor
    seq_lengths: torch.Tensor
    q_starts: torch.Tensor | None
    ready_lengths: torch.Tensor | None
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    page_table: torch.Tensor | None = None
    total_kv_tokens: int | None = None
    total_q_tokens: int | None = None
    hold_request_id: int | None = None
    shared_groups: torch.Tensor | None = None
    mtp_indices: torch.Tensor | None = None
    host_seq_lengths: torch.Tensor | None = None
    host_cu_q: torch.Tensor | None = None
    host_cu_k: torch.Tensor | None = None
    host_ready: DeviceEvent | None = None

    def copy_for_graph(self, new):
        if self.batch_size != new.batch_size or self.microbatch_index != new.microbatch_index:
            raise ValueError("decode Graph metadata must retain its batch size and execution lane")
        if new.host_ready is not None:
            new.host_ready.synchronize()
        for field in fields(self):
            old, value = getattr(self, field.name), getattr(new, field.name)
            if isinstance(value, torch.Tensor):
                if old is None or old.shape != value.shape or old.dtype != value.dtype:
                    raise ValueError(f"Graph metadata shape/dtype changed: {field.name}")
                if old.data_ptr() != value.data_ptr():
                    old.copy_(value, non_blocking=True)


def group_verify_batch(batch, max_tokens):
    """Coalesce consecutive query rows while keeping Graph metadata shapes fixed."""
    from lightllm.common.basemodel.triton_kernel.mtp_utils import build_mtp_shared_group_markers
    from lightllm.common.basemodel.triton_kernel.fa3_utils import build_dynamic_spec_fa3_decode_params
    from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_cumsum_pad0_tensor

    if batch.hold_request_id is None:
        raise ValueError("verification requires an explicit padding request identity")
    groups = build_mtp_shared_group_markers(batch.req_indices, batch.hold_request_id)
    q_lengths, kv_lengths, requests, lengths = build_dynamic_spec_fa3_decode_params(
        batch.req_indices, batch.seq_lengths, groups, batch.batch_size, batch.hold_request_id
    )
    cu_q, cu_k = gen_cumsum_pad0_tensor(q_lengths, kv_lengths)
    return replace(
        batch,
        req_indices=requests,
        seq_lengths=lengths,
        cu_q=cu_q.int(),
        cu_k=cu_k.int(),
        q_starts=cu_q[:-1].int(),
        ready_lengths=lengths - q_lengths,
        max_q_len=max_tokens,
    )
