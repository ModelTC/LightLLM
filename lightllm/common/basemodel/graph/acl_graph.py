import torch
from dataclasses import dataclass, field
from lightllm.common.basemodel.attention.paged_fa3.graph_utils import sync_attn_params
from lightllm.common.basemodel.graph.base.decode_graph import DecodeGraph, register_decode_graph
from lightllm.common.basemodel.infer_struct import InferStateInfo
from typing import Any, Optional


class SeqLenManager:

    def __init__(self, max_batch: int):
        self.max_batch = max_batch

        self.b1_cu_q_seq_len_cpu = torch.empty(max_batch, dtype=torch.int32, device='cpu', pin_memory=True)
        self.b_cu_kv_seq_len_cpu = torch.empty(max_batch, dtype=torch.int32, device='cpu', pin_memory=True)

        self.n_q = -1
        self.n_kv = -1

    def update(self, b1_cu_q_seq_len: torch.Tensor, b_cu_kv_seq_len: torch.Tensor):
        n_q = b1_cu_q_seq_len.numel() - 1
        n_kv = b_cu_kv_seq_len.numel()

        self.b1_cu_q_seq_len_cpu[:n_q].copy_(b1_cu_q_seq_len[1:], non_blocking=False)
        self.b_cu_kv_seq_len_cpu[:n_kv].copy_(b_cu_kv_seq_len, non_blocking=False)

        self.n_q = n_q
        self.n_kv = n_kv

    def get_tensor_slices(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.b1_cu_q_seq_len_cpu[:self.n_q], self.b_cu_kv_seq_len_cpu[:self.n_kv]


@register_decode_graph("ascend")
class AclGraph(DecodeGraph):

    def _init_decode_graph_extra(self):
        init_attn_params(self.graph_batch_sizes)
        self.update_stream = torch.npu.Stream()

    def _sync_attn(self, batch_size: int, *graph_states: InferStateInfo):
        sync_attn_params(
            batch_size=batch_size,
            seqlens_by_microbatch_index=tuple(
                (st.b1_cu_q_seq_len_cpu, st.b_cu_kv_seq_len_cpu) for st in graph_states
            ),
            update_stream=self.update_stream,
        )

    def _replay(self, infer_state: InferStateInfo):
        # Wait for previous step's graph_task_update to complete before overwrite.
        self.update_stream.synchronize()

        batch_size = infer_state.input_ids.shape[0]
        graph_obj, graph_infer_state, graph_output = self.graph[batch_size]
        graph_infer_state.copy_for_cuda_graph(infer_state)
        self.platform_backend.graph.replay_graph(graph_obj)
        self._sync_attn(batch_size, graph_infer_state)
        return graph_output

    def _replay_overlap(self, infer_state: InferStateInfo, infer_state1: InferStateInfo):
        self.update_stream.synchronize()

        batch_size = infer_state.input_ids.shape[0]
        (
            graph_obj,
            graph_infer_state,
            graph_infer_state1,
            graph_model_output,
            graph_model_output1,
        ) = self.graph[batch_size]
        graph_infer_state.copy_for_cuda_graph(infer_state)
        graph_infer_state1.copy_for_cuda_graph(infer_state1)
        self.platform_backend.graph.replay_graph(graph_obj)
        self._sync_attn(batch_size, graph_infer_state, graph_infer_state1)
        return graph_model_output, graph_model_output1


# Adapted from: https://github.com/vllm-project/vllm-ascend/blob/v0.11.0/vllm_ascend/compilation/acl_graph.py
@dataclass
class AclGraphParams:
    # handles/events/attn_params[batch_size][microbatch_index]
    handles: dict[int, dict[int, list[Any]]] = field(default_factory=dict)
    events: dict[int, dict[int, list[Any]]] = field(default_factory=dict)
    workspaces: dict[int, Any] = field(default_factory=dict)
    attn_params: dict[int, dict[int, list[tuple]]] = field(default_factory=dict)


ATTN_PARAMS: Optional[AclGraphParams] = None


def _microbatch_buckets():
    return {0: [], 1: []}


def init_attn_params(batch_sizes: list[int]):
    global ATTN_PARAMS
    ATTN_PARAMS = AclGraphParams(
        handles={bs: _microbatch_buckets() for bs in batch_sizes},
        events={bs: _microbatch_buckets() for bs in batch_sizes},
        workspaces={bs: None for bs in batch_sizes},
        attn_params={bs: _microbatch_buckets() for bs in batch_sizes},
    )


def get_attn_params():
    return ATTN_PARAMS


def add_attn_params(
    batch_size: int,
    event: Any,
    handle: Any,
    attn_params: tuple,
    microbatch_index: int = 0,
):
    global ATTN_PARAMS
    if ATTN_PARAMS is not None:
        ATTN_PARAMS.handles[batch_size][microbatch_index].append(handle)
        ATTN_PARAMS.events[batch_size][microbatch_index].append(event)
        ATTN_PARAMS.attn_params[batch_size][microbatch_index].append(attn_params)
