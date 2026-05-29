import torch
from dataclasses import dataclass, field
from lightllm.common.basemodel.attention.paged_fa3.fp import update_attn_params
from lightllm.common.basemodel.batch_objs import ModelOutput
from lightllm.common.basemodel.graph.base.decode_graph import DecodeGraph
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.utils.log_utils import init_logger
from typing import Any, Optional

logger = init_logger(__name__)


class AclGraph(DecodeGraph):

    def _init_decode_graph_extra(self):
        init_attn_params(self.graph_batch_sizes)
        self.update_stream = torch.npu.Stream()

    @property
    def acl_graph_batch_sizes(self):
        return self.graph_batch_sizes

    def _replay(
        self,
        infer_state: InferStateInfo,
        b1_cu_q_seq_len_cpu: list[int],
        b_cu_kv_seq_len_cpu: list[int],
    ) -> ModelOutput:
        graph_output = super()._replay(infer_state, b1_cu_q_seq_len_cpu, b_cu_kv_seq_len_cpu)
        batch_size = infer_state.input_ids.shape[0]
        update_attn_params(
            batch_size,
            b1_cu_q_seq_len_cpu,
            b_cu_kv_seq_len_cpu.add_(1),
            self.update_stream,
        )
        return graph_output

    def replay(
        self,
        infer_state: InferStateInfo,
        b1_cu_q_seq_len_cpu: list[int],
        b_cu_kv_seq_len_cpu: list[int],
        infer_state1: Optional[InferStateInfo] = None,
    ):
        if self.enable_decode_microbatch_overlap:
            return self._replay_overlap(infer_state, infer_state1)
        assert infer_state1 is None
        return self._replay(infer_state, b1_cu_q_seq_len_cpu, b_cu_kv_seq_len_cpu)


# Adapted from: https://github.com/vllm-project/vllm-ascend/blob/v0.11.0/vllm_ascend/compilation/acl_graph.py
@dataclass
class AclGraphParams:
    handles: dict[int, list[Any]] = field(default_factory=dict)
    events: dict[int, list[torch.npu.ExternalEvent]] = field(default_factory=dict)
    workspaces: dict[int, Any] = field(default_factory=dict)
    attn_params: dict[int, list[tuple]] = field(default_factory=dict)


ATTN_PARAMS: Optional[AclGraphParams] = None


def init_attn_params(batch_sizes: list[int]):
    global ATTN_PARAMS
    ATTN_PARAMS = AclGraphParams(
        handles={bs: [] for bs in batch_sizes},
        events={bs: [] for bs in batch_sizes},
        workspaces={bs: None for bs in batch_sizes},
        attn_params={bs: [] for bs in batch_sizes},
    )


def get_attn_params():
    return ATTN_PARAMS


def add_attn_params(batch_size: int, event: torch.npu.ExternalEvent, handle: Any, attn_params: tuple):
    global ATTN_PARAMS
    if ATTN_PARAMS is not None:
        ATTN_PARAMS.handles[batch_size].append(handle)
        ATTN_PARAMS.events[batch_size].append(event)
        ATTN_PARAMS.attn_params[batch_size].append(attn_params)
