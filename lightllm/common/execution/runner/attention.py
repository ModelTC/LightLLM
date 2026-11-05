from __future__ import annotations
from typing import Hashable, Mapping, TYPE_CHECKING
from lightllm.common.execution.compute.attention.backend import BoundAttention, AttentionMetadata
from lightllm.common.execution.contracts.device import DeviceRuntime, GraphBinding
from lightllm.common.execution.contracts.provider import Phase

if TYPE_CHECKING:
    from lightllm.common.execution.compute.attention.batch import AttentionBatch
    from lightllm.common.execution.planner.plan import BoundAttentionPlan

from lightllm.common.execution.compute.attention.backend import GraphAttention


class AttentionMetadataRunner:
    def __init__(self, plan: BoundAttentionPlan, decode_phase: Phase = "decode"):
        self.plan = plan
        self.decode_phase = decode_phase
        self.metadata_sources: dict[Hashable, BoundAttention] = {}
        for item in plan.bindings:
            self.metadata_sources.setdefault(item.execution.metadata_key, item.execution)

    def prepare_batch(self, batch: AttentionBatch, *, phase: Phase) -> dict[Hashable, AttentionMetadata]:
        executions = [execution for execution in self.metadata_sources.values() if execution.phase == phase]
        if not executions:
            return {}
        return {execution.metadata_key: execution.create_metadata(batch) for execution in executions}

    def bind_graph(self, metadata: Mapping[Hashable, AttentionMetadata], runtime: DeviceRuntime) -> list[GraphBinding]:
        bindings = []
        for key, value in metadata.items():
            execution = self.metadata_sources[key]
            if isinstance(execution, GraphAttention):
                bindings.append(execution.bind_graph(value, runtime))
        return bindings
