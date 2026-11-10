from dataclasses import dataclass, field

import torch

from lightllm.common.attention.api import AttentionPhase, BackendResources, CapabilityResult
from lightllm.common.attention.backends.base import BaseAttentionBackend, PreparedPlan
from lightllm.platform import Platform


@dataclass
class FiaPlan(PreparedPlan):
    page_table: torch.Tensor | None = None
    host_q_lengths: list[int] | None = None
    host_kv_lengths: list[int] | None = None
    mask: torch.Tensor | None = None
    workspace_size: int | None = None
    tasks: list = field(default_factory=list)


class TorchNpuAttBackend(BaseAttentionBackend):
    """普通分页 Dense FIA；主机长度、task handles 均由捕获实例的 Plan 持有。"""

    provider = "torch_npu"
    dependency = "torch_npu"
    supported_platforms = frozenset({Platform.ASCEND})
    supported_graph_phases = frozenset({AttentionPhase.DECODE})
    activation_dtypes = frozenset({torch.float16, torch.bfloat16})
    supports_expert_parallel = False
    workspace_bytes = 128 * 1024 * 1024

    @classmethod
    def check_support(cls, spec, caps=None, execution=None):
        result = super().check_support(spec, caps, execution)
        if not result.supported:
            return result
        if not spec.causal or spec.use_mtp or spec.dynamic_verify:
            return CapabilityResult.reject("FIA requires causal attention without speculative verification")
        if spec.cache.dtype != spec.dtype:
            return CapabilityResult.reject("FIA requires matching plain K/V and activation dtype")
        if (
            spec.value_head_size != spec.head_size
            or spec.cache.key_dim != spec.head_size
            or spec.cache.value_dim != spec.head_size
        ):
            return CapabilityResult.reject("FIA requires equal Q/K/V head dimensions")
        if spec.cache.page_size % 32 or not 32 <= spec.cache.page_size <= 512 or spec.head_size not in (64, 128, 256):
            return CapabilityResult.reject("FIA requires page size divisible by 32 in [32,512] and head dim 64/128/256")
        if execution is not None and (
            execution.microbatch_count != 1 or execution.transfer_paths or execution.max_verify_tokens != 1
        ):
            return CapabilityResult.reject("FIA requires local state without microbatch overlap")
        return result

    def resource_requirements(self, capacity, *, capture):
        workspace = (("fia", self.workspace_bytes, torch.uint8),) if capture else ()
        return BackendResources(workspace, needs_host_lengths=True, zero_initialize_cache=True)

    def _allocate_prefill_plan(self, capacity, capture, workspace):
        return FiaPlan(capacity, capture, workspace)

    _allocate_decode_plan = _allocate_prefill_plan

    def _prepare_prefill(self, plan):
        self._prepare_decode(plan)
        if plan.mask is None:
            plan.mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8, device=plan.batch.device), diagonal=1)

    def _prepare_decode(self, plan):
        batch = plan.batch
        # 图外只复制此实现需要的长度；快照地址在 Graph 和异步 task update 期间不变。
        kv_lengths = batch.b_seq_len.detach().to("cpu").tolist()
        q_lengths = batch.b1_cu_q_seq_len[1:].detach().to("cpu").tolist()
        if plan.host_kv_lengths is None:
            plan.host_kv_lengths = kv_lengths
            plan.host_q_lengths = q_lengths if batch.phase is AttentionPhase.PREFILL else [1] * batch.batch_size
        else:
            plan.host_kv_lengths[:] = kv_lengths
            if batch.phase is AttentionPhase.PREFILL:
                plan.host_q_lengths[:] = q_lengths
        if plan.page_table is None:
            width = (plan.capacity.max_kv_seq_len + self.page_size - 1) // self.page_size
            plan.page_table = torch.empty((batch.batch_size, width), dtype=torch.int32, device=batch.device)
        if plan.capture:
            plan.workspace.get("fia", self.workspace_bytes)

    def _update_prefill(self, plan):
        self._update_decode(plan)

    def _update_decode(self, plan):
        batch = plan.batch
        width = plan.page_table.shape[1]
        pages = batch.req_to_token_indexs[:, : width * self.page_size : self.page_size]
        plan.page_table.copy_(pages.index_select(0, batch.b_req_idx.long()) // self.page_size)

    def bind_graph(self, plan):
        from .graph import FiaGraphBinding

        return FiaGraphBinding(plan, self.runtime)

    def _forward_prefill(self, layer, plan, inputs, *, out=None):
        return self._run(layer, plan, inputs, out=out)

    _forward_decode = _forward_prefill

    def _run(self, layer, plan, inputs, *, out):
        import torch_npu

        q = inputs.query
        prefill = plan.capacity.phase is AttentionPhase.PREFILL
        native_q = q.contiguous() if prefill else q.unsqueeze(2).contiguous()
        key, value = layer.params.key, layer.params.value
        kwargs = dict(
            query=native_q,
            key=key.view(-1, self.page_size, key.shape[1] * key.shape[2]),
            value=value.view(-1, self.page_size, value.shape[1] * value.shape[2]),
            input_layout="TND" if prefill else "BNSD",
            sparse_mode=3 if prefill else 0,
            atten_mask=plan.mask,
            scale=layer.spec.head_size**-0.5,
            next_tokens=0,
            num_heads=layer.spec.num_q_heads,
            num_key_value_heads=layer.spec.cache.key_heads,
            actual_seq_lengths=plan.host_q_lengths,
            actual_seq_lengths_kv=plan.host_kv_lengths,
            block_table=plan.page_table,
            block_size=self.page_size,
        )
        capturing = self.runtime.is_capturing()
        if not plan.capture:
            if capturing:
                raise RuntimeError("FIA capture was not included in the execution requirements")
            result = torch_npu.npu_fused_infer_attention_score(**kwargs)[0]
        else:
            if plan.workspace_size is None:
                if capturing:
                    raise RuntimeError("FIA workspace must be checked during warmup before capture")
                reserve = dict(kwargs, actual_seq_lengths_kv=[plan.capacity.max_kv_seq_len] * plan.batch.batch_size)
                required = torch_npu._npu_fused_infer_attention_score_get_max_workspace(**reserve)
                size = required.numel() * required.element_size()
                if size > self.workspace_bytes:
                    raise ValueError(
                        f"FIA requires {size} workspace bytes, exceeding reservation {self.workspace_bytes}"
                    )
                plan.workspace_size = size
            result = torch.empty_like(native_q)
            lse = torch.empty(1, dtype=torch.float16, device=q.device)
            workspace = plan.workspace.get("fia", self.workspace_bytes)[: plan.workspace_size]
            if capturing:
                stream, event = self.runtime.current_stream(), self.runtime.api.ExternalEvent()
                event.wait(stream)
                event.reset(stream)
                self.runtime.api.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(**kwargs, workspace=workspace, out=[result, lse])
            if capturing:
                handle = self.runtime.api.graph_task_group_end(stream)
                weak = torch_npu._C._weak_ref_tensor
                borrowed = {
                    name: weak(value) if isinstance(value, torch.Tensor) and value.device.type == "npu" else value
                    for name, value in kwargs.items()
                }
                plan.tasks.append((handle, event, borrowed, workspace, [weak(result), weak(lse)]))
        return result if prefill else result.squeeze(2)
