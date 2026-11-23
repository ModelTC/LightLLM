"""Model-side ownership of attention: impl selection, layer groups and their step builders.

A model selects each layer's implementations at load time. Within each phase, implementations
with the same backend and step_builder_key share one builder. Each batch maps its implementations
to the arguments produced by those builders.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch
import triton

from lightllm.common.attention import selector
from lightllm.common.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    AttentionImpl,
    AttentionStepBuilder,
    StepBuilderContext,
)
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

if TYPE_CHECKING:
    from lightllm.common.attention.backends.gqa import GQAKVScales
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo

logger = init_logger(__name__)

_workspace_buffers: dict[tuple, torch.Tensor] = {}
_workspace_buffer_lock = threading.Lock()


def get_gpu_workspace_buffer(key_name: str, workspace_size: int, dtype: torch.dtype = torch.int8) -> torch.Tensor:
    """Return a process-local workspace of workspace_size elements, shared by key name, size, dtype and device."""
    if not key_name:
        raise ValueError("workspace key_name must not be empty")
    if workspace_size <= 0:
        raise ValueError(f"workspace_size must be positive, got {workspace_size}")

    device_id = get_current_device_id()
    buffer_key = (device_id, key_name, workspace_size, dtype)
    with _workspace_buffer_lock:
        workspace_buffer = _workspace_buffers.get(buffer_key)
        if workspace_buffer is None:
            workspace_buffer = torch.empty(workspace_size, dtype=dtype, device=device_id)
            _workspace_buffers[buffer_key] = workspace_buffer
        return workspace_buffer


def uses_dynamic_spec_verify_layout(model: "TpPartBaseModel") -> bool:
    """Whether the main model's decode steps pack a varying number of speculative rows per request."""
    draft_step = model.mtp_manager.get_decode_draft_step(model.is_mtp_draft_model)
    return not model.is_mtp_draft_model and draft_step > 0 and get_env_start_args().mtp_dynamic_verify


def attention_is_causal(model: "TpPartBaseModel") -> bool:
    """Only parallel block-draft models attend non-causally within a block."""
    return not model.is_mtp_draft_model or get_env_start_args().mtp_mode not in ("dspark", "dflash", "dflash2")


def recurrent_mtp_state_index(model: "TpPartBaseModel") -> Optional[torch.Tensor]:
    """The role's recurrent (GDN or KDA) state's slot of each request's last accepted speculative row.

    None when the role has no recurrent state.
    """
    manager = getattr(model, "att_state_manager", None)
    if manager is None:
        return None
    states = [
        state
        for state in manager.get_bounded_attention_states(model.att_state_role)
        if getattr(state, "mtp_state_index", None) is not None
    ]
    if len(states) > 1:
        raise ValueError(f"expected at most one recurrent state for {model.att_state_role}, got {len(states)}")
    return states[0].mtp_state_index if states else None


def build_attention_metadata(
    infer_state: "InferStateInfo",
    model: "TpPartBaseModel",
    uses_dynamic_layout: Optional[Callable[[], bool]] = None,
    req_to_mtp_state_index: Optional[torch.Tensor] = None,
) -> AttentionMetadata:
    """Describe one step's requests for attention step builders, from its InferStateInfo.

    CUDA graph capture and replay steps carry the model's graph kv length capacity, so step builders
    can size and configure captured work for every length a replay may see. Decode steps of
    speculative decoding describe how query rows group into requests and, given
    req_to_mtp_state_index (recurrent_mtp_state_index), where each request's accepted recurrent
    state lies. Fields the step does not set are None. uses_dynamic_layout overrides
    uses_dynamic_spec_verify_layout(model).
    """
    if uses_dynamic_layout is None:
        uses_dynamic_layout = lambda: uses_dynamic_spec_verify_layout(model)  # noqa: E731
    uses_cuda_graph = infer_state.is_cuda_graph_capture or infer_state.is_cuda_graph_replay
    decode_rows_per_request, dynamic_decode_rows = 1, False
    if not infer_state.is_prefill and getattr(model, "mtp_manager", None) is not None:
        decode_rows_per_request = model.mtp_manager.get_decode_draft_step(model.is_mtp_draft_model) + 1
        dynamic_decode_rows = decode_rows_per_request > 1 and uses_dynamic_layout()
    return AttentionMetadata(
        req_to_token_indexs=infer_state.req_manager.req_to_token_indexs,
        b_req_idx=infer_state.b_req_idx,
        b_seq_len=infer_state.b_seq_len,
        batch_size=infer_state.batch_size,
        max_kv_seq_len=infer_state.max_kv_seq_len,
        total_token_num=infer_state.total_token_num,
        cu_q_seq_len=infer_state.b1_cu_q_seq_len,
        cu_kv_seq_len=infer_state.b1_cu_kv_seq_len,
        b_q_start_loc=infer_state.b_q_start_loc,
        b_ready_cache_len=infer_state.b_ready_cache_len,
        max_q_seq_len=infer_state.max_q_seq_len if infer_state.max_q_seq_len is not None else 1,
        decode_rows_per_request=decode_rows_per_request,
        dynamic_decode_rows=dynamic_decode_rows,
        req_to_mtp_state_index=req_to_mtp_state_index if decode_rows_per_request > 1 else None,
        pad_req_idx=getattr(infer_state.req_manager, "HOLD_REQUEST_ID", None),
        b_shared_seq_len=getattr(infer_state, "b_shared_seq_len", None),
        b_mtp_index=getattr(infer_state, "b_mtp_index", None),
        b_image_token_end=getattr(infer_state, "b_image_token_end", None),
        b_image_token_start=getattr(infer_state, "b_image_token_start", None),
        max_ready_cache_len=infer_state.max_cache_len or 0,
        mem_index=infer_state.mem_index,
        b_shared_radix_node_id=getattr(infer_state, "b_shared_radix_node_id", None),
        cuda_graph_max_kv_seq_len=model.graph_max_len_in_batch if uses_cuda_graph else None,
        microbatch_index=infer_state.microbatch_index,
    )


# --llm_kv_type -> GQALayerConfig.kv_quant
_GQA_KV_QUANT = {
    "None": None,
    "int8kv": "int8_group",
    "int4kv": "int4_group",
    "fp8kv_sph": "fp8_per_head",
    "fp8kv_spt": "fp8_per_tensor",
}


def gqa_kv_quant_fields(llm_kv_type: str, kv_scales: Optional["GQAKVScales"], head_dim: int) -> dict[str, Any]:
    """GQALayerConfig's kv_quant, kv_quant_group_size and has_q_scale for a layer's cache.

    kv_scales is the layer's FullAttentionState.get_layer_kv_scales(), None for an unquantized cache.
    """
    if llm_kv_type not in _GQA_KV_QUANT:
        raise ValueError(f"--llm_kv_type {llm_kv_type!r} has no GQA KV cache layout; choose from {list(_GQA_KV_QUANT)}")
    kv_quant = _GQA_KV_QUANT[llm_kv_type]
    if (kv_quant is None) != (kv_scales is None):
        raise ValueError(f"--llm_kv_type {llm_kv_type!r} does not match the cache's scales")
    grouped = kv_quant in ("int8_group", "int4_group")
    return dict(
        kv_quant=kv_quant,
        kv_quant_group_size=head_dim // kv_scales.k_scale.shape[-1] if grouped else None,
        has_q_scale=kv_quant == "fp8_per_head" and kv_scales.q_scale is not None,
    )


def step_builder_context(model: "TpPartBaseModel") -> StepBuilderContext:
    args = model.args
    # Dynamic verification may keep the target row and all MTP draft rows for every running
    # request. TPSP and CUDA graphs may pad that physical batch further.
    max_decode_rows = args.running_max_req_size * (args.mtp_step + 1)
    if args.enable_tpsp_mix_mode:
        tp_size = model.tp_world_size_
        max_decode_rows = triton.cdiv(max_decode_rows, tp_size) * tp_size
    workspace_mib = getattr(args, "flashinfer_workspace_size", 512)
    return StepBuilderContext(
        device=get_current_device_id(),
        page_size=args.page_size,
        # max_seq_length is max_req_total_len plus the MTP headroom reserved at model init.
        max_seq_len=model.max_seq_length,
        max_cuda_graph_batch_size=model.graph_max_batch_size,
        max_decode_rows=max(max_decode_rows, model.graph_max_batch_size),
        microbatch_count=2 if args.enable_decode_microbatch_overlap else 1,
        workspace_bytes=workspace_mib * 1024 * 1024,
        get_workspace=get_gpu_workspace_buffer,
        diverse_decode=bool(getattr(args, "diverse_mode", False)),
    )


@dataclass
class _Group:
    backend: type[AttentionBackend]
    layer_config: Any
    ctx: StepBuilderContext
    impls: list[AttentionImpl] = field(default_factory=list)
    builder: Optional[AttentionStepBuilder] = None

    def build(self, batch: AttentionMetadata):
        if self.builder is None:
            self.builder = self.backend.create_step_builder(self.layer_config, self.ctx)
        return self.builder.build(batch)


def requested_backend(args, phase, layer_config):
    index = selector.arg_index(layer_config)
    values = args.llm_prefill_att_backend if phase == "prefill" else args.llm_decode_att_backend
    return values[index] if index < len(values) else "auto"


def layer_impl_selector(model):
    """Share implementations during model initialization. Only step builders are needed afterward."""
    args = model.args
    ctx = step_builder_context(model)
    model.attention_step_groups = groups = {phase: {} for phase in selector.PHASES}
    model.req_to_mtp_state_index = recurrent_mtp_state_index(model)
    impls = {}
    if args.enable_ep_moe:
        logger.info("Expert parallelism with MoE enabled, excluding flashinfer attention backend")

    def select(phase, layer_config, implementation):
        if layer_config is None:
            return None
        requested = requested_backend(args, phase, layer_config) if implementation is None else implementation
        key = (phase, layer_config, requested)
        if key not in impls:
            selection = selector.select(
                layer_config,
                phase,
                requested,
                priority=selector.auto_priority(layer_config, phase, args.mtp_step),
                excluded=("flashinfer",) if args.enable_ep_moe else (),
            )
            backend = selection.backend
            group_key = (backend, backend.step_builder_key(layer_config))
            if group_key not in groups[phase]:
                groups[phase][group_key] = _Group(backend, layer_config, ctx)
            groups[phase][group_key].impls.append(selection.impl)
            impls[key] = selection.impl
        return impls[key]

    def select_impls(prefill_config, decode_config, *, implementation=None):
        return select("prefill", prefill_config, implementation), select("decode", decode_config, implementation)

    return select_impls


def build_attention_states(model, infer_state):
    phase = "prefill" if infer_state.is_prefill else "decode"
    groups = model.attention_step_groups[phase]
    if not groups:
        return {}
    batch = build_attention_metadata(infer_state, model, req_to_mtp_state_index=model.req_to_mtp_state_index)
    states = {}
    for group in groups.values():
        state = group.build(batch)
        states.update(dict.fromkeys(group.impls, state))
    return states
