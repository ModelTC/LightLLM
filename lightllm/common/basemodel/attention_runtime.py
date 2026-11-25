"""A model's attention runtime: bind each layer's implementations at load time, build each step's states.

In order of lifetime:

- Binding: a layer describes its attention as an AttentionSpec; AttentionRuntime.bind selects
  the prefill and decode implementations.
- Model facts, read once when the runtime is created: AttentionBuildResources for the backends'
  build contexts, and AttentionStepConfig for describing steps.
- Per step: AttentionRuntime.prepare builds one attention state per state_key from InferStateInfo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
import triton

from lightllm.common.attention import selector
from lightllm.common.attention.backend import AttentionBackend, AttentionBuildResources
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger
from lightllm.utils.workspace_utils import get_gpu_workspace_buffer

if TYPE_CHECKING:
    from lightllm.common.basemodel.basemodel import TpPartBaseModel
    from lightllm.common.basemodel.infer_struct import InferStateInfo

logger = init_logger(__name__)


# Binding


@dataclass(frozen=True)
class AttentionSpec:
    """A layer's attention, as returned by its attention_spec(model)."""

    # The layer configs of each phase; None for a phase the layer does not attend in.
    prefill: Any
    decode: Any
    # Required implementation names override the CLI for their respective phases.
    prefill_implementation: Optional[str] = None
    decode_implementation: Optional[str] = None


def attention_is_causal(model: TpPartBaseModel) -> bool:
    """Only parallel block-draft models attend non-causally within a block."""
    return not model.is_mtp_draft_model or get_env_start_args().mtp_mode not in ("dspark", "dflash", "dflash2")


def requested_backend(args, phase: str, layer_config: Any) -> str:
    """The implementation name the command line requests for layer_config's family in phase."""
    index = selector.arg_index(layer_config)
    values = args.llm_prefill_att_backend if phase == "prefill" else args.llm_decode_att_backend
    return values[index] if index < len(values) else "auto"


# Model facts, read once when the runtime is created


def attention_build_resources(model: TpPartBaseModel) -> AttentionBuildResources:
    args = model.args
    # Dynamic verification may keep the target row and all MTP draft rows for every running
    # request. TPSP and CUDA graphs may pad that physical batch further.
    max_decode_rows = args.running_max_req_size * (args.mtp_step + 1)
    if args.enable_tpsp_mix_mode:
        tp_size = model.tp_world_size_
        max_decode_rows = triton.cdiv(max_decode_rows, tp_size) * tp_size
    return AttentionBuildResources(
        device=get_current_device_id(),
        page_size=args.page_size,
        # max_seq_length is max_req_total_len plus the MTP headroom reserved at model init.
        max_seq_len=model.max_seq_length,
        max_cuda_graph_batch_size=model.graph_max_batch_size,
        max_decode_rows=max(max_decode_rows, model.graph_max_batch_size),
        microbatch_count=2 if args.enable_decode_microbatch_overlap else 1,
        workspace_bytes=args.flashinfer_workspace_size * 1024 * 1024,
        get_workspace=get_gpu_workspace_buffer,
        diverse_decode=args.diverse_mode,
    )


def recurrent_mtp_state_index(model: TpPartBaseModel) -> Optional[torch.Tensor]:
    """The slot of each request's last accepted speculative row in the role's recurrent (GDN or KDA) state.

    None when the role has no recurrent state.
    """
    states = [
        state
        for state in model.att_state_manager.get_bounded_attention_states(model.att_state_role)
        if getattr(state, "mtp_state_index", None) is not None
    ]
    if len(states) > 1:
        raise ValueError(f"expected at most one recurrent state for {model.att_state_role}, got {len(states)}")
    return states[0].mtp_state_index if states else None


@dataclass(frozen=True)
class AttentionStepConfig:
    """What a model's steps share when describing themselves to the backends."""

    # Query rows of each request in decode steps; above 1 with speculative decoding.
    decode_rows_per_request: int = 1
    # Decode steps pack a varying number of speculative rows per request (dynamic verify).
    dynamic_decode_rows: bool = False
    # Linear attention: recurrent_mtp_state_index; None without recurrent state.
    req_to_mtp_state_index: Optional[torch.Tensor] = None
    # KV length capacity of the model's CUDA graphs; required for capture and replay steps.
    cuda_graph_max_kv_seq_len: Optional[int] = None


def attention_step_config(model: TpPartBaseModel) -> AttentionStepConfig:
    """Read the step facts from a model whose CUDA graph capacity is final."""
    rows = model.mtp_manager.get_decode_draft_step(model.is_mtp_draft_model) + 1
    # Only the main model verifies a varying number of draft rows per request.
    dynamic = rows > 1 and not model.is_mtp_draft_model and get_env_start_args().mtp_dynamic_verify
    return AttentionStepConfig(
        decode_rows_per_request=rows,
        dynamic_decode_rows=dynamic,
        req_to_mtp_state_index=recurrent_mtp_state_index(model),
        cuda_graph_max_kv_seq_len=model.graph_max_len_in_batch,
    )


# Per step


def configure_attention_step(
    infer_state: InferStateInfo, step_config: AttentionStepConfig = AttentionStepConfig()
) -> InferStateInfo:
    """Set model-dependent attention inputs on the existing step."""
    if infer_state.uses_cuda_graph and step_config.cuda_graph_max_kv_seq_len is None:
        raise ValueError(f"a {infer_state.graph_mode.value} step needs the CUDA graph kv length capacity")
    decode = not infer_state.is_prefill
    rows = step_config.decode_rows_per_request if decode else 1
    infer_state.decode_rows_per_request = rows
    infer_state.dynamic_decode_rows = decode and step_config.dynamic_decode_rows
    infer_state.req_to_mtp_state_index = step_config.req_to_mtp_state_index if rows > 1 else None
    infer_state.cuda_graph_max_kv_seq_len = (
        step_config.cuda_graph_max_kv_seq_len if infer_state.uses_cuda_graph else None
    )
    return infer_state


class AttentionRuntime:
    """A model's attention: binds each layer's implementations at load time and builds each step's states.

    bind() selects a layer's prefill and decode implementations and registers one representative per
    distinct state_key and phase. prepare(infer_state) returns {state_key: att_state} for the step's
    phase; the model stores it as infer_state.att_states, where each layer looks up impl.state_key.
    A model without attention has a runtime with no state groups, whose prepare() returns {}.

    Create it once the model's attention states and CUDA graph capacity are final: it keeps what
    the steps need from the model, not the model itself.
    """

    def __init__(self, model: TpPartBaseModel):
        self.args = model.args
        self.resources = attention_build_resources(model)
        self.step_config = attention_step_config(model)
        self.state_groups: dict[str, dict[tuple, AttentionBackend]] = {phase: {} for phase in selector.PHASES}
        # (phase, layer_config, requested name) -> impl, so layers with the same config share an impl.
        self._impls: dict[tuple, AttentionBackend] = {}
        if self.args.enable_ep_moe:
            logger.info("Expert parallelism with MoE enabled, excluding flashinfer attention backend")

    def bind(self, spec: AttentionSpec) -> tuple[Optional[AttentionBackend], Optional[AttentionBackend]]:
        """The layer's prefill and decode implementations."""
        return (
            self._select("prefill", spec.prefill, spec.prefill_implementation),
            self._select("decode", spec.decode, spec.decode_implementation),
        )

    def _select(self, phase: str, layer_config: Any, implementation: Optional[str]) -> Optional[AttentionBackend]:
        if layer_config is None:
            return None
        requested = implementation or requested_backend(self.args, phase, layer_config)
        key = (phase, layer_config, requested)
        if key not in self._impls:
            selection = selector.select(
                layer_config,
                phase,
                requested,
                priority=selector.auto_priority(layer_config, phase, self.args.mtp_step),
                excluded=("flashinfer",) if self.args.enable_ep_moe else (),
            )
            self.state_groups[phase].setdefault(selection.impl.state_key, selection.impl)
            self._impls[key] = selection.impl
        return self._impls[key]

    def backends(self, phase: str) -> list[type[AttentionBackend]]:
        """The backend of each state built in phase, in binding order."""
        return [type(impl) for impl in self.state_groups[phase].values()]

    @property
    def has_attention(self) -> bool:
        return any(self.state_groups.values())

    def requires_cpu_cache_sync(self) -> bool:
        return any(backend.requires_cpu_cache_sync() for phase in selector.PHASES for backend in self.backends(phase))

    def prepare(self, infer_state: InferStateInfo) -> dict[tuple, Any]:
        """Build one attention state per state_key of infer_state's phase, for this microbatch."""
        groups = self.state_groups["prefill" if infer_state.is_prefill else "decode"]
        if not groups:
            return {}
        configure_attention_step(infer_state, self.step_config)
        return {key: impl.prepare(infer_state, self.resources) for key, impl in groups.items()}
