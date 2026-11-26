"""Bind each layer's attention backends at load time and prepare shared states each step."""

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


@dataclass(frozen=True)
class LayerAttentionConfig:
    """A layer's prefill and decode configuration, as returned by attention_config(model)."""

    # The layer configs of each phase; None for a phase the layer does not attend in.
    prefill: Any
    decode: Any
    # Required backend names override the CLI for their respective phases.
    prefill_backend_name: Optional[str] = None
    decode_backend_name: Optional[str] = None


class AttentionBackendManager:
    """Select layer backends and prepare one state per compatible group in each phase.

    Layers reuse instances by full configuration and requested backend. Each phase's state
    groups track the representative instance for each state_key.

    Create it after the model's attention states and CUDA graph capacity are finalized.
    """

    def __init__(self, model: TpPartBaseModel):
        args = self.args = model.args
        # Dynamic verification may keep the target row and all MTP draft rows for every running
        # request. TPSP and CUDA graphs may pad that physical batch further.
        max_decode_rows = args.running_max_req_size * (args.mtp_step + 1)
        if args.enable_tpsp_mix_mode:
            max_decode_rows = triton.cdiv(max_decode_rows, model.tp_world_size_) * model.tp_world_size_
        self.resources = AttentionBuildResources(
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

        rows = model.mtp_manager.get_decode_draft_step(model.is_mtp_draft_model) + 1
        # The slot of each request's last accepted speculative row in the role's recurrent (GDN or
        # KDA) state; None when the role has no recurrent state.
        recurrent = [
            state.mtp_state_index
            for state in model.att_state_manager.get_bounded_attention_states(model.att_state_role)
            if getattr(state, "mtp_state_index", None) is not None
        ]
        if len(recurrent) > 1:
            raise ValueError(f"expected at most one recurrent state for {model.att_state_role}, got {len(recurrent)}")
        # The model-dependent keyword arguments of configure_attention_step.
        self.step_config = dict(
            decode_rows_per_request=rows,
            # Only the main model verifies a varying number of draft rows per request.
            dynamic_decode_rows=rows > 1 and not model.is_mtp_draft_model and get_env_start_args().mtp_dynamic_verify,
            req_to_mtp_state_index=recurrent[0] if recurrent else None,
            cuda_graph_max_kv_seq_len=model.graph_max_len_in_batch,
        )

        self.state_groups: dict[str, dict[tuple, AttentionBackend]] = {phase: {} for phase in selector.PHASES}
        self._backends: dict[tuple, AttentionBackend] = {}
        if args.enable_ep_moe:
            logger.info("Expert parallelism with MoE enabled, excluding flashinfer attention backend")

    def bind(self, config: LayerAttentionConfig) -> tuple[Optional[AttentionBackend], Optional[AttentionBackend]]:
        """Return the layer's prefill and decode backends."""
        return (
            self._select("prefill", config.prefill, config.prefill_backend_name),
            self._select("decode", config.decode, config.decode_backend_name),
        )

    def prepare(self, infer_state: InferStateInfo) -> dict[tuple, Any]:
        """Build this microbatch's state by state_key for the layers in its phase."""
        groups = self.state_groups["prefill" if infer_state.is_prefill else "decode"]
        if not groups:
            return {}
        configure_attention_step(infer_state, **self.step_config)
        return {key: backend.prepare(infer_state, self.resources) for key, backend in groups.items()}

    def backends(self, phase: str) -> list[type[AttentionBackend]]:
        """Return the backend class of each state group in binding order."""
        return [type(backend) for backend in self.state_groups[phase].values()]

    @property
    def has_attention(self) -> bool:
        return any(self.state_groups.values())

    def requires_cpu_cache_sync(self) -> bool:
        return any(
            backend.requires_cpu_cache_sync() for groups in self.state_groups.values() for backend in groups.values()
        )

    def _select(self, phase: str, layer_config: Any, backend_name: Optional[str]) -> Optional[AttentionBackend]:
        if layer_config is None:
            return None
        requested = backend_name or selector.requested_backend(self.args, phase, layer_config)
        key = (phase, layer_config, requested)
        if key not in self._backends:
            backend = selector.select(
                layer_config,
                phase,
                requested,
                priority=selector.auto_priority(layer_config, phase, self.args.mtp_step),
                excluded=("flashinfer",) if self.args.enable_ep_moe else (),
            )
            self.state_groups[phase].setdefault(backend.state_key, backend)
            self._backends[key] = backend
        return self._backends[key]


def attention_is_causal(model: TpPartBaseModel) -> bool:
    """Only parallel block-draft models attend non-causally within a block."""
    return not model.is_mtp_draft_model or get_env_start_args().mtp_mode not in ("dspark", "dflash", "dflash2")


def configure_attention_step(
    infer_state: InferStateInfo,
    *,
    decode_rows_per_request: int = 1,
    dynamic_decode_rows: bool = False,
    req_to_mtp_state_index: Optional[torch.Tensor] = None,
    cuda_graph_max_kv_seq_len: Optional[int] = None,
) -> InferStateInfo:
    """Set the model-dependent attention inputs on the existing step.

    decode_rows_per_request is the query rows of each request in decode steps, above 1 with speculative
    decoding; dynamic_decode_rows says decode steps pack a varying number of speculative rows per request
    (dynamic verify). req_to_mtp_state_index is the linear-attention recurrent state slot of each request's
    last accepted row, None without recurrent state. cuda_graph_max_kv_seq_len is the KV length capacity of
    the model's CUDA graphs, required for capture and replay steps.
    """
    if infer_state.uses_cuda_graph and cuda_graph_max_kv_seq_len is None:
        raise ValueError(f"a {infer_state.graph_mode.value} step needs the CUDA graph kv length capacity")
    decode = not infer_state.is_prefill
    rows = decode_rows_per_request if decode else 1
    infer_state.decode_rows_per_request = rows
    infer_state.dynamic_decode_rows = decode and dynamic_decode_rows
    infer_state.req_to_mtp_state_index = req_to_mtp_state_index if rows > 1 else None
    infer_state.cuda_graph_max_kv_seq_len = cuda_graph_max_kv_seq_len if infer_state.uses_cuda_graph else None
    return infer_state
