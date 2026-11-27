"""Register each model role's layer backends at load time and prepare their shared states each step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

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
    """Register each model role's layer backends and prepare their shared states each step.

    The main model creates one manager and its drafts share it, as with AttentionStateManager. Roles
    (main, draft0 .. draft35) keep their own backends: identical layer configs share an instance within
    a role, never across roles, because each instance owns its role's cross-step storage. Each phase's
    state groups hold the representative instance of each state_key.
    """

    def __init__(self):
        # By role: the model-wide build resources.
        self.resources: dict[str, AttentionBuildResources] = {}
        # By role and phase: the backend that builds each state_key's state.
        self.state_groups: dict[str, dict[str, dict[tuple, AttentionBackend]]] = {}
        self._backends: dict[tuple[str, str, int], AttentionBackend] = {}

    def register_backends(self, model: TpPartBaseModel) -> None:
        """Select the backends of the model's layers from their attention_config and register them.

        Runs once per role after KV allocation and before warmup, so unsupported layer configs fail at
        load time. A role without attention states (FFN-only drafts) registers no backends.
        """
        role = model.att_state_role
        if role in self.state_groups:
            raise RuntimeError(f"attention backends of {role} are already registered")
        args = model.args
        # Dynamic verification may keep the target row and all MTP draft rows for every running
        # request. TPSP and CUDA graphs may pad that physical batch further.
        max_decode_rows = args.running_max_req_size * (args.mtp_step + 1)
        if args.enable_tpsp_mix_mode:
            max_decode_rows = triton.cdiv(max_decode_rows, model.tp_world_size_) * model.tp_world_size_
        self.resources[role] = AttentionBuildResources(
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
        states = model.att_state_manager
        groups = self.state_groups[role] = {phase: {} for phase in selector.PHASES}
        if not (states.get_full_attention_states(role) or states.get_bounded_attention_states(role)):
            return
        if args.enable_ep_moe:
            logger.info("Expert parallelism with MoE enabled, excluding flashinfer attention backend")
        selected: dict[tuple, AttentionBackend] = {}
        for layer in model.layers_infer:
            config = layer.attention_config(model)
            for phase in selector.PHASES:
                layer_config = getattr(config, phase)
                if layer_config is None:
                    continue
                requested = getattr(config, f"{phase}_backend_name") or selector.requested_backend(
                    args, phase, layer_config
                )
                key = (phase, layer_config, requested)
                if key not in selected:
                    backend = selected[key] = selector.select(
                        layer_config,
                        phase,
                        requested,
                        priority=selector.auto_priority(layer_config, phase, args.mtp_step),
                        excluded=("flashinfer",) if args.enable_ep_moe else (),
                    )
                    groups[phase].setdefault(backend.state_key, backend)
                self._backends[role, phase, layer.layer_num_] = selected[key]

        for phase, phase_groups in groups.items():
            if phase_groups:
                backend_names = ", ".join(backend.name for backend in phase_groups.values())
                logger.info(f"{role} {phase} attention backends: {backend_names}")

    def get_backend(self, model_role: str, phase: str, logical_layer_id: int) -> AttentionBackend:
        """The backend that runs the layer in this phase."""
        return self._backends[model_role, phase, logical_layer_id]

    def create_att_state(self, infer_state: InferStateInfo) -> dict[tuple, Any]:
        """Build this microbatch's state by state_key for the layers of its role and phase."""
        role = infer_state.att_state_role
        groups = self.state_groups[role]["prefill" if infer_state.is_prefill else "decode"]
        if not groups:
            return {}
        resources = self.resources[role]
        return {key: backend.create_att_state(infer_state, resources) for key, backend in groups.items()}

    def has_attention(self, model_role: str) -> bool:
        return any(self.state_groups[model_role].values())

    @property
    def requires_cpu_cache_sync(self) -> bool:
        """Whether some registered backend must not overlap CPU cache copies on the GPU."""
        return any(backend.requires_cpu_cache_sync for backend in self._backends.values())


def attention_is_causal(model: TpPartBaseModel) -> bool:
    """Only parallel block-draft models attend non-causally within a block."""
    return not model.is_mtp_draft_model or get_env_start_args().mtp_mode not in ("dspark", "dflash", "dflash2")
