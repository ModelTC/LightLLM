"""Register each model role's layer backends at load time and prepare their shared metadata each step."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from lightllm.common.attention import selector
from lightllm.common.attention.backend import AttentionBackend
from lightllm.utils.dist_utils import get_current_device_id
from lightllm.utils.envs_utils import get_env_start_args
from lightllm.utils.log_utils import init_logger

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
    """Register each model role's layer backends and prepare their shared metadata each step.

    The main model creates one manager and its drafts share it, as with AttentionStateManager. Roles
    (main, draft0 .. draft35) keep their own backends: identical layer configs share an instance within
    a role, never across roles, because each instance owns its role's cross-step storage. Each phase's
    metadata groups hold the representative instance of each metadata_group_key.
    """

    def __init__(self):
        # (device, name, dtype) -> requested size -> buffer, shared by main and drafts.
        self._workspace_buffers: dict[tuple, dict[int, torch.Tensor]] = {}
        self._workspace_lock = threading.Lock()
        # By role and phase: the backend that builds each metadata_group_key's metadata.
        self.group_backends: dict[str, dict[str, dict[tuple, AttentionBackend]]] = {}
        self._layer_backends: dict[tuple[str, str, int], AttentionBackend] = {}

    def register_backends(self, model: TpPartBaseModel) -> None:
        """Select the backends of the model's layers from their attention_config and register them.

        Runs once per role after KV allocation and before warmup, so unsupported layer configs fail at
        load time. A role without attention states (FFN-only drafts) registers no backends.
        """
        role = model.att_state_role
        if role in self.group_backends:
            raise RuntimeError(f"attention backends of {role} are already registered")
        args = model.args
        states = model.att_state_manager
        groups = self.group_backends[role] = {phase: {} for phase in selector.PHASES}
        if not (states.get_full_attention_states(role) or states.get_bounded_attention_states(role)):
            return
        selection = selector.BackendSelector(args)
        for layer in model.layers_infer:
            config = layer.attention_config(model)
            for phase, layer_config, required_backend_name in (
                ("prefill", config.prefill, config.prefill_backend_name),
                ("decode", config.decode, config.decode_backend_name),
            ):
                if layer_config is None:
                    continue
                backend = selection.select(layer_config, phase, required_backend_name)
                self._layer_backends[role, phase, layer.layer_num_] = backend
                groups[phase].setdefault(backend.metadata_group_key, backend)

        for phase, phase_groups in groups.items():
            if phase_groups:
                backend_names = ", ".join(backend.name for backend in phase_groups.values())
                logger.info(f"{role} {phase} attention backends: {backend_names}")

    def get_backend(self, model_role: str, phase: str, logical_layer_id: int) -> AttentionBackend:
        """The backend that runs the layer in this phase."""
        return self._layer_backends[model_role, phase, logical_layer_id]

    def get_workspace(self, name: str, numel: int, dtype: torch.dtype = torch.int8) -> torch.Tensor:
        """Return shared storage on the current device, keeping repeated requests at one address.

        A new smaller request reuses a larger buffer's prefix, so drafts can borrow the main model's
        reservation. Concurrent users must use different names or disjoint regions of one buffer.
        """
        if not name:
            raise ValueError("workspace name must not be empty")
        if numel <= 0:
            raise ValueError(f"workspace numel must be positive, got {numel}")

        device = get_current_device_id()
        with self._workspace_lock:
            buffers = self._workspace_buffers.setdefault((device, name, dtype), {})
            buffer = buffers.get(numel)
            if buffer is None:
                larger = [buffers[size] for size in sorted(buffers) if size > numel]
                buffer = larger[0][:numel] if larger else torch.empty(numel, dtype=dtype, device=device)
                buffers[numel] = buffer
            return buffer

    def create_att_metadata_by_group(self, infer_state: InferStateInfo) -> dict[tuple, Any]:
        """Build this microbatch's metadata by metadata_group_key for the layers of its role and phase."""
        role = infer_state.att_state_role
        groups = self.group_backends[role]["prefill" if infer_state.is_prefill else "decode"]
        if not groups:
            return {}
        return {key: backend.create_att_metadata(infer_state) for key, backend in groups.items()}

    def has_attention(self, model_role: str) -> bool:
        return any(self.group_backends[model_role].values())

    @property
    def requires_cpu_cache_sync(self) -> bool:
        """Whether some registered backend must not overlap CPU cache copies on the GPU."""
        return any(backend.requires_cpu_cache_sync for backend in self._layer_backends.values())


def attention_is_causal(model: TpPartBaseModel) -> bool:
    """Only parallel block-draft models attend non-causally within a block."""
    return not model.is_mtp_draft_model or get_env_start_args().mtp_mode not in ("dspark", "dflash", "dflash2")
