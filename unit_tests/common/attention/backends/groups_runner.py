"""Run one layer's attention the way a model does: model initialization chooses the implementation and builds
the attention state of the layer's group from each step's InferStateInfo."""

from types import SimpleNamespace
from typing import Any, Optional

import torch

from lightllm.common.attention.backends.gqa import GQAKVScales, GQALayerInputs
from lightllm.common.basemodel.attention_backend_manager import (
    AttentionBackendManager,
    LayerAttentionConfig,
    configure_attention_step,
)


def set_up(backend_cls, layer_config, resources):
    """A backend for layer_config whose storage is set up from resources, as a state group's first prepare does."""
    backend = backend_cls(layer_config)
    backend.setup(resources)
    return backend


class GroupsRunner:
    """model needs what model initialization reads (args, capacities, mtp_manager, is_mtp_draft_model).

    A model without att_state_manager is given one with no recurrent state.
    """

    def __init__(self, model, prefill_config: Any, decode_config: Any) -> None:
        self.model = model
        if not hasattr(model, "att_state_manager"):
            model.att_state_role = "main"
            model.att_state_manager = SimpleNamespace(get_bounded_attention_states=lambda role: ())
        model.attention_backend_manager = AttentionBackendManager(model)
        self.prefill_backend, self.decode_backend = model.attention_backend_manager.bind(
            LayerAttentionConfig(prefill_config, decode_config)
        )

    def step(self, infer_state) -> None:
        infer_state.att_states = self.model.attention_backend_manager.prepare(infer_state)

    def run_gqa(
        self,
        infer_state,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        kv_scales: Optional[GQAKVScales] = None,
        alloc_func=torch.empty,
    ) -> torch.Tensor:
        """Run the layer's GQA attention in infer_state's phase with infer_state's attention state."""
        backend = self.prefill_backend if infer_state.is_prefill else self.decode_backend
        inputs = GQALayerInputs(q=q, k_cache=k, v_cache=v, kv_scales=kv_scales)
        out = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        return backend.forward(infer_state.att_states[backend.state_key], inputs, out=out, alloc_func=alloc_func)


class LayerAttention:
    """One layer's attention outside a model: backends selected by name as a model selects them, each
    preparing attention state from an InferStateInfo as its state group would."""

    def __init__(self, prefill_config: Any, decode_config: Any, name: str, ctx=None) -> None:
        from lightllm.common.attention import selector

        configs = {"prefill": prefill_config, "decode": decode_config}
        self.backends = {phase: selector.select(config, phase, name) for phase, config in configs.items()}
        self.ctx = ctx

    def backend_name(self, phase: str) -> str:
        return self.backends[phase].name

    def att_state(self, infer_state, step_config: Optional[dict] = None) -> Any:
        """Prepare the backend state for this microbatch; step_config holds configure_attention_step's keywords."""
        phase = "prefill" if infer_state.is_prefill else "decode"
        batch = configure_attention_step(infer_state, **(step_config or {}))
        return self.backends[phase].prepare(batch, self.ctx)

    def run(self, phase: str, att_state, layer_inputs, *, out: torch.Tensor, alloc_func=torch.empty) -> torch.Tensor:
        return self.backends[phase].forward(att_state, layer_inputs, out=out, alloc_func=alloc_func)
