"""Run one layer's attention the way a model does: model initialization chooses the implementation and builds
the attention state of the layer's group from each step's InferStateInfo."""

from types import SimpleNamespace
from typing import Any, Optional

import torch

from lightllm.common.attention.backends.gqa import GQAKVScales, GQALayerInputs
from lightllm.common.basemodel.attention_backend_manager import (
    AttentionBackendManager,
    LayerAttentionConfig,
    AttentionStepConfig,
    configure_attention_step,
)


def build_context(backend, layer_config, resources):
    return backend.create_build_context(backend.build_context_key(layer_config), resources)


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
        self.prefill_impl, self.decode_impl = model.attention_backend_manager.bind(
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
        impl = self.prefill_impl if infer_state.is_prefill else self.decode_impl
        inputs = GQALayerInputs(q=q, k_cache=k, v_cache=v, kv_scales=kv_scales)
        out = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        return impl.forward(infer_state.att_states[impl.state_key], inputs, out=out, alloc_func=alloc_func)


class LayerAttention:
    """One layer's attention outside a model: impls selected by name as a model selects them,
    and one step builder per phase that builds attention state from an InferStateInfo."""

    def __init__(self, prefill_config: Any, decode_config: Any, name: str, ctx=None) -> None:
        from lightllm.common.attention import selector

        configs = {"prefill": prefill_config, "decode": decode_config}
        self.selections = {phase: selector.select(config, phase, name) for phase, config in configs.items()}
        self.builders = {
            phase: selection.backend.create_build_context(selection.impl.state_key[1], ctx)
            for phase, selection in self.selections.items()
        }

    def backend_name(self, phase: str) -> str:
        return self.selections[phase].backend.get_name()

    def att_state(self, infer_state, step_config: Optional[AttentionStepConfig] = None) -> Any:
        """Prepare the backend state for this microbatch."""
        phase = "prefill" if infer_state.is_prefill else "decode"
        batch = configure_attention_step(infer_state, step_config or AttentionStepConfig())
        return self.selections[phase].backend.build(batch, self.builders[phase])

    def run(self, phase: str, att_state, layer_inputs, *, out: torch.Tensor, alloc_func=torch.empty) -> torch.Tensor:
        return self.selections[phase].impl.forward(att_state, layer_inputs, out=out, alloc_func=alloc_func)
