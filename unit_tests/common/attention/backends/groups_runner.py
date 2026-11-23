"""Run one layer's attention the way a model does: model initialization chooses the implementation and builds
the attention state of the layer's group from each step's InferStateInfo."""

from typing import Any, Optional

import torch

from lightllm.common.attention.backends.gqa import GQAKVScales, GQALayerInputs
from lightllm.common.basemodel.attention_runtime import layer_impl_selector


class GroupsRunner:
    """model needs what model initialization reads (args, capacities, mtp_manager, is_mtp_draft_model)."""

    def __init__(self, model, prefill_config: Any, decode_config: Any) -> None:
        self.model = model
        select_impls = layer_impl_selector(model)
        self.prefill_impl, self.decode_impl = select_impls(prefill_config, decode_config)

    def step(self, infer_state) -> None:
        """Build the attention state of infer_state's step, as InferStateInfo.init_att_state does."""
        infer_state.init_att_state(self.model)

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
        return impl.forward(infer_state.att_states[impl], inputs, out=out, alloc_func=alloc_func)


class LayerAttention:
    """One layer's attention outside a model: impls selected by name as a model selects them,
    and one step builder per phase that builds attention state from an InferStateInfo."""

    def __init__(self, prefill_config: Any, decode_config: Any, name: str, ctx=None) -> None:
        from lightllm.common.attention import selector

        configs = {"prefill": prefill_config, "decode": decode_config}
        self.selections = {phase: selector.select(config, phase, name) for phase, config in configs.items()}
        self.builders = {
            phase: selection.backend.create_step_builder(configs[phase], ctx)
            for phase, selection in self.selections.items()
        }

    def backend_name(self, phase: str) -> str:
        return self.selections[phase].backend.get_name()

    def att_state(self, infer_state, model=None) -> Any:
        """Prepare the backend state for this microbatch."""
        from types import SimpleNamespace

        from lightllm.common.basemodel.attention_runtime import build_attention_metadata

        phase = "prefill" if infer_state.is_prefill else "decode"
        batch = build_attention_metadata(infer_state, SimpleNamespace() if model is None else model)
        return self.builders[phase].build(batch)

    def run(self, phase: str, att_state, layer_inputs, *, out: torch.Tensor, alloc_func=torch.empty) -> torch.Tensor:
        return self.selections[phase].impl.forward(att_state, layer_inputs, out=out, alloc_func=alloc_func)
