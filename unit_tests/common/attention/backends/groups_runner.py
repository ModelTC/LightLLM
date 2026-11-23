"""Run one layer's attention the way a model does: AttentionGroups chooses the implementation and builds
the step args of the layer's group from each step's InferStateInfo."""

from typing import Any, Optional

import torch

from lightllm.common.attention.backends.gqa import GQAKVScales, GQALayerInputs
from lightllm.common.basemodel.attention_groups import AttentionGroups


class GroupsRunner:
    """model needs what AttentionGroups reads (args, capacities, mtp_manager, is_mtp_draft_model)."""

    def __init__(self, model, prefill_config: Any, decode_config: Any) -> None:
        if not hasattr(model, "att_implementation"):
            model.att_implementation = None
        self.groups = AttentionGroups(model)
        self.attn = self.groups.create_attention(prefill_config, decode_config)

    def step(self, infer_state) -> None:
        """Build the step args of infer_state's step, as InferStateInfo.init_att_state does."""
        infer_state.att_groups = self.groups
        infer_state.init_att_state()

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
        """Run the layer's GQA attention in infer_state's phase with infer_state's step args."""
        run = self.attn.prefill if infer_state.is_prefill else self.attn.decode
        inputs = GQALayerInputs(q=q, k_cache=k, v_cache=v, kv_scales=kv_scales)
        out = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        return run(infer_state.att_step_args[self.attn.group], inputs, out=out, alloc_func=alloc_func)


class LayerAttention:
    """One layer's attention outside a model: impls selected by name as AttentionGroups selects them,
    and one step builder per phase that builds step args from an InferStateInfo."""

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

    def step_args(self, infer_state, model=None) -> Any:
        """The step args of infer_state's step in its phase; model supplies what build_attention_batch reads."""
        from types import SimpleNamespace

        from lightllm.common.basemodel.attention_groups import build_attention_batch

        phase = "prefill" if infer_state.is_prefill else "decode"
        batch = build_attention_batch(infer_state, SimpleNamespace() if model is None else model)
        return self.builders[phase].build(batch)

    def run(self, phase: str, step_args, layer_inputs, *, out: torch.Tensor, alloc_func=torch.empty) -> torch.Tensor:
        return self.selections[phase].impl.forward(step_args, layer_inputs, out=out, alloc_func=alloc_func)
