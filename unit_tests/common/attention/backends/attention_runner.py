"""Run one layer's attention the way a model does: model initialization chooses the implementation and builds
the attention metadata of the layer's group from each step's InferStateInfo."""

from types import SimpleNamespace
from typing import Any, Optional

import torch

from lightllm.common.attention.backends.gqa import GQAKVScales, GQALayerInputs
from lightllm.common.basemodel.attention_backend_manager import (
    AttentionBackendManager,
    LayerAttentionConfig,
)


def init_backend(backend_cls, layer_config, infer_state):
    """A backend for layer_config whose storage is set up from the step, as its first create_att_metadata does."""
    backend = backend_cls(layer_config)
    backend.init_runtime(infer_state)
    return backend


class ModelAttentionRunner:
    """model needs what model initialization reads (args and capacities).

    The runner registers one layer (logical layer 0) with these configs. A model without
    att_state_manager is given one with an attention metadata and no recurrent state.
    """

    def __init__(self, model, prefill_config: Any, decode_config: Any) -> None:
        self.model = model
        if not hasattr(model, "att_state_manager"):
            model.att_state_role = "main"
            model.att_state_manager = SimpleNamespace(
                get_full_attention_states=lambda role: (object(),), get_bounded_attention_states=lambda role: ()
            )
        config = LayerAttentionConfig(prefill_config, decode_config)
        model.layers_infer = [SimpleNamespace(layer_num_=0, attention_config=lambda model: config)]
        manager = model.attention_backend_manager = AttentionBackendManager()
        manager.register_backends(model)
        self.prefill_backend, self.decode_backend = (
            manager.get_backend(model.att_state_role, phase, 0) for phase in ("prefill", "decode")
        )

    def prepare_att_metadata(self, infer_state) -> None:
        model = self.model
        infer_state.attention_backend_manager = model.attention_backend_manager
        infer_state.page_size = model.args.page_size
        infer_state.model_max_seq_len = model.max_seq_length
        infer_state.max_cuda_graph_batch_size = model.graph_max_batch_size
        infer_state.max_decode_rows = max(
            model.args.running_max_req_size * (model.args.mtp_step + 1), model.graph_max_batch_size
        )
        infer_state.diverse_decode = model.args.diverse_mode
        infer_state.att_metadata = self.model.attention_backend_manager.create_att_metadata_by_group(infer_state)

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
        """Run the layer's GQA attention in infer_state's phase with infer_state's attention metadata."""
        backend = self.prefill_backend if infer_state.is_prefill else self.decode_backend
        inputs = GQALayerInputs(q=q, k_cache=k, v_cache=v, kv_scales=kv_scales)
        out = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        return backend.forward(
            infer_state.att_metadata[backend.metadata_group_key], inputs, out=out, alloc_func=alloc_func
        )


class LayerAttention:
    """One layer's attention outside a model: backends selected by name as a model selects them, each
    preparing attention metadata from an InferStateInfo as its state group would."""

    def __init__(self, prefill_config: Any, decode_config: Any, name: str) -> None:
        from lightllm.common.attention import selector

        configs = {"prefill": prefill_config, "decode": decode_config}
        self.backends = {phase: selector.select(config, phase, name) for phase, config in configs.items()}

    def backend_name(self, phase: str) -> str:
        return self.backends[phase].name

    def create_att_metadata(self, infer_state) -> Any:
        """Build attention metadata from the microbatch's prepared inputs."""
        phase = "prefill" if infer_state.is_prefill else "decode"
        return self.backends[phase].create_att_metadata(infer_state)

    def run(self, phase: str, att_metadata, layer_inputs, *, out: torch.Tensor, alloc_func=torch.empty) -> torch.Tensor:
        return self.backends[phase].forward(att_metadata, layer_inputs, out=out, alloc_func=alloc_func)
