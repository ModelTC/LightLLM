from unit_tests.common.attention.backends.step_fixture import make_backend_model

"""Run one layer's attention the way a model does: model initialization chooses the implementation and builds
the attention state of the layer's group from each step's InferStateInfo."""
from unit_tests.common.attention.backends.step_fixture import create_att_state

from collections.abc import Callable
from dataclasses import replace
from typing import Any, Optional

import torch

from lightllm.common.attention.base_backend import PrefillAttBackend

from lightllm.common.attention.backends.gqa import GQAKVScales, GQAAttInputs
from lightllm.common.attention.backends.gqa import selector as gqa_selector
from lightllm.common.attention.attention_backend_manager import AttentionBackendManager
from lightllm.common.attention_state import AttentionStateManager
from unit_tests.common.attention.backends.step_fixture import AttentionStepConfig, configure_attention_step


def run_attention(backend: Any, att_state: Any, layer_inputs: Any, *, out: Any, alloc_func: Any = torch.empty):
    attention = backend.prefill if isinstance(backend, PrefillAttBackend) else backend.decode
    return attention(att_state=att_state, layer_inputs=layer_inputs, out=out, alloc_func=alloc_func)


def register_test_backends(role: Any, logical_layer_id: Any, *backends: Any):
    manager = AttentionBackendManager()
    for backend in backends:
        backend.runtime_config = replace(backend.runtime_config, logical_layer_ids=(logical_layer_id,))
    manager.register_backends(role=role, backends=backends)
    return manager


class GroupsRunner:
    """model needs what model initialization reads (args, capacities, mtp_manager, is_mtp_draft_model).

    A model without att_state_manager is given one with no recurrent state.
    """

    def __init__(
        self,
        model: Any,
        prefill_config: Any,
        decode_config: Any,
        *,
        select_prefill: Callable = gqa_selector.select_gqa_prefill_backend,
        select_decode: Callable = gqa_selector.select_gqa_decode_backend,
    ) -> None:
        self.model = model
        if not hasattr(model, "att_state_manager"):
            model.att_state_role = "main"
            model.att_state_manager = AttentionStateManager(request_capacity=1)
            model.att_state_manager.finish_state_registration()
        self.prefill_impl = select_prefill(att_config=prefill_config, model=model, logical_layer_ids=(0,))
        self.decode_impl = select_decode(att_config=decode_config, model=model, logical_layer_ids=(0,))
        model.attention_backend_manager = AttentionBackendManager()
        model.attention_backend_manager.register_backends(
            role=model.att_state_role, backends=(self.prefill_impl, self.decode_impl)
        )

    def step(self, infer_state) -> None:
        infer_state.att_state_role = self.model.att_state_role
        infer_state.is_draft_model = self.model.is_mtp_draft_model
        infer_state.att_state_manager = self.model.att_state_manager
        infer_state.attention_backend_manager = self.model.attention_backend_manager
        infer_state.att_states = self.model.attention_backend_manager.create_att_state(infer_state=infer_state)

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
        """在infer_state的阶段使用infer_state的注意力状态运行层的GQA注意力。"""
        impl = self.prefill_impl if infer_state.is_prefill else self.decode_impl
        inputs = GQAAttInputs(q=q, k_cache=k, v_cache=v, kv_scales=kv_scales)
        out = alloc_func(q.shape, dtype=q.dtype, device=q.device)
        return run_attention(
            backend=impl, att_state=infer_state.att_states[impl], layer_inputs=inputs, out=out, alloc_func=alloc_func
        )


class LayerAttention:
    """One layer's attention outside a model: impls selected by name as a model selects them,
    and one step builder per phase that builds attention state from an InferStateInfo."""

    def __init__(
        self, prefill_config: Any, decode_config: Any, name: str, *, select_prefill: Any, select_decode: Any
    ) -> None:
        self.backends = {
            "prefill": select_prefill(
                att_config=prefill_config, requested=name, model=make_backend_model(), logical_layer_ids=(0,)
            ),
            "decode": select_decode(
                att_config=decode_config, requested=name, model=make_backend_model(), logical_layer_ids=(0,)
            ),
        }

    def backend_name(self, phase: str) -> str:
        return self.backends[phase].get_name()

    def att_state(self, infer_state, step_config: Optional[AttentionStepConfig] = None) -> Any:
        """Prepare the backend state for this microbatch."""
        phase = "prefill" if infer_state.is_prefill else "decode"
        batch = configure_attention_step(infer_state, step_config or AttentionStepConfig())
        return create_att_state(backend=self.backends[phase], infer_state=batch)

    def run(self, phase: str, att_state, layer_inputs, *, out: torch.Tensor, alloc_func=torch.empty) -> torch.Tensor:
        return run_attention(
            backend=self.backends[phase], att_state=att_state, layer_inputs=layer_inputs, out=out, alloc_func=alloc_func
        )
