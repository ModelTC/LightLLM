from types import SimpleNamespace

import torch

from lightllm.models.qwen4_exp.attention_state import Qwen4ExpAttentionState
from lightllm.models.qwen4_exp.lexical_state import Qwen4ExpLexicalState


class ResourceView:
    def __init__(self, manager):
        self.manager = manager
        self.allocator = manager.full_att_allocator
        self.size = 32
        (self.state,) = manager.get_full_attention_states("main")
        self.qsa_states = {
            layer: state.qsa_view(layer)
            for state in manager.get_full_attention_states()
            for group in state.config.logical_layer_groups
            for layer in group
        }
        self.qsa_raw_keys = self.state.indexer.k_storage[:, :, 0]
        self.qsa_pooled_keys = self.state.indexer.v_storage[:, :, 0]
        self.qsa_key_cos = self.state.positions.k_storage[0, :, 0]
        self.qsa_key_sin = self.state.positions.v_storage[0, :, 0]

    def alloc(self, count):
        return self.allocator.alloc(count)

    def _slot_buffers(self):
        return {
            f"{state.state_name}.{name}": (tensor, 1)
            for state in self.manager.get_full_attention_states()
            for name, tensor in (
                ("k", state.k_storage),
                ("v", state.v_storage),
                ("raw", state.indexer.k_storage),
                ("pooled", state.indexer.v_storage),
                ("cos", state.positions.k_storage),
                ("sin", state.positions.v_storage),
            )
        }

    def get_index_kv_buffer(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.int64, device=self.state.device)
        return {name: tensor.index_select(axis, indices) for name, (tensor, axis) in self._slot_buffers().items()}


def bind_states(model):
    manager = model.att_state_manager
    manager.finish_state_registration()
    if manager.full_att_allocator is None:
        manager.allocate_storage(32, 1, 8, 8, device="cuda")
    else:
        for state in manager.get_full_attention_states():
            if state.k_storage is None:
                state.device = torch.device("cuda")
                state.allocate_storage(32)
    req = model.req_manager
    req.bind_hold_page(manager.full_att_allocator.hold_token_indices)
    lexical = next(
        state for state in manager.get_bounded_attention_states("main") if isinstance(state, Qwen4ExpLexicalState)
    )
    req.lexical_caches = lexical.lexical_caches
    req.init_linear_att_state = lambda request: manager.reset_bounded_att_state([request.req_idx])
    req.snapshot_lexical = lambda request: {
        layer: cache.snapshot(request) for layer, cache in lexical.lexical_caches.items()
    }
    req.restore_lexical = lambda request, snapshots: [
        cache.restore(request, snapshots[layer]) for layer, cache in lexical.lexical_caches.items()
    ]
    req.get_mamba_cache = lambda layer: manager.get_state("main", "gdn", layer).get_layer_view(layer)
    return ResourceView(manager)
