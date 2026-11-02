import gc
import weakref

import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att import GQAAttentionConfig, GQAFullAttentionState


def test_metadata_caches_do_not_retain_storage_after_owner_release(cpu_args):
    def prepare():
        manager = AttentionStateManager(request_capacity=1)
        state = GQAFullAttentionState(GQAAttentionConfig(((0,),), 2, 16, torch.bfloat16), device="cpu")
        manager.register_state("main", state)
        manager.finish_state_registration()
        state.allocate_storage(8)
        manager.get_state("main", "gqa", 0)
        manager.get_cpu_cache_operator().get_page_bytes()
        manager.get_pd_operator().get_page_bytes()
        return weakref.ref(manager), weakref.ref(state), weakref.ref(state.k_storage)

    references = prepare()
    gc.collect()
    assert all(reference() is None for reference in references)


def test_metadata_cache_keeps_live_owner_results(cpu_args):
    manager = AttentionStateManager(request_capacity=1)
    state = GQAFullAttentionState(GQAAttentionConfig(((0,),), 2, 16, torch.bfloat16), device="cpu")
    manager.register_state("main", state)
    manager.finish_state_registration()
    first = manager.get_full_attention_states()
    gc.collect()
    assert manager.get_full_attention_states() is first
    assert manager.get_state("main", "gqa", 0) is state
