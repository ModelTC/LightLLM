from dataclasses import replace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.models.gemma4.model import Gemma4TpPartModel
from lightllm.server.core.objs.start_args_type import StartArgs


@pytest.mark.parametrize("tp", [1, 4])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gemma4_uses_native_width_and_shares_only_declared_content(tp, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    config = dict(
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=256,
        num_global_key_value_heads=4,
        global_head_dim=512,
        hidden_size=4096,
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        layer_types=["sliding_attention", "full_attention"] * 2,
    )
    states = Gemma4TpPartModel.create_full_att_states(
        config, StartArgs(data_type="bfloat16", page_size=4), (3, 7, 9, 11), tp_world_size=tp
    )
    manager = AttentionStateManager(2)
    for state in states:
        state.device = torch.device(device)
        state.allocate_storage(16)
        manager.register_state("main", state)
    manager.finish_state_registration()
    sliding, full = manager.get_state("main", "gqa", 3), manager.get_state("main", "gqa", 7)
    assert sliding.config.logical_layer_groups == ((3, 9),)
    assert full.config.logical_layer_groups == ((7, 11),)
    assert sliding.get_token_bytes() == 16384 // tp
    assert full.get_token_bytes() == 8192 // tp
    assert full.k_storage.shape == (1, 20, 4 // tp, 512)
    assert manager.get_state("main", "gqa", 11) is full
    assert full.get_layer_view(7)[0].data_ptr() == full.get_layer_view(11)[0].data_ptr()
    assert sliding.k_storage.data_ptr() != full.k_storage.data_ptr()
    if device == "cuda":
        indices = torch.tensor([1, 6], dtype=torch.int32, device=device)
        kv = torch.randn(2, 8 // tp, 512, dtype=torch.bfloat16, device=device)
        full.write_full_att_kv(7, indices, kv)
        k, v = full.get_layer_view(11)
        torch.testing.assert_close(torch.cat((k[indices], v[indices]), dim=1), kv, atol=0, rtol=0)
