from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

import torch

from lightllm.common.att_resources.full_att import (
    GroupQuantizedAttentionConfig,
    GroupQuantizedFullAttentionState,
)


def test_quantized_kv_buffers_include_the_full_hold_page():
    allocations = []
    for bits in (8, 4):
        state = GroupQuantizedFullAttentionState(
            GroupQuantizedAttentionConfig((0, 1, 2), 2, 16, torch.float16, bits=bits),
             page_size=4, device="cpu",
        )
        state.initialize_storage(32)
        allocations.extend(tuple(buffer.shape) for buffer in full_state_test_buffers(state).values())

    assert allocations == [
        (3, 36, 2, 16),
        (3, 36, 2, 16),
        (3, 36, 2, 2),
        (3, 36, 2, 2),
        (3, 36, 2, 8),
        (3, 36, 2, 8),
        (3, 36, 2, 2),
        (3, 36, 2, 2),
    ]
