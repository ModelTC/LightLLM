from unit_tests.common.att_resources.model_helpers import full_state_test_buffers

import torch

from lightllm.common.attention_state.full_att import (
    GroupQuantizedAttentionConfig,
    GroupQuantizedFullAttentionState,
)


def test_quantized_kv_buffers_include_the_full_hold_page():
    allocations = []
    for bits in (8, 4):
        state = GroupQuantizedFullAttentionState(
            GroupQuantizedAttentionConfig(
                logical_layer_groups=((0,), (1,), (2,)),
                global_head_num=2,
                head_dim=16,
                scale_dtype=torch.float16,
                bits=bits,
            ),
             page_size=4, device="cpu",
        )
        state.allocate_storage(32)
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
