from types import SimpleNamespace

import torch

from lightllm.server.router.model_infer.infer_batch import InferenceContext


def test_hybrid_state_snapshot_is_skipped_without_radix_cache():
    context = InferenceContext.__new__(InferenceContext)
    context.is_hybrid_att_model = True
    context.radix_cache = None
    context.args = SimpleNamespace(linear_att_hash_page_size=128, linear_att_page_block_num=512)

    class ReqManager:
        def save_state(self, **_):
            raise AssertionError("radix-disabled calibration must not save state")

        def save_big_page_states(self, **_):
            raise AssertionError("radix-disabled calibration must not save big-page state")

    context.req_manager = ReqManager()
    req = SimpleNamespace(
        req_idx=0,
        hybrid_cache_len=128,
        tail_small_page_buffer_id=None,
        get_chuncked_input_token_len=lambda: 128,
    )

    context.save_hybrid_state_to_cache(torch.tensor([0]), [req])

    assert req.tail_small_page_buffer_id is None
