from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.fa3.fp8 import Fp8Fa3DecodeAttState


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


class _MtpManager:
    def get_decode_draft_step(self, _is_draft):
        return 2


class _ReqManager:
    HOLD_REQUEST_ID = -1

    def __init__(self, device):
        self.req_to_token_indexs = torch.arange(128, device=device, dtype=torch.int32).view(1, 128)


class _Backend:
    def __init__(self, device, is_draft):
        self.model = SimpleNamespace(
            is_mtp_draft_model=is_draft,
            mtp_manager=_MtpManager(),
            mem_manager=SimpleNamespace(scales=torch.ones((1, 2), device=device), head_num=1),
            req_manager=_ReqManager(device),
            graph=None,
        )

    def uses_causal_attention(self):
        return not self.model.is_mtp_draft_model

    def uses_dynamic_spec_verify_layout(self):
        return False

    def get_page_table_view(self, att_batch_size, max_kv_len, microbatch_index):
        return torch.empty((att_batch_size, max_kv_len), dtype=torch.int32, device="cuda")

    def _find_layer_index(self, **_kwargs):
        return 0


def _state(is_draft=False):
    device = "cuda"
    infer_state = SimpleNamespace(
        b_req_idx=torch.zeros(3, device=device, dtype=torch.int32),
        b_seq_len=torch.tensor([126, 127, 128], device=device, dtype=torch.int32),
        b1_cu_q_seq_len=torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32),
        b1_cu_kv_seq_len=torch.tensor([0, 126, 253, 381], device=device, dtype=torch.int32),
        max_kv_seq_len=128,
        batch_size=3,
        input_ids=torch.empty(3, device=device, dtype=torch.int64),
        microbatch_index=0,
    )
    state = Fp8Fa3DecodeAttState(backend=_Backend(device, is_draft), infer_state=infer_state)
    state.init_state()
    return state


def test_target_mtp_q_scale_is_prefix_invariant_but_draft_keeps_block_layout():
    """Future speculative target rows must not alter causal row 0 through Q scaling."""
    torch.manual_seed(20260909)
    q_a = torch.randn((3, 6, 256), device="cuda", dtype=torch.bfloat16)
    q_b = q_a.clone()
    q_b[1].mul_(1.37)
    q_b[2].mul_(1.73)
    k = (torch.randn((128, 1, 256), device="cuda", dtype=torch.bfloat16) * 0.1).to(torch.float8_e4m3fn)
    v = (torch.randn((128, 1, 256), device="cuda", dtype=torch.bfloat16) * 0.1).to(torch.float8_e4m3fn)

    state_a = _state(is_draft=False)
    state_b = _state(is_draft=False)
    out_a = state_a.decode_att(q_a, k, v)
    out_b = state_b.decode_att(q_b, k, v)
    torch.cuda.synchronize()
    torch.testing.assert_close(out_a[0], out_b[0], rtol=0, atol=0)

    draft_state = _state(is_draft=True)
    assert draft_state.decode_max_q_seq_len == 3
    assert draft_state.cu_seqlens_q.tolist() == [0, 3]
    assert draft_state.causal is False
