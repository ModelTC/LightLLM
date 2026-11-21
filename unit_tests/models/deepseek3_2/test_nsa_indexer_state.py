"""The DeepSeek V3.2 infer state prepares the NSA indexer's per-step metadata."""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.triton_kernel.gen_nsa_ks_ke import gen_nsa_ks_ke
from lightllm.models.deepseek3_2.infer_struct import Deepseek3_2InferStateInfo

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def make_state(prefill, lengths, q_lens, microbatch_index=0):
    s = Deepseek3_2InferStateInfo()
    s.is_prefill = prefill
    s.batch_size, s.microbatch_index = len(lengths), microbatch_index
    s.b_req_idx = torch.arange(len(lengths), device="cuda", dtype=torch.int32)
    s.b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    s.b_q_seq_len = torch.tensor(q_lens, device="cuda", dtype=torch.int32)
    s.max_kv_seq_len, s.total_token_num = max(lengths), sum(lengths)
    s.input_ids = torch.zeros(sum(q_lens), device="cuda", dtype=torch.int64)
    token_map = torch.randperm(len(lengths) * 64, device="cuda", dtype=torch.int32).view(len(lengths), 64)
    s.req_manager = SimpleNamespace(req_to_token_indexs=token_map, HOLD_REQUEST_ID=63)
    return s


def expected(s, q_token_num):
    ragged = torch.empty(s.total_token_num, dtype=torch.int32, device="cuda")
    ks, ke, lengths = gen_nsa_ks_ke(
        b_seq_len=s.b_seq_len,
        b_q_seq_len=s.b_q_seq_len,
        b_req_idx=s.b_req_idx,
        req_to_token_index=s.req_manager.req_to_token_indexs,
        q_token_num=q_token_num,
        ragged_mem_index=ragged,
        hold_req_idx=63,
    )
    return ks, ke, lengths, ragged


@pytest.mark.parametrize("prefill", [True, False])
def test_indexer_metadata_matches_gen_nsa_ks_ke(prefill):
    model = SimpleNamespace(graph_max_batch_size=2, graph_max_len_in_batch=64, max_seq_length=64)
    lengths = [17, 40, 5]
    s = make_state(prefill, lengths, [9, 40, 2] if prefill else [1, 1, 1])
    s._init_nsa_indexer_state(model)
    ks, ke, lens, ragged = expected(s, s.input_ids.shape[0] if prefill else 3)
    for actual, ref in ((s.nsa_ks, ks), (s.nsa_ke, ke), (s.nsa_lengths, lens)):
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
    torch.testing.assert_close(s.nsa_ragged_mem_index[: s.total_token_num], ragged, rtol=0, atol=0)


def test_decode_steps_a_graph_may_run_use_one_buffer_per_microbatch():
    model = SimpleNamespace(graph_max_batch_size=4, graph_max_len_in_batch=64, max_seq_length=64)
    pointers = []
    for microbatch_index in (0, 0, 1):
        s = make_state(False, [17, 40], [1, 1], microbatch_index)
        s._init_nsa_indexer_state(model)
        pointers.append(s.nsa_ragged_mem_index.data_ptr())
    assert pointers[0] == pointers[1] != pointers[2]
    # Too long for the graph: a fresh buffer the size of the step's tokens.
    s = make_state(False, [17, 40], [1, 1])
    s._init_nsa_indexer_state(SimpleNamespace(graph_max_batch_size=4, graph_max_len_in_batch=32, max_seq_length=64))
    assert s.nsa_ragged_mem_index.data_ptr() not in pointers and s.nsa_ragged_mem_index.numel() == 57
