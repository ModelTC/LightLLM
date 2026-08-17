from types import SimpleNamespace

import torch

from lightllm.server.router.model_infer.mode_backend.dp_backend.impl import DPChunkedPrefillBackend
from lightllm.server.router.model_infer.speculative.proposers.base import SpecProposal


class _RecordingSpecEngine:
    def __init__(self):
        self.propose_args = None
        self.propose_overlap_args = None
        self.scatter_args = None

    def propose_next(self, **kwargs):
        self.propose_args = kwargs
        token_ids = kwargs["next_token_ids"].new_zeros((16, 8))
        return SpecProposal(
            token_ids=token_ids,
            extra_mem_indexes_cpu=torch.tensor([123], dtype=torch.int32),
        )

    def propose_next_overlap(self, **kwargs):
        self.propose_overlap_args = kwargs
        row_count = kwargs["real_verify_rows0"] + kwargs["real_verify_rows1"]
        token_ids = kwargs["next_token_ids0"].new_zeros((row_count, 8))
        return SpecProposal(
            token_ids=token_ids,
            extra_mem_indexes_cpu=torch.tensor([456], dtype=torch.int32),
        )

    def scatter_next_tokens(self, **kwargs):
        self.scatter_args = kwargs


def test_padded_token_ids_support_empty_dp_rank():
    padded_token_ids = DPChunkedPrefillBackend._build_padded_next_token_ids(
        token_ids=None,
        batch_size=4,
        copy_len=0,
        device=torch.device("cpu"),
    )

    assert torch.equal(padded_token_ids, torch.zeros(4, dtype=torch.int64))


def test_dp_eagle_uses_common_extend_then_unit_decode_proposer():
    backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    backend.max_draft_step = 7
    backend.spec_engine = _RecordingSpecEngine()
    model_input = SimpleNamespace(
        batch_size=16,
        b_req_idx=torch.arange(16, dtype=torch.int32),
    )
    model_output = SimpleNamespace(spec_hidden=torch.randn(16, 4))
    next_token_ids = torch.arange(8, dtype=torch.int64)
    real_start_locs = torch.tensor([0], dtype=torch.int32)
    real_accept_len = torch.tensor([2], dtype=torch.int32)

    extra_mem = backend._draft_decode_eagle(
        model_input=model_input,
        model_output=model_output,
        next_token_ids=next_token_ids,
        b_req_mtp_start_loc=real_start_locs,
        mtp_accept_len=real_accept_len,
        req_num=8,
    )

    propose_args = backend.spec_engine.propose_args
    assert propose_args["next_token_ids"].shape == (16,)
    assert torch.equal(propose_args["next_token_ids"][:8], next_token_ids)
    assert torch.equal(propose_args["b_req_mtp_start_loc"], torch.tensor([0, 8], dtype=torch.int32))
    assert torch.equal(propose_args["accept_len"], torch.tensor([2, 1], dtype=torch.int32))
    assert backend.spec_engine.scatter_args["all_next_token_ids"].shape == (8, 8)
    assert torch.equal(extra_mem, torch.tensor([123], dtype=torch.int32))


def test_dp_overlap_eagle_passes_both_fixed_verify_layouts_to_proposer():
    backend = DPChunkedPrefillBackend.__new__(DPChunkedPrefillBackend)
    backend.max_draft_step = 7
    backend.spec_engine = _RecordingSpecEngine()
    model_input0 = SimpleNamespace(
        batch_size=16,
        b_req_idx=torch.arange(16, dtype=torch.int32),
    )
    model_input1 = SimpleNamespace(
        batch_size=16,
        b_req_idx=torch.arange(16, dtype=torch.int32),
    )
    model_output0 = SimpleNamespace(spec_hidden=torch.randn(16, 4))
    model_output1 = SimpleNamespace(spec_hidden=torch.randn(16, 4))
    next_token_ids = torch.arange(24, dtype=torch.int64)
    b_req_idx = torch.arange(24, dtype=torch.int32)
    start_locs = torch.tensor([0, 8, 16], dtype=torch.int32)
    accept_len = torch.tensor([2, 3, 4], dtype=torch.int32)

    extra_mem = backend._draft_decode_eagle_overlap(
        model_input0=model_input0,
        model_output0=model_output0,
        model_input1=model_input1,
        model_output1=model_output1,
        b_req_idx=b_req_idx,
        next_token_ids=next_token_ids,
        mtp_accept_len=accept_len,
        b_req_mtp_start_loc=start_locs,
        req_num0=8,
        req_num1=16,
    )

    propose_args = backend.spec_engine.propose_overlap_args
    assert propose_args["next_token_ids0"].shape == (16,)
    assert propose_args["next_token_ids1"].shape == (16,)
    assert propose_args["real_verify_rows0"] == 8
    assert propose_args["real_verify_rows1"] == 16
    assert torch.equal(propose_args["accept_len0"], torch.tensor([2, 1], dtype=torch.int32))
    assert torch.equal(propose_args["accept_len1"], torch.tensor([3, 4], dtype=torch.int32))
    assert backend.spec_engine.scatter_args["all_next_token_ids"].shape == (24, 8)
    assert torch.equal(extra_mem, torch.tensor([456], dtype=torch.int32))
