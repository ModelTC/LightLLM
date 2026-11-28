from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mode_backend.dp_backend import impl
from lightllm.server.router.model_infer.mtp_speculative.acceptance_metrics import SpecAcceptanceMetrics
from lightllm.server.router.model_infer.mtp_speculative.planner import SpecDecodePlan
from lightllm.server.router.model_infer.mtp_speculative.proposers.base import SpecProposal


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("accepted", [1, 3, 6])
def test_overlap_records_output_ranges_after_output_length_update(monkeypatch, side, accepted):
    class Req:
        def __init__(self):
            self.req_idx = 0
            self.cur_output_len = 1
            self.shm_req = SimpleNamespace()
            self.spec_acceptance_metrics = SpecAcceptanceMetrics()
            self.spec_acceptance_metrics.record(1, 1, output_end=1)
            self.counts = [0, 0, 0]

        def get_cur_total_len(self):
            return 1 + self.cur_output_len

        def update_mtp_accepted_token_num(self, accept_token_num):
            self.counts[0] += accept_token_num

        def update_mtp_verify_token_num(self, verify_token_num):
            self.counts[1] += verify_token_num

        def update_mtp_verify_step_num(self, verify_step_num):
            self.counts[2] += verify_step_num

    req = Req()
    device = "cuda"
    inputs = [
        SimpleNamespace(
            batch_size=6 if index == side else 0,
            b_req_idx=torch.zeros(6 if index == side else 0, dtype=torch.int32, device=device),
            b_mtp_index=torch.arange(6 if index == side else 0, dtype=torch.int32, device=device),
        )
        for index in range(2)
    ]
    runs = [[req] * 6 if index == side else [] for index in range(2)]
    decodes = [[req] if index == side else [] for index in range(2)]

    class Engine:
        def plan_decode(self, **kwargs):
            return SpecDecodePlan(6, 6, 5, 5)

        def prepare_decode_model_inputs(self, **kwargs):
            return inputs[0], None, inputs[1], None

        def propose_next_overlap(self, **kwargs):
            return SpecProposal(token_ids=torch.zeros((1, 5), dtype=torch.int64, device=device))

        def update_planner_statics(self, **kwargs):
            pass

    backend = impl.DPChunkedPrefillBackend.__new__(impl.DPChunkedPrefillBackend)
    backend.decode_draft_engine = Engine()
    backend.is_master_in_dp = True
    backend.eos_id = []
    backend.extra_post_req_handle_func = None
    backend.model = SimpleNamespace(
        microbatch_overlap_decode=lambda *args: tuple(
            SimpleNamespace(logits=torch.zeros((item.batch_size, 8), device=device)) for item in inputs
        )
    )
    backend._get_next_token_ranks = lambda logits, ids: torch.zeros_like(ids)
    backend._async_copy_next_token_infos_to_pin_mem = lambda *items: tuple(item.cpu() for item in items)
    exports = []
    backend._post_handle = lambda **kwargs: exports.append(
        req.spec_acceptance_metrics.export(visible_output_tokens=2, inferred_output_tokens=req.cur_output_len)
    )
    monkeypatch.setattr(
        impl,
        "overlap_prepare_decode_inputs",
        lambda **kwargs: (inputs[0], runs[0], decodes[0], inputs[1], runs[1], decodes[1]),
    )
    monkeypatch.setattr(
        impl,
        "sample",
        lambda *args, **kwargs: (
            torch.zeros(6, dtype=torch.int64, device=device),
            torch.zeros(6, device=device),
            [None] * 6,
        ),
    )
    monkeypatch.setattr(
        impl.mtp_utils,
        "verify_mtp_tokens",
        lambda **kwargs: (
            torch.tensor([accepted], dtype=torch.int32, device=device),
            (torch.arange(6, device=device) < accepted).int(),
        ),
    )
    monkeypatch.setattr(impl.mtp_utils, "scatter_mtp_next_tokens", lambda **kwargs: None)
    monkeypatch.setattr(
        impl,
        "g_pin_mem_manager",
        SimpleNamespace(async_copy_from_gpu_tensor=lambda **kwargs: kwargs["gpu_tensor"].cpu()),
    )
    monkeypatch.setattr(
        impl,
        "g_infer_context",
        SimpleNamespace(
            get_overlap_stream=lambda: torch.cuda.current_stream(),
            req_sampling_manager=SimpleNamespace(update_reqs_out_token_counter_gpu=lambda **kwargs: None),
        ),
    )
    events = SimpleNamespace(
        notify_post_handle_and_wait_pre_post_handle=lambda: None,
        notify_forward_and_wait_post_handle=lambda: None,
        notify_pre_post_handle=lambda: None,
    )
    backend.decode_overlap_mtp(events, [req])
    assert req.cur_output_len == 1 + accepted
    assert req.counts == [accepted - 1, 6, 1]
    assert exports[0]["verification_output_ranges"][-1] == [False, 6, accepted, 2]
    assert exports[0]["committed"]["output_tokens"] == 1
