from contextlib import nullcontext
from threading import Event, Thread
from types import SimpleNamespace

import pytest
import torch

from unit_tests.server.grammar_helpers import (
    init_request,
    make_req,
    make_mask_buffers,
    apply_masks,
    prepare_sampling_tensors,
)
from lightllm.server.router.model_infer.mode_backend.chunked_prefill import impl as chunked_impl
from lightllm.server.router.model_infer.mode_backend.dp_backend import impl as dp_impl
from lightllm.server.router.model_infer.mode_backend.overlap_events import OverlapEventManager


@pytest.mark.parametrize("prefill_first", [False, True])
@pytest.mark.parametrize("constrained", [False, True, "reasoning"])
@pytest.mark.parametrize(
    "dp,microbatch,empty_second",
    [(False, False, False), (True, False, False), (True, False, True), (True, True, False), (True, True, True)],
)
def test_overlap_handoff_completes_postprocessing_before_sampling(
    compiler, monkeypatch, prefill_first, constrained, dp, microbatch, empty_second
):
    """Exercise the real two-thread event protocol and chunked-prefill methods.

    Sampling must see the previous matcher and CPU token counters.
    All paths still submit the next forward before releasing CPU postprocessing.
    """
    impl = dp_impl if dp else chunked_impl
    backend_class = dp_impl.DPChunkedPrefillBackend if dp else chunked_impl.ChunkedPrefillBackend
    backend = object.__new__(backend_class)
    backend.args = SimpleNamespace(penalty_counter_mode="cpu_counter")
    backend.disable_chunked_prefill = False
    backend.extra_post_req_handle_func = backend.pd_prefill_chunked_handle_func = None
    backend._capture_prompt_logprobs_if_needed = lambda *args: None
    req = make_req(
        regular_constraint="b" if constrained == "reasoning" else "ab" if constrained else None,
        guided_reasoning_end=(ord("a"),) if constrained == "reasoning" else (),
    )
    req.out_token_id_count = {}
    init_request(compiler, req)
    mask_buffers = make_mask_buffers([req])
    model_input = SimpleNamespace(
        b_req_idx=torch.tensor([0]), b_mtp_index=torch.tensor([0]), b_prefill_has_output_cpu=[True]
    )
    empty_input = SimpleNamespace(b_req_idx=torch.empty(0), b_mtp_index=torch.empty(0), b_prefill_has_output_cpu=[])
    monkeypatch.setattr(impl, "prepare_decode_inputs", lambda req_objs: (model_input, req_objs))
    monkeypatch.setattr(impl, "prepare_prefill_inputs", lambda req_objs, **kwargs: (model_input, req_objs))
    if microbatch:
        monkeypatch.setattr(
            impl,
            "overlap_prepare_prefill_inputs",
            lambda reqs: (model_input if reqs else empty_input, reqs, empty_input, []),
        )
        monkeypatch.setattr(
            impl,
            "overlap_prepare_decode_inputs",
            lambda req_objs: (model_input if req_objs else empty_input, req_objs, req_objs, empty_input, [], []),
        )
    monkeypatch.setattr(impl.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(
        impl.torch.cuda, "Event", lambda: SimpleNamespace(record=lambda: None, synchronize=lambda: None)
    )
    monkeypatch.setattr(impl.g_infer_context, "get_overlap_stream", lambda: None)
    monkeypatch.setattr(impl.g_infer_context, "save_hybrid_state_to_cache", lambda **kwargs: None)
    monkeypatch.setattr(impl.g_infer_context, "is_hybrid_att_model", False)
    second_forward = Event()
    order, errors = [], []

    def forward(model_input):
        step = sum(event.startswith("forward") for event in order)
        order.append(f"forward{step}")
        if step == 1:
            second_forward.set()
        logits = torch.zeros(model_input.b_req_idx.numel(), 257)
        logits[:, ord("a") + step] = 1
        return SimpleNamespace(logits=logits, prompt_logics=None)

    def forward_microbatch(first, second):
        return forward(first), SimpleNamespace(logits=torch.zeros(0, 257), prompt_logics=None)

    backend.model = SimpleNamespace(
        forward=forward, microbatch_overlap_prefill=forward_microbatch, microbatch_overlap_decode=forward_microbatch
    )

    def sample(**kwargs):
        if second_forward.is_set():
            assert req.out_token_id_count.get(ord("a"), 0) == 1, "Sampling read stale CPU token counters"
        logits = kwargs["logits"]
        reqs = kwargs["run_reqs"]
        if prepare_sampling_tensors(reqs, mask_buffers)[-1]:
            apply_masks(reqs, logits, mask_buffers)
        token = logits.argmax(-1)
        order.append(f"sample{chr(token.item())}")
        return token, token, torch.zeros(1), torch.zeros(1, dtype=torch.int64)

    backend._sample_and_scatter_token = sample

    def pre_post(*args, **kwargs):
        req.cur_output_len += 1
        return []

    backend._pre_post_handle = pre_post

    def post(**kwargs):
        token = kwargs["next_token_ids"].item()
        if token == ord("a"):
            assert second_forward.wait(3), "CPU postprocessing must overlap the next forward"
        if req.output_constraint is not None:
            req.output_constraint.commit(token)
        req.out_token_id_count[token] = req.out_token_id_count.get(token, 0) + 1
        order.append(f"post{chr(token)}")

    backend._post_handle = post
    events = OverlapEventManager()
    packs = [events.get_overlap_event_pack() for _ in range(3)]

    def run(step):
        try:
            packs[step].wait_to_forward()
            run_reqs = [] if step == 1 and empty_second else [req]
            if prefill_first and (step == 0 or empty_second):
                method = backend.prefill_overlap if microbatch else backend.prefill_normal
            else:
                method = backend.decode_overlap if microbatch else backend.decode_normal
            method(packs[step], run_reqs)
        except BaseException as exc:
            errors.append(exc)

    def drain():
        packs[2].wait_to_forward()
        packs[2].notify_post_handle_and_wait_pre_post_handle()

    threads = [Thread(target=run, args=(step,), daemon=True) for step in range(2)]
    threads.append(Thread(target=drain, daemon=True))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert not errors
    assert not any(thread.is_alive() for thread in threads), f"Overlap deadlocked: {order}"
    expected = ["forward0", "samplea", "forward1"]
    if empty_second:
        expected += ["posta"]
    else:
        expected += ["posta", "sampleb", "postb"]
    assert order == expected
