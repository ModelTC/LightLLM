from types import SimpleNamespace

import pytest

from lightllm.models.sensenova_loop.config import validate_inline_vision


def _args(**changes):
    values = dict(
        visual_infer_in_llm=True,
        disable_vision=False,
        disable_audio=True,
        visual_use_proxy_mode=False,
        tp=1,
        dp=1,
        nnodes=1,
        visual_tp=1,
        visual_dp=1,
        running_max_req_size=1,
        run_mode="normal",
        mtp_mode=None,
        enable_cpu_cache=False,
        enable_prefill_cudagraph=False,
        enable_prefill_microbatch_overlap=False,
        enable_decode_microbatch_overlap=False,
    )
    return SimpleNamespace(**(values | changes))


def test_inline_vision_keeps_decode_graph_and_rejects_unsupported_modes():
    validate_inline_vision(_args(), "sensenova_loop")
    validate_inline_vision(_args(visual_infer_in_llm=False, tp=4), "other")
    for change in (
        {"tp": 2},
        {"visual_gpu_ids": [1]},
        {"running_max_req_size": 2},
        {"run_mode": "prefill"},
        {"disable_vision": True},
        {"enable_prefill_cudagraph": True},
        {"enable_decode_microbatch_overlap": True},
        {"enable_cpu_cache": True},
    ):
        with pytest.raises(ValueError):
            validate_inline_vision(_args(**change), "sensenova_loop")
    with pytest.raises(ValueError, match="SenseNova Loop"):
        validate_inline_vision(_args(), "qwen3_5")
