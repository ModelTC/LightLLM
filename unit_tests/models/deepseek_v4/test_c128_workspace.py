"""C128 scratch must cover actual decode/MTP rows and padded graph buckets."""

import bisect
from types import SimpleNamespace as NS, MethodType
from unittest.mock import patch
import pytest
import torch
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.mtp_manager import MtpManager
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.basemodel import cuda_graph as graph_impl
from lightllm.models.deepseek_v4.workspace import DeepseekV4Workspace
from lightllm.server.core.objs.start_args_type import StartArgs
from lightllm.server.router.model_infer.mode_backend.generic_pre_process import (
    prepare_decode_inputs,
    overlap_prepare_decode_inputs,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def model_for(
    batch=256,
    requests=256,
    step=3,
    tp=1,
    overlap=False,
    prefill_overlap=False,
    graph_requests=64,
    graph=True,
    dynamic=False,
):
    args = StartArgs(
        batch_max_tokens=batch,
        running_max_req_size=requests,
        mtp_mode="dspark" if step else None,
        mtp_step=step,
        mtp_dynamic_verify=dynamic,
        enable_decode_microbatch_overlap=overlap,
        enable_prefill_microbatch_overlap=prefill_overlap,
        enable_tpsp_mix_mode=overlap or prefill_overlap or tp > 1,
    )
    mtp = MtpManager.__new__(MtpManager)
    mtp.args = args
    model = NS(
        batch_max_tokens=batch,
        max_req_num=requests,
        tp_world_size_=tp,
        args=args,
        mtp_manager=mtp,
        is_mtp_draft_model=False,
        disable_cudagraph=not graph,
        config={"head_dim": 512},
        run_mode="decode",
        token_lookback_depth=0,
        req_manager=NS(HOLD_REQUEST_ID=requests),
    )
    model._align_decode_batch_size = MethodType(TpPartBaseModel._align_decode_batch_size, model)
    graph_requests = min(graph_requests, requests)
    model.graph_max_batch_size = model._align_decode_batch_size(
        (graph_requests // 2 if overlap else graph_requests) * mtp.get_decode_tokens_per_request(False)
    )
    return model


def reqs_for(count, step):
    return [
        NS(
            req_idx=i,
            mtp_step=step,
            cur_kv_len=124,
            get_cur_total_len=lambda: 125,
            multimodal_params={"images": [], "audios": []},
            get_radix_cache_shared_len=lambda: 0,
            shared_kv_node=None,
        )
        for i in range(count)
    ]


def inputs_for(model, count):
    from lightllm.server.router.model_infer.infer_batch import g_infer_context

    reqs = reqs_for(count, model.args.mtp_step)
    with patch.object(g_infer_context, "backend", NS(model=model)):
        if model.args.enable_decode_microbatch_overlap:
            result = overlap_prepare_decode_inputs(reqs)
            return [result[0], result[3]]
        return [prepare_decode_inputs(reqs)[0]]


def graph_buckets(model):
    with patch.object(graph_impl, "get_env_start_args", lambda: model.args):
        width = model.mtp_manager.get_decode_tokens_per_request(False)
        alignment = model.mtp_manager.get_decode_batch_alignment(False)
        return CudaGraph.gen_cuda_graph_batch_sizes(
            alignment, 32 * width, 8 * alignment, model.graph_max_batch_size, model.tp_world_size_
        )


@pytest.mark.parametrize("overlap", [False, True])
def test_reported_overflow_reproduction_is_now_covered(overlap):
    m = model_for(overlap=overlap)
    w = DeepseekV4Workspace(m)
    w.init_persistent_scratch(m)
    assert (
        w.c128_compress_scratch.shape == (2, 512, 512) if overlap else w.c128_compress_scratch.shape == (1, 1024, 512)
    )
    for region, input in enumerate(inputs_for(m, 256)):
        position = input.b_seq_len - 1
        boundaries = torch.where((position + 1) % 128 == 0)[0]
        assert int(boundaries[boundaries >= 256][0]) == 259
        scratch = w.get_c128_compress_scratch(input.batch_size, region)
        assert int(boundaries.max()) < scratch.shape[0]
    assert w.token_capacity == 256


@pytest.mark.parametrize("requests", [17, 256, 257])
@pytest.mark.parametrize("step", [0, 3, 5])
@pytest.mark.parametrize("tp", [1, 8])
@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_real_decode_mtp_tp_and_graph_padding_fit(requests, step, tp, overlap, graph):
    m = model_for(batch=9, requests=requests, step=step, tp=tp, overlap=overlap, graph=graph)
    w = DeepseekV4Workspace(m)
    w.init_persistent_scratch(m)
    buckets = graph_buckets(m) if graph else []
    for count in [0, 1, requests // 2, requests]:
        for region, input in enumerate(inputs_for(m, count)):
            rows = m._align_decode_batch_size(max(1, input.batch_size))
            if buckets and rows <= buckets[-1]:
                rows = buckets[bisect.bisect_left(buckets, rows)]
            input.input_ids = torch.zeros(input.batch_size, dtype=torch.int64)
            padded = TpPartBaseModel._create_padded_decode_model_input(m, input, rows)
            assert padded.batch_size == rows
            assert w.get_c128_compress_scratch(rows, region).shape == (rows, 512)
    for rows in buckets:
        assert w.get_c128_compress_scratch(rows, 0).shape[0] == rows
    assert w.token_capacity == ((9 + tp - 1) // tp) * tp


@pytest.mark.parametrize("dynamic", [False, True])
def test_dynamic_verify_plans_for_maximum_width_and_odd_microbatch(dynamic):
    m = model_for(batch=1, requests=17, step=5, tp=8, overlap=True, graph=True, dynamic=dynamic)
    w = DeepseekV4Workspace(m)
    w.init_persistent_scratch(m)
    assert w.get_c128_compress_scratch(m._align_decode_batch_size(9 * 6), 0).shape[0] >= 54
    assert w.token_capacity == 8


def test_graph_max_bucket_can_exceed_eager_bound():
    m = model_for(batch=1, requests=17, step=3, tp=8)
    m.graph_max_batch_size = 256  # lower-level model API can explicitly request a larger graph bucket
    w = DeepseekV4Workspace(m)
    w.init_persistent_scratch(m)
    assert w.get_c128_compress_scratch(256, 0).shape[0] == 256


def test_prefill_arena_keeps_its_own_capacity():
    m = model_for(batch=4097, requests=256, step=3, tp=8, prefill_overlap=True)
    w = DeepseekV4Workspace(m)
    w.init_persistent_scratch(m)
    assert w.c128_compress_scratch.shape == (2, 4104, 512)
    w.init_flashmla_prefill_q(2, 4, 512, torch.bfloat16)
    assert w.get_flashmla_prefill_q(4104).shape == (4104, 4, 512)
    with pytest.raises(ValueError, match="capacity"):
        w.get_flashmla_prefill_q(4105)


@pytest.mark.parametrize("overlap", [False, True])
def test_checked_accessor_rejects_bad_rows_and_regions(overlap):
    m = model_for(overlap=overlap)
    w = DeepseekV4Workspace(m)
    w.init_persistent_scratch(m)
    for index in [-1, 2 if overlap else 1]:
        with pytest.raises(ValueError, match="region"):
            w.get_c128_compress_scratch(1, index)
    for rows in [-1, w.c128_compress_scratch.shape[1] + 1]:
        with pytest.raises(ValueError, match="capacity"):
            w.get_c128_compress_scratch(rows, 0)
    assert w.get_c128_compress_scratch(0, 0).shape == (0, 512)
