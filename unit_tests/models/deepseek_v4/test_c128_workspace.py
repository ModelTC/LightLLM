"""C128 capacity, compression reference and graph replay regressions."""

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
    reqs = reqs_for(count, model.args.mtp_step)
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
    w.init_c128_compress_scratch(m)
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
    w.init_c128_compress_scratch(m)
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
    w.init_c128_compress_scratch(m)
    assert w.get_c128_compress_scratch(m._align_decode_batch_size(9 * 6), 0).shape[0] >= 54
    assert w.token_capacity == 8


def test_graph_max_bucket_can_exceed_eager_bound():
    m = model_for(batch=1, requests=17, step=3, tp=8)
    m.graph_max_batch_size = 256  # lower-level model API can explicitly request a larger graph bucket
    w = DeepseekV4Workspace(m)
    w.init_c128_compress_scratch(m)
    assert w.get_c128_compress_scratch(256, 0).shape[0] == 256


def test_prefill_arena_keeps_its_own_capacity():
    m = model_for(batch=4097, requests=256, step=3, tp=8, prefill_overlap=True)
    w = DeepseekV4Workspace(m)
    w.init_c128_compress_scratch(m)
    assert w.c128_compress_scratch.shape == (2, 4104, 512)
    w.init_flashmla_prefill_q(2, 4, 512, torch.bfloat16)
    assert w.get_flashmla_prefill_q(4104).shape == (4104, 4, 512)
    with pytest.raises(ValueError, match="capacity"):
        w.get_flashmla_prefill_q(4105)


@pytest.mark.parametrize("overlap", [False, True])
def test_checked_accessor_rejects_bad_rows_and_regions(overlap):
    m = model_for(overlap=overlap)
    w = DeepseekV4Workspace(m)
    w.init_c128_compress_scratch(m)
    for index in [-1, 2 if overlap else 1]:
        with pytest.raises(ValueError, match="region"):
            w.get_c128_compress_scratch(1, index)
    for rows in [-1, w.c128_compress_scratch.shape[1] + 1]:
        with pytest.raises(ValueError, match="capacity"):
            w.get_c128_compress_scratch(rows, 0)
    assert w.get_c128_compress_scratch(0, 0).shape == (0, 512)


def compress_case(workspace, batch, region=0, boundary=True):
    """Four verify rows per request, with known C128 history and packed output."""
    from lightllm.models.deepseek_v4.layer_infer.compressor import fused_compress

    steps = torch.arange(4, device="cuda", dtype=torch.int32).repeat(batch)
    positions = (124 if boundary else 2) + steps
    request_ids = torch.arange(batch, device="cuda", dtype=torch.int32).repeat_interleave(4)
    runtime = torch.randn(batch + 1, 131, 1024, device="cuda") * 0.4
    values = torch.randn(batch * 4, 1024, device="cuda") * 0.4
    packed = torch.full((batch + 1, 1168), 85, dtype=torch.uint8, device="cuda")
    state = NS(config=NS(window_size=128), get_layer_view=lambda **_: runtime)
    compressed = NS(config=NS(block_token_size=256), get_layer_view=lambda **_: NS(kv=packed))
    infer = NS(
        position_ids=positions,
        mem_index=request_ids * 256 + steps,
        b_req_idx=request_ids,
        b_mtp_index=steps,
        b_seq_len=positions + 1,
        b_ready_cache_len=None,
        b_q_start_loc=None,
        _dsv4_token_to_batch_idx=None,
        is_prefill=False,
        microbatch_index=region,
        dsv4_workspace=workspace,
        get_compressor_state=lambda **_: state,
        get_compressed_kv_state=lambda **_: compressed,
    )
    norm = torch.ones(512, device="cuda", dtype=torch.bfloat16)
    cos = torch.ones(256, 32, device="cuda")
    sin = torch.zeros_like(cos)

    def run():
        return fused_compress(
            kv_score=values,
            infer_state=infer,
            layer_idx=0,
            norm_weight=norm,
            eps=1e-6,
            head_dim=512,
            qk_rope_head_dim=64,
            compress_ratio=128,
            cos_table=cos,
            sin_table=sin,
        )

    return NS(run=run, infer=infer, runtime=runtime, values=values, packed=packed, batch=batch, region=region)


def check_compressed_reference(case, history, workspace):
    current = case.values.view(case.batch, 4, 1024)
    window = torch.cat((history[: case.batch, :124], current), dim=1)
    expected = (window[..., :512] * window[..., 512:].softmax(dim=1)).sum(dim=1)
    actual = workspace.get_c128_compress_scratch(case.batch * 4, case.region)[3::4]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-7)
    torch.testing.assert_close(case.runtime[: case.batch, 124:128], current, rtol=0, atol=0)
    normalized = expected * torch.rsqrt(expected.square().mean(dim=1, keepdim=True) + 1e-6)
    rope = case.packed[: case.batch, 448:576].contiguous().view(torch.bfloat16)
    torch.testing.assert_close(rope.float(), normalized[:, -64:].bfloat16().float(), rtol=0.008, atol=2e-6)
    # Only the first packed slot belongs to a boundary; the second one stays untouched.
    assert (case.packed[: case.batch, 576:1152] == 85).all()
    assert (case.packed[: case.batch, 1160:1168] == 85).all()
    assert (case.packed[case.batch] == 85).all()


@pytest.mark.parametrize("overlap", [False, True])
def test_c128_kernel_preserves_guards_regions_and_reference(overlap):
    torch.manual_seed(1013)
    model = model_for(overlap=overlap, graph=False)
    workspace = DeepseekV4Workspace(model)
    workspace.init_c128_compress_scratch(model)
    regions, rows, head = workspace.c128_compress_scratch.shape
    arena = torch.full((regions, rows + 2, head), 17.0, device="cuda")
    workspace.c128_compress_scratch = arena[:, 1:-1]
    batch = 128 if overlap else 256
    cases = [compress_case(workspace, batch, region) for region in range(regions)]
    histories = [case.runtime.clone() for case in cases]
    streams = [torch.cuda.Stream() for _ in cases]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    for case, stream in zip(cases, streams):
        with torch.cuda.stream(stream):
            case.run()
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for case, history in zip(cases, histories):
        check_compressed_reference(case, history, workspace)
    assert (arena[:, 0] == 17).all() and (arena[:, -1] == 17).all()


def test_c128_capture_without_boundary_then_replay_with_boundary():
    torch.manual_seed(1013)
    workspace = DeepseekV4Workspace(model_for(graph=True))
    workspace.init_c128_compress_scratch(model_for(graph=True))
    case = compress_case(workspace, 256, boundary=False)
    for _ in range(2):
        case.run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        case.run()
    history = case.runtime.clone()
    case.infer.position_ids.copy_(124 + case.infer.b_mtp_index)
    case.infer.b_seq_len.copy_(case.infer.position_ids + 1)
    graph.replay()
    check_compressed_reference(case, history, workspace)


@pytest.mark.parametrize("boundary", [False, True])
def test_short_scratch_is_rejected_before_any_compression_launch(monkeypatch, boundary):
    from lightllm.models.deepseek_v4.layer_infer import compressor

    model = model_for(graph=False)
    workspace = DeepseekV4Workspace(model)
    workspace.init_c128_compress_scratch(model)
    workspace.c128_compress_scratch = torch.empty((1, 256, 512), device="cuda")
    case = compress_case(workspace, 256, boundary=boundary)

    class RejectLaunch:
        def __getitem__(self, grid):
            pytest.fail("unsafe compression kernel was launched")

    monkeypatch.setattr(compressor, "_fused_compress_norm_rope_kernel", RejectLaunch())
    with pytest.raises(ValueError, match="capacity"):
        case.run()
