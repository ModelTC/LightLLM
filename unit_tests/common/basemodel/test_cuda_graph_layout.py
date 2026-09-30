import math
from types import SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.basemodel as basemodel_module
import lightllm.common.basemodel.cuda_graph as cuda_graph_module
import lightllm.common.basemodel.mtp_manager as mtp_manager_module
from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.basemodel.cuda_graph import CudaGraph
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.mtp_manager import MtpManager


@pytest.fixture(autouse=True)
def _graph_args(monkeypatch):
    args = SimpleNamespace(
        enable_decode_microbatch_overlap=False,
        enable_tpsp_mix_mode=False,
        enable_torch_memory_saver=False,
    )
    monkeypatch.setattr(cuda_graph_module, "get_env_start_args", lambda: args)
    return args


def _batch_sizes(max_batch_size, batch_stride=1):
    physical_max_batch_size = max_batch_size * batch_stride
    graph = CudaGraph(
        batch_step_size_before_split=batch_stride,
        split_batch_size=4 * batch_stride,
        batch_step_size_after_split=2 * batch_stride,
        max_batch_size=physical_max_batch_size,
    )
    return graph.cuda_graph_batch_sizes


def test_dynamic_schedule_uses_compacted_physical_rows(_graph_args):
    assert _batch_sizes(max_batch_size=128) == [1, 2, 3, 4, *range(6, 129, 2)]


def test_public_static_schedule_preserves_original_static_mtp_default(_graph_args):
    assert CudaGraph.gen_cuda_graph_batch_sizes(
        batch_step_size_before_split=8,
        split_batch_size=32,
        batch_step_size_after_split=16,
        max_batch_size=32,
    ) == [
        8,
        16,
        24,
        32,
    ]


def test_instance_and_public_static_schedule_match(_graph_args):
    graph = CudaGraph(
        batch_step_size_before_split=8,
        split_batch_size=32,
        batch_step_size_after_split=16,
        max_batch_size=128,
    )

    assert graph.cuda_graph_batch_sizes == CudaGraph.gen_cuda_graph_batch_sizes(
        batch_step_size_before_split=8,
        split_batch_size=32,
        batch_step_size_after_split=16,
        max_batch_size=graph.max_batch_size,
        tp_world_size=graph.tp_world_size,
    )


def test_batch_step_size_before_split_controls_capture_range(_graph_args):
    assert _batch_sizes(max_batch_size=4, batch_stride=8) == [8, 16, 24, 32]


def test_batch_step_size_after_split_controls_capture_range(_graph_args):
    assert _batch_sizes(max_batch_size=8, batch_stride=7) == [
        7,
        14,
        21,
        28,
        42,
        56,
    ]


def test_token_forward_keeps_mtp_hidden_capture_buffer_for_replay():
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.layers_num = 0
    model.layers_infer = []
    model.trans_layers_weight = []
    model.pre_post_weight = None
    model.pre_infer = SimpleNamespace(
        token_forward=lambda input_ids, infer_state, layer_weight: infer_state.mtp_draft_input_hiddens,
        _tpsp_sp_split=lambda input, infer_state: input,
    )
    model.post_infer = SimpleNamespace(
        _tpsp_allgather=lambda input, infer_state: input,
        token_forward=lambda input, infer_state, layer_weight: object(),
    )
    output = SimpleNamespace(to_no_ref_tensor=lambda: None)
    model._create_model_output = lambda post_output, infer_state: output

    def make_state(hidden, is_cuda_graph):
        state = InferStateInfo()
        state.input_ids = torch.zeros(hidden.shape[0], dtype=torch.int64)
        state.mtp_draft_input_hiddens = hidden
        state.is_cuda_graph = is_cuda_graph
        state.hidden_collector = SimpleNamespace(add_final_hidden=lambda value: None)
        state.decode_att_state = SimpleNamespace(copy_for_decode_cuda_graph=lambda value: None)
        return state

    capture_hidden = torch.zeros((2, 4))
    graph_state = make_state(capture_hidden, is_cuda_graph=True)
    model._token_forward(graph_state)

    assert graph_state.mtp_draft_input_hiddens is capture_hidden
    replay_state = make_state(torch.ones((2, 4)), is_cuda_graph=False)
    graph_state.copy_for_cuda_graph(replay_state)
    assert graph_state.mtp_draft_input_hiddens.sum().item() == 8

    eager_state = make_state(torch.ones((2, 4)), is_cuda_graph=False)
    model._token_forward(eager_state)
    assert eager_state.mtp_draft_input_hiddens is None


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
@pytest.mark.parametrize("mtp_step", [1, 2])
@pytest.mark.parametrize("dynamic,is_draft", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("overlap", [False, True])
def test_mtp_tpsp_layout(monkeypatch, _graph_args, tp_size, mtp_step, dynamic, is_draft, overlap):
    args = _graph_args
    args.enable_tpsp_mix_mode = True
    args.enable_decode_microbatch_overlap = overlap
    args.mtp_mode = "eagle_with_att"
    args.mtp_step = mtp_step
    args.mtp_dynamic_verify = dynamic
    monkeypatch.setattr(mtp_manager_module, "get_env_start_args", lambda: args)
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.is_mtp_draft_model = is_draft
    monkeypatch.setattr(basemodel_module, "get_env_start_args", lambda: args)
    monkeypatch.setattr(basemodel_module, "get_llm_data_type", lambda: None)
    monkeypatch.setattr(basemodel_module, "get_dp_world_size", lambda: tp_size)
    monkeypatch.setattr(MtpManager, "_instance", MtpManager())

    class StopBeforeWeights(Exception):
        pass

    def stop_before_weights():
        raise StopBeforeWeights

    # 执行真实构造函数的容量计算，在读取模型配置、分配 GPU 权重前停止。
    monkeypatch.setattr(model, "_init_config", stop_before_weights)
    with pytest.raises(StopBeforeWeights):
        model.__init__(dict(run_mode="normal", weight_dir="", max_total_token_num=1024, graph_max_batch_size=7))

    width = 1 if dynamic or is_draft else mtp_step + 1
    logical_max = 7 // 2 if overlap else 7
    physical_max = logical_max * (1 if is_draft else mtp_step + 1)
    alignment = math.lcm(width, tp_size)
    assert model.graph_max_batch_size % alignment == 0
    assert physical_max <= model.graph_max_batch_size < physical_max + alignment

    sizes = CudaGraph.gen_cuda_graph_batch_sizes(
        batch_step_size_before_split=width,
        split_batch_size=4 * width,
        batch_step_size_after_split=2 * width,
        max_batch_size=model.graph_max_batch_size,
        tp_world_size=tp_size,
    )
    assert sizes[-1] == model.graph_max_batch_size
    assert all(size % width == 0 and size % tp_size == 0 for size in sizes)
    if tp_size == 8 and width == 3:
        assert sizes == [24]
