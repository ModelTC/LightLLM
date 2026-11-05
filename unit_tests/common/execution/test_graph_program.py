from types import SimpleNamespace

import pytest
import torch

from lightllm.common.execution.runner.graph_program import GraphProgram


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("probe", [False, True])
def test_segments_use_current_batch_and_keep_dynamic_outputs_alive(probe):
    source = torch.arange(16, device="cuda", dtype=torch.float32)
    initial = SimpleNamespace(offset=3, bias=5)
    calls = []
    program = GraphProgram(torch.cuda.graph_pool_handle())

    def attention(inputs, batch):
        assert inputs[1] is None
        calls.append(("attention", batch))
        return (inputs[0] + batch.offset,)

    def update(inputs, batch):
        calls.append(("update", batch))
        inputs[0].add_(batch.bias)

    with program.capture():
        hidden = source * 2
        specs = None if probe else ((source.shape, source.dtype, source.device),)
        (dynamic,) = program.call(attention, (hidden, None), initial, specs)
        program.defer(update, (dynamic,))
        output = dynamic * 3
    assert len(calls) == int(probe)
    del hidden, dynamic
    for offset, bias in ((7, 11), (-3, 2)):
        source.add_(1)
        batch = SimpleNamespace(offset=offset, bias=bias)
        program.replay(batch)
        torch.testing.assert_close(output, (source * 2 + offset + bias) * 3, rtol=0, atol=0)
        assert calls[-2:] == [("attention", batch), ("update", batch)]


@pytest.mark.parametrize("where", ["probe", "segment"])
def test_failed_capture_exits_cuda_capture_and_is_not_replayable(where):
    source = torch.ones(8, device="cuda")
    program = GraphProgram(torch.cuda.graph_pool_handle())

    def fail(inputs, batch):
        raise ValueError("injected shape probe failure")

    with pytest.raises(ValueError):
        with program.capture():
            hidden = source + 1
            if where == "probe":
                program.call(fail, (hidden,), None)
            else:
                raise ValueError("injected segment failure")
    assert not torch.cuda.is_current_stream_capturing()
    assert not program.is_capturing
    with pytest.raises(RuntimeError, match="incomplete"):
        program.replay(None)
    fresh = GraphProgram(torch.cuda.graph_pool_handle())
    with fresh.capture():
        output = source + 2
    fresh.replay(None)
    torch.testing.assert_close(output, source + 2)


def test_replay_rejects_changed_dynamic_output_layout():
    source = torch.ones(8, device="cuda")
    program = GraphProgram(torch.cuda.graph_pool_handle())
    with program.capture():
        hidden = source + 1
        program.call(
            lambda inputs, batch: (inputs[0][: batch.count],),
            (hidden,),
            SimpleNamespace(count=8),
            ((source.shape, source.dtype, source.device),),
        )
    with pytest.raises(ValueError, match="layout changed"):
        program.replay(SimpleNamespace(count=4))


@pytest.mark.parametrize("moe", [False, True])
def test_multimodal_recipe_reads_current_image_features(moe):
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from lightllm.models.qwen3_vl.layer_infer.transformer_layer_infer import Qwen3VLTransformerLayerInfer
    from lightllm.models.qwen3_vl_moe.layer_infer.transformer_layer_infer import Qwen3VLMOETransformerLayerInfer

    cls = Qwen3VLMOETransformerLayerInfer if moe else Qwen3VLTransformerLayerInfer
    layer = cls.__new__(cls)
    source = torch.randn(4, 32, device="cuda")
    program = GraphProgram(torch.cuda.graph_pool_handle())
    initial = InferStateInfo()
    initial.prefill_program = program
    with program.capture():
        hidden = source * 2
        layer._apply_deepstack_features_wrapper_run(hidden, initial, 0)
        output = hidden + 3
    for ids in ([1, 100, 102, 2], [101, 2, 100, 102]):
        batch = InferStateInfo()
        batch.input_ids = torch.tensor(ids, device="cuda")
        batch.cpu_embed_cache_tensor = torch.randn(3, 2, 32, pin_memory=True)
        batch.img_token_lens = torch.tensor([3], device="cuda")
        batch.img_start_token_ids = torch.tensor([100], device="cuda")
        batch.img_start_locs_in_cache = torch.tensor([0], device="cuda")
        program.replay(batch)
        expected = source * 2 + 3
        for row, token in enumerate(ids):
            if 100 <= token < 103:
                expected[row] += batch.cpu_embed_cache_tensor[token - 100, 1].cuda()
        torch.testing.assert_close(output, expected)


def test_stateful_recipe_uses_declared_outputs_without_probe():
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from lightllm.models.glm5_next.layer_infer.transformer_layer_infer import Glm5NextTransformerLayerInfer

    layer = Glm5NextTransformerLayerInfer.__new__(Glm5NextTransformerLayerInfer)
    layer.layer_num_ = 0
    program = GraphProgram(torch.cuda.graph_pool_handle())
    source = torch.randn(4, 32, device="cuda")
    initial = InferStateInfo()
    initial.prefill_program = program
    initial.prefill_att_state1 = SimpleNamespace(backend=SimpleNamespace(tp_num_heads=2, head_dim=16))
    with program.capture():
        qkv, gate, beta = source * 2, source + 1, source - 1
        output = layer._kda_prefill_cuda_graph_wrapper(qkv, gate, beta, initial, None)
        result = output + 3
    for offset in (4, 9):
        calls = []

        def attention(*, att_control, **kwargs):
            # The host synchronization must execute outside CUDA capture.
            torch.cuda.synchronize()
            calls.append(offset)
            tensors = att_control.linear_att_prefill_dict
            return (tensors["mixed_qkv"] + tensors["raw_gate"] + tensors["raw_beta"] + offset).view(1, 4, 2, 16)

        batch = InferStateInfo()
        batch.prefill_att_state1 = SimpleNamespace(prefill_att=attention)
        program.replay(batch)
        assert calls == [offset]
        torch.testing.assert_close(result, (source * 4 + offset).view(1, 4, 2, 16) + 3)
