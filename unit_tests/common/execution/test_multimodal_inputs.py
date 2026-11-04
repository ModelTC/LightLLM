from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.execution.batch_state import BatchStateFactory
from lightllm.common.execution.multimodal import MultimodalInputs
from unit_tests.common.execution.test_batch_runner import prefill_input


def factory(binding):
    return BatchStateFactory(
        state_type=InferStateInfo,
        hidden_collector=SimpleNamespace(new_instance=lambda: None),
        requests=None,
        state_store=None,
        role="main",
        is_draft=False,
        return_all_prompt_logics=False,
        group_for_lane=lambda lane: lane,
        select_indices=lambda batch: torch.arange(batch.input_ids.numel(), device=batch.input_ids.device),
        prepare_inputs=binding.prepare,
    )


def item(token_id, count, offset):
    return dict(token_id=token_id, token_num=count, start_index_in_embed_cache=offset)


def test_batch_metadata_retains_native_storage_and_separates_lanes():
    cache = torch.randn(6, 3, 16)
    binding = MultimodalInputs.bind(torch.empty(8, 16), cache)
    build = factory(binding)
    batch = prefill_input(4)
    batch.multimodal_params = [dict(images=[item(100, 3, 1)], audios=[item(200, 2, 4)])] * 2
    first = build.create(batch)
    batch.multimodal_params = [dict(images=[], audios=[item(200, 2, 4)])]
    second = build.create(batch, 1)
    assert first.cpu_embed_cache_tensor is second.cpu_embed_cache_tensor is cache
    assert first.img_start_token_ids.tolist() == [100, 200]
    assert first.img_token_lens.tolist() == [3, 2]
    assert first.img_start_locs_in_cache.tolist() == [1, 4]
    assert second.img_start_token_ids.tolist() == [200]
    assert first.dist_group == 0 and second.dist_group == 1
    cache[4, 0].fill_(7)
    assert torch.all(second.cpu_embed_cache_tensor[4, 0] == 7)


@pytest.mark.parametrize("cache", [torch.empty(3, 16), torch.empty(3, 1, 8), torch.empty(3, 1, 16).half()])
def test_binding_rejects_incompatible_storage(cache):
    with pytest.raises(ValueError, match="dimensions or dtype"):
        MultimodalInputs.bind(torch.empty(8, 16), cache)


def test_text_only_binding_reuses_empty_view_and_rejects_missing_image():
    binding = MultimodalInputs.bind(torch.empty(8, 16))
    build = factory(binding)
    batch = prefill_input(2)
    batch.multimodal_params = None
    a, b = build.create(batch), build.create(batch)
    assert a.cpu_embed_cache_tensor is b.cpu_embed_cache_tensor is binding.cache
    assert a.img_token_lens.numel() == 0
    batch.multimodal_params = [dict(images=[item(100, 1, 0)], audios=[])]
    with pytest.raises(ValueError, match="outside the bound"):
        build.create(batch)


@pytest.mark.parametrize("method", ["copy_for_cuda_graph", "copy_for_prefill_cuda_graph"])
def test_graph_cannot_overwrite_embedding_storage_when_binding_changes(method):
    old, new = InferStateInfo(), InferStateInfo()
    old.cpu_embed_cache_tensor = torch.zeros(3, 2, 16)
    new.cpu_embed_cache_tensor = torch.ones_like(old.cpu_embed_cache_tensor)
    with pytest.raises(ValueError, match="embedding storage binding changed"):
        getattr(old, method)(new)
    assert not old.cpu_embed_cache_tensor.any()
    new.cpu_embed_cache_tensor = old.cpu_embed_cache_tensor
    getattr(old, method)(new)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gemma_preserves_image_and_text_embedding_scales():
    from lightllm.models.gemma3.layer_infer.pre_layer_infer import Gemma3PreLayerInfer

    embedding = Gemma3PreLayerInfer.__new__(Gemma3PreLayerInfer)
    embedding.tp_world_size_ = 1
    embedding.embed_scale = torch.tensor(4.0)
    embedding.boi_token_index, embedding.eoi_token_index = 6, 7
    text = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(2, 1, 16, device="cuda", dtype=text.dtype)
    weight = SimpleNamespace(wte_weight_=SimpleNamespace(weight=text, tp_vocab_start_id=0, tp_vocab_end_id=8))
    binding = embedding.bind_inputs(weight, cache).__self__
    batch = prefill_input(5)
    batch.input_ids = torch.tensor([1, 6, 100, 101, 7], device="cuda")
    batch.multimodal_params = [dict(images=[item(100, 2, 0)], audios=[])]
    state = factory(binding).create(batch)
    actual = embedding.context_forward(batch.input_ids, state, weight)
    expected = torch.stack((text[1] * 4, text[6] * 4, cache[0, 0], cache[1, 0], text[7] * 4))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("cache_device", ["cpu", "cuda"])
def test_bound_embedding_and_deepstack_use_current_chunk_without_global_context(dtype, cache_device, monkeypatch):
    from lightllm.common.execution.graph_program import GraphProgram
    from lightllm.models.qwen3_vl.layer_infer.pre_layer_infer import Qwen3VLMultimodalPreLayerInfer
    from lightllm.models.qwen3_vl.layer_infer.transformer_layer_infer import Qwen3VLTransformerLayerInfer
    from lightllm.server.router.model_infer import infer_batch

    class UnavailableContext:
        def __getattr__(self, name):
            pytest.fail(f"compute queried global context: {name}")

    monkeypatch.setattr(infer_batch, "g_infer_context", UnavailableContext())
    weights = torch.randn(8, 16, device="cuda", dtype=dtype)
    cache = torch.randn(6, 3, 16, device=cache_device, dtype=dtype, pin_memory=cache_device == "cpu")
    cache[4:, 1:].zero_()  # Audio rows have no deepstack features.
    embedding = Qwen3VLMultimodalPreLayerInfer.__new__(Qwen3VLMultimodalPreLayerInfer)
    embedding.tp_world_size_ = 1
    weight = SimpleNamespace(wte_weight_=SimpleNamespace(weight=weights, tp_vocab_start_id=0, tp_vocab_end_id=8))
    prepare = embedding.bind_inputs(weight, cache)
    build = factory(prepare.__self__)
    virtual = 2 ** 62
    image, audio = item(virtual, 3, 1), item(virtual + 20, 2, 4)
    program = GraphProgram(torch.cuda.graph_pool_handle())
    layer = Qwen3VLTransformerLayerInfer.__new__(Qwen3VLTransformerLayerInfer)
    source = torch.zeros(5, 16, device="cuda", dtype=dtype)
    initial = InferStateInfo()
    initial.prefill_program = program
    try:
        with program.capture():
            hidden = source.clone()
            layer._apply_deepstack_features_wrapper_run(hidden, initial, 0)
            output = hidden.clone()
        for ids, params in (
            ([1, virtual + 1, virtual + 20, virtual + 2, 4], [dict(images=[image], audios=[audio])] * 2),
            ([virtual + 21, 5, 2, virtual + 20, 0], [dict(images=[], audios=[audio])]),
        ):
            batch = prefill_input(5)
            batch.input_ids = torch.tensor(ids, device="cuda")
            batch.multimodal_params = params
            state = build.create(batch)
            actual = embedding.context_forward(batch.input_ids, state, weight)
            expected = torch.empty_like(actual)
            for row, token in enumerate(ids):
                if token < virtual:
                    expected[row] = weights[token]
                else:
                    obj = image if token < virtual + 20 else audio
                    index = obj["start_index_in_embed_cache"] + token - obj["token_id"]
                    expected[row] = cache[index, 0].to("cuda")
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            source.copy_(actual)
            program.replay(state)
            for row, token in enumerate(ids):
                if virtual <= token < virtual + 3:
                    expected[row] += cache[1 + token - virtual, 1].to("cuda")
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        program.close()
