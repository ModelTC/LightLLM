from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from lightllm.models.qwen4_exp.configuration import normalize_model_config
from lightllm.models.qwen4_exp.multimodal import canonical_image_ids
from lightllm.server.router.model_infer.infer_batch import g_infer_context
from lightllm.common.basemodel.hidden_collector import NoopHiddenCollector
from .runtime import assemble_model, state_for
from .reference import reference_model


def test_model_types_resolve_through_public_registry():
    from lightllm.models.registry import ModelRegistry
    from lightllm.models.qwen4_exp.model import Qwen4ExpTpPartModel

    for model_type in ("qwen4_exp", "qwen4_exp_text"):
        assert ModelRegistry.get_model_class({"model_type": model_type}) is Qwen4ExpTpPartModel


@pytest.mark.parametrize("model_type", ["qwen4_exp", "qwen4_exp_text"])
def test_startup_detects_hybrid_cache_and_optional_vision(model_type, tmp_path):
    from lightllm.utils.config_utils import is_linear_att_mixed_model, has_vision_module, auto_set_fused_shared_experts

    config = {"model_type": model_type, "architectures": ["Qwen4ExpForConditionalGeneration"]}
    if model_type == "qwen4_exp":
        config["vision_config"] = {"hidden_size": 32}
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert is_linear_att_mixed_model(str(tmp_path))
    assert has_vision_module(str(tmp_path)) == (model_type == "qwen4_exp")
    args = SimpleNamespace(model_dir=str(tmp_path), enable_ep_moe=False, enable_fused_shared_experts=False)
    auto_set_fused_shared_experts(args)
    assert args.enable_fused_shared_experts


def test_tokenizer_uses_native_processor_and_virtual_image_ranges(monkeypatch):
    from lightllm.server import tokenizer as tokenizer_module
    from lightllm.models.qwen4_exp.image_processing import Qwen4ExpImageProcessor
    from lightllm.models.qwen4_exp.tokenizer import Qwen4ExpTokenizer

    config = dict(
        model_type="qwen4_exp",
        architectures=["Qwen4ExpForConditionalGeneration"],
        vision_config={},
        vision_start_token_id=61,
        vision_end_token_id=62,
        image_token_id=60,
    )
    monkeypatch.setattr(tokenizer_module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr(tokenizer_module.PretrainedConfig, "get_config_dict", lambda *args: (config, {}))
    processor = SimpleNamespace(min_pixels=16, max_pixels=64, patch_size=2, merge_size=2)
    monkeypatch.setattr(Qwen4ExpImageProcessor, "from_pretrained", lambda *args: processor)
    tokenizer = tokenizer_module.get_tokenizer("checkpoint")
    assert type(tokenizer) is Qwen4ExpTokenizer
    image = SimpleNamespace(token_id=1000000, token_num=2, start_idx=None)
    actual = tokenizer.encode([7, 61, 60, 62, 11], SimpleNamespace(images=[image]))
    assert actual == [7, 61, 1000000, 1000001, 62, 11]
    assert image.start_idx == 2
    assert tokenizer.image_processor is processor


@pytest.mark.parametrize("image_count", [0, 1, 2])
@pytest.mark.parametrize("qwen4", [False, True])
def test_token_count_hook_matches_qwen4_expansion_and_preserves_legacy(image_count, qwen4):
    from lightllm.models.qwen2_vl.model import QWen2VLTokenizer
    from lightllm.models.qwen4_exp.tokenizer import Qwen4ExpTokenizer
    from lightllm.server.httpserver.manager import HttpServerManager

    cls = Qwen4ExpTokenizer if qwen4 else QWen2VLTokenizer
    processor = SimpleNamespace(min_pixels=16, max_pixels=64, patch_size=2, merge_size=2)
    tokenizer = cls(
        tokenizer=object(),
        image_processor=processor,
        model_cfg=dict(vision_start_token_id=61, vision_end_token_id=62, image_token_id=60),
    )
    images = []
    for index in range(image_count):
        image = SimpleNamespace(image_w=4 * (index + 1), image_h=4, token_id=1000 + index * 100)
        image.token_num = tokenizer.get_image_token_length(image)
        images.append(image)
    multimodal = SimpleNamespace(images=images, audios=[])
    prompt = [7] + [61, 60, 62] * image_count + [11]
    expanded = tokenizer.encode(prompt, multimodal)
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.tokenizer = tokenizer
    checked = []
    manager._assert_image_token_count = checked.append
    actual = manager.tokens(prompt, multimodal, SimpleNamespace())
    assert checked == [image.token_num for image in images]
    assert actual == len(expanded)


def test_plain_text_counting_fallback_is_unchanged():
    from lightllm.server.httpserver.manager import HttpServerManager

    manager = HttpServerManager.__new__(HttpServerManager)
    manager.tokenizer = SimpleNamespace(encode=lambda prompt, *_args, **_kwargs: prompt)
    assert manager.tokens([7, 11], SimpleNamespace(images=[], audios=[]), SimpleNamespace()) == 2


def test_token_count_hook_cannot_bypass_image_limits():
    from lightllm.server.httpserver.manager import HttpServerManager
    from lightllm.models.qwen4_exp.tokenizer import Qwen4ExpTokenizer

    tokenizer = Qwen4ExpTokenizer(
        tokenizer=object(),
        image_processor=SimpleNamespace(min_pixels=16, max_pixels=64, patch_size=2, merge_size=2),
        model_cfg=dict(vision_start_token_id=61, vision_end_token_id=62, image_token_id=60),
    )
    manager = HttpServerManager.__new__(HttpServerManager)
    manager.tokenizer = tokenizer

    def reject(_count):
        raise ValueError("image token limit")

    manager._assert_image_token_count = reject
    with pytest.raises(ValueError, match="image token limit"):
        manager.tokens(
            [61, 60, 62],
            SimpleNamespace(images=[SimpleNamespace(image_w=4, image_h=4)], audios=[]),
            SimpleNamespace(),
        )


def test_image_processor_loads_real_config_without_runtime_enum_serialization(tmp_path):
    from lightllm.models.qwen4_exp.image_processing import Qwen4ExpImageProcessor

    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            dict(
                size={"shortest_edge": 16, "longest_edge": 64},
                patch_size=2,
                merge_size=2,
                image_mean=[0.5] * 3,
                image_std=[0.5] * 3,
                image_processor_type="Qwen2VLImageProcessorFast",
            )
        )
    )
    processor = Qwen4ExpImageProcessor.from_pretrained(str(tmp_path))
    assert processor.min_pixels == 16 and processor.max_pixels == 64
    assert processor.patch_size == processor.merge_size == 2
    assert processor.image_mean == processor.image_std == [0.5] * 3


def test_normalize_outer_and_text_config_preserves_checkpoint(config):
    text = dict(config, rope_parameters={"rope_theta": 1000, "partial_rotary_factor": 0.5})
    text.pop("n_layer")
    outer = {"text_config": text, "vision_config": {"hidden_size": 32}, "image_token_id": 60}
    original = deepcopy(outer)
    actual, vision = normalize_model_config(outer)
    standalone, no_vision = normalize_model_config(text)
    assert actual["n_layer"] == standalone["n_layer"] == 4
    assert actual["image_token_id"] == 60
    assert actual["rope_theta"] == standalone["rope_theta"] == 1000
    assert vision == outer["vision_config"] and no_vision is None
    assert outer == original
    customized, _ = normalize_model_config(text, SimpleNamespace(vocab_size=128))
    assert customized["vocab_size"] == 128 and text["vocab_size"] == 64


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_image_ranges_preserve_text_and_physical_ids(dtype):
    second = 2**34 if dtype == torch.int64 else 2000000
    tokens = torch.tensor(
        [7, 60, 999999, 1000000, 1000001, 1000002, second + 4, second + 5, 63], device="cuda", dtype=dtype
    )
    starts = torch.tensor([1000000, second], device="cuda", dtype=torch.int64)
    lengths = torch.tensor([2, 5], device="cuda", dtype=torch.int64)
    before = tokens.clone()
    actual = canonical_image_ids(tokens, starts, lengths, 60)
    expected = torch.tensor([7, 60, 999999, 60, 60, 1000002, 60, second + 5, 63], device="cuda", dtype=dtype)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(tokens, before, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = canonical_image_ids(tokens, starts, lengths, 60)
    tokens[3:5] = 11
    graph.replay()
    expected[3:5] = 11
    torch.testing.assert_close(captured, expected, atol=0, rtol=0)
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_image_cache_ids_are_canonicalized_before_ple_commit(runtime, config, monkeypatch):
    config["image_token_id"] = 60
    model, source = assemble_model(runtime, config)
    runtime.req.init_linear_att_state(SimpleNamespace(req_idx=1))
    image_cache = torch.randn(4, 1, 32, dtype=torch.bfloat16).pin_memory()
    monkeypatch.setattr(g_infer_context, "cpu_embed_cache_client", SimpleNamespace(cpu_embed_cache_tensor=image_cache))
    virtual = [7, 1000000, 1000001, 11]
    canonical = torch.tensor([7, 60, 60, 11], device="cuda", dtype=torch.int64)
    state = state_for(runtime, [1], [4], virtual, ready=[0])
    state.hidden_collector = NoopHiddenCollector()
    state.multimodal_params = [
        {"images": [{"token_id": 1000000, "token_num": 2, "start_index_in_embed_cache": 1}], "audios": []}
    ]
    actual = model._context_forward(state)
    torch.testing.assert_close(state.input_ids, torch.tensor(virtual, device="cuda"), atol=0, rtol=0)
    torch.testing.assert_close(state.qwen4_exp_ple[1].input_tokens, canonical, atol=0, rtol=0)
    torch.testing.assert_close(runtime.req.lexical_caches[1].ngram[1], canonical[-2:], atol=0, rtol=0)
    embeddings = F.embedding(canonical.cpu(), source["model.language_model.embed_tokens.weight"])
    embeddings[1:3] = image_cache[1:3, 0]
    positions = torch.arange(4, device="cuda")
    cos, sin = model._cos_cached[positions].repeat(1, 2), model._sin_cached[positions].repeat(1, 2)
    expected = reference_model(canonical, cos, sin, [4], source, config, input_embeddings=embeddings.cuda())[-1:]
    torch.testing.assert_close(actual.logits, expected, atol=0.025, rtol=0.025)
