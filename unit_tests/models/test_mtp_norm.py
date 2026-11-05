from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.basemodel import TpPartBaseModel
from lightllm.common.mtp_config import FinalHiddenPolicy
from lightllm.common.basemodel.layer_weights.meta_weights import NoTpGEMMANormWeight, RMSNormWeight
from lightllm.common.basemodel.mtp_manager import MtpManager
from lightllm.models.deepseek_mtp.layer_infer.pre_layer_infer import Deepseek3MTPPreLayerInfer
from lightllm.models.glm5_next_mtp.layer_infer.pre_layer_infer import Glm5NextMTPPreLayerInfer
from lightllm.models.mistral_mtp.layer_infer.pre_layer_infer import MistralMTPPreLayerInfer
from lightllm.models.llama.layer_infer import post_layer_infer as llama_post
from lightllm.models.qwen3_5_mtp.layer_infer.pre_layer_infer import Qwen3_5MTPPreLayerInfer


@pytest.mark.parametrize("family", ["qwen35", "glm5_next", "deepseek", "glm4_moe_lite", "mistral", "qwen3_moe"])
@pytest.mark.parametrize("prefill", [False, True])
@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float32),
        pytest.param(
            "cuda", torch.bfloat16, marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
        ),
    ],
)
def test_main_to_draft_two_steps(monkeypatch, tmp_path, family, prefill, device, dtype) -> None:
    from importlib import import_module

    classes = {
        "qwen35": ("qwen3_5.model:Qwen3_5TpPartModel", "qwen3_5_mtp.model:Qwen3_5MTPModel"),
        "glm5_next": ("glm5_next.model:Glm5NextTpPartModel", "glm5_next_mtp.model:Glm5NextMTPModel"),
        "deepseek": ("deepseek2.model:Deepseek2TpPartModel", "deepseek_mtp.model:Deepseek3MTPModel"),
        "glm4_moe_lite": ("glm4_moe_lite.model:Glm4MoeLiteTpPartModel", "glm4_moe_lite_mtp.model:Glm4MoeLiteMTPModel"),
        "mistral": ("mistral.model:MistralTpPartModel", "mistral_mtp.model:MistralMTPModel"),
        "qwen3_moe": ("qwen3_moe.model:Qwen3MOEModel", "qwen3_moe_mtp.model:Qwen3MOEMTPModel"),
    }
    producers = []
    for path in classes[family]:
        module, name = path.split(":")
        cls = getattr(import_module("lightllm.models." + module), name)
        producers.append(cls.__new__(cls))
    import json

    draft_type = {"qwen35": "qwen3_5", "deepseek": "deepseek_v3"}.get(family, family)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": draft_type}))
    producers[1].weight_dir_ = str(tmp_path)
    manager = MtpManager.__new__(MtpManager)
    manager.args = SimpleNamespace(
        mtp_mode="vanilla_no_att" if family in ("mistral", "qwen3_moe") else "vanilla_with_att",
        mtp_draft_model_dir=[str(tmp_path)],
    )
    norm_class = NoTpGEMMANormWeight if family == "qwen35" else RMSNormWeight

    def norm_weight(values):
        norm = norm_class.__new__(norm_class)
        norm.dim = 3
        norm.data_type_ = dtype
        norm.weight_name = "norm.weight"
        norm.weight = torch.empty(3, device=device, dtype=dtype)
        norm.load_hf_weights({"norm.weight": torch.tensor(values, device=device, dtype=dtype)})
        norm._forward = norm._cuda_forward if device == "cuda" else norm._native_forward
        return norm

    main_norm = norm_weight([0.5, -0.25, 1.0])
    draft_norm = norm_weight([-0.5, 1.0, 0.25])
    enorm = norm_weight([0.0, 0.25, -0.25])
    hnorm = norm_weight([0.25, -0.5, 0.5])
    projection = torch.arange(1, 16, device=device, dtype=dtype).reshape(5, 3) / 10
    fusion = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]],
        device=device,
        dtype=dtype,
    )

    class LMHead:
        vocab_size = 5

        def __call__(self, input, alloc_func):
            return projection @ input

    main_weight = SimpleNamespace(final_norm_weight_=main_norm, lm_head_weight_=LMHead())
    draft_weight = SimpleNamespace(
        final_norm_weight_=draft_norm,
        lm_head_weight_=LMHead(),
        enorm_weight_=enorm,
        hnorm_weight_=hnorm,
        eh_proj_weight_=SimpleNamespace(mm=lambda hidden: hidden @ fusion.T),
    )
    post = llama_post.LlamaPostLayerInfer.__new__(llama_post.LlamaPostLayerInfer)
    post.eps_ = 1e-6
    post.tp_world_size_ = 1
    post.alloc_tensor = lambda shape, dtype, **kwargs: torch.empty(shape, device=device, dtype=dtype)
    pre_class = {
        "qwen35": Qwen3_5MTPPreLayerInfer,
        "glm5_next": Glm5NextMTPPreLayerInfer,
        "mistral": MistralMTPPreLayerInfer,
    }.get(family, Deepseek3MTPPreLayerInfer)
    pre = pre_class.__new__(pre_class)
    pre.eps_ = post.eps_
    model = TpPartBaseModel.__new__(TpPartBaseModel)
    model.layers_num = 0
    model.post_infer = post
    post._tpsp_allgather = lambda input, infer_state: input

    def forward(hidden, weight, producer):
        state.hidden_collector = manager.create_hidden_collector(producer)
        state.final_hidden_policy = manager.get_final_hidden_policy(producer)
        model.pre_post_weight = weight
        model.pre_infer = SimpleNamespace(
            token_forward=lambda *args: hidden,
            _tpsp_sp_split=lambda input, infer_state: input,
        )
        return model._token_forward(state)

    state = SimpleNamespace(
        is_prefill=False,
        is_cuda_graph=False,
        input_ids=torch.ones(2, device=device, dtype=torch.long),
        is_draft_model=False,
        batch_size=2,
        prompt_logics=None,
    )
    monkeypatch.setattr(
        llama_post,
        "get_env_start_args",
        lambda: SimpleNamespace(target_vocab_topk_sampling=None, draft_vocab_topk_sampling=None),
    )

    def reference_norm(hidden, weight):
        value = hidden.float()
        variance = value.square().mean(dim=-1, keepdim=True)
        return (value * torch.rsqrt(variance + post.eps_) * weight.weight.float()).to(dtype)

    if family in ("mistral", "qwen3_moe"):
        draft_weight.final_norm_weight_ = main_norm

    def collected(hidden, weight, producer):
        if manager.get_final_hidden_policy(producer) is FinalHiddenPolicy.POST_NORM:
            return reference_norm(hidden, weight.final_norm_weight_)
        return hidden.clone()

    hidden = torch.tensor([[1.0, 2.0, -3.0], [4.0, -2.0, 1.0]], device=device, dtype=dtype)
    expected = collected(hidden, main_weight, producers[0])
    output = forward(hidden, main_weight, producers[0])
    torch.testing.assert_close(output.mtp_collector.spec_hidden, expected)
    torch.testing.assert_close(output.logits, (reference_norm(hidden, main_norm) @ projection.T).float())
    assert state.hidden_collector.final_hidden is None

    state.is_draft_model = True
    for step in range(2):
        embedding = torch.tensor([[2.0, -1.0, 3.0], [-3.0, 2.0, 1.0]], device=device, dtype=dtype) + step
        state.mtp_draft_input_hiddens = output.mtp_collector.spec_hidden
        fusion_hidden = reference_norm(expected, main_norm) if family == "mistral" else expected
        expected_fused = (
            torch.cat(
                (reference_norm(embedding, enorm), reference_norm(fusion_hidden, hnorm)),
                dim=-1,
            )
            @ fusion.T
        )
        if family == "qwen35":
            fused = pre._mtp_fuse(embedding, state, draft_weight)
        elif family == "glm5_next":
            fused = pre._fuse_hidden(embedding, state, draft_weight)
        else:
            fuse = pre._mtp_context_forward if prefill else pre._mtp_token_forward
            fused = fuse(embedding, state, draft_weight)
        torch.testing.assert_close(fused, expected_fused)
        expected = collected(expected_fused, draft_weight, producers[1])
        output = forward(fused, draft_weight, producers[1])
        torch.testing.assert_close(output.mtp_collector.spec_hidden, expected)
        torch.testing.assert_close(
            output.logits,
            (reference_norm(expected_fused, draft_weight.final_norm_weight_) @ projection.T).float(),
            atol=0.02 if device == "cuda" else 1e-5,
            rtol=0.02 if device == "cuda" else 1e-5,
        )
        assert state.hidden_collector.final_hidden is None
        if family not in ("mistral", "qwen3_moe"):
            assert not torch.allclose(
                reference_norm(expected_fused, draft_norm), reference_norm(expected_fused, main_norm)
            )
