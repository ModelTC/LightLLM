from lightllm.core.formats.kv.spec import DenseKVStateSpec
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.core.compute.attention.dense.backend import DenseAttentionBackend
from lightllm.models.base.adapter import DenseModelAdapter
from lightllm.core.build import BuildConfig, prepare
from lightllm.core.registry import default_providers
from lightllm.models.base.build import bind_attention_execution
from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
from lightllm.core.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.core.contracts.state import FormatSpec
from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel
from lightllm.models.qwen3_dflash2.model import Qwen3DFlash2Model
from lightllm.models.qwen3_dflash2.adapter import Qwen3DFlash2ModelAdapter
from lightllm.models.qwen3_dspark.model import Qwen3DSparkModel
from lightllm.models.qwen3_5_dflash.model import Qwen3_5DFlashModel
from lightllm.models.qwen3_5_dspark.model import Qwen3_5DSparkModel


@pytest.mark.parametrize("model_class", [Qwen3DFlashModel, Qwen3DSparkModel, Qwen3_5DFlashModel, Qwen3_5DSparkModel])
def test_block_model_declares_noncausal_compute_and_global_state_ids(model_class):
    config = dict(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=256,
        head_dim=128,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
    )
    model = model_class.__new__(model_class)
    model.config = model_class.normalize_model_config(config)
    model.args = SimpleNamespace(mtp_mode="dflash")
    model.data_type, model.tp_world_size_ = torch.float16, 1
    model.main_model = SimpleNamespace(layers_infer=[None] * 4, pre_post_weight=object())
    model.mtp_previous_draft_models = [SimpleNamespace(layers_infer=[None] * 2)]
    model.weight_dir_ = "unused"
    model.weight_dict = {"embed_tokens.weight": None, "lm_head.weight": None}
    model._init_model_adapter()
    adapter = model.model_adapter
    assert adapter.attention_phases == ("decode",)
    assert tuple(adapter.attention_specs()) == (6, 7)
    assert all(not spec.causal and spec.head_dim == 128 for spec in adapter.attention_specs().values())
    assert [(state.role, state.producer, state.consumers) for state in adapter.state_specs()] == [
        ("draft1", 6, (6,)),
        ("draft1", 7, (7,)),
    ]


def test_dflash2_does_not_inherit_plain_block_semantics():
    from lightllm.models.qwen3_dflash2.layer_infer.transformer_layer_infer import Qwen3DFlash2TransformerLayerInfer
    from lightllm.models.qwen3_dflash2.layer_infer.post_layer_infer import Qwen3DFlash2PostLayerInfer

    assert Qwen3DFlash2Model.__dict__["model_adapter_class"] is Qwen3DFlash2ModelAdapter
    for width, window in ((0, (-1, -1)), (3, (2, 2))):
        adapter = Qwen3DFlash2ModelAdapter(
            dict(
                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, hidden_size=256, sliding_window=width
            ),
            role="draft1",
            dtype=torch.float16,
            layer_ids=(6, 7),
            main_weights=object(),
        )
        assert adapter.attention_phases == ("decode",) and adapter.share_main_weights
        assert all(not spec.causal and spec.window == window for spec in adapter.attention_specs().values())
        assert adapter.transformer_layer_infer_class is Qwen3DFlash2TransformerLayerInfer
        assert adapter.post_layer_infer_class is Qwen3DFlash2PostLayerInfer
        assert [spec.producer for spec in adapter.state_specs()] == [6, 7]


@pytest.mark.parametrize("layout", ["own", "shared", "partial", "unknown", "shape", "unloaded"])
def test_dspark_declares_complete_vocabulary_weight_layout_before_allocation(layout):
    from lightllm.models.qwen3_dspark.adapter import Qwen3DSparkModelAdapter

    config = dict(
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=256,
        vocab_size=128,
        architectures=[] if layout == "unknown" else ["DSparkDraftModel"],
    )
    weight = SimpleNamespace(
        dim=128 if layout == "shape" else 256,
        vocab_size=128,
        data_type_=torch.float16,
        verify_load=lambda: layout != "unloaded",
    )
    adapter = Qwen3DSparkModelAdapter(
        config,
        role="draft0",
        dtype=torch.float16,
        main_weights=SimpleNamespace(wte_weight_=weight, lm_head_weight_=weight),
    )
    names = {"embed_tokens.weight", "lm_head.weight"} if layout == "own" else set()
    if layout == "partial":
        names.add("embed_tokens.weight")
    if layout in ("partial", "unknown", "shape", "unloaded"):
        with pytest.raises(ValueError, match="DSpark"):
            adapter.declare_weight_layout(names)
        return
    adapter.declare_weight_layout(names)
    assert adapter.share_main_weights == (layout == "shared")
    assert adapter.pre_and_post_weight_class.owns_vocab_weights == (layout == "own")


def test_writer_only_prefill_has_no_attention_binding_or_metadata():
    adapter = DenseModelAdapter(
        dict(num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2, hidden_size=256),
        role="draft0",
        dtype=torch.float16,
        layer_ids=(3,),
    )
    layer = SimpleNamespace(layer_num_=3)
    manager = AttentionStateManager(2)
    runtime = SimpleNamespace(
        caps=DeviceCaps("cuda", 0, (9, 0)),
        allocate_workspace=lambda size: torch.empty(size, dtype=torch.uint8),
        allocate_tensor=torch.empty,
    )
    prepared = prepare(
        model_adapter=adapter,
        providers=default_providers(),
        device=runtime,
        config=BuildConfig(
            limits=ExecutionLimits(1024 * 1024, 2, 16),
            formats={"history": FormatSpec("plain", torch.float16, "token_head_dim", ("k", "v"), 1)},
            requested={"history": {"decode": "triton"}},
            static_params=adapter.attention_params({3: None}, runtime),
        ),
    )
    resources = prepared.allocate(manager=manager, device=runtime)
    manager.finish_state_registration()
    state = manager.get_full_attention_states()[0]
    state.device = torch.device("cpu")
    state.allocate_storage(32)
    execution = bind_attention_execution(resources, [layer])
    plan = execution.plan
    assert [(item.layer_id, item.phase) for item in plan.bindings] == [(3, "decode")]
    assert layer.bound_prefill_attention is None and layer.bound_kv_writer is not None
    assert execution.metadata.prepare_batch(None, phase="prefill") == {}


def test_noncausal_backend_support_is_selected_from_semantics():
    spec = DenseAttentionSpec(4, 2, 64, causal=False, dtype=torch.float16)
    format = FormatSpec("plain", torch.float16, "token_head_dim", ("k", "v"), 1)
    caps, limits = DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(1024 * 1024, 4, 16, max_verify_tokens=2)
    assert DenseAttentionBackend("fa3", "verify").probe(spec, format, caps, limits).offers
    assert DenseAttentionBackend("triton", "verify").probe(spec, format, caps, limits).reason
    assert DenseAttentionBackend("fa3", "verify").probe(spec, format, replace(caps, capability=(8, 0)), limits).reason


@pytest.mark.parametrize(
    "backend,causal,supported", [("fa3", False, True), ("fa3", True, False), ("triton", False, False)]
)
def test_right_window_requires_noncausal_fa3(backend, causal, supported):
    spec = DenseAttentionSpec(4, 2, 64, causal=causal, window=(2, 2), dtype=torch.float16)
    format = FormatSpec("plain", torch.float16, "token_head_dim", ("k", "v"), 1)
    caps, limits = DeviceCaps("cuda", 0, (9, 0)), ExecutionLimits(1024 * 1024, 4, 16, max_verify_tokens=4)
    assert bool(DenseAttentionBackend(backend, "verify").probe(spec, format, caps, limits).offers) == supported


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("model_class", [Qwen3DFlashModel, Qwen3DSparkModel, Qwen3_5DFlashModel, Qwen3_5DSparkModel])
@pytest.mark.parametrize("block_size", [1, 3])
@torch.no_grad()
def test_block_checkpoint_commit_proposal_and_graph(model_class, block_size, monkeypatch, tmp_path):
    _check_block_checkpoint(model_class, block_size, monkeypatch, tmp_path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("model_class", [Qwen3DSparkModel, Qwen3_5DSparkModel])
@pytest.mark.parametrize("block_size", [1, 3])
@torch.no_grad()
def test_shared_dspark_checkpoint_commit_proposal_and_graph(model_class, block_size, monkeypatch, tmp_path):
    _check_block_checkpoint(model_class, block_size, monkeypatch, tmp_path, shared_target=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("draft_width", [1, 3])
@pytest.mark.parametrize("window", [0, 3])
@pytest.mark.parametrize("dynamic", [False, True])
@torch.no_grad()
def test_dflash2_checkpoint_window_convolution_selector_and_graph(draft_width, window, dynamic, monkeypatch, tmp_path):
    _check_block_checkpoint(
        Qwen3DFlash2Model, draft_width + 1, monkeypatch, tmp_path, dflash2_options=(window, dynamic)
    )


def _check_block_checkpoint(model_class, block_size, monkeypatch, tmp_path, shared_target=False, dflash2_options=None):
    from dataclasses import asdict
    import json
    import torch.nn.functional as F
    from safetensors.torch import save_file
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf35
    from lightllm.common.basemodel.batch_objs import ModelInput
    from lightllm.common.basemodel.mtp_manager import MtpManager
    from lightllm.core.runner.decode_graph import CudaGraph
    from lightllm.common.quantization.no_quant import NoQuantization
    from lightllm.distributed.communication_op import dist_group_manager
    from lightllm.models.qwen3.model import Qwen3TpPartModel
    from lightllm.models.qwen3_5.model import Qwen3_5TpPartModel
    from lightllm.server.core.objs.start_args_type import StartArgs
    from lightllm.utils.envs_utils import get_env_start_args, get_llm_data_type
    from unit_tests.common.execution.test_model_cuda import prepare_model_batch

    torch.manual_seed(79)
    is35 = shared_target or model_class in (Qwen3_5DFlashModel, Qwen3_5DSparkModel)
    dspark = model_class in (Qwen3DSparkModel, Qwen3_5DSparkModel)
    dflash2 = model_class is Qwen3DFlash2Model
    target_cls = Qwen3_5TpPartModel if is35 else Qwen3TpPartModel
    config_type, reference_type = (
        (hf35.Qwen3_5TextConfig, hf35.Qwen3_5ForCausalLM)
        if is35
        else (
            Qwen3Config,
            Qwen3ForCausalLM,
        )
    )
    extras = {}
    if is35:
        extras = dict(
            layer_types=["full_attention", "linear_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
            rope_parameters=dict(
                rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.25, mrope_section=[6, 5, 5]
            ),
        )
    config = config_type(
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128 if is35 else 64,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        **extras,
    )
    target = reference_type(config).half()
    target_dir, draft_dir = tmp_path / "target", tmp_path / "draft"
    target_dir.mkdir()
    draft_dir.mkdir()
    config.save_pretrained(target_dir)
    save_file(
        {k: v.detach().contiguous() for k, v in target.state_dict().items()}, str(target_dir / "model.safetensors")
    )
    head_type = "gated" if block_size == 1 else "rnn"
    draft_config = dict(
        model_type="qwen3",
        vocab_size=128,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_parameters=dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=1.0),
        block_size=4,
        mask_token_id=0,
        target_layer_ids=[0, 1],
        markov_rank=16 if dspark else 0,
        markov_head_type=head_type if dspark else None,
        enable_confidence_head=dspark,
        confidence_head_with_markov=dspark,
    )
    if shared_target:
        draft_config["architectures"] = ["DSparkDraftModel"]
        draft_config["dflash_config"] = {name: draft_config.pop(name) for name in ("target_layer_ids", "mask_token_id")}
    if dflash2:
        draft_config.update(
            sliding_window=dflash2_options[0],
            selector_top_k=16,
            selector_rank=16,
            conv_kernel_size=3,
            conv_group_size=32,
            output_multiplier=1.2,
            final_logit_softcapping=2.0,
        )
    (draft_dir / "config.json").write_text(json.dumps(draft_config))
    matrix = lambda *shape: (torch.randn(*shape) * 0.02).half()
    norm = lambda width=256: (1 + torch.randn(width) * 0.01).half()
    weights = {"fc.weight": matrix(256, 512), "hidden_norm.weight": norm(), "norm.weight": norm()}
    shared = shared_target or model_class is Qwen3_5DFlashModel or dflash2
    if shared:
        embedding, output_weight = target.model.embed_tokens.weight.detach(), target.lm_head.weight.detach()
    else:
        weights.update({"embed_tokens.weight": matrix(128, 256), "lm_head.weight": matrix(128, 256)})
        embedding, output_weight = weights["embed_tokens.weight"], weights["lm_head.weight"]
    embedding, output_weight = embedding.float().cuda(), output_weight.float().cuda()
    for i in range(2):
        prefix = f"layers.{i}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            weights[prefix + name + ".weight"] = norm()
        for name in ("q_norm", "k_norm"):
            weights[prefix + "self_attn." + name + ".weight"] = norm(64)
        for name, width in (("q", 256), ("k", 128), ("v", 128)):
            weights[prefix + f"self_attn.{name}_proj.weight"] = matrix(width, 256)
        weights[prefix + "self_attn.o_proj.weight"] = matrix(256, 256)
        for name in ("gate", "up"):
            weights[prefix + f"mlp.{name}_proj.weight"] = matrix(512, 256)
        weights[prefix + "mlp.down_proj.weight"] = matrix(256, 512)
        if dflash2:
            for name in ("attention_conv", "mlp_conv"):
                weights[prefix + name + ".kernel_projection.weight"] = matrix(48, 256) * 0.1
                base = matrix(2, 3, 256)
                base[:, 0].add_(1)
                weights[prefix + name + ".base_kernel"] = base
    if dflash2:
        weights["candidate_selector.hidden_projection.weight"] = matrix(16, 256)
        for name in ("predecessor", "successor"):
            weights[f"candidate_selector.{name}_codebook"] = matrix(128, 16)
    if dspark:
        weights["markov_head.markov_w1.weight"] = matrix(128, 16)
        weights["markov_head.markov_w2.weight"] = matrix(128, 16)
        name, width, inputs = ("gate", 16, 272) if head_type == "gated" else ("joint", 48, 288)
        weights[f"markov_head.{name}_proj.weight"] = matrix(width, inputs)
        weights[f"markov_head.{name}_proj.bias"] = matrix(width)
        weights["confidence_head.proj.weight"] = matrix(1, 272)
        weights["confidence_head.proj.bias"] = matrix(1)
    save_file(weights, str(draft_dir / "model.safetensors"))
    reference = {name: value.float().cuda() for name, value in weights.items()}
    del target

    for name in (
        "GLOBAL_RANK",
        "GLOBAL_DP_RANK",
        "DP_RANK_IN_NODE",
        "CURRENT_RANK_IN_DP",
        "CURRENT_RANK_IN_NODE",
        "CURRENT_DEVICE_ID",
    ):
        monkeypatch.setenv("LIGHTLLM_" + name, "0")
    for name in ("GLOBAL_WORLD_SIZE", "DP_SIZE", "DP_WORLD_SIZE", "NODE_WORLD_SIZE"):
        monkeypatch.setenv("LIGHTLLM_" + name, "1")
    args = StartArgs(
        model_dir=str(target_dir),
        data_type="float16",
        page_size=1,
        running_max_req_size=2,
        llm_prefill_att_backend=["triton", "triton"],
        llm_decode_att_backend=["fa3", "triton"],
        mtp_mode="dspark" if dspark else "dflash2" if dflash2 else "dflash",
        mtp_step=block_size - 1 if dflash2 else block_size,
        mtp_dynamic_verify=dflash2_options[1] if dflash2 else False,
        mtp_draft_model_dir=[str(draft_dir)],
    )
    monkeypatch.setenv("LIGHTLLM_START_ARGS", json.dumps(asdict(args)))
    get_env_start_args.cache_clear()
    get_llm_data_type.cache_clear()
    monkeypatch.setattr(MtpManager, "_instance", None)
    monkeypatch.setattr(dist_group_manager, "groups", [SimpleNamespace()])
    for cls in (target_cls, model_class):
        for name in (
            "pre_and_post_weight_class",
            "transformer_weight_class",
            "pre_layer_infer_class",
            "post_layer_infer_class",
            "transformer_layer_infer_class",
        ):
            monkeypatch.setattr(cls, name, lambda *a, **kw: pytest.fail("legacy model factory"))
    kwargs = dict(
        run_mode="normal",
        max_total_token_num=128,
        batch_max_tokens=32,
        max_req_num=2,
        max_seq_length=32,
        graph_max_batch_size=2,
        graph_max_len_in_batch=32,
        disable_cudagraph=True,
    )
    main = target_cls(dict(kwargs, weight_dir=str(target_dir)))
    draft = model_class(dict(kwargs, weight_dir=str(draft_dir), main_model=main, mtp_previous_draft_models=[]))
    assert draft.linear_plan.bindings and not draft.linear_plan.legacy
    assert [layer.layer_num_ for layer in draft.layers_infer] == [2, 3]
    assert [weight.layer_num_ for weight in draft.trans_layers_weight] == [0, 1]
    assert (draft.pre_post_weight.wte_weight_ is main.pre_post_weight.wte_weight_) == shared
    assert (draft.pre_post_weight.lm_head_weight_ is main.pre_post_weight.lm_head_weight_) == shared
    assert (draft._cos_cached is main._cos_cached) == (model_class is Qwen3DFlashModel)
    main.prepare_attention_states()
    draft.prepare_attention_states()
    manager = main.att_state_manager
    manager.finish_state_registration()
    for state in manager.get_full_attention_states():
        state.device = torch.device("cuda")
        state.allocate_storage(128)
    for model in (main, draft):
        model.req_manager = SimpleNamespace(
            req_to_token_indexs=torch.arange(96, dtype=torch.int32, device="cuda").view(3, 32), HOLD_REQUEST_ID=2
        )
        model.prefill_att_backend = model.decode_att_backend = None
        model.prefill_att_backend1 = model.decode_att_backend1 = None
        model._init_execution_plan()
        model._init_hidden_collector()
        for view in model.attention_execution.resources.stores.get("history").views().values():
            for value in view.fields:
                value.zero_()
    phase = "decode" if block_size == 1 else "verify"
    assert {item.phase for item in draft.execution_plan.bindings} == {phase}
    assert all(layer.bound_prefill_attention is None for layer in draft.layers_infer)
    main_views = tuple(main.attention_execution.resources.stores.get("history").views().values())
    main_snapshots = [(value, value.clone()) for view in main_views for value in view.fields]
    draft_views = tuple(draft.attention_execution.resources.stores.get("history").views().values())
    monkeypatch.setattr(manager, "get_state", lambda *a, **kw: pytest.fail("runtime manager lookup"))
    monkeypatch.setattr(NoQuantization, "apply", lambda *a, **kw: pytest.fail("legacy weight compute"))
    observed_hidden = []
    if dflash2:
        compute_candidates = draft.post_infer._compute_candidates

        def candidates(hidden, infer_state, layer_weight):
            observed_hidden.append(hidden)
            return compute_candidates(hidden, infer_state, layer_weight)

        monkeypatch.setattr(draft.post_infer, "_compute_candidates", candidates)
    target_hidden = torch.randn(2, 32, 512, dtype=torch.float16, device="cuda") * 0.3
    prefixes = [3, 2]

    def linear(x, name):
        return F.linear(x.float(), reference[name + ".weight"], reference.get(name + ".bias"))

    def normalize(x, name):
        return F.rms_norm(x.float(), (x.shape[-1],), reference[name + ".weight"], 1e-6)

    def convolve(x, dynamic, name, side):
        base = reference[name + ".base_kernel"][side]
        dynamic = dynamic.view(block_size, 2, 3, 8)[:, side].repeat_interleave(32, dim=-1)
        return torch.stack(
            [
                sum(x[t - offset] * (base[offset] + dynamic[t, offset]) for offset in range(min(t + 1, 3)))
                for t in range(block_size)
            ]
        )

    def rope(x, start=0):
        positions = torch.arange(start, start + x.shape[0], device="cuda").float()
        angle = positions[:, None] * 10000.0 ** (-torch.arange(0, 64, 2, device="cuda").float() / 64)
        left, right = x.chunk(2, dim=-1)
        return torch.cat(
            (
                left * angle.cos()[:, None] - right * angle.sin()[:, None],
                right * angle.cos()[:, None] + left * angle.sin()[:, None],
            ),
            -1,
        )

    def committed_kv(request, length, layer):
        projected = normalize(linear(target_hidden[request, :length], "fc"), "hidden_norm")
        prefix = f"layers.{layer}.self_attn."
        k = normalize(linear(projected, prefix + "k_proj").view(length, 2, 64), prefix + "k_norm")
        return rope(k), linear(projected, prefix + "v_proj").view(length, 2, 64)

    def check_commit():
        for layer, view in enumerate(draft_views):
            for request, length in enumerate(prefixes):
                for actual, expected in zip(view.fields, committed_kv(request, length, layer)):
                    torch.testing.assert_close(
                        actual[request * 32 : request * 32 + length].float(), expected, rtol=2e-2, atol=5e-3
                    )

    inp = prepare_model_batch(draft, [[11] * n for n in prefixes], [0, 0], True, input_only=True)
    inp.mtp_draft_input_hiddens = torch.cat([target_hidden[r, :n] for r, n in enumerate(prefixes)])
    with monkeypatch.context() as local:
        for layer in draft.layers_infer:
            local.setattr(
                layer.bound_decode_attention, "run", lambda *a, **kw: pytest.fail("attention during KV commit")
            )
        assert draft.forward(inp).logits.numel() == 0
    check_commit()

    def rows_input(rows, hidden=None):
        tensor = lambda values, dtype=torch.int32: torch.tensor(values, dtype=dtype, device="cuda")
        return ModelInput(
            batch_size=len(rows),
            total_token_num=sum(prefixes) + len(rows),
            max_q_seq_len=1,
            max_kv_seq_len=max(position + 1 for _, position, _ in rows),
            max_cache_len=max(prefixes),
            input_ids=tensor([token for _, _, token in rows], torch.int64),
            b_req_idx=tensor([request for request, _, _ in rows]),
            b_seq_len=tensor([position + 1 for _, position, _ in rows]),
            b_mtp_index=tensor([i % block_size for i in range(len(rows))]),
            is_prefill=False,
            b_position_delta=tensor([0] * len(rows)),
            b_shared_seq_len=tensor([0] * len(rows)),
            b_shared_radix_node_id=tensor([0] * len(rows)),
            mtp_draft_input_hiddens=hidden,
            multimodal_params=[{"images": [], "audios": []} for _ in rows],
        )

    def proposal(tokens):
        rows = [(r, prefixes[r] + step, token) for r, block in enumerate(tokens) for step, token in enumerate(block)]
        batch = draft._create_inferstate(rows_input(rows))
        draft.batch_metadata(batch)
        batch.init_att_state()
        return batch

    def check_proposal(output, tokens):
        all_logits, all_ids, all_confidence = [], [], []
        candidate_hiddens = []
        for request, block in enumerate(tokens):
            hidden = F.embedding(torch.tensor(block, device="cuda"), embedding)
            for layer in range(2):
                prefix = f"layers.{layer}."
                x = normalize(hidden, prefix + "input_layernorm")
                if dflash2:
                    attention_dynamic = linear(x, prefix + "attention_conv.kernel_projection")
                    x = convolve(x, attention_dynamic, prefix + "attention_conv", 0)
                q = normalize(
                    linear(x, prefix + "self_attn.q_proj").view(block_size, 4, 64), prefix + "self_attn.q_norm"
                )
                k = normalize(
                    linear(x, prefix + "self_attn.k_proj").view(block_size, 2, 64), prefix + "self_attn.k_norm"
                )
                v = linear(x, prefix + "self_attn.v_proj").view(block_size, 2, 64)
                q, k = rope(q, prefixes[request]), rope(k, prefixes[request])
                old_k, old_v = committed_kv(request, prefixes[request], layer)
                k = torch.cat((old_k, k)).repeat_interleave(2, dim=1).transpose(0, 1)
                v = torch.cat((old_v, v)).repeat_interleave(2, dim=1).transpose(0, 1)
                scores = q.transpose(0, 1) @ k.transpose(-1, -2) / 8
                if dflash2 and dflash2_options[0]:
                    distance = torch.arange(k.shape[1], device="cuda")[None, :] - (
                        prefixes[request] + torch.arange(block_size, device="cuda")[:, None]
                    )
                    scores.masked_fill_(distance.abs() >= dflash2_options[0], -torch.inf)
                attention = (scores.softmax(-1) @ v).transpose(0, 1).flatten(1)
                attention = linear(attention, prefix + "self_attn.o_proj")
                if dflash2:
                    attention = convolve(attention, attention_dynamic, prefix + "attention_conv", 1)
                hidden = hidden + attention
                x = normalize(hidden, prefix + "post_attention_layernorm")
                if dflash2:
                    mlp_dynamic = linear(x, prefix + "mlp_conv.kernel_projection")
                    x = convolve(x, mlp_dynamic, prefix + "mlp_conv", 0)
                mlp = linear(
                    F.silu(linear(x, prefix + "mlp.gate_proj")) * linear(x, prefix + "mlp.up_proj"),
                    prefix + "mlp.down_proj",
                )
                if dflash2:
                    mlp = convolve(mlp, mlp_dynamic, prefix + "mlp_conv", 1)
                hidden = hidden + mlp
            logits = F.linear(normalize(hidden, "norm"), output_weight)
            all_logits.append(logits)
            if dflash2:
                candidate_hidden = normalize(hidden, "norm")[1:]
                candidate_hiddens.append(candidate_hidden)
                gates = linear(candidate_hidden, "candidate_selector.hidden_projection")
                values, ids = logits[1:].topk(16, dim=-1)
                unary = (values * 1.2 / 2.0).tanh() * 2.0
                previous, predictions, confidence = block[0], [], []
                for step in range(block_size - 1):
                    predecessor = reference["candidate_selector.predecessor_codebook"][previous]
                    successor = reference["candidate_selector.successor_codebook"][ids[step]]
                    scores = unary[step] + (predecessor * gates[step] * successor).sum(-1)
                    winner = scores.argmax()
                    previous = ids[step, winner]
                    predictions.append(previous)
                    rest = scores.clone()
                    rest[winner] = -torch.inf
                    confidence.append(scores[winner] - rest.logsumexp(0))
                all_ids.append(torch.cat((torch.tensor(block[:1], device="cuda"), torch.stack(predictions))))
                all_confidence.append(torch.stack(confidence))
            if dspark:
                if shared_target:
                    hidden = normalize(hidden, "norm")
                previous, state, predictions, confidence = block[0], torch.zeros(16, device="cuda"), [], []
                for step in range(block_size):
                    previous_embedding = reference["markov_head.markov_w1.weight"][previous]
                    if head_type == "gated":
                        latent = (
                            linear(torch.cat((hidden[step], previous_embedding)), "markov_head.gate_proj").sigmoid()
                            * previous_embedding
                        )
                    else:
                        gate, candidate, raw = linear(
                            torch.cat((state, previous_embedding, hidden[step])), "markov_head.joint_proj"
                        ).chunk(3)
                        state = gate.sigmoid() * state + (1 - gate.sigmoid()) * candidate.tanh()
                        latent = raw.tanh()
                    score = logits[step] + linear(latent, "markov_head.markov_w2")
                    confidence.append(linear(torch.cat((hidden[step], previous_embedding)), "confidence_head.proj"))
                    previous = score.argmax()
                    predictions.append(previous)
                all_ids.append(torch.stack(predictions))
                all_confidence.append(torch.cat(confidence))
        if dflash2:
            expected = torch.cat(candidate_hiddens)
            error = observed_hidden[-1].float() - expected
            assert (error.square().mean() / expected.square().mean()).sqrt().item() <= 0.01
            assert error.abs().max().item() <= expected.abs().max().item() / 32
            assert torch.equal(output.mtp_collector.draft_token_ids, torch.stack(all_ids).flatten())
            if dflash2_options[1]:
                torch.testing.assert_close(
                    output.mtp_collector.confidence_logits, torch.stack(all_confidence), rtol=2e-2, atol=2e-3
                )
            else:
                assert output.mtp_collector.confidence_logits is None
        elif dspark:
            assert torch.equal(output.mtp_collector.draft_token_ids, torch.stack(all_ids).flatten())
            torch.testing.assert_close(
                output.mtp_collector.confidence_logits, torch.stack(all_confidence), rtol=2e-2, atol=2e-3
            )
        else:
            expected = torch.cat(all_logits)
            error = output.logits.float() - expected
            assert (error.square().mean() / expected.square().mean()).sqrt().item() <= 0.01
            assert error.abs().max().item() <= expected.abs().max().item() / 32
            assert torch.equal(output.logits.argmax(-1), expected.argmax(-1))

    tokens = [[21] + [0] * (block_size - 1), [41] + [0] * (block_size - 1)]
    batch = proposal(tokens)
    check_proposal(draft._token_forward(batch), tokens)
    draft.graph = CudaGraph(1, 2 * block_size, block_size, max_batch_size=2 * block_size, max_len_in_batch=32)
    batch.is_cuda_graph_capture = True
    captured = draft.graph.capture_decode(draft._token_forward, batch)
    check_proposal(captured, tokens)

    counts = [1, min(block_size, 2)]
    rows = [(r, prefixes[r] + i, 51 + i) for r, count in enumerate(counts) for i in range(count)]
    hidden = torch.stack([target_hidden[r, position] for r, position, _ in rows])
    with monkeypatch.context() as local:
        for layer in draft.layers_infer:
            local.setattr(
                layer.bound_decode_attention, "run", lambda *a, **kw: pytest.fail("attention during KV commit")
            )
        draft._decode(rows_input(rows, hidden))
    prefixes = [length + count for length, count in zip(prefixes, counts)]
    check_commit()
    tokens = [[61] + [3] * (block_size - 1), [71] + [5] * (block_size - 1)]
    batch = proposal(tokens)
    batch.is_cuda_graph_replay = True
    draft.graph.replay(batch)
    check_proposal(captured, tokens)
    for value, before in main_snapshots:
        torch.testing.assert_close(value, before, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("per_head,static_q", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_noncausal_block_matches_decoded_reference_and_graph(per_head, static_q, dtype):
    import torch.nn.functional as F
    from lightllm.core.compute.attention.batch import AttentionBatch
    from lightllm.core.compute.attention.dense.backend import DenseStaticParams
    from lightllm.core.planner.planner import Planner
    from lightllm.core.contracts.state import StateSpec
    from lightllm.core.storage.attention import DenseStateStore
    from lightllm.core.compute.attention.dense.static_fp8 import StaticFP8AttentionBackend
    from lightllm.core.formats.kv.static_fp8 import StaticFP8KVWriter
    from unit_tests.common.execution.test_static_fp8 import fp8_format

    torch.manual_seed(80)
    spec = DenseAttentionSpec(8, 2, 64, dtype=dtype, causal=False)
    declaration = StateSpec("draft0", "kv.0", 0, (0,), DenseKVStateSpec(kv_heads=2, head_dim=64))
    manager = AttentionStateManager(3)
    store = DenseStateStore((declaration,), fp8_format(per_head), manager)
    manager.finish_state_registration()
    state = manager.get_full_attention_states()[0]
    scales = [0.015625, 0.0078125, 0.0078125, 0.015625] if per_head else [0.015625, 0.0078125]
    calibration = dict(num_layers=1, num_head=2, scales=[scales], scales_shape=[1, len(scales)])
    if static_q:
        calibration["q_calibration"] = dict(num_head=2, scales_shape=[1, 2], scales=[[0.015625, 0.0078125]])
    state.bind_calibration(calibration, 0)
    state.device = torch.device("cuda")
    state.allocate_storage(512)
    view = store.views()[declaration.key]
    alloc = lambda shape, dtype: torch.empty(shape, dtype=dtype, device="cuda")
    planner = Planner(
        (StaticFP8AttentionBackend("verify"),),
        DeviceCaps("cuda", 0, (9, 0)),
        ExecutionLimits(1024 * 1024, 6, 128, max_verify_tokens=3),
    )
    bound = (
        planner.bind(
            [(0, "verify", spec, view, DenseStaticParams(spec, alloc), "fa3")], lambda n: alloc(n, torch.uint8)
        )
        .bindings[0]
        .execution
    )
    ids = torch.randperm(512, device="cuda")[:384].int()
    table = ids.reshape(3, 128)
    kv = torch.randn(384, 4, 64, dtype=dtype, device="cuda")
    StaticFP8KVWriter(view)(token_indices=ids, kv=kv)
    key = bound.k.float() * bound.k_scale[None, :, None]
    value = bound.v.float() * bound.v_scale[None, :, None]
    q = torch.randn(6, 8, 64, dtype=dtype, device="cuda")
    lengths = [65, 17]

    def metadata():
        tensor = lambda values: torch.tensor(values, dtype=torch.int32, device="cuda")
        return bound.create_metadata(
            AttentionBatch(
                6,
                1,
                max(lengths),
                0,
                table,
                tensor([0, 0, 0, 1, 1, 1]),
                tensor([length - 2 + i for length in lengths for i in range(3)]),
                None,
                None,
                tensor(range(7)),
                tensor(range(7)),
                hold_request_id=2,
            )
        )

    def check(output):
        expected = []
        for request, length in enumerate(lengths):
            grouped = q[request * 3 : (request + 1) * 3].float().reshape(3, 2, -1)
            scale = bound.q_scale if static_q else grouped.abs().amax((0, 2)) / 448
            decoded = (grouped / scale[None, :, None]).clamp(-448, 448).to(torch.float8_e4m3fn).float()
            decoded = (decoded * scale[None, :, None]).reshape(3, 8, 64)
            expected.append(
                F.scaled_dot_product_attention(
                    decoded.transpose(0, 1),
                    key[table[request, :length]].transpose(0, 1),
                    value[table[request, :length]].transpose(0, 1),
                    enable_gqa=True,
                ).transpose(0, 1)
            )
        expected = torch.cat(expected)
        error = output.float() - expected
        assert (error.square().mean() / expected.square().mean()).sqrt().item() < 0.035
        assert error.abs().max().item() < torch.finfo(torch.float8_e4m3fn).eps / 2 * expected.abs().max().item()

    meta = metadata()
    check(bound.run(q, meta))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = bound.run(q, meta)
    lengths = [69, 21]
    meta.copy_for_graph(metadata())
    q.add_(0.1)
    graph.replay()
    check(output)
