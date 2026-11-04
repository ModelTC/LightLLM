from types import SimpleNamespace

import pytest
import torch

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.attention_state.full_att import GQAAttentionConfig, GQAFullAttentionState
from lightllm.core.build import PreparedExecution as Runner
from lightllm.core.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.core.contracts.state import FormatSpec
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.qwen2.model import Qwen2TpPartModel
from lightllm.models.qwen3.model import Qwen3TpPartModel
from lightllm.models.qwen2_vl.model import Qwen2VLTpPartModel
from lightllm.models.qwen3_vl.model import Qwen3VLTpPartModel
from lightllm.models.qwen3_vl_moe.model import Qwen3VLMOETpPartModel


def make_model(model_class, device="cpu", implementation="triton", microbatches=1):
    model = model_class.__new__(model_class)
    model.args = SimpleNamespace(mtp_mode=None, llm_kv_type="None")
    model.tp_world_size_ = 1
    model.data_type = torch.float16
    model.config = dict(num_key_value_heads=2, num_attention_heads=4, hidden_size=256, num_hidden_layers=2)
    model._init_model_adapter()
    model.att_state_manager = AttentionStateManager(request_capacity=2)
    model.layers_infer = []
    for i in range(2):
        layer = model.transformer_layer_infer_class.__new__(model.transformer_layer_infer_class)
        layer.layer_num_, layer.tp_q_head_num_, layer.tp_k_head_num_, layer.head_dim_ = i, 4, 2, 64
        layer.alloc_tensor = lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device)
        model.layers_infer.append(layer)
    runtime = SimpleNamespace(
        caps=DeviceCaps("cuda", 0, (9, 0)),
        allocate_workspace=lambda size: torch.empty(size, dtype=torch.uint8, device=device),
        allocate_tensor=lambda shape, dtype: torch.empty(shape, dtype=dtype, device=device),
    )
    model.execution_runner = Runner(
        model.model_adapter,
        model.layers_infer,
        model.att_state_manager,
        runtime,
        ExecutionLimits((512 if implementation == "flashinfer" else 16) * 1024 * 1024, 2, 8, microbatches),
        FormatSpec("plain", torch.float16, "token_head_dim", ("k", "v"), 1),
        {"prefill": implementation, "decode": implementation},
    )
    model.att_state_manager.finish_state_registration()
    state = model.att_state_manager.get_full_attention_states()[0]
    state.device = torch.device(device)
    state.allocate_storage(8)
    return model, state


@pytest.mark.parametrize(
    "model_class",
    [
        LlamaTpPartModel,
        Qwen2TpPartModel,
        Qwen3TpPartModel,
        Qwen2VLTpPartModel,
        Qwen3VLTpPartModel,
        Qwen3VLMOETpPartModel,
    ],
)
def test_model_startup_binds_layers_and_writer(model_class, monkeypatch):
    model, state = make_model(model_class)
    writes = []
    monkeypatch.setattr(
        "lightllm.common.execution.dense.DenseKVWriter.__call__",
        lambda self, **kw: writes.append(dict(kw, view=self.view)),
    )
    model._init_execution_plan()
    assert len(model.execution_plan.bindings) == 4
    model.att_state_manager = None
    layer = model.layers_infer[1]
    q = torch.randn(2, 256, dtype=torch.float16)
    batch = SimpleNamespace(
        execution_metadata={"triton.prefill": object(), "triton.decode": object()}, mem_index=torch.tensor([0, 1])
    )
    for phase in ("prefill", "decode"):
        bound = getattr(layer, f"bound_{phase}_attention")
        assert bound.k.data_ptr() == state.k_storage[1].data_ptr()
        monkeypatch.setattr(bound, "run", lambda q, metadata: q)
    assert layer._context_attention_kernel(q, None, batch, None).shape == q.shape
    assert layer._token_attention_kernel(q, batch, None).shape == q.shape
    layer._post_cache_kv(q, batch, None)
    assert writes[0]["view"].state.producer == 1
    assert writes[0]["token_indices"] is batch.mem_index


def test_unmigrated_subclass_does_not_inherit_opt_in():
    class DifferentAttentionModel(LlamaTpPartModel):
        pass

    model = DifferentAttentionModel.__new__(DifferentAttentionModel)
    model._init_execution_plan()
    assert model.execution_plan is None


def test_dense_semantics_use_model_positions_with_external_layer_ids():
    from lightllm.models.base.adapter import DenseModelAdapter

    adapter = DenseModelAdapter(
        dict(
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=32,
        ),
        role="draft1",
        dtype=torch.float16,
        layer_ids=(24, 27),
    )
    assert adapter.attention_specs()[24].window == (-1, -1)
    assert adapter.attention_specs()[27].window == (31, 0)
    assert [(s.role, s.producer, s.consumers) for s in adapter.state_specs()] == [
        ("draft1", 24, (24,)),
        ("draft1", 27, (27,)),
    ]


@pytest.mark.parametrize("main_layers,previous_layers", [(24, []), (56, [1, 1])])
def test_qwen35_draft_declares_execution_ids_and_shared_weights(main_layers, previous_layers):
    from lightllm.models.qwen3_5_mtp.model import Qwen3_5MTPModel

    model = Qwen3_5MTPModel.__new__(Qwen3_5MTPModel)
    model.args = SimpleNamespace(mtp_mode="eagle_with_att")
    model.data_type, model.tp_world_size_ = torch.float16, 2
    model.config = dict(
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=256,
        layer_types=["full_attention"],
    )
    weights = SimpleNamespace(wte_weight_=object(), lm_head_weight_=object(), final_norm_weight_=object())
    model.main_model = SimpleNamespace(layers_infer=[None] * main_layers, pre_post_weight=weights)
    model.mtp_previous_draft_models = [SimpleNamespace(layers_infer=[None] * n) for n in previous_layers]
    model._init_model_adapter()
    adapter = model.model_adapter
    layer = main_layers + sum(previous_layers)
    assert adapter.layer_ids == (layer,)
    assert adapter.checkpoint_layer_id == len(previous_layers)
    assert adapter.shared_weights == (weights.wte_weight_, weights.lm_head_weight_, weights.final_norm_weight_)
    (state,) = adapter.state_specs()
    assert (state.role, state.producer) == (f"draft{len(previous_layers)}", layer)


@pytest.mark.parametrize("previous_layers", [[], [1, 1]])
def test_qwen35_moe_draft_binds_experts_at_execution_layer(previous_layers):
    from lightllm.models.qwen3_5_moe_mtp.model import Qwen3_5MoeMTPModel
    from lightllm.models.qwen3_5_moe_mtp.layer_weights.transformer_layer_weight import (
        Qwen3_5MoeMTPTransformerLayerWeight,
    )

    model = Qwen3_5MoeMTPModel.__new__(Qwen3_5MoeMTPModel)
    model.args = SimpleNamespace(mtp_mode="eagle_with_att")
    model.data_type, model.tp_world_size_ = torch.float16, 2
    model.config = model.normalize_model_config(
        dict(
            num_hidden_layers=24,
            num_attention_heads=4,
            num_key_value_heads=2,
            hidden_size=256,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=256,
            shared_expert_intermediate_size=256,
        )
    )
    weights = SimpleNamespace(wte_weight_=object(), lm_head_weight_=object(), final_norm_weight_=object())
    model.main_model = SimpleNamespace(layers_infer=[None] * 24, pre_post_weight=weights)
    model.mtp_previous_draft_models = [SimpleNamespace(layers_infer=[None] * n) for n in previous_layers]
    model._init_model_adapter()
    adapter = model.model_adapter
    layer = 24 + sum(previous_layers)
    assert set(adapter.attention_specs()) == set(adapter.moe_specs()) == {layer}
    assert adapter.checkpoint_layer_id == len(previous_layers)
    assert adapter.transformer_weight_class is Qwen3_5MoeMTPTransformerLayerWeight
    assert adapter.shared_weights == (weights.wte_weight_, weights.lm_head_weight_, weights.final_norm_weight_)
    spec = adapter.moe_specs()[layer]
    assert spec.renormalize and spec.gated_shared and spec.shared_experts == 1 and spec.round_intermediates


@pytest.mark.parametrize("model_class", [Qwen2VLTpPartModel, Qwen3VLTpPartModel, Qwen3VLMOETpPartModel])
@pytest.mark.parametrize("nested", [False, True])
def test_vl_startup_normalizes_rope_and_expert_config(model_class, nested, tmp_path):
    import json

    text = dict(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=256,
        head_dim=64,
        vocab_size=128,
        num_local_experts=4,
        rope_parameters=dict(rope_type="default", rope_theta=10000.0, mrope_section=[11, 11, 10]),
    )
    raw = {"text_config": text, "vision_config": {}} if nested else text
    (tmp_path / "config.json").write_text(json.dumps(raw))
    model = model_class.__new__(model_class)
    model.weight_dir_, model.finetune_config, model.max_total_token_num = str(tmp_path), None, 64
    model._init_config()
    assert model.config["rope_scaling"] == text["rope_parameters"]
    assert model.config["rope_theta"] == 10000.0
    assert model.config["head_dim"] == 64
    if model_class is Qwen3VLMOETpPartModel:
        assert model.config["num_experts"] == 4
    assert "rope_scaling" not in text


@pytest.mark.parametrize("renormalize", [None, False, True])
@pytest.mark.parametrize("expert_field", ["num_experts", "num_local_experts"])
def test_qwen3_moe_declares_selected_expert_layers(renormalize, expert_field):
    from lightllm.models.qwen3_moe.adapter import Qwen3MoEModelAdapter
    from lightllm.models.qwen3_moe.model import Qwen3MOEModel

    config = dict(
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=256,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        mlp_only_layers=[3],
        decoder_sparse_step=2,
    )
    if renormalize is not None:
        config["norm_topk_prob"] = renormalize
    config[expert_field] = config.pop("num_experts")
    normalized = Qwen3MOEModel.normalize_model_config(config)
    adapter = Qwen3MoEModelAdapter(normalized, role="main", dtype=torch.float16)
    specs = adapter.moe_specs()
    assert set(specs) == {1, 5}
    assert len(adapter.attention_specs()) == len(adapter.state_specs()) == 6
    assert all(s.renormalize == (renormalize is not False) for s in specs.values())
    assert adapter.config["norm_topk_prob"] == (renormalize is not False)
    assert ("norm_topk_prob" in config) == (renormalize is not None)


def test_mtp_cohort_keeps_existing_execution_path():
    from lightllm.models.base.build import prepare_runner

    model = LlamaTpPartModel.__new__(LlamaTpPartModel)
    model.args = SimpleNamespace(mtp_mode="vanilla_with_att")
    assert prepare_runner(model) is None


@pytest.mark.parametrize("nested", [False, True])
def test_qwen35_moe_declares_hybrid_state_and_gated_experts(nested):
    from lightllm.core.compute.attention.dense.spec import DenseAttentionSpec
    from lightllm.core.compute.attention.gdn.spec import GatedDeltaNetSpec
    from lightllm.models.registry import ModelRegistry
    from lightllm.models.qwen3_5_moe.model import Qwen3_5MOETpPartModel
    from lightllm.models.qwen3_5_moe.adapter import Qwen35MoEModelAdapter
    from lightllm.models.qwen3_5_moe.layer_weights.transformer_layer_weight import Qwen35MOETransformerLayerWeight

    text = dict(
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=256,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
    )
    raw = dict(model_type="qwen3_5_moe", text_config=text) if nested else dict(text, model_type="qwen3_5_moe")
    assert ModelRegistry.get_model_class(raw) is Qwen3_5MOETpPartModel
    model = Qwen3_5MOETpPartModel.__new__(Qwen3_5MOETpPartModel)
    model.args = SimpleNamespace(mtp_mode=None, llm_kv_type="None")
    model.data_type, model.tp_world_size_ = torch.float16, 2
    model.config = model.normalize_model_config(raw)
    model._init_model_adapter()
    adapter = model.model_adapter
    assert isinstance(adapter, Qwen35MoEModelAdapter)
    assert adapter.transformer_weight_class is Qwen35MOETransformerLayerWeight
    attention = adapter.attention_specs()
    assert isinstance(attention[0], GatedDeltaNetSpec) and isinstance(attention[1], DenseAttentionSpec)
    assert attention[0].k_heads == 1 and attention[1].q_heads == 2
    assert [(s.producer, s.retention) for s in adapter.state_specs()] == [(0, "recurrent"), (1, "history")]
    assert set(adapter.moe_specs()) == {0, 1}
    assert all(s.renormalize and s.gated_shared and s.shared_experts == 1 for s in adapter.moe_specs().values())
    assert all(s.round_intermediates for s in adapter.moe_specs().values())
    assert "norm_topk_prob" not in text


def test_fa3_page_tables_are_owned_by_backend_and_microbatch(monkeypatch):
    from lightllm.common.basemodel.attention.fa3 import fp

    monkeypatch.setattr(fp, "get_current_device_id", lambda: "cpu")
    args = SimpleNamespace(
        running_max_req_size=2, mtp_step=0, enable_tpsp_mix_mode=False, enable_decode_microbatch_overlap=True
    )

    def create():
        backend = object.__new__(fp.Fa3AttBackend)
        backend.model = SimpleNamespace(args=args, graph_max_batch_size=2, max_seq_length=16)
        backend.infer_page_size = 1
        backend._init_page_table_buffers()
        return backend

    first, second = create(), create()
    buffers = first.page_table_buffers + second.page_table_buffers
    assert len({b.data_ptr() for b in buffers}) == 4
    buffers[0].fill_(7)
    buffers[2].zero_()
    assert torch.all(buffers[0] == 7)


def test_rebinding_a_live_plan_is_rejected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (9, 0))
    model, _ = make_model(LlamaTpPartModel)
    model._init_execution_plan()
    with pytest.raises(RuntimeError, match="cannot be replaced"):
        model._init_execution_plan()
