from inspect import isabstract
from types import SimpleNamespace

import pytest
import torch

import lightllm.common.basemodel.hidden_collector as hidden_collector_module
from lightllm.common.basemodel.hidden_collector import (
    FinalHiddenCollector,
    HiddenCollector,
    LayerHiddenCollector,
    MtpHeadOutputCollector,
    NoopHiddenCollector,
)


class _IdentityPreInfer:
    @staticmethod
    def _tpsp_allgather(input, infer_state):
        del infer_state
        return input


def _mock_target_layer_ids(monkeypatch, layer_ids):
    monkeypatch.setattr(
        hidden_collector_module,
        "get_env_start_args",
        lambda: SimpleNamespace(mtp_draft_model_dir=["/models/draft"]),
    )
    monkeypatch.setattr(
        hidden_collector_module.PretrainedConfig,
        "get_config_dict",
        lambda _: ({"target_layer_ids": layer_ids}, {}),
    )


def test_hidden_collector_is_base_class_for_implementations():
    assert isabstract(HiddenCollector)
    assert issubclass(NoopHiddenCollector, HiddenCollector)
    assert issubclass(FinalHiddenCollector, HiddenCollector)
    assert issubclass(LayerHiddenCollector, HiddenCollector)


def test_final_hidden_collectors_are_independent_instances():
    prototype = FinalHiddenCollector()
    collector0 = prototype.new_instance()
    collector1 = prototype.new_instance()
    hidden0 = torch.randn(2, 3)
    hidden1 = torch.randn(2, 3)
    infer_state = SimpleNamespace(need_dp_prefill_balance=False)

    collector0.add_final_hidden(hidden0)
    collected = collector0.finish_output(infer_state=infer_state).spec_hidden
    assert collected.data_ptr() == hidden0.data_ptr()
    assert collector0.final_hidden is None

    collector0.add_final_hidden(hidden0)
    collector1.add_final_hidden(hidden1)
    collected0 = collector0.finish_output(infer_state=infer_state).spec_hidden
    collected1 = collector1.finish_output(infer_state=infer_state).spec_hidden
    assert collected0.data_ptr() == hidden0.data_ptr()
    assert collected1.data_ptr() == hidden1.data_ptr()


def test_layer_hidden_collector_keeps_microbatch_state_separate(monkeypatch):
    _mock_target_layer_ids(monkeypatch, [0])
    model = SimpleNamespace(is_mtp_draft_model=False, layers_num=2, pre_infer=_IdentityPreInfer())
    prototype = LayerHiddenCollector(model=model)
    collector0 = prototype.new_instance()
    collector1 = prototype.new_instance()
    hidden0 = torch.full((2, 3), 1.0)
    hidden1 = torch.full((2, 3), 2.0)
    infer_state = SimpleNamespace(need_dp_prefill_balance=False)

    collector0.add(layer_index=0, hidden=hidden0)
    collector1.add(layer_index=0, hidden=hidden1)

    collected0 = collector0.finish_output(infer_state=infer_state).spec_hidden
    collected1 = collector1.finish_output(infer_state=infer_state).spec_hidden

    assert torch.equal(collected0, hidden0)
    assert torch.equal(collected1, hidden1)


def test_layer_hidden_collector_requires_target_layer_ids_from_config(monkeypatch):
    _mock_target_layer_ids(monkeypatch, None)
    model = SimpleNamespace(layers_num=2, pre_infer=_IdentityPreInfer())

    with pytest.raises(AssertionError, match="target_layer_ids is required in draft config"):
        LayerHiddenCollector(model=model)


def test_noop_collector_keeps_normal_forward_output_minimal():
    final_hidden = torch.randn(2, 3)
    collector = NoopHiddenCollector()

    collector.add(layer_index=0, hidden=final_hidden)
    collector.add_final_hidden(final_hidden)

    assert collector.finish_output(infer_state=None).spec_hidden is None


def test_mtp_head_output_collector_returns_and_clears_outputs():
    collector = MtpHeadOutputCollector()
    draft_token_ids = torch.arange(6)
    confidence_logits = torch.arange(6).view(2, 3)

    collector.add_mtp_outputs(
        draft_token_ids=draft_token_ids,
        confidence_logits=confidence_logits,
    )
    output = collector.finish_output(infer_state=None)

    assert output.spec_hidden is None
    assert output.draft_token_ids is draft_token_ids
    assert output.confidence_logits is confidence_logits
    assert collector.draft_token_ids is None
    assert collector.confidence_logits is None


def test_final_collector_returns_final_hidden_without_layer_bookkeeping():
    final_hidden = torch.randn(2, 3)
    collector = FinalHiddenCollector()
    collector.add_final_hidden(final_hidden)
    collected = collector.finish_output(infer_state=None).spec_hidden

    assert collected.data_ptr() == final_hidden.data_ptr()


def test_layer_collector_preserves_selected_layers_in_model_order(monkeypatch):
    _mock_target_layer_ids(monkeypatch, [0, 2])
    layer0 = torch.full((2, 2), 1.0)
    layer1 = torch.full((2, 2), 2.0)
    layer2 = torch.full((2, 2), 3.0)
    model = SimpleNamespace(layers_num=3, pre_infer=_IdentityPreInfer())
    collector = LayerHiddenCollector(model=model)

    collector.add(layer_index=0, hidden=layer0)
    collector.add(layer_index=1, hidden=layer1)
    collector.add(layer_index=2, hidden=layer2)
    layer0.fill_(9.0)

    collected = collector.finish_output(infer_state=SimpleNamespace(need_dp_prefill_balance=False)).spec_hidden

    assert torch.equal(collected, torch.cat([torch.full((2, 2), 1.0), layer2], dim=-1))
    assert not collector.layer_hiddens

    collector.add(layer_index=0, hidden=layer0)
    collector.add(layer_index=2, hidden=layer2)
    collected = collector.finish_output(infer_state=SimpleNamespace(need_dp_prefill_balance=False)).spec_hidden

    assert torch.equal(collected, torch.cat([layer0, layer2], dim=-1))
    assert not collector.layer_hiddens


def test_layer_collector_restores_graph_state_without_sharing_runtime_container(monkeypatch):
    _mock_target_layer_ids(monkeypatch, [0])
    model = SimpleNamespace(layers_num=2, pre_infer=_IdentityPreInfer())
    graph_collector = LayerHiddenCollector(model=model)
    collector = graph_collector.new_instance()
    infer_state = SimpleNamespace(need_dp_prefill_balance=False)
    graph_collector.add(layer_index=0, hidden=torch.full((2, 3), 1.0))
    collector.restore_graph_state(graph_collector)

    collected = collector.finish_output(infer_state=infer_state).spec_hidden

    assert torch.equal(collected, torch.full((2, 3), 1.0))
    assert not collector.layer_hiddens
    assert len(graph_collector.layer_hiddens) == 1


def test_layer_collector_releases_graph_tensor_ownership_with_statistics(monkeypatch):
    _mock_target_layer_ids(monkeypatch, [0, 2])
    model = SimpleNamespace(layers_num=3, pre_infer=_IdentityPreInfer())
    collector = LayerHiddenCollector(model=model)
    hidden0 = torch.randn(2, 3)
    hidden2 = torch.randn(2, 3)
    converted = []

    def to_no_ref(hidden):
        converted.append(hidden)
        return hidden

    monkeypatch.setattr(hidden_collector_module, "tensor_to_no_ref_tensor", to_no_ref)
    collector.add(layer_index=0, hidden=hidden0)
    collector.add(layer_index=2, hidden=hidden2)

    tensor_count, total_nbytes = collector.release_graph_tensor_ownership()

    assert tensor_count == 2
    assert total_nbytes == hidden0.numel() * hidden0.element_size() + hidden2.numel() * hidden2.element_size()
    assert all(actual is expected for actual, expected in zip(converted, collector.layer_hiddens))
    assert NoopHiddenCollector().release_graph_tensor_ownership() == (0, 0)
