from types import SimpleNamespace

import torch

from lightllm.common.basemodel.hidden_collector import (
    FinalHiddenCollector,
    HiddenCollector,
    LayerHiddenCollector,
    NoopHiddenCollector,
)


class _IdentityPreInfer:
    @staticmethod
    def _tpsp_allgather(input, infer_state):
        del infer_state
        return input


def test_hidden_collector_selects_implementation():
    model = SimpleNamespace(is_mtp_draft_model=False, layers_num=3, pre_infer=_IdentityPreInfer())
    noop_collector = HiddenCollector()
    final_hidden_collector = HiddenCollector(model=model, spec_mode="vanilla_with_att")
    layer_hidden_collector = HiddenCollector(
        model=model,
        spec_mode="eagle3",
        layer_ids=[0, 2],
        microbatch_count=2,
    )

    assert isinstance(noop_collector, HiddenCollector)
    assert isinstance(noop_collector.collectors[0], NoopHiddenCollector)
    assert isinstance(final_hidden_collector.collectors[0], FinalHiddenCollector)
    assert all(isinstance(collector, LayerHiddenCollector) for collector in layer_hidden_collector.collectors)


def test_draft_hidden_collector_follows_spec_mode():
    model = SimpleNamespace(is_mtp_draft_model=True)

    recurrent_collector = HiddenCollector(model=model, spec_mode="eagle3")
    block_collector = HiddenCollector(model=model, spec_mode="dspark")

    assert isinstance(recurrent_collector.collectors[0], FinalHiddenCollector)
    assert isinstance(block_collector.collectors[0], NoopHiddenCollector)


def test_hidden_collector_supports_single_and_overlap_forward():
    model = SimpleNamespace(is_mtp_draft_model=False)
    collector = HiddenCollector(model=model, spec_mode="vanilla_with_att", microbatch_count=2)
    hidden0 = torch.randn(2, 3)
    hidden1 = torch.randn(2, 3)
    infer_state = SimpleNamespace(need_dp_prefill_balance=False)

    collector.add(layer_index=0, hidden=hidden0)
    collected = collector.finish(
        infer_state=infer_state,
        final_hidden=hidden0,
    )
    assert collected.data_ptr() == hidden0.data_ptr()

    collector.add(layer_index=0, hidden=hidden0)
    collector.add(layer_index=0, hidden=hidden1, microbatch_index=1)
    collected0 = collector.finish(
        infer_state=infer_state,
        final_hidden=hidden0,
    )
    collected1 = collector.finish(
        infer_state=infer_state,
        final_hidden=hidden1,
        microbatch_index=1,
    )
    assert collected0.data_ptr() == hidden0.data_ptr()
    assert collected1.data_ptr() == hidden1.data_ptr()


def test_layer_hidden_collector_keeps_microbatch_state_separate():
    model = SimpleNamespace(is_mtp_draft_model=False, layers_num=2, pre_infer=_IdentityPreInfer())
    collector = HiddenCollector(model=model, spec_mode="eagle3", layer_ids=[0], microbatch_count=2)
    hidden0 = torch.full((2, 3), 1.0)
    hidden1 = torch.full((2, 3), 2.0)
    infer_state = SimpleNamespace(need_dp_prefill_balance=False)

    collector.add(layer_index=0, hidden=hidden0)
    collector.add(layer_index=0, hidden=hidden1, microbatch_index=1)

    collected0 = collector.finish(infer_state=infer_state, final_hidden=hidden0)
    collected1 = collector.finish(infer_state=infer_state, final_hidden=hidden1, microbatch_index=1)

    assert torch.equal(collected0, hidden0)
    assert torch.equal(collected1, hidden1)


def test_noop_collector_keeps_normal_forward_output_minimal():
    final_hidden = torch.randn(2, 3)
    collector = NoopHiddenCollector()

    collector.add(layer_index=0, hidden=final_hidden)

    assert collector.prefill_outputs(final_hidden) == [final_hidden]
    assert (
        collector.finish(
            infer_state=None,
            final_hidden=final_hidden,
        )
        is None
    )


def test_final_collector_returns_final_hidden_without_layer_bookkeeping():
    final_hidden = torch.randn(2, 3)
    collected = FinalHiddenCollector().finish(
        infer_state=None,
        final_hidden=final_hidden,
    )

    assert collected.data_ptr() == final_hidden.data_ptr()


def test_layer_collector_preserves_selected_layers_in_model_order():
    layer0 = torch.full((2, 2), 1.0)
    layer1 = torch.full((2, 2), 2.0)
    layer2 = torch.full((2, 2), 3.0)
    model = SimpleNamespace(layers_num=3, pre_infer=_IdentityPreInfer())
    collector = LayerHiddenCollector(model=model, layer_ids=[0, 2])

    collector.add(layer_index=0, hidden=layer0)
    collector.add(layer_index=1, hidden=layer1)
    collector.add(layer_index=2, hidden=layer2)
    layer0.fill_(9.0)

    forward_outputs = collector.prefill_outputs(layer2)
    collected = collector.finish(
        infer_state=SimpleNamespace(need_dp_prefill_balance=False),
        final_hidden=layer2,
        forward_outputs=forward_outputs,
    )

    assert torch.equal(collected, torch.cat([torch.full((2, 2), 1.0), layer2], dim=-1))
    assert not collector.layer_hiddens

    collector.add(layer_index=0, hidden=layer0)
    collector.add(layer_index=2, hidden=layer2)
    collected = collector.finish(
        infer_state=SimpleNamespace(need_dp_prefill_balance=False),
        final_hidden=layer2,
    )

    assert torch.equal(collected, torch.cat([layer0, layer2], dim=-1))
    assert not collector.layer_hiddens
