from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight import FusedMoeWeight
from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.triton_impl import FuseMoeTriton
from lightllm.common.basemodel.triton_kernel.fused_moe.append_shared_expert_topk import (
    append_fused_shared_experts,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton kernels")


def test_experts_with_topk_appends_the_fused_shared_expert(monkeypatch):
    monkeypatch.setattr(
        "lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.fused_moe_weight.get_moe_capture_callback",
        lambda *args, **kwargs: None,
    )
    seen = {}

    def run(**kwargs):
        seen.update(kwargs)
        return kwargs["input_tensor"]

    impl = SimpleNamespace(
        num_fused_shared_experts=1,
        n_routed_experts=4,
        _fused_experts=run,
    )
    impl._prepare_expert_execution = lambda **kwargs: FuseMoeTriton._prepare_expert_execution(impl, **kwargs)
    weight = SimpleNamespace(
        layer_num_=0,
        w13=torch.ones(1, device="cuda"),
        w2=torch.ones(1, device="cuda"),
        fuse_moe_impl=impl,
    )
    topk_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int64, device="cuda")
    topk_weights = torch.tensor([[0.2, 0.8], [0.4, 0.6]], device="cuda")
    FusedMoeWeight.experts_with_topk(
        weight,
        torch.ones(2, 4, device="cuda"),
        topk_weights,
        topk_ids,
        is_prefill=False,
    )
    assert torch.equal(seen["topk_ids"], torch.tensor([[0, 2, 4], [1, 3, 4]], device="cuda"))
    assert torch.allclose(seen["topk_weights"], torch.tensor([[0.2, 0.8, 1.0], [0.4, 0.6, 1.0]], device="cuda"))


def test_capture_observes_logical_ids_before_execution_mapping():
    logical_ids = torch.tensor([[0, 2]], dtype=torch.int64, device="cuda")
    logical_weights = torch.tensor([[0.2, 0.8]], device="cuda")
    captured = {}
    executed = {}
    impl = FuseMoeTriton.__new__(FuseMoeTriton)
    impl.n_routed_experts = 4
    impl.num_fused_shared_experts = 1
    impl._select_experts = lambda **kwargs: (logical_weights, logical_ids)

    def fused(**kwargs):
        executed.update(kwargs)
        return kwargs["input_tensor"]

    impl._fused_experts = fused
    hidden = torch.ones(1, 4, device="cuda")
    impl(
        input_tensor=hidden,
        router_logits=hidden,
        w13=None,
        w2=None,
        correction_bias=None,
        scoring_func="softmax",
        top_k=2,
        renormalize=False,
        use_grouped_topk=False,
        topk_group=0,
        num_expert_group=0,
        is_prefill=False,
        moe_capture_callback=lambda ids: captured.setdefault("ids", ids),
    )

    assert captured["ids"] is logical_ids
    assert torch.equal(logical_ids, torch.tensor([[0, 2]], device="cuda"))
    assert torch.equal(executed["topk_ids"], torch.tensor([[0, 2, 4]], device="cuda"))


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("execution_index_dtype", [None, torch.int64])
def test_padding_mask_preserves_captured_routes_and_replays(index_dtype, execution_index_dtype):
    logical_ids = torch.tensor([[0, 2], [1, 3]], dtype=index_dtype, device="cuda")
    logical_weights = torch.tensor([[0.2, 0.8], [0.4, 0.6]], device="cuda")
    impl = FuseMoeTriton.__new__(FuseMoeTriton)
    impl.n_routed_experts = 4
    impl.num_fused_shared_experts = 1
    impl.execution_index_dtype = execution_index_dtype
    impl._select_experts = lambda **kwargs: (logical_weights, logical_ids)
    impl._fused_experts = lambda **kwargs: (kwargs["topk_weights"], kwargs["topk_ids"])
    hidden = torch.ones(2, 4, device="cuda")
    mask = torch.tensor([True, False], device="cuda")
    captured = {}

    def run():
        return impl(
            input_tensor=hidden,
            router_logits=hidden,
            w13=None,
            w2=None,
            correction_bias=None,
            scoring_func="softmax",
            top_k=2,
            renormalize=False,
            use_grouped_topk=False,
            topk_group=0,
            num_expert_group=0,
            is_prefill=False,
            moe_capture_callback=lambda ids: captured.setdefault("ids", ids),
            token_mask=mask,
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = run()
    for values in ([True, False], [False, True], [False, False], [True, True]):
        mask.copy_(torch.tensor(values, device="cuda"))
        graph.replay()
        expected_ids = torch.tensor(
            [[0, 2, 4], [1, 3, 4]], dtype=execution_index_dtype or index_dtype, device="cuda"
        ).masked_fill(~mask[:, None], -1)
        expected_weights = torch.tensor([[0.2, 0.8, 1.0], [0.4, 0.6, 1.0]], device="cuda")
        torch.testing.assert_close(ids, expected_ids, atol=0, rtol=0)
        torch.testing.assert_close(weights, expected_weights.masked_fill(~mask[:, None], 0), atol=0, rtol=0)
    assert captured["ids"] is logical_ids
    torch.testing.assert_close(
        logical_ids, torch.tensor([[0, 2], [1, 3]], dtype=index_dtype, device="cuda"), atol=0, rtol=0
    )
    torch.testing.assert_close(logical_weights, torch.tensor([[0.2, 0.8], [0.4, 0.6]], device="cuda"))


@pytest.mark.parametrize("rows,topk,strided", [(0, 10, False), (17, 3, True), (129, 10, False), (1024, 10, False)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_mask_topk_preserves_inputs_and_dynamic_graph(rows, topk, strided, dtype, index_dtype):
    from lightllm.common.basemodel.triton_kernel.fused_moe.mask_topk import mask_topk

    stride = 2 if strided else 1
    ids = torch.randint(512, (rows, topk * stride), dtype=index_dtype, device="cuda")[:, ::stride]
    weights = torch.randn(rows, topk * stride, dtype=dtype, device="cuda")[:, ::stride]
    mask = torch.arange(rows * stride, device="cuda").remainder(3).bool()[::stride]
    saved_ids, saved_weights = ids.clone(), weights.clone()

    def run():
        return mask_topk(weights, ids, mask, torch.int64)

    graph = torch.cuda.CUDAGraph()
    run()
    with torch.cuda.graph(graph):
        output_weights, output_ids = run()
    for active in (True, False, True):
        mask.fill_(active)
        graph.replay()
        torch.testing.assert_close(output_weights, weights.masked_fill(~mask[:, None], 0), atol=0, rtol=0)
        torch.testing.assert_close(output_ids, ids.masked_fill(~mask[:, None], -1).long(), atol=0, rtol=0)
        torch.testing.assert_close(ids, saved_ids, atol=0, rtol=0)
        torch.testing.assert_close(weights, saved_weights, atol=0, rtol=0)
    graph.reset()


def test_redundancy_repair_does_not_overwrite_logical_ids(monkeypatch):
    from lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl import (
        FuseMoeDeepGEMM,
    )

    logical_ids = torch.tensor([[1, 3]], dtype=torch.int64, device="cuda")
    seen = {}

    def repair(topk_ids, **kwargs):
        seen["ptr"] = topk_ids.data_ptr()
        topk_ids.fill_(-1)

    monkeypatch.setattr(
        "lightllm.common.basemodel.layer_weights.meta_weights.fused_moe.impl.deepgemm_impl.redundancy_topk_ids_repair",
        repair,
    )
    impl = FuseMoeDeepGEMM.__new__(FuseMoeDeepGEMM)
    impl.redundancy_expert_num = 1
    impl.redundancy_expert_ids_tensor = None
    impl.ep_n_routed_experts = 2
    impl.global_rank_ = 0
    impl.routed_expert_counter_tensor = None
    impl.auto_update_redundancy_expert = False
    impl.num_fused_shared_experts = 0
    impl.n_routed_experts = 4
    _, execution_ids = FuseMoeDeepGEMM._prepare_expert_execution(
        impl,
        torch.ones(1, 2, device="cuda"),
        logical_ids,
        is_prefill=False,
    )
    assert seen["ptr"] != logical_ids.data_ptr()
    assert torch.equal(logical_ids, torch.tensor([[1, 3]], device="cuda"))
    assert torch.equal(execution_ids, torch.full_like(logical_ids, -1))


def test_append_fused_shared_experts_without_gate():
    topk_ids = torch.tensor([[0, 2], [1, 3], [2, 0]], dtype=torch.int32, device="cuda")
    topk_weights = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3]], dtype=torch.float32, device="cuda")

    out_weights, out_ids = append_fused_shared_experts(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_expert_start_id=4,
        num_fused_shared_experts=1,
        shared_expert_gate=None,
    )

    expect_ids = torch.tensor([[0, 2, 4], [1, 3, 4], [2, 0, 4]], dtype=torch.int32, device="cuda")
    expect_weights = torch.tensor(
        [[0.2, 0.8, 1.0], [0.4, 0.6, 1.0], [0.7, 0.3, 1.0]], dtype=torch.float32, device="cuda"
    )
    assert torch.equal(out_ids, expect_ids)
    assert torch.allclose(out_weights, expect_weights)


def test_append_fused_shared_experts_with_gate():
    topk_ids = torch.tensor([[0, 2], [1, 3], [2, 0]], dtype=torch.int32, device="cuda")
    topk_weights = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3]], dtype=torch.float32, device="cuda")
    shared_expert_gate = torch.tensor([[0.0], [2.0], [-2.0]], dtype=torch.float32, device="cuda")

    out_weights, out_ids = append_fused_shared_experts(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_expert_start_id=4,
        num_fused_shared_experts=1,
        shared_expert_gate=shared_expert_gate,
    )

    expect_ids = torch.tensor([[0, 2, 4], [1, 3, 4], [2, 0, 4]], dtype=torch.int32, device="cuda")
    expect_weights = torch.cat([topk_weights, torch.sigmoid(shared_expert_gate)], dim=1)
    assert torch.equal(out_ids, expect_ids)
    assert torch.allclose(out_weights, expect_weights)


def test_append_fused_shared_experts_multiple_tokens_per_grid():
    token_num = 4097
    topk_ids = torch.stack(
        [
            torch.arange(token_num, dtype=torch.int32, device="cuda") % 4,
            (torch.arange(token_num, dtype=torch.int32, device="cuda") + 1) % 4,
        ],
        dim=1,
    )
    topk_weights = torch.rand((token_num, 2), dtype=torch.float32, device="cuda")
    shared_expert_gate = torch.randn((token_num, 1), dtype=torch.float32, device="cuda")

    out_weights, out_ids = append_fused_shared_experts(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_expert_start_id=4,
        num_fused_shared_experts=1,
        shared_expert_gate=shared_expert_gate,
    )

    expect_ids = torch.cat(
        [topk_ids, torch.full((token_num, 1), 4, dtype=torch.int32, device="cuda")],
        dim=1,
    )
    expect_weights = torch.cat([topk_weights, torch.sigmoid(shared_expert_gate)], dim=1)
    assert torch.equal(out_ids, expect_ids)
    assert torch.allclose(out_weights, expect_weights)


def test_append_fused_shared_experts_multi_shared_gate():
    token_num = 7
    topk_ids = torch.stack(
        [
            torch.arange(token_num, dtype=torch.int32, device="cuda") % 4,
            (torch.arange(token_num, dtype=torch.int32, device="cuda") + 1) % 4,
        ],
        dim=1,
    )
    topk_weights = torch.rand((token_num, 2), dtype=torch.float32, device="cuda")
    shared_expert_gate = torch.randn((token_num, 2), dtype=torch.float32, device="cuda")

    out_weights, out_ids = append_fused_shared_experts(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_expert_start_id=4,
        num_fused_shared_experts=2,
        shared_expert_gate=shared_expert_gate,
    )

    expect_ids = torch.cat(
        [
            topk_ids,
            torch.tensor([[4, 5]], dtype=torch.int32, device="cuda").repeat(token_num, 1),
        ],
        dim=1,
    )
    expect_weights = torch.cat([topk_weights, torch.sigmoid(shared_expert_gate)], dim=1)
    assert torch.equal(out_ids, expect_ids)
    assert torch.allclose(out_weights, expect_weights)
