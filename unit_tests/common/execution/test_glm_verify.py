from lightllm.common.execution.formats.recurrent.spec import ConvSSMStateSpec
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from lightllm.common.attention_state import AttentionStateManager
from lightllm.common.execution.compute.attention.kda.backend import KDAInputs, KDAStaticParams, KimiDeltaAttentionBackend
from lightllm.common.execution.planner.planner import Planner
from lightllm.common.execution.contracts.device import DeviceCaps, ExecutionLimits
from lightllm.common.execution.contracts.state import FormatSpec, StateSpec
from lightllm.common.execution.compute.attention.kda.spec import KimiDeltaAttentionSpec
from lightllm.common.execution.storage.attention import ConvSSMStateStore
from unit_tests.common.execution.test_native_glm import make_batch


@pytest.mark.parametrize("previous_depth", [0, 2])
def test_glm_draft_declares_checkpoint_and_state_domains(previous_depth):
    from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel
    from lightllm.models.glm5_next_mtp.adapter import Glm5NextMTPModelAdapter
    from lightllm.common.execution.compute.attention.pooled_nope.spec import PooledNoPEAttentionSpec

    raw = dict(
        num_hidden_layers=4,
        num_nextn_predict_layers=1,
        hidden_size=128,
        num_attention_heads=4,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=0,
        index_n_heads=4,
        index_head_dim=128,
        index_topk=256,
        index_kpool=4,
        layer_types=["linear_attention"] * 3 + ["deepseek_sparse_attention"],
        linear_attn_config=dict(num_heads=2, head_dim=128, short_conv_kernel_size=4),
        mhc=True,
        n_routed_experts=4,
        n_shared_experts=1,
        first_k_dense_replace=1,
        num_experts_per_tok=2,
        moe_intermediate_size=128,
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        norm_topk_prob=True,
        swiglu_limit=10.0,
    )
    config = Glm5NextMTPModel.normalize_model_config(raw)
    assert Glm5NextMTPModel.normalize_model_config(config) == config
    layer = 4 + previous_depth
    adapter = Glm5NextMTPModelAdapter(
        config,
        role="draft",
        dtype=torch.bfloat16,
        layer_ids=(layer,),
        shared_weights=(None, None, None),
    )
    assert raw["num_hidden_layers"] == 4 and raw["mhc"]
    assert adapter.checkpoint_layer_id == 4 and adapter.layer_ids == (layer,)
    assert isinstance(adapter.attention_specs()[layer], PooledNoPEAttentionSpec)
    assert {s.producer for s in adapter.state_specs()} == {layer}
    assert {s.retention for s in adapter.state_specs()} == {"history", "window"}
    assert set(adapter.moe_specs()) == {layer}
    assert not adapter.config["mhc"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "implementation,quantized",
    [("triton", False), ("triton", True), ("flashmla_sparse", False), ("flashmla_sparse", True)],
)
@pytest.mark.parametrize("layout", ["fixed", "dynamic"])
def test_glm_mtp_verification(implementation, quantized, layout, monkeypatch, tmp_path):
    from unit_tests.common.execution.test_native_glm import test_glm_model_binding_prefix_and_graph

    test_glm_model_binding_prefix_and_graph(
        implementation, quantized, True, True, monkeypatch, tmp_path, verify_layout=layout
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "implementation,quantized,dtype",
    [
        ("triton", False, torch.float16),
        ("triton", False, torch.bfloat16),
        ("triton", True, torch.bfloat16),
        ("flashmla_sparse", False, torch.float16),
        ("flashmla_sparse", False, torch.bfloat16),
        ("flashmla_sparse", True, torch.bfloat16),
    ],
)
def test_pooled_verify_rejected_tail_and_graph(implementation, quantized, dtype, monkeypatch):
    from unit_tests.common.execution.test_native_glm import test_pooled_nope_dense_and_sparse_prefix_graph

    test_pooled_nope_dense_and_sparse_prefix_graph(
        implementation, quantized, dtype, monkeypatch, verify_candidates=True
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["vanilla_with_att", "eagle_with_att"])
@pytest.mark.parametrize(
    "implementation,quantized",
    [
        ("triton", False),
        ("triton", True),
        ("flashmla_sparse", False),
        ("flashmla_sparse", True),
    ],
)
def test_glm_mtp_draft_binding_and_graph(mode, implementation, quantized, monkeypatch, tmp_path):
    from unit_tests.common.execution.test_native_glm import test_glm_model_binding_prefix_and_graph

    test_glm_model_binding_prefix_and_graph(
        implementation, quantized, True, True, monkeypatch, tmp_path, draft_mode=mode
    )


@torch.no_grad()
def exercise_glm_drafts(main, drafts, references, monkeypatch):
    from lightllm.common.execution.runner.decode_graph import CudaGraph
    from lightllm.common.execution.runner.prefill_graph import PrefillCudaGraph
    from unit_tests.common.execution.test_model_cuda import prepare_model_batch

    histories = [torch.randn(64, 128, dtype=main.data_type, device="cuda") * 0.3 for _ in range(2)]
    main_fields = [
        t for view in main.attention_execution.resources.stores.get("history").views().values() for t in view.fields
    ]
    for value in main_fields:
        value.zero_()

    def prepare(model, sequences, cached, prefill):
        inp = prepare_model_batch(model, sequences, cached, prefill, input_only=True)
        inp.mtp_draft_input_hiddens = torch.cat([h[n : len(s)] for h, n, s in zip(histories, cached, sequences)])
        state = model._create_inferstate(inp)
        model.batch_metadata(state)
        state.init_att_state()
        return state

    for index, (draft, reference) in enumerate(zip(drafts, references, strict=True)):
        assert draft.model_adapter.layer_ids == (4 + index,)
        assert draft.model_adapter.checkpoint_layer_id == 4
        assert draft.pre_post_weight.wte_weight_ is main.pre_post_weight.wte_weight_
        assert draft.pre_post_weight.lm_head_weight_ is main.pre_post_weight.lm_head_weight_
        assert draft.pre_post_weight.main_norm_weight_ is main.pre_post_weight.final_norm_weight_
        assert draft.linear_plan.bindings and not draft.linear_plan.legacy
        assert len(draft.moe_plan.bindings) == 1 and not draft.moe_plan.legacy
        assert set(draft.model_adapter.attention_specs()) == {4 + index}
        monkeypatch.setattr(
            draft.trans_layers_weight[0].experts, "experts", lambda *a, **kw: pytest.fail("legacy draft MoE")
        )
        fields = [
            t
            for store in (
                draft.attention_execution.resources.stores.get("history"),
                draft.attention_execution.resources.stores.get("window"),
            )
            for view in store.views().values()
            for t in view.fields
        ]
        for field in fields:
            field.zero_()

        def expected(sequences, cached, prefill):
            if reference.args.llm_kv_type == "fp8kv_dsa":
                # FP8 prefill retains current tokens at BF16; decode reads encoded KV.
                # Keep the same phase/history when comparing these representations.
                batch = prepare(reference, sequences, cached, prefill)
                output = reference._context_forward(batch) if prefill else reference._token_forward(batch)
                return output.logits.clone()
            return reference._context_forward(prepare(reference, sequences, [0, 0], True)).logits.clone()

        def check(output, target):
            torch.testing.assert_close(output.logits, target, rtol=0.02, atol=0.005)
            assert torch.equal(output.logits.argmax(-1), target.argmax(-1))

        def forward(sequences, cached, prefill):
            batch = prepare(draft, sequences, cached, prefill)
            before = batch.mtp_draft_input_hiddens.clone()
            output = draft._context_forward(batch) if prefill else draft._token_forward(batch)
            torch.testing.assert_close(batch.mtp_draft_input_hiddens, before, rtol=0, atol=0)
            output.logits = output.logits.clone()
            return output

        sequences = [[11, 12, 13], [21, 22]]
        check(forward(sequences, [0, 0], True), expected(sequences, [0, 0], True))
        sequences = [sequences[0] + [14], sequences[1] + [23]]
        check(forward(sequences, [3, 2], False), expected(sequences, [3, 2], False))
        saved = [(t, t.clone()) for t in fields]
        draft.graph = CudaGraph(1, 2, 1, max_batch_size=2, max_len_in_batch=64)
        batch = prepare(draft, sequences, [3, 2], False)
        before = batch.mtp_draft_input_hiddens.clone()
        batch.is_cuda_graph_capture = True
        captured = draft.graph.capture_decode(draft._token_forward, batch)
        torch.testing.assert_close(batch.mtp_draft_input_hiddens, before, rtol=0, atol=0)
        for tensor, value in saved:
            tensor.copy_(value)
        sequences = [sequences[0] + [15], sequences[1] + [24]]
        batch = prepare(draft, sequences, [4, 3], False)
        before = batch.mtp_draft_input_hiddens.clone()
        batch.is_cuda_graph_replay = True
        draft.graph.replay(batch)
        torch.testing.assert_close(batch.mtp_draft_input_hiddens, before, rtol=0, atol=0)
        check(captured, expected(sequences, [4, 3], False))
        sequences = [sequences[0] + [16, 17], sequences[1] + [25, 26]]
        check(forward(sequences, [5, 4], True), expected(sequences, [5, 4], True))
        draft.prefill_graph = PrefillCudaGraph(draft.graph, 1)
        sequences = [sequences[0] + [18, 19], sequences[1] + [27, 28]]
        check(forward(sequences, [7, 6], True), expected(sequences, [7, 6], True))
        sequences = [sequences[0] + [20, 21], sequences[1] + [29, 30]]
        check(forward(sequences, [9, 8], True), expected(sequences, [9, 8], True))
        for value in main_fields:
            assert torch.count_nonzero(value) == 0


def kda_contracts(dtype, ssm_dtype, slots=3):
    spec = KimiDeltaAttentionSpec(2, 128, 4, dtype)
    state = StateSpec(
        "main",
        "kda.0",
        0,
        (0,),
        ConvSSMStateSpec(key_heads=2, value_heads=2, key_dim=128, value_dim=128, conv_history_length=3),
        "recurrent",
    )
    fmt = FormatSpec(
        "conv_ssm",
        dtype,
        "request_conv_ssm",
        ("conv", "ssm", "accepted_index"),
        1,
        field_dtypes=(dtype, ssm_dtype, torch.int32),
        candidate_slots=slots,
    )
    return spec, state, fmt


def test_kda_candidate_layout_is_explicit():
    spec, state, fmt = kda_contracts(torch.bfloat16, torch.float32)
    limits = ExecutionLimits(0, 6, 64, max_verify_tokens=3)
    backend = KimiDeltaAttentionBackend("verify")
    assert backend.probe(spec, fmt, DeviceCaps("cuda"), limits).offers
    for bad in (
        replace(fmt, candidate_slots=2),
        replace(fmt, field_dtypes=(torch.bfloat16, torch.float32, torch.int64)),
    ):
        assert not backend.probe(spec, bad, DeviceCaps("cuda"), limits).offers
    assert not KimiDeltaAttentionBackend("decode").probe(spec, fmt, DeviceCaps("cuda"), limits).offers
    store = ConvSSMStateStore((state,), fmt, AttentionStateManager(4), 4, "cpu", kind="kda")
    conv, ssm, cursor = store.views()[state.key].fields
    assert conv.shape == (4, 768, 5) and ssm.shape == (12, 2, 128, 128)
    assert cursor.shape == (4,) and cursor.dtype == torch.int32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ssm_dtype", [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_kda_verify_accept_reject_suffix_and_graph(dtype, ssm_dtype, monkeypatch):
    torch.manual_seed(617)
    spec, state, fmt = kda_contracts(dtype, ssm_dtype)
    manager = AttentionStateManager(4)
    store = ConvSSMStateStore((state,), fmt, manager, 4, "cuda", kind="kda")
    view = store.views()[state.key]
    conv, ssm, cursor = view.fields
    conv.normal_(std=0.1)
    ssm.normal_(std=0.01)
    params = KDAStaticParams(
        spec,
        torch.randn(768, 4, device="cuda", dtype=dtype) * 0.2,
        torch.randn(2, device="cuda") * 0.1,
        torch.randn(256, device="cuda") * 0.1,
    )
    phases = ("prefill", "verify")
    planner = Planner(
        tuple(KimiDeltaAttentionBackend(p) for p in phases),
        DeviceCaps("cuda", 0, (9, 0)),
        ExecutionLimits(0, 6, 64, 2, max_verify_tokens=3),
    )
    plan = planner.bind([(0, p, spec, view, params, "auto") for p in phases], lambda _: pytest.fail("scratch"))
    prefill, verify = [binding.execution for binding in plan.bindings]
    monkeypatch.setattr(manager, "get_state", lambda *a, **kw: pytest.fail("runtime manager lookup"))
    untouched = [(tensor[1].clone() if tensor is conv else tensor[3:6].clone()) for tensor in (conv, ssm)]
    histories, memories = {}, {}

    def inputs(count):
        return KDAInputs(*(torch.randn(count, width, device="cuda", dtype=dtype) * 0.3 for width in (768, 256, 2)))

    def clone(data):
        return KDAInputs(*(t.clone() for t in (data.qkv, data.gate, data.beta)))

    def reference(data, rows, lengths, fresh=False, candidates=False):
        outputs, saved, offset = [], {}, 0
        for row, length in zip(rows, lengths):
            history = torch.zeros(768, 3, device="cuda") if fresh else histories[row].clone()
            memory = torch.zeros(2, 128, 128, device="cuda") if fresh else memories[row].clone()
            states = []
            for t in range(offset, offset + length):
                window = torch.cat((history, data.qkv[t].float()[:, None]), -1)
                mixed = F.silu((window * params.conv_weight.float()).sum(-1)).to(dtype).float()
                history = window[:, 1:]
                q, k, v = (part.reshape(2, 128) for part in mixed.split(256))
                q = q * (q.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
                k = k * (k.square().sum(-1, keepdim=True) + 1e-6).rsqrt()
                decay = (
                    spec.gate_lower_bound
                    * torch.sigmoid(
                        params.a_log.exp()[:, None]
                        * (data.gate[t].float().reshape(2, 128) + params.dt_bias.reshape(2, 128))
                    )
                ).exp()
                memory = memory * decay[:, :, None]
                delta = (v - torch.einsum("hkv,hk->hv", memory, k)) * data.beta[t].float().sigmoid()[:, None]
                memory = memory + k[:, :, None] * delta[:, None, :]
                outputs.append(torch.einsum("hkv,hk->hv", memory, q) * 128 ** -0.5)
                if candidates:
                    memory = memory.to(ssm_dtype).float()
                    states.append((history.clone(), memory.clone()))
            saved[row] = states
            if not candidates:
                histories[row], memories[row] = history, memory.to(ssm_dtype).float()
            offset += length
        return torch.stack(outputs), saved

    def check(output, target):
        actual = output.reshape(-1, 2, 128)[: len(target)].float()
        torch.testing.assert_close(actual, target, rtol=0.03, atol=0.003)

    data = inputs(12)
    target, _ = reference(data, [2, 0], [7, 5], fresh=True)
    meta = prefill.create_metadata(make_batch([7, 5], [2, 0], [0, 0]))
    check(prefill.run(clone(data), meta), target)
    for row in [2, 0]:
        torch.testing.assert_close(conv[row, :, :3].float(), histories[row], rtol=0, atol=0)
        torch.testing.assert_close(ssm[row * 3].float(), memories[row], rtol=0.03, atol=0.004)

    def batch(rows, widths, prefixes, lane=0):
        requests, offsets, ready = [], [], []
        for row, width, prefix in zip(rows, widths, prefixes):
            requests += [row] * width
            offsets += list(range(width))
            ready += list(range(prefix, prefix + width))
        pad = 6 - len(requests)
        requests += [3] * pad
        offsets += [0] * pad
        ready += [0] * pad
        return replace(
            make_batch([1] * 6, requests, ready, lane=lane),
            hold_request_id=3,
            mtp_indices=torch.tensor(offsets, dtype=torch.int32, device="cuda"),
        )

    def check_candidates(saved):
        for row, states in saved.items():
            for index, (history, memory) in enumerate(states):
                torch.testing.assert_close(conv[row, :, index : index + 3].float(), history, rtol=0, atol=0)
                torch.testing.assert_close(ssm[row * 3 + index].float(), memory, rtol=0.03, atol=0.004)

    data = inputs(6)
    target, candidates = reference(data, [2, 0], [3, 2], candidates=True)
    meta = verify.create_metadata(batch([2, 0], [3, 2], [7, 5]))
    check(verify.run(clone(data), meta), target)
    check_candidates(candidates)
    for row, index in [(2, 0), (0, 1)]:
        histories[row], memories[row] = candidates[row][index]
        cursor[row] = index

    data = inputs(6)
    original = clone(data)
    meta = verify.create_metadata(batch([0, 2], [1, 3], [7, 8], lane=1))
    saved = [(t, t.clone()) for t in view.fields]
    verify.run(data, meta)
    for tensor, value in saved:
        tensor.copy_(value)
    data.qkv.copy_(original.qkv)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = verify.run(data, meta)
    for tensor, value in saved:
        tensor.copy_(value)
    data.qkv.copy_(original.qkv)
    meta.copy_for_graph(verify.create_metadata(batch([2, 0], [2, 3], [8, 7], lane=1)))
    target, candidates = reference(original, [2, 0], [2, 3], candidates=True)
    graph.replay()
    check(output, target)
    check_candidates(candidates)
    for row, index in [(2, 1), (0, 0)]:
        histories[row], memories[row] = candidates[row][index]
        cursor[row] = index
    data = inputs(5)
    target, _ = reference(data, [0, 2], [2, 3])
    meta = prefill.create_metadata(make_batch([2, 3], [0, 2], [8, 10]))
    check(prefill.run(clone(data), meta), target)
    assert cursor.tolist() == [0, 0, 0, 0]
    for row in [0, 2]:
        torch.testing.assert_close(conv[row, :, :3].float(), histories[row], rtol=0, atol=0)
        torch.testing.assert_close(ssm[row * 3].float(), memories[row], rtol=0.03, atol=0.004)
    torch.testing.assert_close(conv[1], untouched[0], rtol=0, atol=0)
    torch.testing.assert_close(ssm[3:6], untouched[1], rtol=0, atol=0)
