from types import SimpleNamespace

import pytest
import torch

from lightllm.server.router.model_infer.mtp_speculative.proposers.dflash import DFlashProposer
from lightllm.server.router.model_infer.mtp_speculative.proposers.proposal_type import DFlashSpecProposal
from lightllm.server.router.model_infer.pin_mem_manager import g_pin_mem_manager


def test_dflash_prefill_uses_a_shallow_copy_for_target_hidden():
    forwarded_inputs = []
    draft_model = SimpleNamespace(forward=forwarded_inputs.append)
    proposer = DFlashProposer(
        backend=SimpleNamespace(draft_models=[draft_model]),
        enable_dynmaic_mtp=False,
    )
    model_input = SimpleNamespace(
        is_prefill=True,
        b_position_delta=None,
        b_req_idx=torch.tensor([3, 5], dtype=torch.int32),
        input_ids=torch.arange(5, dtype=torch.int64),
        mtp_draft_input_hiddens=None,
    )
    target_hidden = torch.empty((5, 8))

    proposer.fill_draft_model_kv_state(
        target_model_input=model_input,
        target_model_output=SimpleNamespace(
            mtp_collector=SimpleNamespace(spec_hidden=target_hidden),
        ),
        target_next_token_ids=torch.tensor([11, 13], dtype=torch.int64),
    )

    assert len(forwarded_inputs) == 1
    assert forwarded_inputs[0] is not model_input
    assert forwarded_inputs[0].mtp_draft_input_hiddens is target_hidden
    assert model_input.mtp_draft_input_hiddens is None


def test_dflash_commits_verify_kv_and_builds_parallel_block(monkeypatch):
    block_size = 3
    flat_draft_token_ids = torch.tensor([30, 31, 32, 40, 41, 42], dtype=torch.int64)
    flat_draft_token_probs = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4], dtype=torch.float16)
    forwarded_inputs = []

    def forward(model_input):
        forwarded_inputs.append(model_input)
        return SimpleNamespace()

    draft_model = SimpleNamespace(
        block_size=block_size,
        mask_token_id=99,
        forward=forward,
    )
    proposer = DFlashProposer(
        backend=SimpleNamespace(
            draft_models=[draft_model],
            _gen_argmax_token_ids_and_prob=lambda _: (flat_draft_token_ids, flat_draft_token_probs),
        ),
        enable_dynmaic_mtp=True,
    )
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    monkeypatch.setattr(
        g_pin_mem_manager,
        "get_const_gpu_tensor",
        lambda *, shape, fill_value, dtype, **_: torch.full(shape, fill_value, dtype=dtype),
    )

    target_hidden = torch.empty((5, 8))
    model_input = SimpleNamespace(
        is_prefill=False,
        batch_size=5,
        total_token_num=40,
        max_q_seq_len=1,
        max_kv_seq_len=9,
        input_ids=torch.tensor([10, 11, 12, 20, 21], dtype=torch.int64),
        b_req_idx=torch.tensor([7, 7, 7, 9, 9], dtype=torch.int32),
        b_mtp_index=torch.arange(5, dtype=torch.int32),
        b_seq_len=torch.tensor([4, 5, 6, 8, 9], dtype=torch.int32),
        b_position_delta=torch.tensor([1, 1, 1, 2, 2], dtype=torch.int32),
        b_shared_seq_len=torch.tensor([3, 3, 3, 6, 6], dtype=torch.int32),
        b_shared_radix_node_id=torch.tensor([17, 17, 17, 19, 19], dtype=torch.int64),
        multimodal_params=[{"images": [], "audios": []} for _ in range(5)],
        mtp_draft_input_hiddens=None,
    )

    proposal = proposer.propose_next(
        target_model_input=model_input,
        target_model_output=SimpleNamespace(
            mtp_collector=SimpleNamespace(spec_hidden=target_hidden),
        ),
        target_next_token_ids=model_input.input_ids,
        b_req_mtp_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        draft_step=2,
        accept_len=torch.tensor([2, 2], dtype=torch.int32),
    )

    assert isinstance(proposal, DFlashSpecProposal)
    assert len(forwarded_inputs) == 2
    verify_draft_input, block_draft_input = forwarded_inputs
    assert verify_draft_input is not model_input
    assert verify_draft_input.mtp_draft_input_hiddens is target_hidden
    assert model_input.mtp_draft_input_hiddens is None

    assert torch.equal(block_draft_input.input_ids, torch.tensor([11, 99, 99, 21, 99, 99]))
    assert torch.equal(block_draft_input.b_req_idx, torch.tensor([7, 7, 7, 9, 9, 9], dtype=torch.int32))
    assert torch.equal(block_draft_input.b_mtp_index, torch.zeros(6, dtype=torch.int32))
    assert torch.equal(block_draft_input.b_seq_len, torch.tensor([6, 7, 8, 10, 11, 12], dtype=torch.int32))
    assert torch.equal(block_draft_input.b_position_delta, torch.tensor([1, 1, 1, 2, 2, 2], dtype=torch.int32))
    assert block_draft_input.mtp_draft_input_hiddens is None

    assert torch.equal(proposal.token_ids, torch.tensor([[30, 31], [40, 41]], dtype=torch.int64))
    torch.testing.assert_close(
        proposal.schedule_scores,
        flat_draft_token_probs.reshape(2, block_size)[:, :2].float(),
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dflash_hidden_commit_uses_draft_state(device, monkeypatch):
    from lightllm.common.attention_state import AttentionStateManager
    from lightllm.common.attention_state.full_att import GQAFullAttentionState, GQAAttentionConfig
    from lightllm.common.basemodel.infer_struct import InferStateInfo
    from lightllm.common.basemodel.layer_infer.template.transformer_layer_infer_template import TransformerLayerInferTpl
    from lightllm.models.qwen3_dflash.model import Qwen3DFlashModel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    manager = AttentionStateManager(request_capacity=1)
    states = []
    for role, layers in (("main", ((0,), (1,), (2,), (3,))), ("draft0", ((4,),))):
        state = GQAFullAttentionState(
            GQAAttentionConfig(logical_layer_groups=layers, global_head_num=1, head_dim=16, dtype=torch.float32),
            device=device,
        )
        state.allocate_storage(8)
        state.k_storage.zero_()
        state.v_storage.zero_()
        manager.register_state(role, state)
        states.append(state)
    manager.finish_state_registration()
    main, draft = states
    ids = torch.tensor([1, 3], dtype=torch.int32, device=device)
    if device == "cpu":

        def write(logical_layer_id, token_indices, kv):
            assert logical_layer_id == 4
            draft.k_storage[0, token_indices.long()] = kv[:, :1]
            draft.v_storage[0, token_indices.long()] = kv[:, 1:]

        monkeypatch.setattr(draft, "write_full_att_kv", write)
    model = Qwen3DFlashModel.__new__(Qwen3DFlashModel)
    model.infer_state_class = InferStateInfo
    model.att_state_manager = manager
    model.is_mtp_draft_model = True
    model.mtp_previous_draft_models = []
    model._cos_cached = model._sin_cached = torch.zeros(8, 16, device=device)
    model.req_manager = SimpleNamespace(req_to_token_indexs=object())
    monkeypatch.setattr(
        "lightllm.models.qwen3_dflash.model.build_kv_indexes_and_input_ids", lambda inputs, table: (ids, None)
    )
    model.pre_post_weight = None
    model.pre_infer = SimpleNamespace(context_forward=lambda _, state, weight: state.mtp_draft_input_hiddens)

    def commit(hidden, state, weight):
        TransformerLayerInferTpl._post_cache_kv(SimpleNamespace(layer_num_=4), hidden, state, weight)
        return hidden

    model.layers_infer = [SimpleNamespace(context_forward=commit)]
    model.trans_layers_weight = [None]
    hidden = torch.arange(64, dtype=torch.float32, device=device).view(2, 2, 16)
    output = model._decode(
        SimpleNamespace(
            mtp_draft_input_hiddens=hidden,
            batch_size=2,
            b_seq_len=torch.tensor([2, 4], device=device),
        )
    )
    assert output.logits.shape == (2, 1)
    torch.testing.assert_close(draft.k_storage[0, ids.long()], hidden[:, :1])
    torch.testing.assert_close(draft.v_storage[0, ids.long()], hidden[:, 1:])
    assert not main.k_storage.any() and not main.v_storage.any()
