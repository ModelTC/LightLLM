from types import SimpleNamespace

import pytest
import torch

from lightllm.models.deepseek_v4.model import DeepseekV4TpPartModel
from lightllm.models.deepseek_v4.workspace import DeepseekV4Workspace
from lightllm.models.deepseek_v4_dspark.model import DeepseekV4DSparkModel
from lightllm.models.deepseek_v4_mtp.model import DeepseekV4MTPModel


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("draft_class", [DeepseekV4MTPModel, DeepseekV4DSparkModel])
def test_draft_initialization_preserves_captured_prefill_buffer(monkeypatch, draft_class):
    args = SimpleNamespace(llm_kv_type="fp8kv_dsa")
    monkeypatch.setattr("lightllm.models.deepseek_v4.model.get_env_start_args", lambda: args)

    def backend_class(**kwargs):
        return lambda model: SimpleNamespace(tp_q_head_num=2, padded_q_head_num=4)

    monkeypatch.setattr("lightllm.models.deepseek_v4.model.get_nsa_prefill_att_backend_class", backend_class)
    monkeypatch.setattr("lightllm.models.deepseek_v4.model.get_nsa_decode_att_backend_class", backend_class)
    main = DeepseekV4TpPartModel.__new__(DeepseekV4TpPartModel)
    main.batch_max_tokens, main.tp_world_size_, main.data_type = 8, 1, torch.bfloat16
    main.config = {"head_dim": 512, "block_size": 3, "dspark_noise_token_id": 1}
    main.layers_infer = [SimpleNamespace(compress_ratio=0)]
    main.dsv4_workspace = DeepseekV4Workspace(main)
    for name in (
        "_freqs_cis_sliding",
        "_freqs_cis_compress",
        "_cos_cached_sliding",
        "_sin_cached_sliding",
        "_cos_cached_compress",
        "_sin_cached_compress",
    ):
        setattr(main, name, torch.ones((1, 32), device="cuda"))
    main._init_att_backend()
    buffer = main.dsv4_workspace.flashmla_prefill_q
    buffer[:, :2].fill_(2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = main.dsv4_workspace.flashmla_prefill_q + 1

    draft = draft_class.__new__(draft_class)
    draft.main_model, draft.config, draft.data_type = main, main.config, main.data_type
    draft.layers_infer = [SimpleNamespace(compress_ratio=0)]
    draft.context_wkv_weight = object()
    draft._init_custom()
    draft._init_att_backend()

    assert draft.dsv4_workspace is main.dsv4_workspace
    assert main.dsv4_workspace.flashmla_prefill_q is buffer
    assert draft.layers_infer[0].flashmla_q_head_num_ == 4
    main.dsv4_workspace.flashmla_prefill_q[:, :2].fill_(4)
    graph.replay()
    torch.testing.assert_close(result[:, :2], torch.full_like(result[:, :2], 5), rtol=0, atol=0)
    torch.testing.assert_close(result[:, 2:], torch.ones_like(result[:, 2:]), rtol=0, atol=0)


@pytest.mark.parametrize("batch_max_tokens,tp,capacity", [(4097, 8, 4104), (4096, 8, 4096), (1, 8, 8), (3, 1, 3)])
def test_workspace_covers_tp_padding_and_checks_q_capacity(batch_max_tokens, tp, capacity):
    from lightllm.models.deepseek_v4.triton_kernel.norm_rope import fused_q_norm_rope

    model = SimpleNamespace(
        batch_max_tokens=batch_max_tokens, tp_world_size_=tp, config={"index_n_heads": 2, "index_head_dim": 128}
    )
    workspace = DeepseekV4Workspace(model)
    assert workspace.token_capacity == capacity
    workspace.init_flashmla_prefill_q(2, 4, 512, torch.bfloat16)
    workspace.init_c4_prefill_aux(model)
    q = workspace.get_flashmla_prefill_q(capacity)
    q_input = torch.ones((capacity, 2, 512), dtype=torch.bfloat16, device="cuda")
    freqs = torch.ones((1, 32), dtype=torch.complex64, device="cuda")
    positions = torch.zeros(capacity, dtype=torch.int32, device="cuda")
    fused_q_norm_rope(q_input, q[:, :2], 1e-6, freqs, positions)
    torch.testing.assert_close(q[:, :2], q_input, rtol=0, atol=0)
    assert q[:, 2:].count_nonzero() == 0
    assert workspace.c4_prefill_aux.indexer_q_outputs(capacity)[0].shape[0] == capacity
    with pytest.raises(ValueError, match="capacity"):
        workspace.get_flashmla_prefill_q(capacity + 1)
