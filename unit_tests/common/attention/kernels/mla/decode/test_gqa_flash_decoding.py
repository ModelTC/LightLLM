import torch
import time
import pytest
import numpy as np
import torch.nn.functional as F
import flashinfer
from lightllm.utils.log_utils import init_logger
from unit_tests.common.attention.backends.step_fixture import make_infer_state
from lightllm.common.attention.backends.mla import MLADecodeLayerInputs, MLADecodeLayerConfig
from lightllm.common.attention.backends.mla.triton import TritonMLADecodeBackend

logger = init_logger(__name__)

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.mark.parametrize(
    "batch, seqlen, heads, nope_head, rope_head",
    [
        (a, b, c, d, e)
        for a in [1, 16, 32, 128, 512]
        for b in [16, 32, 512, 2048]
        for c in [16]
        for d in [512]
        for e in [64]
    ],
)
def test_gqa_flash_decoding(batch, seqlen, heads, nope_head, rope_head):
    Z, N_CTX, H, D_HEAD, ROPE_HEAD = batch, seqlen, heads, nope_head, rope_head
    dtype = torch.bfloat16
    sm_scale = 1.0 / ((D_HEAD + ROPE_HEAD) ** 0.5)
    q_nope = torch.randn((Z, H, D_HEAD), dtype=dtype, device="cuda")
    q_rope = torch.randn((Z, H, ROPE_HEAD), dtype=dtype, device="cuda")

    kv = torch.randn((Z * N_CTX, 1, D_HEAD + ROPE_HEAD), dtype=dtype, device="cuda")

    max_input_len = Z * N_CTX
    req_to_token_indexs = torch.randperm(max_input_len, dtype=torch.int32).cuda().view(Z, N_CTX)
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda") * N_CTX
    b_start_loc = torch.arange(Z).cuda().int() * N_CTX
    b_req_idx = torch.randperm(Z, dtype=torch.int32).cuda()
    kv_starts = torch.cat([b_start_loc, b_start_loc[-1:] + b_seq_len[-1:]], dim=0)

    o = torch.zeros((Z, H, D_HEAD), dtype=dtype, device="cuda")
    o1 = torch.zeros((Z, H, D_HEAD), dtype=dtype, device="cuda")

    layer_config = MLADecodeLayerConfig(
        num_heads=H, kv_lora_rank=D_HEAD, qk_rope_head_dim=ROPE_HEAD, softmax_scale=sm_scale, dtype=dtype
    )
    backend = TritonMLADecodeBackend(layer_config)
    # Every request has N_CTX tokens, so the eager tuning length (the average) is N_CTX.
    att_state = backend.build_att_metadata(
        make_infer_state(
            req_to_token_indexs=req_to_token_indexs,
            b_req_idx=b_req_idx,
            b_seq_len=b_seq_len,
            batch_size=Z,
            max_kv_seq_len=N_CTX,
            total_token_num=Z * N_CTX,
        )
    )
    assert backend.forward(att_state, MLADecodeLayerInputs(q_nope=q_nope, q_rope=q_rope, kv_cache=kv), out=o) is o

    kv_nope = kv[:, :, :D_HEAD]
    kv_rope = kv[:, :, D_HEAD:]
    batch_size = Z
    head_dim_ckv = D_HEAD
    head_dim_kpe = ROPE_HEAD
    num_heads = H
    page_size = 1
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    q_indptr = torch.arange(batch_size + 1).to(0).int()
    kv_indptr = kv_starts
    kv_indices = torch.arange(Z * N_CTX).cuda().int()
    for b, sl, start in zip(b_req_idx, b_seq_len, b_start_loc):
        kv_indices[start : start + sl] = req_to_token_indexs[b][:sl]
    kv_lens = b_seq_len
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer,
        use_cuda_graph=True,
        qo_indptr=q_indptr,
        kv_indices=kv_indices,
        kv_indptr=kv_indptr,
        kv_len_arr=kv_lens,
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        False,  # causal
        sm_scale,
        q_nope.dtype,
        kv.dtype,
    )
    wrapper.run(q_nope, q_rope, kv_nope, kv_rope, out=o1, return_lse=False)

    cos_sim1 = F.cosine_similarity(o, o1).mean()
    assert cos_sim1 == 1.0
