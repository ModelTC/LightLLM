import torch
import pytest
import numpy as np
import torch.nn.functional as F
from lightllm.utils.log_utils import init_logger
from lightllm.models.deepseek2.triton_kernel.gqa_flash_decoding import gqa_token_decode_attention_flash_decoding
from lightllm.models.deepseek2.triton_kernel.gqa_flash_decoding_fp8 import gqa_token_decode_attention_flash_decoding_fp8
from lightllm.models.deepseek2.infer_struct import Deepseek2InferStateInfo
from lightllm.common.req_manager import ReqManager

logger = init_logger(__name__)

seed = 42
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.mark.parametrize(
    "batch, seqlen, heads, nope_head, rope_head",
    [(a, b, c, d, e) for a in [1, 16, 32, 128] for b in [16, 32, 512, 2048] for c in [16] for d in [512] for e in [64]],
)
def test_gqa_flash_decoding_fp8(batch, seqlen, heads, nope_head, rope_head):
    Z, N_CTX, H, D_HEAD, ROPE_HEAD = batch, seqlen, heads, nope_head, rope_head
    dtype = torch.bfloat16
    sm_scale = 1.0 / ((D_HEAD + ROPE_HEAD) ** 0.5)
    q = torch.randn((Z, H, D_HEAD), dtype=dtype, device="cuda")
    q_rope = torch.randn((Z, H, ROPE_HEAD), dtype=dtype, device="cuda")

    kv = torch.randn((Z * N_CTX, 1, D_HEAD + ROPE_HEAD), dtype=dtype, device="cuda")
    kv_scale = torch.randn((Z * N_CTX, 1, 1), dtype=dtype, device="cuda")
    kv_fp8 = kv.to(torch.float8_e4m3fn)

    req_to_token_indexs = torch.zeros((10, Z * N_CTX), dtype=torch.int32, device="cuda")
    b_seq_len = torch.ones((Z,), dtype=torch.int32, device="cuda")
    b_req_idx = torch.ones((Z,), dtype=torch.int32, device="cuda")

    b_seq_len[0] = N_CTX
    b_req_idx[0] = 0
    req_to_token_indexs[0][:N_CTX] = torch.tensor(np.arange(N_CTX), dtype=torch.int32).cuda()

    o = torch.empty((Z, H, D_HEAD), dtype=dtype, device="cuda")
    o1 = torch.empty((Z, H, D_HEAD), dtype=dtype, device="cuda")

    infer_state = Deepseek2InferStateInfo()
    infer_state.batch_size = Z
    infer_state.max_len_in_batch = N_CTX
    infer_state.total_token_num = Z * N_CTX
    infer_state.req_manager = ReqManager(Z, N_CTX, None)
    infer_state.req_manager.req_to_token_indexs = req_to_token_indexs
    infer_state.b_req_idx = b_req_idx
    infer_state.b_seq_len = b_seq_len

    kv_nope = kv_fp8[:, :, :D_HEAD].to(dtype) * kv_scale
    kv_rope = kv_fp8[:, :, D_HEAD:].to(dtype) * kv_scale
    gqa_token_decode_attention_flash_decoding(
        q,
        q_rope,
        kv_nope,
        kv_rope,
        infer_state,
        H,
        D_HEAD,
        ROPE_HEAD,
        D_HEAD,
        sm_scale,
        o,
    )

    kv_nope_fp8 = kv_fp8[:, :, :D_HEAD]
    kv_rope_fp8 = kv_fp8[:, :, D_HEAD:]
    gqa_token_decode_attention_flash_decoding_fp8(
        q, q_rope, kv_nope_fp8, kv_rope_fp8, kv_scale, infer_state, H, D_HEAD, ROPE_HEAD, D_HEAD, sm_scale, o1
    )

    cos_sim = F.cosine_similarity(o, o1).mean()
    assert cos_sim > 0.99
