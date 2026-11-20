from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.nsa.fp8_flashmla_sparse import NsaFlashMlaFp8SparseAttBackend
from lightllm.common.basemodel.infer_struct import InferStateInfo


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class Model:
    graph_max_batch_size = 2
    graph_max_len_in_batch = 12
    max_seq_length = 12


def packed_cache():
    # Keep padding after each physical row, as in a per-layer State view.
    packed = torch.zeros((12, 1, 672), dtype=torch.uint8, device="cuda")[:, :, :656]
    latent = torch.randn((12, 1, 512), device="cuda").to(torch.float8_e4m3fn)
    scales = torch.tensor([0.5, 1, 0.25, 2], device="cuda")
    rope = torch.randn((12, 1, 64), dtype=torch.bfloat16, device="cuda")
    packed[..., :512].copy_(latent.view(torch.uint8))
    packed[..., 512:528].view(torch.float32).copy_(scales)
    packed[..., 528:].view(torch.bfloat16).copy_(rope)
    rows = torch.cat((latent.float() * scales.repeat_interleave(128), rope.float()), dim=-1)
    return packed, rows[:, 0]


def reference(q, rows, indices):
    result = []
    for i, ix in enumerate(indices):
        selected = rows[ix[ix >= 0].long()].float()
        weights = ((q[i].float() @ selected.T) * 0.04).softmax(-1)
        result.append(weights @ selected[:, :512])
    return torch.stack(result).to(q.dtype)


@pytest.mark.parametrize("cached", [False, True])
def test_prefill_indices(cached):
    backend = NsaFlashMlaFp8SparseAttBackend(model=Model())
    state = InferStateInfo()
    state.max_cache_len = int(cached)
    state.mem_index = torch.tensor([9, 2], dtype=torch.int32, device="cuda")
    # One request whose two new tokens follow an optional cached token.
    state.total_token_num = 2 + int(cached)
    state.b_seq_len = torch.tensor([state.total_token_num], dtype=torch.int32, device="cuda")
    state.b_q_seq_len = torch.tensor([2], dtype=torch.int32, device="cuda")
    state.b_req_idx = torch.tensor([0], dtype=torch.int32, device="cuda")
    state.input_ids = torch.zeros(2, dtype=torch.int64, device="cuda")
    state.req_manager = SimpleNamespace(
        req_to_token_indexs=torch.arange(12, dtype=torch.int32, device="cuda").view(1, 12), HOLD_REQUEST_ID=1
    )
    att = backend.create_att_prefill_state(infer_state=state)
    att.init_state()
    q = torch.randn((2, 64, 576), dtype=torch.bfloat16, device="cuda")
    packed, rows = packed_cache()
    original = packed.clone()
    new_kv = torch.randn((2, 1, 576), device=q.device, dtype=q.dtype)
    physical = torch.full((2, 128), -1, device=q.device, dtype=torch.int32)
    physical[0, :2] = torch.tensor([7, 9] if cached else [-1, 9], device=q.device)
    physical[1, :3] = torch.tensor([4, 9, 2] if cached else [-1, 9, 2], device=q.device)
    local = torch.full_like(physical, -1)
    local[physical == 9], local[physical == 2] = 0, 1
    control = AttControl(
        nsa_prefill=True,
        nsa_prefill_dict=dict(
            topk_mem_indices=physical, topk_indices=local, softmax_scale=0.04, kv_lora_rank=512, prefill_cache_kv=new_kv
        ),
    )
    out = torch.empty((2, 64, 512), dtype=q.dtype, device=q.device)
    actual = att.prefill_att(q=q, k=packed, v=None, att_control=control, alloc_func=lambda *a, **kw: out)
    assert actual is out
    rows[state.mem_index.long()] = new_kv[:, 0].float()
    torch.testing.assert_close(out, reference(q, rows, physical), atol=0.02, rtol=0.02)
    assert torch.equal(original, packed)


def test_decode_replay_reads_physical_indices():
    backend = NsaFlashMlaFp8SparseAttBackend(model=Model())
    state = InferStateInfo()
    state.batch_size, state.total_token_num, state.max_kv_seq_len, state.microbatch_index = 2, 12, 6, 0
    state.b_seq_len = torch.tensor([6, 6], dtype=torch.int32, device="cuda")
    state.b_q_seq_len = torch.ones_like(state.b_seq_len)
    state.b_req_idx = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    state.req_manager = SimpleNamespace(
        req_to_token_indexs=torch.arange(12, dtype=torch.int32, device="cuda").view(2, 6), HOLD_REQUEST_ID=2
    )
    att = backend.create_att_decode_state(infer_state=state)
    att.init_state()
    packed, rows = packed_cache()
    original = packed.clone()
    q = torch.randn((2, 64, 576), dtype=torch.bfloat16, device="cuda")
    indices = torch.full((2, 128), -1, dtype=torch.int32, device="cuda")
    indices[:, :2] = torch.tensor([[1, 8], [11, 2]], device="cuda")
    control = AttControl(
        nsa_decode=True, nsa_decode_dict=dict(topk_mem_indices=indices, softmax_scale=0.04, kv_lora_rank=512)
    )
    out = torch.empty((2, 64, 512), dtype=q.dtype, device=q.device)

    def allocate(shape, *, dtype, device):
        return out if tuple(shape) == tuple(out.shape) else torch.empty(shape, dtype=dtype, device=device)

    def decode():
        return att.decode_att(
            q=(q[..., :512], q[..., 512:]), k=packed, v=None, att_control=control, alloc_func=allocate
        )

    assert decode() is out
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert decode() is out
    for slots in ([[1, 8, -1], [11, 2, -1]], [[6, 3, 0], [10, 4, 7]]):
        indices[:, :3] = torch.tensor(slots, device=q.device)
        graph.replay()
        torch.testing.assert_close(out, reference(q, rows, indices), atol=0.02, rtol=0.02)
    assert torch.equal(original, packed)
