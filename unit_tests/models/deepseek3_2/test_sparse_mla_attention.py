"""DeepSeek V3.2 sparse MLA through the model-facing adapters, against a torch reference.

The bf16 adapter runs FA3 decode with rotary channels; the FP8 adapter runs FlashMLA over DeepSeek's packed
"fp8_ds_mla" rows (512 FP8 latent channels, four fp32 scales per 128 channels and 64 bf16 rotary channels).
CUDA graph steps rebuild their step args on replay, so a replay with new lengths and indices must match eager.
"""

from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.infer_struct import InferStateInfo
from lightllm.common.basemodel.triton_kernel.gen_decode_params import gen_decode_params
from lightllm.common.basemodel.triton_kernel.gen_prefill_params import gen_prefill_params

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] != 9,
        reason="FA3 and FlashMLA sparse kernels target Hopper",
    ),
]

# softmax_scale = (kv_lora_rank + qk_rope_head_dim) ** -0.5
RANK, ROPE, TOPK, SCALE = 512, 64, 2048, 1 / 24
SLOTS, MAX_LEN = 16384, 4096


class _Model(SimpleNamespace):
    # Attention backends are cached per model object.
    __hash__ = object.__hash__
    __eq__ = object.__eq__


def make_model():
    return _Model(
        config={"index_topk": TOPK},
        graph_max_batch_size=8,
        graph_max_len_in_batch=MAX_LEN,
        max_seq_length=MAX_LEN,
        mtp_manager=None,
    )


def gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


def token_map(requests, seed=0):
    slots = torch.randperm(SLOTS, device="cuda", dtype=torch.int32, generator=gen(seed))
    return slots[: requests * MAX_LEN].view(requests, MAX_LEN)


def make_infer(req_to_token, lengths, *, capture=False, replay=False, microbatch=0):
    infer = InferStateInfo()
    infer.is_prefill = False
    infer.batch_size = len(lengths)
    infer.b_req_idx = torch.arange(len(lengths), device="cuda", dtype=torch.int32)
    infer.b_seq_len = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    infer.max_kv_seq_len, infer.total_token_num = max(lengths), sum(lengths)
    infer.max_q_seq_len = 1
    infer.req_manager = SimpleNamespace(req_to_token_indexs=req_to_token, HOLD_REQUEST_ID=len(lengths) + 1)
    infer.is_cuda_graph_capture, infer.is_cuda_graph_replay = capture, replay
    infer.microbatch_index = microbatch
    (
        infer.b_q_seq_len,
        infer.b1_cu_q_seq_len,
        infer.b_kv_seq_len,
        infer.b1_cu_kv_seq_len,
        infer.position_ids,
    ) = gen_decode_params(infer.b_seq_len)
    return infer


def decode_state(backend, infer):
    infer.decode_att_state = backend.create_att_decode_state(infer)
    infer.decode_att_state.init_state()
    return infer.decode_att_state


def select(req_to_token, lengths, out, seed):
    """Each row selects min(length, TOPK) distinct visible cache slots, then -1 padding."""
    out.fill_(-1)
    for row, length in enumerate(lengths):
        count = min(length, TOPK)
        positions = torch.randperm(length, device="cuda", generator=gen(seed + row))[:count]
        out[row, :count] = req_to_token[row, positions]
    return out


def attend(q_row, keys, scale=SCALE):
    """q_row [heads, 576] over keys [selected, 576]; values are the latent part of each key."""
    keys = keys.float()
    return (q_row.float() @ keys.T * scale).softmax(-1) @ keys[:, :RANK]


def write_fp8_ds_mla(kv):
    from lightllm.common.basemodel.triton_kernel.kv_copy.destindex_copy_kv_flashmla_fp8 import (
        destindex_copy_kv_flashmla_fp8,
    )

    packed = torch.zeros(kv.shape[0], 1, 656, device="cuda", dtype=torch.uint8)
    destindex_copy_kv_flashmla_fp8(
        KV_nope=kv[..., :RANK],
        KV_rope=kv[..., RANK:],
        DestLoc=torch.arange(kv.shape[0], device="cuda"),
        O_nope=packed[..., :RANK].view(torch.float8_e4m3fn),
        O_scale=packed[..., RANK : RANK + 16].view(torch.float32),
        O_rope=packed[..., RANK + 16 :].view(torch.bfloat16),
    )
    return packed


def dequantize_fp8_ds_mla(packed):
    latent = packed[:, 0, :RANK].view(torch.float8_e4m3fn).float().view(-1, 4, 128)
    scales = packed[:, 0, RANK : RANK + 16].view(torch.float32).view(-1, 4, 1)
    rope = packed[:, 0, RANK + 16 :].view(torch.bfloat16).float()
    return torch.cat(((latent * scales).view(-1, RANK), rope), dim=-1)


def capture(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    return graph, output


def check_decode_and_graph(backend, cache, reference_cache, heads):
    """Eager decode matches the reference; a graph captured for one step replays another step correctly."""
    req_to_token = token_map(4)
    # Every row selects at least as many keys on replay, so stale captured lengths would read valid but too few
    # entries and fail the comparison instead of faulting on -1 padding.
    first, second = [100, 3000, 2048, 7], [4096, 2100, 2049, 2000]
    indices = select(req_to_token, first, torch.empty((4, TOPK), device="cuda", dtype=torch.int32), seed=10)
    q_nope = torch.randn(4, heads, RANK, device="cuda", dtype=torch.bfloat16, generator=gen(1)) * 0.5
    q_rope = torch.randn(4, heads, ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(2)) * 0.5
    control = AttControl(
        nsa_decode=True,
        nsa_decode_dict={
            "topk_mem_indices": indices,
            "softmax_scale": SCALE,
            "kv_lora_rank": RANK,
            "qk_rope_head_dim": ROPE,
        },
    )

    def run(state):
        return state.decode_att((q_nope, q_rope), cache, None, control)

    def reference():
        q = torch.cat((q_nope, q_rope), dim=-1)
        return torch.stack([attend(q[row], reference_cache[ids[ids >= 0].long()]) for row, ids in enumerate(indices)])

    def eager(lengths):
        return run(decode_state(backend, make_infer(req_to_token, lengths))).clone()

    first_eager = eager(first)
    torch.testing.assert_close(first_eager.float(), reference(), atol=0.02, rtol=0.02)

    graph_infer = make_infer(req_to_token, first, capture=True)
    graph_state = decode_state(backend, graph_infer)
    graph, graph_out = capture(lambda: run(graph_state))
    graph.replay()
    assert torch.equal(graph_out, first_eager)

    # Serving replay: the replay step builds its own state, then its request tensors are copied into the graph's.
    select(req_to_token, second, indices, seed=20)
    q_nope.normal_(generator=gen(3)).mul_(0.5)
    replay_infer = make_infer(req_to_token, second, replay=True)
    decode_state(backend, replay_infer)
    graph_infer.copy_for_cuda_graph(replay_infer)
    graph.replay()
    replayed = graph_out.clone()
    assert torch.equal(replayed, eager(second))
    torch.testing.assert_close(replayed.float(), reference(), atol=0.02, rtol=0.02)


def test_bf16_fa3_decode_with_rope_matches_reference_and_cuda_graph_replay():
    pytest.importorskip("sgl_kernel.flash_attn")
    from lightllm.common.basemodel.attention.nsa.flashmla_sparse import NsaFlashMlaSparseAttBackend

    kv = torch.randn(SLOTS, 1, RANK + ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(0)) * 0.5
    check_decode_and_graph(NsaFlashMlaSparseAttBackend(model=make_model()), kv, kv[:, 0].float(), heads=16)


def test_fp8_ds_mla_flashmla_decode_matches_reference_and_cuda_graph_replay():
    pytest.importorskip("flash_mla")
    from lightllm.common.basemodel.attention.nsa.fp8_flashmla_sparse import NsaFlashMlaFp8SparseAttBackend

    kv = torch.randn(SLOTS, 1, RANK + ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(0)) * 0.5
    packed = write_fp8_ds_mla(kv)
    # FlashMLA's sparse FP8 decode runs 64-head tiles.
    check_decode_and_graph(
        NsaFlashMlaFp8SparseAttBackend(model=make_model()), packed, dequantize_fp8_ds_mla(packed), heads=64
    )


@pytest.mark.parametrize(
    "ready, ends",
    [([0, 0], [40, 150]), ([100, 2100], [164, 2200])],
    ids=["no_cached_prefix", "cached_prefix"],
)
def test_fp8_ds_mla_flashmla_prefill_matches_reference(ready, ends):
    """Cached history is read from the packed FP8 cache; the step's own keys come from current_kv in bf16."""
    pytest.importorskip("flash_mla")
    from lightllm.common.basemodel.attention.nsa.fp8_flashmla_sparse import NsaFlashMlaFp8SparseAttBackend

    heads = 64
    req_to_token = token_map(len(ends), seed=5)
    kv = torch.randn(SLOTS, 1, RANK + ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(6)) * 0.5
    packed = write_fp8_ds_mla(kv)
    history = dequantize_fp8_ds_mla(packed)
    q_lens = [end - start for start, end in zip(ready, ends)]
    rows = sum(q_lens)

    infer = InferStateInfo()
    infer.is_prefill = True
    infer.batch_size = len(ends)
    infer.b_req_idx = torch.arange(len(ends), device="cuda", dtype=torch.int32)
    infer.b_seq_len = torch.tensor(ends, device="cuda", dtype=torch.int32)
    infer.b_ready_cache_len = torch.tensor(ready, device="cuda", dtype=torch.int32)
    infer.max_kv_seq_len, infer.total_token_num, infer.max_cache_len = max(ends), sum(ends), max(ready)
    infer.max_q_seq_len = max(q_lens)
    infer.req_manager = SimpleNamespace(req_to_token_indexs=req_to_token, HOLD_REQUEST_ID=len(ends) + 1)
    infer.input_ids = torch.zeros(rows, device="cuda", dtype=torch.int64)
    (
        infer.b_q_seq_len,
        infer.b1_cu_q_seq_len,
        infer.b_kv_seq_len,
        infer.b1_cu_kv_seq_len,
        infer.position_ids,
    ) = gen_prefill_params(input_token_num=rows, b_ready_cache_len=infer.b_ready_cache_len, b_seq_len=infer.b_seq_len)
    infer.b_q_start_loc = infer.b1_cu_q_seq_len[:-1]
    infer.mem_index = torch.cat([req_to_token[req, start:end] for req, (start, end) in enumerate(zip(ready, ends))])
    # The step's keys in full precision, deliberately different from what the FP8 cache holds for those slots.
    current_kv = torch.randn(rows, 1, RANK + ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(7)) * 0.5

    topk_mem_indices = torch.full((rows, TOPK), -1, device="cuda", dtype=torch.int32)
    topk_indices = torch.full((rows, TOPK), -1, device="cuda", dtype=torch.int32)
    q = torch.randn(rows, heads, RANK + ROPE, device="cuda", dtype=torch.bfloat16, generator=gen(8)) * 0.5
    expected, row, step_offset = [], 0, 0
    for req, (start, end) in enumerate(zip(ready, ends)):
        for position in range(start, end):
            count = min(position + 1, TOPK)
            selected = torch.randperm(position + 1, device="cuda", generator=gen(100 + row))[:count]
            topk_mem_indices[row, :count] = req_to_token[req, selected]
            # Without cached history every selected key is one of the step's own keys, in ragged step order.
            topk_indices[row, :count] = (step_offset + selected - start).int()
            from_step = selected >= start
            keys = history[req_to_token[req, selected].long()]
            keys[from_step] = current_kv[step_offset + selected[from_step] - start, 0].float()
            expected.append(attend(q[row], keys))
            row += 1
        step_offset += end - start

    backend = NsaFlashMlaFp8SparseAttBackend(model=make_model())
    infer.prefill_att_state = backend.create_att_prefill_state(infer)
    infer.prefill_att_state.init_state()
    control = AttControl(
        nsa_prefill=True,
        nsa_prefill_dict={
            "topk_mem_indices": topk_mem_indices,
            "topk_indices": topk_indices,
            "prefill_cache_kv": current_kv,
            "softmax_scale": SCALE,
            "kv_lora_rank": RANK,
        },
    )
    out = infer.prefill_att_state.prefill_att(q, packed, None, control)
    assert out.shape == (rows, heads, RANK)
    torch.testing.assert_close(out.float(), torch.stack(expected), atol=0.02, rtol=0.02)
