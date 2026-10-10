from types import SimpleNamespace

import pytest
import torch

from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.common.basemodel.attention.nsa.glm5_next import (
    Glm5NextSparsePrefillState,
    Glm5NextSparseDecodeState,
)
from lightllm.common.basemodel.attention.nsa.fp8_glm5_next import (
    Fp8Glm5NextSparsePrefillState,
    Fp8Glm5NextSparseDecodeState,
)
from lightllm.models.glm5_next.triton_kernel.destindex_copy_kv_flashmla_fp8 import destindex_copy_kv_flashmla_fp8
from lightllm.models.glm5_next.triton_kernel.prefill_gather_kv_flashmla_fp8 import (
    gather_prefill_kv_cache_triton,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize("kv_type", ["None", "fp8kv_dsa"])
def test_main_and_native_mtp_backend_selection(monkeypatch, draft, kv_type):
    from lightllm.models.glm5_next.model import Glm5NextTpPartModel
    from lightllm.models.glm5_next_mtp.model import Glm5NextMTPModel

    monkeypatch.setenv("LIGHTLLM_CURRENT_DEVICE_ID", "0")
    model = object.__new__(Glm5NextMTPModel if draft else Glm5NextTpPartModel)
    model.args = SimpleNamespace(llm_kv_type=kv_type)
    model.graph_max_batch_size = 4
    model.max_seq_length = 128
    model._init_att_backend()
    prefill = model.prefill_att_backend.create_att_prefill_state(SimpleNamespace())
    decode = model.decode_att_backend.create_att_decode_state(SimpleNamespace())
    assert type(prefill) is (Fp8Glm5NextSparsePrefillState if kv_type == "fp8kv_dsa" else Glm5NextSparsePrefillState)
    assert type(decode) is (Fp8Glm5NextSparseDecodeState if kv_type == "fp8kv_dsa" else Glm5NextSparseDecodeState)


def _dequantize_kv_cache(packed_kv):
    values = packed_kv[..., :512].view(torch.float8_e4m3fn).float().reshape(-1, 4, 128)
    scales = packed_kv[..., 512:528].view(torch.float32).reshape(-1, 4, 1)
    return (values * scales).reshape(-1, 1, 512).bfloat16()


def _torch_sparse_attention(q, kv, indices):
    outputs = []
    for row in range(q.shape[0]):
        keys = kv[indices[row][indices[row] >= 0].long(), 0].float()
        outputs.append((q[row].float() @ keys.T * 0.0625).softmax(-1) @ keys)
    return torch.stack(outputs)


def _capture_cuda_graph(run):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    return graph, output


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_fp8_scatter_quantization_and_gather_prefill(index_dtype):
    torch.manual_seed(53)
    # Noncontiguous KV from a fused projection, and a strided packed cache.
    source = torch.randn(11, 1, 1024, device="cuda", dtype=torch.bfloat16)[..., :512]
    source[0].zero_()
    source[1].mul_(1e-5)
    mem_index = torch.randperm(31, device="cuda")[:11].to(index_dtype)
    packed_kv = torch.full((31, 1, 800), 71, device="cuda", dtype=torch.uint8)
    destindex_copy_kv_flashmla_fp8(source, mem_index, packed_kv)
    groups = source[..., :512].float().reshape(11, 1, 4, 128)
    scales = torch.exp2(torch.ceil(torch.log2((groups.abs().amax(-1) / 448).clamp_min(1e-4))))
    values = (groups / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
    torch.testing.assert_close(packed_kv[..., 512:528].view(torch.float32)[mem_index], scales, rtol=0, atol=0)
    assert torch.equal(packed_kv[mem_index, :, :512], values.view(torch.uint8).reshape(11, 1, 512))
    assert not packed_kv[mem_index, :, 528:656].any()
    assert (packed_kv[..., 656:] == 71).all()
    unused = torch.ones(31, device="cuda", dtype=torch.bool)
    unused[mem_index] = False
    assert (packed_kv[unused] == 71).all()

    indices = mem_index[[0, 2, 8, 2, 5, 1, 9, 3]]
    prefill_kv = source[8:].clone().add_(0.25)
    gathered_kv = gather_prefill_kv_cache_triton(packed_kv[..., :656], indices, mem_index[8:], prefill_kv)
    expected = _dequantize_kv_cache(packed_kv)
    expected[mem_index[8:]] = prefill_kv
    assert torch.equal(gathered_kv, expected[indices.long()])


@pytest.mark.parametrize("num_heads", [8, 16, 64])
def test_fp8_nope_decode_and_cuda_graph(num_heads):
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("Validated FlashMLA sparse decode target is Hopper")
    import flash_mla

    torch.manual_seed(53)
    packed_kv = torch.zeros(4096, 1, 800, device="cuda", dtype=torch.uint8)
    source = torch.randn(4096, 1, 512, device="cuda", dtype=torch.bfloat16)
    mem_index = torch.arange(4096, device="cuda", dtype=torch.int32)
    destindex_copy_kv_flashmla_fp8(source, mem_index, packed_kv)
    q = torch.randn(num_heads, 9, 512, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    # Includes a HOLD row, short tails, and three consecutive MTP positions.
    lengths = torch.tensor([0, 1, 3, 127, 128, 511, 2048, 2049, 2051], device="cuda", dtype=torch.int32)
    indices = torch.randint(0, 4096, (9, 2176), device="cuda", dtype=torch.int32)
    lanes = torch.arange(2176, device="cuda")
    indices.masked_fill_(lanes[None, :] >= lengths[:, None], -1)
    control = AttControl(
        nsa_decode=True, nsa_decode_dict={"topk_mem_indices": indices, "softmax_scale": 0.0625, "kv_lora_rank": 512}
    )
    state = Fp8Glm5NextSparseDecodeState(flashmla_sched_meta=flash_mla.get_mla_metadata()[0])

    def run():
        return state.decode_att((q, q[..., :0]), packed_kv[..., :656], None, control)

    def check(actual):
        expected = _torch_sparse_attention(q, _dequantize_kv_cache(packed_kv), indices)
        torch.testing.assert_close(actual.float(), expected, atol=0.012, rtol=0.015)

    check(run())
    graph, out = _capture_cuda_graph(run)
    graph.replay()
    check(out)
    q.normal_()
    source.normal_()
    destindex_copy_kv_flashmla_fp8(source, mem_index, packed_kv)
    lengths.copy_(lengths.roll(1))
    indices.random_(0, 4096)
    indices.masked_fill_(lanes[None, :] >= lengths[:, None], -1)
    graph.replay()
    check(out)


@pytest.mark.parametrize("cached", [False, True])
def test_fp8_nope_batched_prefill_and_graph(cached):
    torch.manual_seed(53)
    # Two requests whose physical cache locations are unrelated to batch order.
    prefix_len = 4 if cached else 0
    seq_len = prefix_len + 3
    ragged_mem_index = torch.randperm(64, device="cuda")[: 2 * seq_len].int()
    prefill_rows = torch.cat(
        (torch.arange(prefix_len, seq_len), torch.arange(seq_len + prefix_len, 2 * seq_len))
    ).cuda()
    prefill_mem_index = ragged_mem_index[prefill_rows]
    source = torch.randn(2 * seq_len, 1, 512, device="cuda", dtype=torch.bfloat16)
    prefill_kv = source[prefill_rows].clone()
    packed_kv = torch.zeros(64, 1, 800, device="cuda", dtype=torch.uint8)
    destindex_copy_kv_flashmla_fp8(source, ragged_mem_index, packed_kv)
    q = torch.randn(6, 16, 512, device="cuda", dtype=torch.bfloat16)
    kv_start_offsets = torch.tensor([0] * 3 + [seq_len] * 3, device="cuda", dtype=torch.int32)
    topk_indices = torch.full((6, 128), -1, device="cuda", dtype=torch.int32)
    for row in range(6):
        length = prefix_len + row % 3 + 1
        topk_indices[row, :length] = torch.arange(length, device="cuda")
    control = AttControl(
        nsa_prefill=True,
        nsa_prefill_dict={
            "topk_indices": topk_indices,
            "prefill_cache_kv": prefill_kv,
            "softmax_scale": 0.0625,
            "kv_lora_rank": 512,
        },
    )
    state = Fp8Glm5NextSparsePrefillState(
        infer_state=SimpleNamespace(max_cache_len=prefix_len, mem_index=prefill_mem_index),
        ks=kv_start_offsets,
        ragged_mem_index=ragged_mem_index,
    )

    def run():
        return state.prefill_att(q, packed_kv[..., :656], None, control)

    def check(out):
        expected_kv = _dequantize_kv_cache(packed_kv)[ragged_mem_index.long()]
        expected_kv[prefill_rows] = prefill_kv
        indices = torch.where(topk_indices >= 0, topk_indices + kv_start_offsets[:, None], -1)
        expected = _torch_sparse_attention(q, expected_kv, indices)
        torch.testing.assert_close(out.float(), expected, atol=0.012, rtol=0.015)

    check(run())
    graph, out = _capture_cuda_graph(run)
    graph.replay()
    check(out)
    q.normal_()
    source.normal_()
    prefill_kv.copy_(source[prefill_rows])
    ragged_mem_index.copy_(ragged_mem_index.roll(1))
    prefill_mem_index.copy_(ragged_mem_index[prefill_rows])
    destindex_copy_kv_flashmla_fp8(source, ragged_mem_index, packed_kv)
    graph.replay()
    check(out)
