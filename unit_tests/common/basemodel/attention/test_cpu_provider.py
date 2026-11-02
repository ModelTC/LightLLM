from types import SimpleNamespace

import torch

from lightllm.common.basemodel.attention.runtime import DeviceRuntime
from lightllm.common.basemodel.attention.torch_cpu import TorchCPUAttentionBackend
from unit_tests.common.basemodel.attention.test_binding import bind_test_model, make_state


def test_cpu_provider_binds_native_storage_and_computes_ragged_gqa(monkeypatch):
    def reject_cuda(*args, **kwargs):
        raise AssertionError("CPU execution touched CUDA")

    monkeypatch.setattr(torch.cuda, "current_device", reject_cuda)
    runtime = DeviceRuntime(torch.device("cpu"))
    state = make_state(((0,),), heads=2, dim=16)
    backend = TorchCPUAttentionBackend(object())
    model = bind_test_model([state], backend)
    bound = model.layers_infer[0].attention
    torch.manual_seed(43)
    kv = torch.randn(8, 4, 16, dtype=torch.bfloat16)
    bound.write_kv(token_indices=torch.arange(8), kv=kv)

    class Batch(SimpleNamespace):
        pass

    infer = Batch(
        b_req_idx=torch.tensor([0, 1]),
        b_seq_len=torch.tensor([5, 3]),
        b_ready_cache_len=torch.tensor([2, 1]),
        req_manager=SimpleNamespace(req_to_token_indexs=torch.tensor([[7, 2, 4, 1, 5], [6, 0, 3, 0, 0]])),
    )
    metadata = backend.create_att_prefill_state(infer)
    metadata.init_state()
    q = torch.randn(5, 4, 16, dtype=torch.bfloat16)
    result = bound.prefill(q, (metadata, None), runtime.allocate)
    expected = []
    for row, (request, end) in enumerate(((0, 3), (0, 4), (0, 5), (1, 2), (1, 3))):
        indices = infer.req_manager.req_to_token_indexs[request, :end]
        k = kv[indices, :2].float().repeat_interleave(2, 1)
        v = kv[indices, 2:].float().repeat_interleave(2, 1)
        scores = torch.einsum("hd,thd->ht", q[row].float(), k) / 4
        expected.append(torch.einsum("ht,thd->hd", scores.softmax(-1), v))
    torch.testing.assert_close(result, torch.stack(expected).to(result.dtype), atol=0.02, rtol=0.02)
    fence = runtime.record_completion()
    assert fence.query()
    runtime.wait(fence)
