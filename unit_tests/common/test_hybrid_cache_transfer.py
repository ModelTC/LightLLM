from types import SimpleNamespace

import torch

from lightllm.common import hybrid_cache_transfer as transfer
from lightllm.common.basemodel.triton_kernel import linear_att_cpu_cache_copy as kernels


def test_cpu_checkpoint_metadata(monkeypatch):
    class CudaIndexes(list):
        is_cuda = True

    args = SimpleNamespace(cpu_cache_token_page_size=8, linear_att_hash_page_size=4)
    monkeypatch.setattr(transfer, "get_env_start_args", lambda: args)
    monkeypatch.setattr(transfer, "get_current_rank_in_dp", lambda: 0)
    monkeypatch.setattr(transfer, "get_dp_world_size", lambda: 1)
    copies = []
    monkeypatch.setattr(kernels, "copy_kv_buffer_to_cpu_cache", lambda **kwargs: copies.append(kwargs))
    tensor = torch.tensor

    def cpu_tensor(data, **kwargs):
        kwargs.pop("pin_memory", None)
        return tensor(data, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)
    pages = torch.full((4, 1, 1, 1, 64), 37, dtype=torch.uint8)
    pages[1].reshape(-1)[-16:].view(torch.int64).copy_(torch.tensor([24, 1]))
    before = pages.clone()
    pool = SimpleNamespace(
        checkpoint_lens=[None, 16, None, 32],
        conv_state_cache=SimpleNamespace(buffer=None),
        ssm_state_cache=SimpleNamespace(buffer=None),
    )
    writer = transfer.HybridCpuCacheTransfer(None, None, pool)
    writer.big_page_ids_buffer_store = torch.empty(4, dtype=torch.int64)
    client = SimpleNamespace(cpu_kv_cache_tensor=pages)
    page_indexes = [2, 0, 1, -1]
    page_readies = [False, False, True, False]

    writer.offload(
        CudaIndexes(range(32)),
        CudaIndexes(page_indexes),
        CudaIndexes(page_readies),
        client,
        [0, 1, 2, 3],
        None,
        page_indexes_cpu=page_indexes,
        page_readies_cpu=page_readies,
    )

    assert len(copies) == 1
    assert copies[0]["big_page_buffer_ids"].tolist() == [0, 1, 2, 3]
    assert pages[2].reshape(-1)[-16:].view(torch.int64).tolist() == [-1, 1]
    assert pages[0].reshape(-1)[-16:].view(torch.int64).tolist() == [16, 1]
    assert torch.equal(pages[1], before[1])
    assert torch.equal(pages[3], before[3])
    assert torch.equal(pages.reshape(4, -1)[:, :-16], before.reshape(4, -1)[:, :-16])

    serialized = pages.numpy().tobytes()
    restored = torch.frombuffer(bytearray(serialized), dtype=torch.uint8).reshape_as(pages)
    reader = transfer.HybridCpuCacheTransfer(None, None, None)
    restored_client = SimpleNamespace(cpu_kv_cache_tensor=restored)
    assert reader.get_checkpoint_lens([2, 0, 1], restored_client) == [None, 16, 24]
    restored[0].reshape(-1)[-8:].view(torch.int64).fill_(0)
    assert reader.get_checkpoint_lens([0], restored_client) == [None]
