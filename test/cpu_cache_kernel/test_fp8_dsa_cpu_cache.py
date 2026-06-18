from types import SimpleNamespace

import torch


def test_fp8_dsa_cpu_cache_meta_includes_flashmla_and_indexer(monkeypatch):
    import lightllm.utils.kv_cache_utils as kv_cache_utils
    from lightllm.common.kv_cache_mem_manager import FP8PerTokenGroupQuantDeepseek3_2MemoryManager

    kv_cache_utils.calcu_cpu_cache_meta.cache_clear()

    monkeypatch.setattr(
        kv_cache_utils,
        "get_env_start_args",
        lambda: SimpleNamespace(
            enable_cpu_cache=True,
            model_dir="/fake/glm-5.2",
            cpu_cache_token_page_size=64,
            cpu_cache_storage_size=1,
            mtp_mode=None,
        ),
    )
    monkeypatch.setattr(kv_cache_utils, "is_linear_att_mixed_model", lambda _: False)
    monkeypatch.setattr(kv_cache_utils, "select_mem_manager_class", lambda: FP8PerTokenGroupQuantDeepseek3_2MemoryManager)
    monkeypatch.setattr(kv_cache_utils, "get_layer_num", lambda _: 2)

    meta = kv_cache_utils.calcu_cpu_cache_meta()

    expected_head_dim = (
        FP8PerTokenGroupQuantDeepseek3_2MemoryManager.flashmla_bytes_per_token
        + FP8PerTokenGroupQuantDeepseek3_2MemoryManager.indexer_bytes_per_token
    )
    assert meta.data_type is torch.uint8
    assert meta.num_heads == 1
    assert meta.head_dim == expected_head_dim
    assert meta.scale_head_dim == 0
    assert meta.calcu_one_page_size() == 64 * 2 * expected_head_dim


def test_bf16_dsa_cpu_cache_meta_keeps_padded_indexer_bytes(monkeypatch):
    import lightllm.utils.kv_cache_utils as kv_cache_utils
    from lightllm.common.kv_cache_mem_manager import Deepseek3_2MemoryManager

    kv_cache_utils.calcu_cpu_cache_meta.cache_clear()

    monkeypatch.setattr(
        kv_cache_utils,
        "get_env_start_args",
        lambda: SimpleNamespace(
            enable_cpu_cache=True,
            model_dir="/fake/glm-5.2",
            cpu_cache_token_page_size=64,
            cpu_cache_storage_size=1,
            mtp_mode=None,
        ),
    )
    monkeypatch.setattr(kv_cache_utils, "is_linear_att_mixed_model", lambda _: False)
    monkeypatch.setattr(kv_cache_utils, "select_mem_manager_class", lambda: Deepseek3_2MemoryManager)
    monkeypatch.setattr(kv_cache_utils, "get_layer_num", lambda _: 2)
    monkeypatch.setattr(kv_cache_utils, "get_llm_data_type", lambda: torch.bfloat16)

    meta = kv_cache_utils.calcu_cpu_cache_meta()

    assert meta.data_type is torch.bfloat16
    assert meta.num_heads == 1
    assert meta.head_dim == 512 + 64 + (144 // 2)
    assert meta.scale_head_dim == 0
    assert meta.calcu_one_page_size() == 64 * 2 * meta.head_dim * torch.bfloat16.itemsize


def test_fp8_dsa_operator_implements_cpu_cache_methods():
    from lightllm.common.kv_cache_mem_manager.operator.base import BaseMemManagerOperator
    from lightllm.common.kv_cache_mem_manager.operator.deepseek import FP8PerTokenGroupQuantDeepseek3_2MemOperator

    assert (
        FP8PerTokenGroupQuantDeepseek3_2MemOperator.load_cpu_cache_to_gpu
        is not BaseMemManagerOperator.load_cpu_cache_to_gpu
    )
    assert (
        FP8PerTokenGroupQuantDeepseek3_2MemOperator.offload_gpu_kv_to_cpu_cache
        is not BaseMemManagerOperator.offload_gpu_kv_to_cpu_cache
    )


def test_fp8_dsa_operator_moves_flashmla_and_indexer_views(monkeypatch):
    import lightllm.common.basemodel.triton_kernel.kv_cache_offload as kv_cache_offload
    import lightllm.utils.dist_utils as dist_utils
    import lightllm.utils.envs_utils as envs_utils
    from lightllm.common.kv_cache_mem_manager import FP8PerTokenGroupQuantDeepseek3_2MemoryManager
    from lightllm.common.kv_cache_mem_manager.operator.deepseek import FP8PerTokenGroupQuantDeepseek3_2MemOperator

    class CudaSeq:
        is_cuda = True

        def __init__(self, n):
            self.n = n

        def __len__(self):
            return self.n

    split = FP8PerTokenGroupQuantDeepseek3_2MemoryManager.flashmla_bytes_per_token
    indexer_size = FP8PerTokenGroupQuantDeepseek3_2MemoryManager.indexer_bytes_per_token
    cpu_tensor = torch.empty((4, 2, 64, 1, split + indexer_size), dtype=torch.uint8)
    client = SimpleNamespace(
        kv_cache_tensor_meta=SimpleNamespace(data_type=torch.uint8, num_heads=1, head_dim=split + indexer_size),
        cpu_kv_cache_tensor=cpu_tensor,
    )
    mem_manager = SimpleNamespace(
        flashmla_bytes_per_token=split,
        indexer_bytes_per_token=indexer_size,
        kv_buffer=object(),
        indexer_k_buffer=object(),
    )
    op = FP8PerTokenGroupQuantDeepseek3_2MemOperator(mem_manager)
    mem_indexes = CudaSeq(128)
    page_indexes = CudaSeq(2)
    page_readies = CudaSeq(2)

    monkeypatch.setattr(envs_utils, "get_env_start_args", lambda: SimpleNamespace(cpu_cache_token_page_size=64))
    monkeypatch.setattr(dist_utils, "get_current_rank_in_dp", lambda: 0)
    monkeypatch.setattr(dist_utils, "get_dp_world_size", lambda: 1)

    load_calls = []
    offload_calls = []
    monkeypatch.setattr(kv_cache_offload, "load_cpu_kv_to_gpu", lambda **kwargs: load_calls.append(kwargs))
    monkeypatch.setattr(kv_cache_offload, "offload_gpu_kv_to_cpu", lambda **kwargs: offload_calls.append(kwargs))

    op.load_cpu_cache_to_gpu(mem_indexes, page_indexes, client, req=None)
    op.offload_gpu_kv_to_cpu_cache(mem_indexes, page_indexes, page_readies, client, req=None)

    assert [call["gpu_kv_cache"] for call in load_calls] == [mem_manager.kv_buffer, mem_manager.indexer_k_buffer]
    assert [call["cpu_kv_cache"].shape[-1] for call in load_calls] == [split, indexer_size]
    assert [call["gpu_kv_cache"] for call in offload_calls] == [mem_manager.kv_buffer, mem_manager.indexer_k_buffer]
    assert [call["cpu_kv_cache"].shape[-1] for call in offload_calls] == [split, indexer_size]
