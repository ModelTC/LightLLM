import torch

from lightllm.server.x2i_server.kv_offload import offload_separate_kv_to_cpu_for_x2i


def test_tp2_split_kv_to_shared_pages():
    layers, storage_tokens, local_heads, head_dim = 2, 9, 2, 3
    page_size, global_heads = 3, 4
    token_indexes = torch.tensor([7, 1, 8, 0, 4], dtype=torch.int32)
    page_indexes = torch.tensor([2, 0], dtype=torch.int32)
    shared = torch.full((3, layers, page_size, 2 * global_heads, head_dim), -1.0)

    expected_k = []
    expected_v = []
    for tp_rank in range(2):
        k = torch.arange(
            layers * storage_tokens * local_heads * head_dim,
            dtype=torch.float32,
        ).reshape(layers, storage_tokens, local_heads, head_dim)
        k = k + tp_rank * 10000
        v = k + 5000
        offload_separate_kv_to_cpu_for_x2i(
            token_indexes=token_indexes,
            k_buffer=k,
            v_buffer=v,
            cpu_kv_cache=shared,
            page_indexes=page_indexes,
            tp_index=tp_rank,
            tp_world_size=2,
        )
        expected_k.append(k[:, token_indexes.long()])
        expected_v.append(v[:, token_indexes.long()])

    restored = torch.cat([shared[2, :, :3], shared[0, :, :2]], dim=1)
    assert torch.equal(restored[:, :, :global_heads], torch.cat(expected_k, dim=2))
    assert torch.equal(restored[:, :, global_heads:], torch.cat(expected_v, dim=2))
    assert torch.all(shared[1] == -1)


def test_replicated_kv_heads_are_written_once():
    # TP=4 with two global KV heads means each logical KV shard is replicated.
    token_indexes = torch.tensor([0], dtype=torch.int32)
    page_indexes = torch.tensor([0], dtype=torch.int32)
    shared = torch.zeros((1, 1, 1, 4, 1))

    for tp_rank in range(4):
        k = torch.tensor([[[[10.0 + tp_rank]]]])
        v = torch.tensor([[[[20.0 + tp_rank]]]])
        offload_separate_kv_to_cpu_for_x2i(
            token_indexes,
            k,
            v,
            shared,
            page_indexes,
            tp_rank,
            4,
        )

    # Ranks 0 and 2 are the canonical writers for the two replica groups.
    assert torch.equal(shared.flatten(), torch.tensor([10.0, 12.0, 20.0, 22.0]))
