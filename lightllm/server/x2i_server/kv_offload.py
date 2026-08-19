import torch


@torch.no_grad()
def offload_separate_kv_to_cpu_for_x2i(
    token_indexes: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    cpu_kv_cache: torch.Tensor,
    page_indexes: torch.Tensor,
    tp_index: int,
    tp_world_size: int,
) -> None:
    """Copy split K/V tensors into the shared X2I page layout.

    Ascend stores K and V independently, while the shared cache uses
    ``[page, layer, token, 2 * global_kv_heads, head_dim]``. TP ranks write
    disjoint global-head slices into the same shared pages.
    """
    if k_buffer.shape != v_buffer.shape:
        raise ValueError(f"K/V cache shapes differ: {k_buffer.shape} vs {v_buffer.shape}")
    if cpu_kv_cache.ndim != 5 or k_buffer.ndim != 4:
        raise ValueError(f"unexpected KV shapes: device={k_buffer.shape}, cpu={cpu_kv_cache.shape}")

    token_count = token_indexes.numel()
    page_size = cpu_kv_cache.shape[2]
    if token_count > page_indexes.numel() * page_size:
        raise ValueError("not enough shared-memory pages for the requested KV tokens")
    if token_count == 0:
        return

    merged_cpu_heads = cpu_kv_cache.shape[3]
    if merged_cpu_heads % 2 != 0:
        raise ValueError(f"shared KV head count must be even, got {merged_cpu_heads}")
    global_kv_heads = merged_cpu_heads // 2
    local_kv_heads = k_buffer.shape[2]
    if k_buffer.shape[0] != cpu_kv_cache.shape[1] or k_buffer.shape[3] != cpu_kv_cache.shape[4]:
        raise ValueError(f"device/shared KV metadata mismatch: {k_buffer.shape} vs {cpu_kv_cache.shape}")
    if (tp_world_size * local_kv_heads) % global_kv_heads != 0:
        raise ValueError(
            f"cannot map {local_kv_heads} local KV heads across TP={tp_world_size} "
            f"to {global_kv_heads} global KV heads"
        )

    replication = (tp_world_size * local_kv_heads) // global_kv_heads
    if replication < 1:
        raise ValueError("invalid KV-head replication factor")
    if tp_index % replication != 0:
        return
    shard_index = tp_index // replication
    head_start = shard_index * local_kv_heads
    head_end = head_start + local_kv_heads
    if head_end > global_kv_heads:
        raise ValueError(f"TP head slice [{head_start}, {head_end}) exceeds {global_kv_heads}")

    indexes = token_indexes.reshape(-1).to(dtype=torch.long)
    selected_k = torch.index_select(k_buffer, 1, indexes).to(device="cpu")
    selected_v = torch.index_select(v_buffer, 1, indexes).to(device="cpu")
    page_ids = page_indexes.reshape(-1).to(device="cpu", dtype=torch.long).tolist()

    copied = 0
    for page_id in page_ids:
        page_tokens = min(page_size, token_count - copied)
        if page_tokens <= 0:
            break
        token_slice = slice(copied, copied + page_tokens)
        cpu_kv_cache[page_id, :, :page_tokens, head_start:head_end, :].copy_(selected_k[:, token_slice])
        cpu_kv_cache[
            page_id,
            :,
            :page_tokens,
            global_kv_heads + head_start : global_kv_heads + head_end,
            :,
        ].copy_(selected_v[:, token_slice])
        copied += page_tokens
