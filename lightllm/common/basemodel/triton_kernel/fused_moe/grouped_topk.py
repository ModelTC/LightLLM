import torch
import triton
import triton.language as tl


def get_group_score_used_topk_num(topk_group: int, num_expert_group: int, top_k: int) -> int:
    return 2 if topk_group == 4 and num_expert_group == 8 and top_k == 8 else 1


@triton.jit
def _eplb_replica_index(token_index, logical_id, replica_count):
    """Mix high and low bits before modulo, including periodic token indices."""
    value = token_index.to(tl.uint32) ^ ((logical_id.to(tl.uint32) + 1) * 0x9E3779B9)
    value = (value ^ (value >> 16)) * 0x7FEB352D
    value = (value ^ (value >> 15)) * 0x846CA68B
    value = value ^ (value >> 16)
    return value % replica_count.to(tl.uint32)


@triton.jit(do_not_specialize=["sample_index", "return_logical_ids", "record_load", "use_replicas"])
def grouped_topk_kernel(
    gating_output_ptr,
    gating_output_stride_m,
    gating_output_stride_n,
    correction_bias_ptr,
    out_topk_weights,
    out_topk_weights_stride_m,
    out_topk_weights_stride_n,
    out_topk_ids,
    out_topk_ids_stride_m,
    out_topk_ids_stride_n,
    out_logical_ids,
    out_logical_ids_stride_m,
    out_logical_ids_stride_n,
    logical_to_physical_ptr,
    logical_replica_count_ptr,
    expert_counter_ptr,
    sample_index,
    group_num,
    group_expert_num,
    total_expert_num,
    group_topk_num,
    return_logical_ids,
    record_load,
    use_replicas,
    IS_SIGMOID: tl.constexpr,
    USE_GROUPED_TOPK: tl.constexpr,
    HAS_CORRECTION_BIAS: tl.constexpr,
    EXPERT_GROUP_NUM: tl.constexpr,
    EXPERT_GROUP_SIZE: tl.constexpr,
    TOPK_NUM: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    GROUP_SCORE_USED_TOPK_NUM: tl.constexpr,
    COUNTER_NUM_EXPERTS: tl.constexpr,
    COUNTER_CHANNELS: tl.constexpr,
    HASH_REPLICA_COUNTS: tl.constexpr,
    REPLICA_ROTATION_RANK: tl.constexpr,
    MAP_SLOTS: tl.constexpr,
    APPLY_EPLB: tl.constexpr,
    HAS_COUNTER: tl.constexpr,
    SINGLE_TOKEN: tl.constexpr,
):
    """Shared logical top-k; ties choose the lowest expert ID, with optional EPLB mapping."""
    token_index = tl.program_id(axis=0)
    offs_group = tl.arange(0, EXPERT_GROUP_NUM)
    offs_group_v = tl.arange(0, EXPERT_GROUP_SIZE)
    logical_ids = offs_group[:, None] * group_expert_num + offs_group_v[None, :]
    valid_expert = (
        (offs_group < group_num)[:, None]
        & (offs_group_v < group_expert_num)[None, :]
        & (logical_ids < total_expert_num)
    )
    hidden_states = tl.load(
        gating_output_ptr + token_index * gating_output_stride_m + logical_ids * gating_output_stride_n,
        mask=valid_expert,
        other=-float("inf"),
    ).to(tl.float32)

    if IS_SIGMOID:
        old_scores = tl.sigmoid(hidden_states)
    else:
        group_max = tl.max(hidden_states, axis=1)
        global_max = tl.max(group_max, axis=0)
        numerators = tl.where(valid_expert, tl.exp(hidden_states - global_max), 0.0)
        denominator = tl.sum(tl.sum(numerators, axis=1), axis=0)
        old_scores = numerators / denominator

    if HAS_CORRECTION_BIAS:
        correction_bias = tl.load(correction_bias_ptr + logical_ids, mask=valid_expert, other=0.0)
        scores = tl.where(valid_expert, old_scores + correction_bias, -float("inf"))
    else:
        scores = tl.where(valid_expert, old_scores, -float("inf"))

    if USE_GROUPED_TOPK:
        if GROUP_SCORE_USED_TOPK_NUM == 1:
            group_value = tl.max(scores, axis=1)
        elif GROUP_SCORE_USED_TOPK_NUM == 2:
            first_score, first_index = tl.max(scores, axis=1, return_indices=True)
            second_score = tl.max(
                tl.where(offs_group_v[None, :] == first_index[:, None], -float("inf"), scores),
                axis=1,
            )
            group_value = first_score + second_score
        else:
            sorted_group_scores = tl.sort(scores, dim=1, descending=True)
            group_value = tl.sum(
                tl.where(offs_group_v[None, :] < GROUP_SCORE_USED_TOPK_NUM, sorted_group_scores, 0.0),
                axis=1,
            )

        if EXPERT_GROUP_NUM > 1:
            sorted_group_value = tl.sort(group_value, descending=True)
        else:
            sorted_group_value = group_value
        group_topk_value = tl.sum(tl.where(offs_group == group_topk_num - 1, sorted_group_value, 0.0))
        candidate_scores = tl.where(
            (group_value >= group_topk_value)[:, None] & valid_expert,
            scores,
            -float("inf"),
        )
    else:
        # Softmax rounding can merge distinct logits; rank the original values.
        candidate_scores = tl.where(valid_expert, hidden_states, -float("inf"))

    sort_block_size: tl.constexpr = EXPERT_GROUP_NUM * EXPERT_GROUP_SIZE
    flat_offsets = tl.arange(0, sort_block_size)
    candidate_scores = tl.reshape(candidate_scores, (sort_block_size,))
    topk_offsets = tl.arange(0, TOPK_BLOCK_SIZE)
    selected_weights = tl.zeros((TOPK_BLOCK_SIZE,), tl.float32)
    selected_logical_ids = tl.zeros((TOPK_BLOCK_SIZE,), tl.int32)
    sum_scores = 0.0
    for topk_index in range(TOPK_NUM):
        selected_score = tl.max(candidate_scores, axis=0)
        selected_offset = tl.min(tl.where(candidate_scores == selected_score, flat_offsets, sort_block_size), axis=0)
        selected_group = selected_offset // EXPERT_GROUP_SIZE
        selected_group_offset = selected_offset % EXPERT_GROUP_SIZE
        selected_logical_id = selected_group * group_expert_num + selected_group_offset
        selected_hidden_state = tl.load(
            gating_output_ptr + token_index * gating_output_stride_m + selected_logical_id * gating_output_stride_n
        ).to(tl.float32)
        if IS_SIGMOID:
            selected_weight = tl.sigmoid(selected_hidden_state)
        else:
            selected_weight = tl.exp(selected_hidden_state - global_max) / denominator
        sum_scores += selected_weight
        topk_lane = topk_offsets == topk_index
        selected_weights = tl.where(topk_lane, selected_weight, selected_weights)
        selected_logical_ids = tl.where(topk_lane, selected_logical_id, selected_logical_ids)
        candidate_scores = tl.where(flat_offsets == selected_offset, -float("inf"), candidate_scores)

    topk_mask = topk_offsets < TOPK_NUM
    if HAS_COUNTER:
        if record_load:
            tl.atomic_add(
                expert_counter_ptr + (sample_index * COUNTER_NUM_EXPERTS + selected_logical_ids) * COUNTER_CHANNELS,
                1,
                mask=topk_mask,
                sem="relaxed",
            )
            if COUNTER_CHANNELS > 1:
                for copies in tl.static_range(1, HASH_REPLICA_COUNTS + 1):
                    counts = tl.full((TOPK_BLOCK_SIZE,), copies, tl.int32)
                    if SINGLE_TOKEN:
                        bucket = tl.full((TOPK_BLOCK_SIZE,), REPLICA_ROTATION_RANK % copies, tl.int32)
                    else:
                        bucket = (
                            _eplb_replica_index(token_index, selected_logical_ids, counts) + REPLICA_ROTATION_RANK
                        ) % copies
                    channel = 1 + copies * (copies - 1) // 2 + bucket
                    tl.atomic_add(
                        expert_counter_ptr
                        + (sample_index * COUNTER_NUM_EXPERTS + selected_logical_ids) * COUNTER_CHANNELS
                        + channel,
                        1,
                        mask=topk_mask,
                        sem="relaxed",
                    )
    if APPLY_EPLB:
        if SINGLE_TOKEN or not use_replicas:
            replica_indices = tl.zeros((TOPK_BLOCK_SIZE,), tl.int32)
        else:
            replica_counts = tl.load(
                logical_replica_count_ptr + selected_logical_ids,
                mask=topk_mask,
                other=1,
            )
            replica_indices = _eplb_replica_index(token_index, selected_logical_ids, replica_counts).to(tl.int32)
        selected_physical_ids = tl.load(
            logical_to_physical_ptr + selected_logical_ids * MAP_SLOTS + replica_indices,
            mask=topk_mask,
            other=-1,
        )
    else:
        selected_physical_ids = selected_logical_ids
    if RENORMALIZE:
        selected_weights /= sum_scores
    tl.store(
        out_topk_weights + token_index * out_topk_weights_stride_m + topk_offsets * out_topk_weights_stride_n,
        selected_weights,
        mask=topk_mask,
    )
    tl.store(
        out_topk_ids + token_index * out_topk_ids_stride_m + topk_offsets * out_topk_ids_stride_n,
        selected_physical_ids,
        mask=topk_mask,
    )
    if return_logical_ids:
        tl.store(
            out_logical_ids + token_index * out_logical_ids_stride_m + topk_offsets * out_logical_ids_stride_n,
            selected_logical_ids,
            mask=topk_mask,
        )


def triton_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    group_score_used_topk_num=2,
):
    weights, ids, _ = triton_grouped_topk_eplb(
        hidden_states=hidden_states,
        gating_output=gating_output,
        correction_bias=correction_bias,
        topk=topk,
        renormalize=renormalize,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        scoring_func=scoring_func,
        logical_to_physical_map=None,
        logical_replica_count=None,
        expert_counter=None,
        sample_index=0,
        record_load=False,
        use_grouped_topk=True,
        group_score_used_topk_num=group_score_used_topk_num,
    )
    return weights, ids


def triton_grouped_topk_eplb(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    scoring_func: str,
    logical_to_physical_map: torch.Tensor,
    logical_replica_count: torch.Tensor,
    expert_counter: torch.Tensor,
    sample_index: int,
    record_load: bool,
    use_grouped_topk: bool,
    return_logical_ids: bool = False,
    group_score_used_topk_num: int = 2,
    replica_rotation_rank: int = 0,
    use_replicas: bool = True,
):
    """One logical selector; absent EPLB maps compile out accounting and replica selection."""
    from math import isqrt

    if record_load and expert_counter is None:
        raise ValueError("EPLB load recording requires an expert counter")
    channels = expert_counter.shape[2] if expert_counter is not None and expert_counter.ndim == 3 else 1
    max_hash_replicas = (isqrt(1 + 8 * (channels - 1)) - 1) // 2
    if expert_counter is not None and (
        expert_counter.ndim not in (2, 3)
        or (channels > 1 and max_hash_replicas * (max_hash_replicas + 1) // 2 != channels - 1)
    ):
        raise ValueError("EPLB counters must contain logical counts and complete replica hash buckets")
    token_num, total_expert_num = gating_output.shape
    out_topk_weights = torch.empty((token_num, topk), dtype=torch.float32, device=gating_output.device)
    out_topk_ids = torch.empty((token_num, topk), dtype=torch.long, device=gating_output.device)
    out_logical_ids = (
        torch.empty((token_num, topk), dtype=torch.long, device=gating_output.device) if return_logical_ids else None
    )
    if token_num == 0:
        return out_topk_weights, out_topk_ids, out_logical_ids
    if use_grouped_topk:
        assert total_expert_num % num_expert_group == 0
        group_num = num_expert_group
        group_expert_num = total_expert_num // num_expert_group
        group_topk_num = topk_group
    else:
        group_num = 1
        group_expert_num = total_expert_num
        group_topk_num = 1
    expert_group_num = triton.next_power_of_2(group_num)
    expert_group_size = triton.next_power_of_2(group_expert_num)
    sort_block_size = expert_group_num * expert_group_size
    num_warps = min(max(1, sort_block_size // 256), 8)
    grouped_topk_kernel[(token_num,)](
        gating_output,
        *gating_output.stride(),
        correction_bias,
        out_topk_weights,
        *out_topk_weights.stride(),
        out_topk_ids,
        *out_topk_ids.stride(),
        out_logical_ids if out_logical_ids is not None else out_topk_ids,
        *(out_logical_ids.stride() if out_logical_ids is not None else out_topk_ids.stride()),
        logical_to_physical_map,
        logical_replica_count,
        expert_counter,
        sample_index,
        group_num=group_num,
        group_expert_num=group_expert_num,
        total_expert_num=total_expert_num,
        group_topk_num=group_topk_num,
        IS_SIGMOID=use_grouped_topk and scoring_func == "sigmoid",
        USE_GROUPED_TOPK=use_grouped_topk,
        HAS_CORRECTION_BIAS=use_grouped_topk and correction_bias is not None,
        return_logical_ids=return_logical_ids,
        record_load=record_load,
        use_replicas=use_replicas,
        EXPERT_GROUP_NUM=expert_group_num,
        EXPERT_GROUP_SIZE=expert_group_size,
        TOPK_NUM=topk,
        TOPK_BLOCK_SIZE=triton.next_power_of_2(topk),
        RENORMALIZE=renormalize,
        GROUP_SCORE_USED_TOPK_NUM=group_score_used_topk_num,
        COUNTER_NUM_EXPERTS=expert_counter.shape[1] if expert_counter is not None else total_expert_num,
        COUNTER_CHANNELS=channels,
        HASH_REPLICA_COUNTS=max_hash_replicas,
        REPLICA_ROTATION_RANK=replica_rotation_rank,
        MAP_SLOTS=logical_to_physical_map.shape[1] if logical_to_physical_map is not None else 1,
        APPLY_EPLB=logical_to_physical_map is not None,
        HAS_COUNTER=expert_counter is not None,
        SINGLE_TOKEN=token_num == 1,
        num_warps=num_warps,
        num_stages=1,
    )
    return out_topk_weights, out_topk_ids, out_logical_ids
