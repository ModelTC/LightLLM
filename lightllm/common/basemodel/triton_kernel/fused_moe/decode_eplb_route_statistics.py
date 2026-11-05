"""Decode 路由共用的 logical 专家共现统计。"""

import triton
import triton.language as tl


@triton.jit
def _record_decode_route_cooccurrence(
    logical_topk_ids_ptr,
    topk_id_offsets,
    row_expert_ids,
    valid_mask,
    num_topk_ids,
    decode_route_counter_ptr,
    decode_route_counter_row_stride,
    decode_route_recording_enabled_ptr,
    TOP_K: tl.constexpr,
):
    """把 decode top-k 的无序 expert 对记录到共现矩阵上三角。"""
    recording_enabled = tl.load(pointer=decode_route_recording_enabled_ptr) != 0
    # 设备端标志是整个 program 共用的标量。关闭时跳过全部统计计算，
    # 而非只屏蔽 load / atomic_add；Graph 重放仍可原地切换标志，无需重新捕获。
    if recording_enabled:
        # 每个有效 lane 对应当前 token 的一个 logical expert，并以该 expert
        # 作为矩阵行。先由展平 offset 定位所属 token 的 top-k 起点。
        token_topk_start_offsets = (topk_id_offsets // TOP_K) * TOP_K
        counter_row_offsets = row_expert_ids * decode_route_counter_row_stride

        # 例如 top-k 为 [7, 11, 23]：expert 7 的 lane 写 [7,7]、[7,11]、
        # [7,23]；expert 11 的 lane 跳过重复的 [11,7]，继续写 [11,11]
        # 和 [11,23]。因此每个非对角无序 expert 对只记录一次。
        for partner_slot in tl.static_range(arg1=0, arg2=TOP_K):
            partner_offsets = token_topk_start_offsets + partner_slot
            partner_mask = valid_mask & (partner_offsets < num_topk_ids)
            partner_expert_ids = tl.load(
                pointer=logical_topk_ids_ptr + partner_offsets,
                mask=partner_mask,
                other=-1,
            )

            # 使用 <= 而不是 <，明确把 [i, i] 纳入上三角。top-k ID 唯一时，
            # 每个被选中的 expert 恰好有一个自身槽位，因此对角线准确加一；
            # i < j 的 pair 由较小 ID 对应的 lane 唯一写入，不存在重复计数。
            upper_triangle_mask = partner_mask & (row_expert_ids <= partner_expert_ids)
            tl.atomic_add(
                pointer=decode_route_counter_ptr + counter_row_offsets + partner_expert_ids,
                val=1,
                mask=upper_triangle_mask,
                sem="relaxed",
            )
