import torch
import triton
import triton.language as tl

from lightllm.utils.envs_utils import get_diverse_max_batch_shared_group_size


@triton.jit
def _fwd_kernel_build_diverse_shared_group_markers(
    b_shared_radix_node_id,
    b_mark_shared_group,
    batch_size,
    MAX_GROUP_SIZE: tl.constexpr,
    SCAN_BLOCK_SIZE: tl.constexpr,
):
    current_row = tl.program_id(axis=0)
    current_node_id = tl.load(b_shared_radix_node_id + current_row)

    # -1 means that this row has no reusable radix node. This includes padded
    # HOLD rows, and every such row must remain an independent group.
    if current_node_id == -1:
        tl.store(b_mark_shared_group + current_row, 1)
        return

    has_next_row = current_row + 1 < batch_size
    next_node_id = tl.load(
        b_shared_radix_node_id + current_row + 1,
        mask=has_next_row,
        other=-1,
    )
    is_shared_run_end = (~has_next_row) | (next_node_id != current_node_id)
    if not is_shared_run_end:
        return

    backward_offsets = tl.arange(0, SCAN_BLOCK_SIZE)
    scanned_rows = current_row - backward_offsets
    valid_scanned_rows = scanned_rows >= 0
    scanned_node_id = tl.load(
        b_shared_radix_node_id + scanned_rows,
        mask=valid_scanned_rows,
        other=-1,
    )
    has_same_node_id = valid_scanned_rows & (scanned_node_id == current_node_id)
    mismatch_count = tl.cumsum((~has_same_node_id).to(tl.int32), axis=0)
    belongs_to_shared_run = has_same_node_id & (mismatch_count == 0)
    shared_run_size = tl.sum(belongs_to_shared_run, axis=0)
    shared_run_start = current_row - shared_run_size + 1

    for group_start_offset in tl.range(0, shared_run_size, MAX_GROUP_SIZE):
        group_size = tl.minimum(MAX_GROUP_SIZE, shared_run_size - group_start_offset)
        group_end_row = shared_run_start + group_start_offset + group_size - 1
        tl.store(b_mark_shared_group + group_end_row, group_size)


def build_diverse_shared_group_markers(
    b_shared_radix_node_id: torch.Tensor,
) -> torch.Tensor:
    """根据各 decode 行对应的 radix node ``time_id`` 构建 diverse 共享组标记。

    连续且相同的非负 ``time_id`` 会组成一个共享组；每个组只在最后一行写入
    组大小，组内其他行写入 0。超过 ``max_group_size`` 的连续请求会被拆分成
    多个共享组。``-1`` 表示该行没有可复用的 radix node，因此每个 ``-1``
    都独立成组并写入 1。

    例如 ``max_group_size = 3`` 时::

        b_shared_radix_node_id: [7, 7, 7, 7, 11, 11, -1, -1]
        b_mark_shared_group:     [0, 0, 3, 1,  0,  2,  1,  1]

    前四个 ``7`` 被拆成大小为 3 和 1 的两个组，两个 ``11`` 组成大小为 2
    的组，最后两个 ``-1`` 则分别作为独立的一行组。
    """

    assert b_shared_radix_node_id.is_cuda
    batch_size = b_shared_radix_node_id.shape[0]
    if batch_size == 0:
        return torch.empty((0,), dtype=torch.int32, device=b_shared_radix_node_id.device)

    max_group_size = int(get_diverse_max_batch_shared_group_size())
    assert max_group_size > 0
    b_mark_shared_group = torch.zeros((batch_size,), dtype=torch.int32, device=b_shared_radix_node_id.device)
    _fwd_kernel_build_diverse_shared_group_markers[(batch_size,)](
        b_shared_radix_node_id=b_shared_radix_node_id,
        b_mark_shared_group=b_mark_shared_group,
        batch_size=batch_size,
        MAX_GROUP_SIZE=max_group_size,
        SCAN_BLOCK_SIZE=triton.next_power_of_2(batch_size),
        num_warps=8,
        num_stages=1,
    )
    return b_mark_shared_group
