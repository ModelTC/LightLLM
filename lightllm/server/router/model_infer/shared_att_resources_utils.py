"""分别发布 PD 运行态和 DP shared 资源，使用独立共享内存及读取计数。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from multiprocessing.reduction import ForkingPickler

import torch.multiprocessing as mp
from filelock import FileLock

from lightllm.utils.device_utils import kv_trans_use_p2p
from lightllm.utils.dist_utils import get_current_rank_in_node
from lightllm.utils.shm_utils import create_or_link_shm, get_service_shm_name

if TYPE_CHECKING:
    import torch
    from lightllm.common.attention_state.manager import AttentionStateManager


def publish_att_resources(
    resources: AttentionStateManager,
    request_mapping: torch.Tensor,
    kind: Literal["pd", "dp"],
    reader_count: int,
) -> None:
    """按用途选择导出副本；每个读者独立消费对应类别的 IPC 引用。"""
    assert kind in ("pd", "dp"), f"unknown IPC storage kind {kind}"
    if reader_count < 0:
        raise ValueError("storage reader count cannot be negative")
    if reader_count == 0:
        return
    if kv_trans_use_p2p():
        from .mode_backend.pd.p2p_fix import reduce_tensor

        mp.reductions.reduce_tensor.__code__ = reduce_tensor.__code__
    count = reader_count
    if kind == "pd":
        ipc_resources = resources.create_pd_ipc_copy()
    else:
        ipc_resources = resources.create_dp_ipc_copy()
    with FileLock(f"/tmp/{get_service_shm_name(f'rank_storage_{kind}_lock')}"):
        # Each importer consumes its own CUDA IPC reference counters.
        payloads = [ForkingPickler.dumps((ipc_resources, request_mapping)).tobytes() for _ in range(count)]
        size = len(payloads[0])
        if any(len(payload) != size for payload in payloads):
            raise RuntimeError("rank storage exports have inconsistent sizes")
        shm = create_or_link_shm(
            name=f"rank_storage_{kind}_{get_current_rank_in_node()}", expected_size=8 + count * size, force_mode="create"
        )
        shm.buf[:4] = count.to_bytes(4, "little")
        shm.buf[4:8] = size.to_bytes(4, "little")
        for index, payload in enumerate(payloads):
            shm.buf[8 + index * size : 8 + (index + 1) * size] = payload
        shm.close()


def load_att_resources(
    rank_in_node: int, kind: Literal["pd", "dp"],
) -> tuple[AttentionStateManager, torch.Tensor]:
    """读取指定用途的资源；DP 接收方随后通过 manager 统一注册共享页池。"""
    assert kind in ("pd", "dp"), f"unknown IPC storage kind {kind}"
    with FileLock(f"/tmp/{get_service_shm_name(f'rank_storage_{kind}_lock')}"):
        shm = create_or_link_shm(name=f"rank_storage_{kind}_{rank_in_node}", expected_size=-1, force_mode="link")
        try:
            count = int.from_bytes(shm.buf[:4], "little")
            size = int.from_bytes(shm.buf[4:8], "little")
            if count <= 0:
                raise RuntimeError("rank storage IPC exports are exhausted")
            payload = bytes(shm.buf[8 + (count - 1) * size : 8 + count * size])
            resources, request_mapping = ForkingPickler.loads(payload)
            shm.buf[:4] = (count - 1).to_bytes(4, "little")
            return resources, request_mapping
        finally:
            shm.close()
