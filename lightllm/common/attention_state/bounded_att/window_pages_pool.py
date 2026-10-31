"""单种时间有序窗口快照的页池，槽位由 manager 的共享 allocator 管理。"""

import torch

from .utils import create_page_pool_tensor, register_page_pool_tensor


class WindowPagesPool:
    def __init__(
        self,
        size: int,
        snapshot_shape: tuple[int, int, int],
        dtype: torch.dtype,
        device: str | torch.device | int = "cpu",
        pin_memory: bool = True,
    ) -> None:
        # [页槽位, 物理层, 时间窗口, 字段宽度]；只保存快照，不保存请求或长度元数据。
        self.buffer = create_page_pool_tensor(
            size=(size, *snapshot_shape),
            dtype=dtype,
            device=device,
            pin_memory=pin_memory,
        )

    def register_pin_memory(self) -> None:
        """跨 DP 导入后，在接收进程当前设备上注册共享页池。"""
        register_page_pool_tensor(tensor=self.buffer)
