"""EPLB 原始路由负载的异步汇集任务。"""

from typing import Optional

import torch
import torch.distributed as dist

from .async_task import EPLBAsyncTask


class EPLBLoadGatherTask(EPLBAsyncTask):
    """在独立 Gloo 通信组中汇集每个 rank 的逐样本路由负载。"""

    def __init__(
        self,
        *,
        local_load: torch.Tensor,
        load_gather_group: dist.ProcessGroup,
        world_size: int,
    ) -> None:
        assert local_load.device.type == "cpu"
        assert local_load.ndim == 3
        assert world_size > 0
        self.local_load = local_load.contiguous()
        self.load_gather_group = load_gather_group
        self.world_size = world_size
        self.result: Optional[torch.Tensor] = None
        super().__init__(thread_name="eplb-load-gather")

    def execute(self) -> None:
        """生成 ``[rank, layer, sample, logical_expert]`` 的连续结果。"""
        gathered_load = torch.empty(
            (self.world_size, *self.local_load.shape),
            dtype=self.local_load.dtype,
            device=self.local_load.device,
        )
        load_by_rank = list(gathered_load.unbind(dim=0))
        dist.all_gather(
            load_by_rank,
            self.local_load,
            group=self.load_gather_group,
        )
        self.result = gathered_load
