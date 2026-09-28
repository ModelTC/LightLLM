"""EPLB 原始路由统计的异步汇集任务。"""

from typing import Optional

import torch
import torch.distributed as dist

from .async_task import EPLBAsyncTask


class EPLBLoadGatherTask(EPLBAsyncTask):
    """在独立 Gloo 通信组中汇集每个 rank 的三维路由统计。"""

    def __init__(
        self,
        local_route_statistics: torch.Tensor,
        load_gather_group: dist.ProcessGroup,
    ) -> None:
        assert local_route_statistics.device.type == "cpu"
        assert local_route_statistics.ndim == 3
        self.local_route_statistics = local_route_statistics.contiguous()
        self.load_gather_group = load_gather_group
        self.world_size = dist.get_world_size(group=load_gather_group)
        assert self.world_size > 0
        self.result: Optional[torch.Tensor] = None
        super().__init__(thread_name="eplb-load-gather")

    def execute(self) -> None:
        """汇集各 rank 的统计：prefill 为 ``[rank, layer, sample, expert_num]``，
        decode 为 ``[rank, layer, expert_num, expert_num]`` 的上三角共现矩阵。
        """
        gathered_route_statistics = torch.empty(
            (self.world_size, *self.local_route_statistics.shape),
            dtype=self.local_route_statistics.dtype,
            device=self.local_route_statistics.device,
        )
        route_statistics_by_rank = list(gathered_route_statistics.unbind(dim=0))
        dist.all_gather(
            route_statistics_by_rank,
            self.local_route_statistics,
            group=self.load_gather_group,
        )
        self.result = gathered_route_statistics
