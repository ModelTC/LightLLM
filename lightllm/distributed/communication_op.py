# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.6.3.post1/vllm/distributed/communication_op.py
# of the vllm-project/vllm GitHub repository.
#
# Copyright 2023 ModelTC Team
# Copyright 2023 vLLM Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp, ProcessGroup
from typing import List, Dict, Optional, Union
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import has_nvlink
from lightllm.utils.envs_utils import (
    get_env_start_args,
    get_deepep_num_max_dispatch_tokens_per_rank,
    get_redundancy_expert_num,
)
from lightllm.utils.dist_utils import (
    get_global_world_size,
    get_dp_world_size,
    get_global_rank,
    get_current_rank_in_dp,
    create_new_group_for_current_dp,
)
from lightllm.utils.device_utils import get_device_sm_count
from lightllm.utils.sgl_utils import HAS_SGL_KERNEL
from lightllm.utils.light_utils import HAS_LIGHTLLM_KERNEL
from contextlib import nullcontext, contextmanager

logger = init_logger(__name__)

from .custom_all_reduce import CustomAllreduce
from .custom_all_gather import CustomAllgather

try:
    import deep_ep

    HAS_DEEPEP = True
except:
    HAS_DEEPEP = False
    logger.info("deep_ep is not installed, you can't use the api of it.")


class CustomProcessGroup:
    def __init__(self):
        self.custom_reduce = None
        self.custom_gather = None
        self.dp_world_size = get_dp_world_size()
        self.device_group = create_new_group_for_current_dp("nccl")
        self.autotune_group = dist.new_group([i for i in range(get_global_world_size())], backend="gloo")

    def init_custom_reduce(self) -> None:
        if not HAS_SGL_KERNEL or not has_nvlink() or self.dp_world_size not in [2, 4, 6, 8]:
            return
        args = get_env_start_args()
        if args.disable_custom_allreduce:
            return
        cpu_group = create_new_group_for_current_dp("gloo")
        self.custom_reduce = CustomAllreduce(cpu_group, torch.cuda.current_device())
        logger.info("Enable Custom ALLReduce. You can disable it by settting --disable_custom_allreduce.")

    def init_custom_gather(self) -> None:
        if not HAS_LIGHTLLM_KERNEL or not has_nvlink() or self.dp_world_size not in [2, 4, 6, 8]:
            return

        args = get_env_start_args()
        if not args.enable_custom_allgather:
            return

        cpu_group = create_new_group_for_current_dp("gloo")
        self.custom_gather = CustomAllgather(cpu_group, torch.cuda.current_device())
        logger.info("Enable Custom ALLGather.  You can disable it by settting --disable_custom_allgather")

    def all_reduce(self, input_: torch.Tensor) -> None:
        if self.custom_reduce is not None and self.custom_reduce.should_custom_ar(input_):
            input_.data = self.custom_reduce.custom_all_reduce(input_)
            return
        else:
            return dist.all_reduce(input_, group=self.device_group)

    def all_gather_into_tensor(self, output_: torch.Tensor, input_: torch.Tensor, async_op: bool = False) -> None:
        if self.custom_gather is not None and self.custom_gather.should_custom_ar(input_):
            self.custom_gather.custom_all_gather(output_, input_)
            return
        else:
            return dist.all_gather_into_tensor(output_, input_, group=self.device_group, async_op=async_op)


@contextmanager
def lightllm_capture_graph(group: CustomProcessGroup = None):
    with group.custom_reduce.capture() if group and group.custom_reduce else nullcontext():
        with group.custom_gather.capture() if group and group.custom_gather else nullcontext():
            yield


class DistributeGroupManager:
    def __init__(self):
        self.groups = []

    def __len__(self):
        return len(self.groups)

    def create_groups(self, group_size: int):
        for i in range(group_size):
            group = CustomProcessGroup()
            group.init_custom_gather()
            group.init_custom_reduce()
            self.groups.append(group)
        return

    def get_default_group(self) -> CustomProcessGroup:
        return self.groups[0]

    def get_group(self, group_index: int) -> CustomProcessGroup:
        return self.groups[group_index]

    def new_deepep_group(self, n_routed_experts, hidden_size):
        moe_mode = os.getenv("MOE_MODE", "TP")
        num_max_dispatch_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank()
        if moe_mode == "TP":
            self.ep_buffer = None
            return
        assert HAS_DEEPEP, "deep_ep is required for expert parallelism"
        self._set_num_sms_for_deep_gemm()

        global_world_size = get_global_world_size()
        deepep_group = dist.new_group(list(range(global_world_size)))
        low_latency_mode, num_rdma_bytes = True, 0
        if low_latency_mode:
            self.ll_num_tokens, self.ll_hidden = num_max_dispatch_tokens_per_rank, hidden_size
            self.ll_num_experts = n_routed_experts + get_redundancy_expert_num() * global_world_size
            num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
                self.ll_num_tokens, self.ll_hidden, global_world_size, self.ll_num_experts
            )
        self.ep_buffer = deep_ep.Buffer(
            deepep_group,
            int(1e9),
            num_rdma_bytes,
            low_latency_mode=low_latency_mode,
            num_qps_per_rank=(self.ll_num_experts // global_world_size if low_latency_mode else 1),
        )

    def _set_num_sms_for_deep_gemm(self):
        try:
            # set num sms for deep_gemm
            from deep_gemm.jit_kernels.utils import set_num_sms

            deepep_sms = int(os.getenv("DEEPEP_SMS", deep_ep.Buffer.num_sms))
            device_sms = get_device_sm_count()
            deep_ep.Buffer.set_num_sms(deepep_sms)
            set_num_sms(device_sms - deepep_sms)
        except:
            pass

    def clear_deepep_buffer(self):
        """
        prefill 之后需要clean 一下，ep buffer 才能正常执行 decode。
        """
        if hasattr(self, "ep_buffer") and self.ep_buffer is not None:
            self.ep_buffer.clean_low_latency_buffer(self.ll_num_tokens, self.ll_hidden, self.ll_num_experts)


def all_reduce(
    input_: torch.Tensor,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    op: ReduceOp = ReduceOp.SUM,
    async_op: bool = False,
) -> None:
    if isinstance(group, CustomProcessGroup):
        return group.all_reduce(input_)
    else:
        return dist.all_reduce(input_, op, group, async_op)


def all_gather_into_tensor(
    output_: torch.Tensor,
    input_: torch.Tensor,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    async_op: bool = False,
) -> None:
    if isinstance(group, CustomProcessGroup):
        return group.all_gather_into_tensor(output_, input_)
    else:
        return dist.all_gather_into_tensor(output_, input_, group, async_op)


def all_gather(
    output_: List[torch.Tensor],
    input_: torch.Tensor,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    async_op: bool = False,
) -> None:
    # todo 目前还没有定制算子的支持。
    if isinstance(group, CustomProcessGroup):
        return dist.all_gather(output_, input_, group.device_group, async_op)
    else:
        return dist.all_gather(output_, input_, group, async_op)


def reduce_scatter_tensor(
    output: torch.Tensor,
    input: torch.Tensor,
    op: ReduceOp = ReduceOp.SUM,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    async_op=False,
):
    # 目前还没有定制算子实现。
    if isinstance(group, CustomProcessGroup):
        return dist.reduce_scatter_tensor(output, input, op=op, group=group.device_group, async_op=async_op)
    else:
        return dist.reduce_scatter_tensor(output, input, op=op, group=group, async_op=async_op)


dist_group_manager = DistributeGroupManager()
