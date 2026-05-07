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
    create_new_group_for_current_dp,
    create_dp_special_inter_group,
)
from lightllm.utils.device_utils import get_device_sm_count
from lightllm.utils.torch_dtype_utils import get_torch_dtype

logger = init_logger(__name__)


try:
    import deep_ep

    HAS_DEEPEP = True
except:
    HAS_DEEPEP = False
    logger.info("deep_ep is not installed, you can't use the api of it.")


class CustomProcessGroup:
    def __init__(self):
        self.symm_mem_reduce = None
        self.flashinfer_reduce = None
        self.dp_world_size = get_dp_world_size()
        self.device_group = create_new_group_for_current_dp("nccl")
        if get_env_start_args().enable_dp_prefill_balance:
            self.dp_prefill_balance_group = create_dp_special_inter_group("nccl")
        else:
            self.dp_prefill_balance_group = None

        self.autotune_group = dist.new_group([i for i in range(get_global_world_size())], backend="gloo")

    def _support_custom_allreduce(self) -> bool:
        return has_nvlink() and self.dp_world_size in [2, 4, 6, 8]

    def init_symm_mem_reduce(self) -> None:
        if not self._support_custom_allreduce():
            return
        from .symm_mem_all_reduce import SymmMemAllreduce

        data_type = get_torch_dtype(get_env_start_args().data_type)
        symm = SymmMemAllreduce(self.device_group, torch.cuda.current_device(), dtype=data_type)
        if not symm.disabled:
            self.symm_mem_reduce = symm
            logger.info("Enable SymmMem ALLReduce.")

    def init_flashinfer_reduce(self) -> None:
        if not self._support_custom_allreduce():
            return
        from .flashinfer_all_reduce import FlashInferAllReduce

        fi_cpu_group = create_new_group_for_current_dp("gloo")
        fi = FlashInferAllReduce(fi_cpu_group, torch.cuda.current_device())
        if not fi.disabled:
            self.flashinfer_reduce = fi
            logger.info("Enable FlashInfer ALLReduce.")

    def all_reduce(self, input_: torch.Tensor) -> None:
        # Dispatch chain: FlashInfer -> SymmMem -> NCCL.
        if self.flashinfer_reduce is not None and self.flashinfer_reduce.should_use(input_):
            input_.data = self.flashinfer_reduce.all_reduce(input_)
            return
        if self.symm_mem_reduce is not None and self.symm_mem_reduce.should_use(input_):
            self.symm_mem_reduce.all_reduce(input_)
            return
        return dist.all_reduce(input_, group=self.device_group)

    def all_gather_into_tensor(self, output_: torch.Tensor, input_: torch.Tensor, async_op: bool = False) -> None:
        return dist.all_gather_into_tensor(output_, input_, group=self.device_group, async_op=async_op)


class DistributeGroupManager:
    def __init__(self):
        self.groups = []

    def __len__(self):
        return len(self.groups)

    def create_groups(self, group_size: int):
        args = get_env_start_args()
        for i in range(group_size):
            group = CustomProcessGroup()
            if not args.disable_symm_mem_allreduce:
                group.init_symm_mem_reduce()
            if not args.disable_flashinfer_allreduce:
                group.init_flashinfer_reduce()
            self.groups.append(group)
        return

    def get_default_group(self) -> CustomProcessGroup:
        return self.groups[0]

    def get_group(self, group_index: int) -> CustomProcessGroup:
        return self.groups[group_index]

    def new_deepep_group(self, n_routed_experts, hidden_size):
        enable_ep_moe = get_env_start_args().enable_ep_moe
        num_max_dispatch_tokens_per_rank = get_deepep_num_max_dispatch_tokens_per_rank()
        if not enable_ep_moe:
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
            try:
                from deep_gemm.jit_kernels.utils import set_num_sms
            except:
                from deep_gemm import set_num_sms

            deepep_sms = int(os.getenv("DEEPEP_SMS", deep_ep.Buffer.num_sms))
            device_sms = get_device_sm_count()
            deep_ep.Buffer.set_num_sms(deepep_sms)
            set_num_sms(device_sms - deepep_sms)
        except BaseException as e:
            logger.warning(f"set num sms for deep_gemm failed: {e}")

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
    if _is_single_group(group=group):
        return
    if isinstance(group, CustomProcessGroup):
        if op == ReduceOp.SUM:
            return group.all_reduce(input_)
        return dist.all_reduce(input_, op, group.device_group, async_op)
    return dist.all_reduce(input_, op, group, async_op)


def all_gather_into_tensor(
    output_: torch.Tensor,
    input_: torch.Tensor,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    async_op: bool = False,
) -> None:
    if _is_single_group(group=group):
        output_.copy_(input_)
        return
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
    if _is_single_group(group=group):
        if len(output_) > 0:
            output_[0].copy_(input_)
        return
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
    if _is_single_group(group=group):
        output.copy_(input)
        return
    # 目前还没有定制算子实现。
    if isinstance(group, CustomProcessGroup):
        return dist.reduce_scatter_tensor(output, input, op=op, group=group.device_group, async_op=async_op)
    else:
        return dist.reduce_scatter_tensor(output, input, op=op, group=group, async_op=async_op)


def broadcast(
    tensor: torch.Tensor,
    src: int,
    group: Optional[Union[ProcessGroup, CustomProcessGroup]] = None,
    async_op: bool = False,
) -> None:
    if _is_single_group(group=group):
        return
    if isinstance(group, CustomProcessGroup):
        return dist.broadcast(tensor, src=src, group=group.device_group, async_op=async_op)
    else:
        return dist.broadcast(tensor, src=src, group=group, async_op=async_op)


def _is_single_group(group: Optional[Union[ProcessGroup, CustomProcessGroup]]) -> bool:
    if isinstance(group, CustomProcessGroup):
        return group.dp_world_size == 1
    else:
        return dist.get_world_size(group=group) == 1


dist_group_manager = DistributeGroupManager()
