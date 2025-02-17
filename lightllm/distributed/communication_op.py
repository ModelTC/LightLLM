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

from typing import Any, Dict, Optional, Union

import os
import torch
import torch.distributed as dist
from torch.distributed import ReduceOp
from lightllm.utils.log_utils import init_logger
from lightllm.utils.device_utils import has_nvlink
from lightllm.utils.envs_utils import get_env_start_args

original_all_reduce = torch.distributed.all_reduce
original_all_gather_into_tensor = torch.distributed.all_gather_into_tensor

from contextlib import nullcontext, contextmanager

logger = init_logger(__name__)

try:
    HAS_VLLM = True
    from .custom_all_reduce import CustomAllreduce
except:
    HAS_VLLM = False
    logger.info("vllm or lightllm_kernel is not installed, you can't use custom allreduce")

try:
    HAS_LIGHTLLM_KERNEL = True
    from .custom_all_gather import CustomAllgather
except:
    HAS_LIGHTLLM_KERNEL = False
    logger.info("lightllm_kernel is not installed, you can't use custom allgather")


class CustomCommunicationOp:
    def __init__(self):
        self.vllm_reduce = None
        self.custom_gather = None
        self.device_group = None

    @contextmanager
    def lightllm_capture_graph(self):
        if self.vllm_reduce is not None:
            with self.vllm_reduce.capture():
                if self.custom_gather is not None:
                    with self.custom_gather.capture():
                        yield
                else:
                    yield
        else:
            yield

    def set_custom_reduce(self):
        ENABLE_VLLM_REDUCE = os.getenv("ENABLE_VLLM_REDUCE", "True").upper() in ["ON", "TRUE", "1"]
        world_size = dist.get_world_size()
        ranks = list(range(world_size))

        if not has_nvlink() or world_size not in [2, 4, 6, 8]:
            ENABLE_VLLM_REDUCE = False

        # 创建新的 NCCL 组以防止原始 all_reduce 与 cudagraph 卡住
        if self.device_group is None:
            self.device_group = dist.new_group(ranks, backend="nccl")

        if ENABLE_VLLM_REDUCE and HAS_VLLM:
            cpu_group = dist.new_group(ranks, backend="gloo")
            self.vllm_reduce = CustomAllreduce(cpu_group, torch.cuda.current_device())
            logger.info("Enable VLLM ALLReduce.")

        def _all_reduce_closure(input_, op=ReduceOp.SUM, group=self.device_group, async_op=False):
            if op != ReduceOp.SUM or async_op:
                original_all_reduce(input_, op, group, async_op)
            else:
                if self.vllm_reduce is not None and self.vllm_reduce.should_custom_ar(input_):
                    input_.data = self.vllm_reduce.custom_all_reduce(input_)
                else:
                    original_all_reduce(input_, op, group, async_op)

        dist.all_reduce = _all_reduce_closure

    def set_custom_gather(self):
        ENABLE_CUSTOM_GATHER = os.getenv("ENABLE_CUSTOM_GATHER", "False").upper() in ["ON", "TRUE", "1"]
        args = get_env_start_args()
        world_size = dist.get_world_size()
        ranks = list(range(world_size))
        if self.device_group is None:
            self.device_group = dist.new_group(ranks, backend="nccl")

        if ENABLE_CUSTOM_GATHER and HAS_LIGHTLLM_KERNEL or args.disable_custom_allreduce:
            cpu_group = dist.new_group(ranks, backend="gloo")
            self.custom_gather = CustomAllgather(cpu_group, torch.cuda.current_device())
            logger.info("Enable Custom ALLGather.")

        def _all_gather_closure(output_, input_, group=self.device_group, async_op=False):
            if async_op:
                original_all_gather_into_tensor(output_, input_, group, async_op)
            else:
                if self.custom_gather is not None and self.custom_gather.should_custom_ar(input_):
                    self.custom_gather.custom_all_gather(output_, input_)
                else:
                    original_all_gather_into_tensor(output_, input_, group, async_op)

        dist.all_gather_into_tensor = _all_gather_closure


custom_comm_ops = CustomCommunicationOp()
