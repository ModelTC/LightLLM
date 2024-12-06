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
from functools import partial

original_all_reduce = torch.distributed.all_reduce
from contextlib import nullcontext, contextmanager

try:
    HAS_VLLM = True
    from .custom_all_reduce import CustomAllreduce
except:
    HAS_VLLM = False

vllm_reduce = None
logger = init_logger(__name__)


@contextmanager
def lightllm_capture_graph():
    if vllm_reduce is not None:
        with vllm_reduce.capture():
            yield
    else:
        yield
    pass


def set_custom_reduce():
    global vllm_reduce
    global device_group
    ENABLE_VLLM_REDUCE = os.getenv("ENABLE_VLLM_REDUCE", "False").upper() in [
        "ON",
        "TRUE",
        "1",
    ]
    world_size = dist.get_world_size()
    ranks = list(range(world_size))
    # new_group prevent stuck of torch origin all_reduce with cudagraph
    device_group = torch.distributed.new_group(ranks, backend="nccl")
    if ENABLE_VLLM_REDUCE and HAS_VLLM:
        cpu_group = torch.distributed.new_group(ranks, backend="gloo")
        vllm_reduce = CustomAllreduce(cpu_group, torch.cuda.current_device())
        logger.info("Enable VLLM ALLReduce.")

    def _all_reduce_closure(input_, op=ReduceOp.SUM, group=device_group, async_op=False):
        if op != ReduceOp.SUM or async_op:
            original_all_reduce(input_, op, group, async_op)
        else:
            if vllm_reduce is not None and vllm_reduce.should_custom_ar(input_):
                input_.data = vllm_reduce.custom_all_reduce(input_)
            else:
                original_all_reduce(input_, op, group, async_op)

    dist.all_reduce = _all_reduce_closure
