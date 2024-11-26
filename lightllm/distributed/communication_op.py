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

import torch
import torch.distributed

from .parallel_state import get_tp_group
from torch.distributed import ReduceOp
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)
# if op != ReduceOp.SUM or group != None or async_op != False:
#     logger.warning("This function op, group, async_op will only run with default values")


def all_reduce(input_, op=ReduceOp.SUM, group=None, async_op=False):
    input_.data = tensor_model_parallel_all_reduce(input_)


def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)


def tensor_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)


def tensor_model_parallel_gather(input_: torch.Tensor, dst: int = 0, dim: int = -1) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
