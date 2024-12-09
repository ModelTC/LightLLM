# Adapted from
# https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/utils.py
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

from functools import wraps
from torch.library import Library
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Literal,
    Optional,
    OrderedDict,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
)
from typing_extensions import ParamSpec
from lightllm.utils.log_utils import init_logger
import subprocess

vllm_lib = Library("vllm", "FRAGMENT")  # noqa
logger = init_logger(__name__)

_P = ParamSpec("_P")
_R = TypeVar("_R")


def is_full_nvlink() -> bool:
    try:
        output = subprocess.check_output("nvidia-smi topo --matrix", shell=True).decode()
        if "NVLink" in output:
            return True
        else:
            return False
    except Exception:
        return True


def direct_register_custom_op(
    op_name: str,
    op_func: Callable,
    mutates_args: List[str],
    fake_impl: Optional[Callable] = None,
    target_lib: Optional[Library] = None,
):
    """
    `torch.library.custom_op` can have significant overhead because it
    needs to consider complicated dispatching logic. This function
    directly registers a custom op and dispatches it to the CUDA backend.
    See https://gist.github.com/youkaichao/ecbea9ec9fc79a45d2adce1784d7a9a5
    for more details.

    By default, the custom op is registered to the vLLM library. If you
    want to register it to a different library, you can pass the library
    object to the `target_lib` argument.

    IMPORTANT: the lifetime of the operator is tied to the lifetime of the
    library object. If you want to bind the operator to a different library,
    make sure the library object is alive when the operator is used.
    """
    import torch.library

    if hasattr(torch.library, "infer_schema"):
        schema_str = torch.library.infer_schema(op_func, mutates_args=mutates_args)
    else:
        # for pytorch 2.4
        import torch._custom_op.impl

        schema_str = torch._custom_op.impl.infer_schema(op_func, mutates_args)
    my_lib = target_lib or vllm_lib
    my_lib.define(op_name + schema_str)
    my_lib.impl(op_name, op_func, "CUDA")
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
