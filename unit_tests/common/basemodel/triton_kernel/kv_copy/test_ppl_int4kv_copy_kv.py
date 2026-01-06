import torch
import pytest
import numpy as np
from typing import Tuple
from lightllm.common.basemodel.triton_kernel.kv_copy.ppl_int4kv_copy_kv import destindex_copy_int4kv
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)
