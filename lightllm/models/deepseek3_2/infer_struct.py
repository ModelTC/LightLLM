import os
import torch
import numpy as np
import torch.distributed as dist
from lightllm.models.deepseek2.flashattention_infer_struct import Deepseek2FlashAttentionStateInfo


class Deepseek3_2FlashAttentionInferStateInfo(Deepseek2FlashAttentionStateInfo):
    pass