from typing import Optional

import torch

from .fp import FlashInferAttBackend
from .env_utils import set_flashinfer_envs
from lightllm.utils.log_utils import init_logger


logger = init_logger(__name__)


class Fp8FlashInferAttBackend(FlashInferAttBackend):
    """FlashInfer over a static per-tensor FP8 KV cache; scales come per layer in AttControl.kv_scales."""

    kv_quant: Optional[str] = "fp8_per_tensor"

    def __init__(self, model):
        set_flashinfer_envs()
        super().__init__(model=model)

    def _kv_data_type(self) -> torch.dtype:
        return torch.float8_e4m3fn

    def _init_infer_page_size(self):
        # TODO: FP8 FlashInfer 完成多 token 推理页适配后，改为继承模型 page_size。
        self.infer_page_size = 1
        assert self.model.args.page_size % self.infer_page_size == 0, (
            f"model page_size {self.model.args.page_size} "
            f"must be divisible by infer_page_size {self.infer_page_size}"
        )
        logger.warning(
            "Fp8FlashInferAttBackend temporarily uses infer_page_size=1 with "
            f"model page_size={self.model.args.page_size}; multi-token FP8 inference pages are not implemented yet."
        )
