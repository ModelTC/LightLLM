from typing import Optional

from lightllm.utils.log_utils import init_logger
from .fp import Fa3AttBackend


logger = init_logger(__name__)


class Fp8Fa3AttBackend(Fa3AttBackend):
    """FA3 over a static per-head FP8 KV cache; scales come per layer in AttControl.kv_scales."""

    kv_quant: Optional[str] = "fp8_per_head"

    def _init_infer_page_size(self):
        # TODO: FP8 FA3 完成多 token 推理页适配后，改为继承模型 page_size。
        self.infer_page_size = 1
        assert self.model.args.page_size % self.infer_page_size == 0, (
            f"model page_size {self.model.args.page_size} "
            f"must be divisible by infer_page_size {self.infer_page_size}"
        )
        logger.warning(
            f"Fp8Fa3AttBackend temporarily uses infer_page_size=1 with model page_size={self.model.args.page_size}; "
            "multi-token FP8 inference pages are not implemented yet."
        )
