import os
import torch
from .impl import ChunkedPrefillBackend
from typing import List
from lightllm.server.router.model_infer.infer_batch import g_infer_context, InferReq
from lightllm.server.router.model_infer.mode_backend.continues_batch.impl import ContinuesBatchBackend
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


class FirstTokenConstraintBackend(ChunkedPrefillBackend):
    def __init__(self) -> None:
        super().__init__()
        self.prefill_mask_func = self._mask_first_gen_token_logits
        self.decode_mask_func = self._mask_first_gen_token_logits
        self.extra_post_req_handle_func = None

    def init_custom(self):
        first_allowed_tokens_strs: str = os.environ.get("FIRST_ALLOWED_TOKENS", None)
        logger.info(f"first_allowed_tokens_strs : {first_allowed_tokens_strs}")
        # 使用该模式需要设置FIRST_ALLOWED_TOKENS 环境变量，格式为 "1,2" 或 "1,2,3"  等数字字符串
        assert first_allowed_tokens_strs is not None
        first_allowed_tokens_strs.split(",")
        self.first_allowed_tokens = [int(e.strip()) for e in first_allowed_tokens_strs.split(",") if len(e.strip()) > 0]
        logger.info(f"first_allowed_tokens : {self.first_allowed_tokens}")
        # check token_id < vocab_size
        assert all(e < self.model.vocab_size for e in self.first_allowed_tokens)
        self.fill_value = torch.tensor(-1000000.0)
        return

    def _mask_first_gen_token_logits(self, run_reqs: List[InferReq], logits: torch.Tensor):
        # 这个函数中的实现会造成全局同步，造成折叠失效，主要是
        # mask[i, self.first_allowed_tokens] 切片复制还有
        # 后续出现 .cpu() .cuda() 等操作也会造成异常的全局同步
        # to do remove all zeros_like, ones_like, use triton kernel
        # replace
        if any([req.cur_output_len == 0 for req in run_reqs]):
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for i, req in enumerate(run_reqs):
                if req.cur_output_len == 0:
                    mask[i, :] = True
                    mask[i, self.first_allowed_tokens] = False
            # 不能使用 logits[mask] = -1000000.0
            # 会存在诡异的多流异步问题, 可能是torch的bug
            new_logits = torch.where(mask, self.fill_value, logits)
            logits.copy_(new_logits)
        return
