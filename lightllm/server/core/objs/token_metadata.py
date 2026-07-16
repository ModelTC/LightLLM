import base64
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.log_utils import init_logger
from lightllm.utils.shm_utils import create_or_link_shm

from .logprob_utils import logprob_info

logger = init_logger(__name__)


class ReqFinalTokenMetadata:
    """请求结束时一次性写出的 token 元信息（shm + pickle）。

    对外接口只有 ``save`` / ``read``。
    """

    def __init__(self, req):
        self.req = req

    def save(
        self,
        prompt_top_token_ids: Optional[np.ndarray] = None,
        prompt_top_logprobs: Optional[np.ndarray] = None,
        routed_experts: Optional[np.ndarray] = None,
    ) -> None:
        if prompt_top_token_ids is None and prompt_top_logprobs is None and routed_experts is None:
            return

        has_token_ids = prompt_top_token_ids is not None
        has_logprobs = prompt_top_logprobs is not None
        if has_token_ids != has_logprobs:
            raise ValueError("prompt_top_token_ids and prompt_top_logprobs must be set together")

        payload = {
            "prompt_top_token_ids": prompt_top_token_ids,
            "prompt_top_logprobs": prompt_top_logprobs,
            "routed_experts": routed_experts,
        }
        blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        shm = create_or_link_shm(self._shm_name(), len(blob), force_mode="create")
        try:
            shm.buf[: len(blob)] = blob
        finally:
            shm.close()

    def read(self, tokenizer=None) -> Dict[str, Any]:
        """读取并组装 HTTP 侧需要的 metadata。

        Returns:
            dict with keys:
              - prompt_logprobs / prompt_token_ids（始终返回；无数据时为空占位）
              - routed_experts（无数据时为 None）
        """
        packed_prompt_ids = None
        packed_prompt_logprobs = None
        packed_routed = None
        shm = None
        try:
            shm = create_or_link_shm(self._shm_name(), -1, force_mode="link")
            payload = pickle.loads(shm.buf)
            packed_prompt_ids = payload.get("prompt_top_token_ids")
            packed_prompt_logprobs = payload.get("prompt_top_logprobs")
            packed_routed = payload.get("routed_experts")
        except BaseException as e:
            logger.warning(
                f"Failed to read final token metadata shm for req "
                f"{getattr(self.req, 'request_id', None)}, name={self._shm_name()}: {type(e).__name__}: {e}"
            )
        finally:
            if shm is not None:
                shm.close()
                shm.unlink()

        return {
            "prompt_token_ids": [int(x) for x in self.req.shm_prompt_ids.arr[: self.req.input_len]],
            "prompt_logprobs": self._build_prompt_logprobs_response(
                tokenizer=tokenizer,
                packed_prompt_ids=packed_prompt_ids,
                packed_prompt_logprobs=packed_prompt_logprobs,
            ),
            "routed_experts": self._build_routed_experts_response(packed_routed),
        }

    def _build_prompt_logprobs_response(
        self,
        tokenizer,
        packed_prompt_ids: Optional[np.ndarray],
        packed_prompt_logprobs: Optional[np.ndarray],
    ) -> List[Any]:
        """组装 OpenAI 风格的 prompt_logprobs 列表。

        场景说明：
        1. ``input_len <= 1``：没有可预测的 prompt 位置（首 token 无前文），
           仅返回 ``[None]`` 占位。
        2. ``prompt_logprobs == 0``：不要求 top-k，只返回每个位置**真实命中**的
           prompt token；其 logprob/rank 写在逐 token 的 ``shm_logprobs`` 里，
           不依赖本类 shm 中的 pickle 载荷。
        3. ``prompt_logprobs > 0``：返回每个位置的 top-k 候选。数据来自 Infer
           侧 ``save`` 写入的 ``prompt_top_token_ids/logprobs``；若 shm 缺失或
           未写入对应字段，则用空 dict 占位，保证列表长度仍为 ``input_len``。
        """
        req = self.req
        topk = req.sample_params.prompt_logprobs
        if req.input_len <= 1:
            return [None]

        if topk == 0:
            # prompt_logprobs=0 返回每个位置真实命中的 prompt token，
            # logprob/rank 存在逐 token 元信息里。
            prompt_logprobs = [None]
            for token_index in range(1, req.input_len):
                token_id = int(req.shm_prompt_ids.arr[token_index])
                rank = int(req.shm_logprobs.arr["rank"][token_index])
                rank = None if rank < 0 else rank
                prompt_logprobs.append(
                    {
                        token_id: logprob_info(
                            tokenizer,
                            token_id,
                            req.shm_logprobs.arr["logprob"][token_index],
                            rank,
                        )
                    }
                )
            return prompt_logprobs

        if topk > 0:
            prompt_logprobs = [None]
            if packed_prompt_ids is None or packed_prompt_logprobs is None:
                prompt_logprobs.extend({} for _ in range(req.input_len - 1))
                return prompt_logprobs

            rows = min(req.input_len - 1, packed_prompt_ids.shape[0])
            use_topk = min(topk, packed_prompt_ids.shape[1])
            for row_index in range(rows):
                position_logprobs = {}
                for index in range(use_topk):
                    top_token_id = int(packed_prompt_ids[row_index, index])
                    if top_token_id >= 0:
                        position_logprobs[top_token_id] = logprob_info(
                            tokenizer,
                            top_token_id,
                            packed_prompt_logprobs[row_index, index],
                            index + 1,
                        )
                prompt_logprobs.append(position_logprobs)
            if rows < req.input_len - 1:
                prompt_logprobs.extend({} for _ in range(req.input_len - 1 - rows))
            return prompt_logprobs

        # prompt_logprobs < 0：请求未开启该字段，仍给最小占位，避免调用方 KeyError。
        return [None]

    def _build_routed_experts_response(self, packed_routed: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """组装 HTTP 响应中的 routed_experts 字段。

        场景说明：
        1. Infer 未开启 ``--enable_return_routed_experts``，或该请求未写入
           routing 数据时，``packed_routed`` 为 None，直接返回 None。
        2. 有数据时，将 ndarray 编码为 ``{shape, dtype, data}``：``data`` 为
           C-order 原始字节的 base64，供 HTTP / 客户端按 shape+dtype 还原，
           避免在 JSON 里展开巨大嵌套列表。
        """
        if packed_routed is None:
            return None
        return {
            "shape": list(packed_routed.shape),
            "dtype": str(packed_routed.dtype),
            "data": base64.b64encode(packed_routed.tobytes()).decode("ascii"),
        }

    def _shm_name(self) -> str:
        service_uni_name = get_unique_server_name()
        return f"{service_uni_name}_shm_final_token_metadata_{self.req.index_in_shm_mem}"
