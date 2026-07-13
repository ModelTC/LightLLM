import base64
from multiprocessing import shared_memory
from typing import Optional, Tuple

import numpy as np

from lightllm.utils.envs_utils import get_unique_server_name
from lightllm.utils.shm_utils import create_or_link_shm

from .logprob_utils import logprob_info


class ReqFinalTokenMetadata:
    """请求结束时一次性写出的 token 元信息。

    布局保持简单：[prompt_logprobs][routed_experts]。
    读取侧可以从 Req 和 routing 配置推导尺寸，所以 shm 里不再存 header。
    """

    def __init__(self, req):
        self.req = req
        self.shm: Optional[shared_memory.SharedMemory] = None

    @staticmethod
    def prompt_logprobs_dtype(topk: int):
        return np.dtype(
            [
                ("top_token_ids", np.int32, (topk,)),
                ("top_logprobs", np.float32, (topk,)),
            ]
        )

    def shm_name(self) -> str:
        service_uni_name = get_unique_server_name()
        return f"{service_uni_name}_shm_final_token_metadata_{self.req.index_in_shm_mem}"

    def create(
        self,
        prompt_logprobs: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        routed_experts: Optional[np.ndarray] = None,
    ) -> None:
        prompt_nbytes = self._prompt_logprobs_nbytes()
        routing_nbytes = 0
        if prompt_nbytes > 0 and prompt_logprobs is None:
            raise ValueError("prompt_logprobs data is required by request layout")

        if prompt_logprobs is not None:
            prompt_token_ids, prompt_logprobs_values = prompt_logprobs
            prompt_token_ids = np.asarray(prompt_token_ids)
            prompt_logprobs_values = np.asarray(prompt_logprobs_values)
            assert prompt_token_ids.shape == prompt_logprobs_values.shape
            assert prompt_nbytes > 0
            assert self.req.input_len - 1 == prompt_token_ids.shape[0]
            assert self.req.sample_params.prompt_logprobs == prompt_token_ids.shape[1]

        if routed_experts is not None:
            routed_experts = np.asarray(routed_experts)
            routing_nbytes = routed_experts.nbytes

        if prompt_nbytes == 0 and routing_nbytes == 0:
            return

        total_nbytes = prompt_nbytes + routing_nbytes
        self.shm = create_or_link_shm(self.shm_name(), total_nbytes, force_mode="create")

        # prompt top-k 放在前面，routing 紧跟其后；offset 只从请求和配置推导。
        offset = 0
        if prompt_logprobs is not None:
            prompt_array = np.ndarray(
                (self.req.input_len - 1,),
                dtype=self.prompt_logprobs_dtype(self.req.sample_params.prompt_logprobs),
                buffer=self.shm.buf,
                offset=offset,
            )
            prompt_array["top_token_ids"][:] = prompt_logprobs[0]
            prompt_array["top_logprobs"][:] = prompt_logprobs[1]
            offset += prompt_array.nbytes

        if routed_experts is not None:
            routing_array = np.ndarray(
                routed_experts.shape,
                dtype=routed_experts.dtype,
                buffer=self.shm.buf,
                offset=offset,
            )
            routing_array[:] = routed_experts

        self.close_handle()
        return

    def link(self) -> None:
        if self.shm is None:
            self.shm = create_or_link_shm(self.shm_name(), -1, force_mode="link")
        return

    def close_handle(self) -> None:
        if self.shm is not None:
            self.shm.close()
            self.shm = None
        return

    def close_and_unlink(self) -> None:
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            self.shm = None
            return

        try:
            shm = shared_memory.SharedMemory(name=self.shm_name())
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        return

    def link_prompt_logprobs_array(self):
        prompt_nbytes = self._prompt_logprobs_nbytes()
        if prompt_nbytes == 0:
            return None
        self.link()
        return np.ndarray(
            (self.req.input_len - 1,),
            dtype=self.prompt_logprobs_dtype(self.req.sample_params.prompt_logprobs),
            buffer=self.shm.buf,
            offset=0,
        )

    def routed_experts_response(self, num_tokens: int, num_moe_layers: int, topk: int, np_dtype):
        self.link()
        try:
            offset = self._prompt_logprobs_nbytes()
            shape = (num_tokens, num_moe_layers, topk)
            dtype = np.dtype(np_dtype)
            required_nbytes = offset + int(np.prod(shape)) * dtype.itemsize
            if self.shm.size < required_nbytes:
                return None
            routing_data = np.ndarray(shape, dtype=dtype, buffer=self.shm.buf, offset=offset)
            return {
                "shape": list(routing_data.shape),
                "dtype": str(routing_data.dtype),
                "data": base64.b64encode(routing_data.tobytes()).decode("ascii"),
            }
        finally:
            self.close_handle()

    def _prompt_logprobs_nbytes(self) -> int:
        topk = self.req.sample_params.prompt_logprobs
        rows = self.req.input_len - 1
        if topk <= 0 or rows <= 0:
            return 0
        return rows * self.prompt_logprobs_dtype(topk).itemsize

    def prompt_logprobs_response(self, tokenizer=None):
        req = self.req
        prompt_token_ids = req.shm_prompt_ids.arr[: req.input_len]
        prompt_token_ids_list = [int(token_id) for token_id in prompt_token_ids]
        if req.input_len <= 1:
            return {
                "prompt_logprobs": [None],
                "prompt_token_ids": prompt_token_ids_list,
            }

        topk = req.sample_params.prompt_logprobs
        prompt_logprobs = [None]
        if topk == 0:
            # prompt_logprobs=0 返回每个位置真实命中的 prompt token，
            # logprob/rank 存在逐 token 元信息里。
            for token_index in range(1, req.input_len):
                token_id = int(prompt_token_ids[token_index])
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
            return {
                "prompt_logprobs": prompt_logprobs,
                "prompt_token_ids": prompt_token_ids_list,
            }

        try:
            captured = self.link_prompt_logprobs_array()
            if captured is None:
                prompt_logprobs.extend({} for _ in range(req.input_len - 1))
                return {
                    "prompt_logprobs": prompt_logprobs,
                    "prompt_token_ids": prompt_token_ids_list,
                }

            for row_index in range(req.input_len - 1):
                position_logprobs = {}
                for index in range(topk):
                    top_token_id = int(captured["top_token_ids"][row_index, index])
                    if top_token_id >= 0:
                        position_logprobs[top_token_id] = logprob_info(
                            tokenizer,
                            top_token_id,
                            captured["top_logprobs"][row_index, index],
                            index + 1,
                        )

                prompt_logprobs.append(position_logprobs)
        except FileNotFoundError:
            prompt_logprobs.extend({} for _ in range(req.input_len - 1))
        finally:
            self.close_handle()

        return {
            "prompt_logprobs": prompt_logprobs,
            "prompt_token_ids": prompt_token_ids_list,
        }
