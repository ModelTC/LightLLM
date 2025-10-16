import os
from typing import List, Dict
from lightllm.server.core.objs import Req
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


LIGHTLLM_DECODE_PREFIX_LENGTH = int(os.getenv("LIGHTLLM_DECODE_PREFIX_LENGTH", 5))


class DecodeReq:
    def __init__(
        self,
        req: Req,
        is_pd_decode_mode: bool,
    ) -> None:
        self.request_id = req.request_id
        self.group_req_id = req.group_req_id
        self.prompt_ids = req.shm_prompt_ids.arr[0 : req.input_len].tolist()
        self.output_ids = []
        self.output_strs = []
        self.prefix_offset = max(len(self.prompt_ids) - LIGHTLLM_DECODE_PREFIX_LENGTH, 0)

        if is_pd_decode_mode:
            # pd decode mode 需要模拟一下 prefill 输出的第一个token
            self.read_offset = max(0, len(self.prompt_ids) - 1)
        else:
            self.read_offset = len(self.prompt_ids)

        self.req = req
        self.input_len = self.req.input_len
        self.prefix_str = ""
        self.stop_strs: List[str] = self.req.sample_params.stop_sequences.to_strings()
        # to_strings()已经做了倒序排列，第一个元素就是最长字符串
        self.stop_str_max_len = len(self.stop_strs[0]) if self.stop_strs else 0

    def init_token_healing_prefix_str(self, token_id_to_token: Dict[int, str], tokenizer):
        tokens = [token_id_to_token[token_id] for token_id in self.req.prefix_token_ids.get_token_ids()]
        if tokens:
            self.prefix_str = tokenizer.convert_tokens_to_string(tokens)
        else:
            self.prefix_str = ""
        return

    def stop_sequences_str_match(self) -> bool:
        stop_strs = self.stop_strs
        if not stop_strs or self.stop_str_max_len == 0:
            return False

        tail_token_len = self.stop_str_max_len + 10  # 10 for safety
        tail_token_strs = self.output_strs[-tail_token_len:]
        tail_str = "".join(tail_token_strs)

        for stop_str in stop_strs:
            if stop_str in tail_str:
                logger.debug(
                    f"req_id {self.request_id} Found stop sequence in tail: stop_str='{stop_str}', "
                    f"tail_str='{tail_str}'"
                )
                return True
        return False

    def need_detoken(self):
        if (
            (not self.req.is_aborted)
            and (not self.req.stop_str_matched)
            and len(self.output_ids) < self.req.candetoken_out_len
        ):
            return True
        return False

    def out_queue_is_full(self):
        return self.req.out_tokens_queue.is_full()

    def get_next_token_id_and_index(self):
        src_index = self.input_len + len(self.output_ids)
        return self.req.shm_prompt_ids.arr[src_index], src_index

    def get_decode_tokens(self):
        prefix_tokens = self.req.shm_prompt_ids.arr[self.prefix_offset : self.read_offset].tolist()
        read_tokens = self.req.shm_prompt_ids.arr[self.prefix_offset : self.input_len + len(self.output_ids)].tolist()
        return prefix_tokens, read_tokens

    def can_set_release_mark(self):
        if self.req.is_aborted:
            return True
        if self.req.stop_str_matched:
            return True
        if (
            self.req.finish_status.is_finished()
            and self.req.candetoken_out_len == len(self.output_ids)
            and self.req.finish_token_index == self.input_len + len(self.output_ids) - 1
        ):
            return True
        return False
