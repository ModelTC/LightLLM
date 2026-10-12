import os
from lightllm.server.core.objs import Req


LIGHTLLM_DECODE_PREFIX_LENGTH = int(os.getenv("LIGHTLLM_DECODE_PREFIX_LENGTH", 5))


class DecodeReq:
    def __init__(
        self,
        req: Req,
    ) -> None:
        self.request_id = req.request_id
        self.group_req_id = req.group_req_id
        self.prompt_ids = req.shm_prompt_ids.arr[0 : req.input_len].tolist()
        self.output_ids = []
        self.prefix_offset = max(len(self.prompt_ids) - LIGHTLLM_DECODE_PREFIX_LENGTH, 0)

        self.read_offset = len(self.prompt_ids)

        self.req = req
        self.input_len = self.req.input_len

    def need_detoken(self):
        return len(self.output_ids) < self.req.candetoken_out_len

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
        return (
            self.req.finish_status.is_finished()
            and self.req.candetoken_out_len == len(self.output_ids)
            and self.req.finish_token_index == self.input_len + len(self.output_ids) - 1
        )
