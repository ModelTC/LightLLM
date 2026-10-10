import os
from collections import deque
from typing import List
from lightllm.server.core.objs import Req
from lightllm.server.reasoning_parser import ReasoningStopState
from lightllm.utils.envs_utils import get_env_start_args, get_stop_in_reasoning
from lightllm.utils.log_utils import init_logger

logger = init_logger(__name__)


LIGHTLLM_DECODE_PREFIX_LENGTH = int(os.getenv("LIGHTLLM_DECODE_PREFIX_LENGTH", 5))


class DecodeReq:
    def __init__(
        self,
        req: Req,
        is_pd_decode_mode: bool,
        tokenizer,
    ) -> None:
        self.request_id = req.request_id
        self.group_req_id = req.group_req_id
        self.prompt_ids = req.shm_prompt_ids.arr[0 : req.input_len].tolist()
        self.output_ids = []
        self.prefix_offset = max(len(self.prompt_ids) - LIGHTLLM_DECODE_PREFIX_LENGTH, 0)

        if is_pd_decode_mode:
            # pd decode mode 需要模拟一下 prefill 输出的第一个token
            self.read_offset = max(0, len(self.prompt_ids) - 1)
        else:
            self.read_offset = len(self.prompt_ids)

        self.req = req
        self.input_len = self.req.input_len
        self.stop_strs: List[str] = self.req.sample_params.stop_sequences.to_strings()
        # to_strings()已经做了倒序排列，第一个元素就是最长字符串
        self.stop_str_max_len = len(self.stop_strs[0]) if self.stop_strs else 0
        self.stop_str_tail = ""
        # 非 None 表示 reasoning 忽略 stop；当前是否在思考由 in_reasoning 表示。
        self.reasoning_stop_state = None
        self.stop_token_sequences = [
            group.to_list()
            for group in self.req.sample_params.stop_sequences.groups[: self.req.sample_params.stop_sequences.size]
            if group.sequence_str_len == 0
        ]
        self.stop_token_tail = deque(maxlen=max((len(ids) for ids in self.stop_token_sequences), default=1))
        if self.req.sample_params.stop_sequences.size:
            args = get_env_start_args()
            if args.reasoning_parser and not get_stop_in_reasoning():
                self.reasoning_stop_state = ReasoningStopState(
                    args.reasoning_parser,
                    tokenizer,
                    self.req.sample_params._reasoning_status,
                    self.prompt_ids[:-1] if is_pd_decode_mode else self.prompt_ids,
                )
                self.req._in_reasoning = self.reasoning_stop_state.in_reasoning

    def match_stop_sequences(self, token_id, new_text) -> bool:
        # Simulated finish markers must not change the reasoning state of a PD continuation.
        src_index = self.input_len + len(self.output_ids) - 1
        if self.req.finish_token_index == src_index and (
            self.req.finish_status.is_error_finished() or self.req.finish_status.is_finished_pd_decode_capacity()
        ):
            return False
        if self.reasoning_stop_state is not None:
            # 按 token ID 更新reasoning状态；返回值表示当前 token 是否允许参与 stop 匹配。
            can_match_stop = self.reasoning_stop_state.update(token_id)
            self.req._in_reasoning = self.reasoning_stop_state.in_reasoning
            if not can_match_stop:
                self.stop_str_tail = ""
                self.stop_token_tail.clear()
                return False

        tail_str = self.stop_str_tail + new_text
        stop_index = len(tail_str)
        matched_stop = ""
        for stop_str in self.stop_strs:
            index = tail_str.find(stop_str)
            if index != -1 and index < stop_index:
                stop_index = index
                matched_stop = stop_str
        self.stop_str_tail = tail_str[-(self.stop_str_max_len - 1) :] if self.stop_str_max_len > 1 else ""
        if matched_stop:
            stop_end = stop_index + (len(matched_stop) if self.req.sample_params.include_stop_str_in_output else 0)
            self.req.stop_output_offset = stop_end - len(tail_str)
            logger.debug(
                f"req_id {self.request_id} Found stop sequence: stop_str='{matched_stop}', tail_str='{tail_str}'"
            )
            return True

        # 普通模式由推理进程匹配 token stop；思考阶段忽略 stop 时，推理进程跳过匹配，交给这里处理。
        # 思考内容和边界标记已在上面返回并清空尾巴，这里只匹配正文中连续的 token 序列。
        if self.reasoning_stop_state is not None and self.stop_token_sequences:
            self.stop_token_tail.append(int(token_id))
            tail_ids = list(self.stop_token_tail)
            for stop_ids in self.stop_token_sequences:
                # 例如 stop_ids=[101, 102]，仅当最近两个 token 按此顺序连续出现时命中。
                if tail_ids[-len(stop_ids) :] == stop_ids:
                    return True
        return False

    def need_detoken(self):
        if (not self.req.stop_str_matched) and len(self.output_ids) < self.req.candetoken_out_len:
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
        if self.req.stop_str_matched:
            return True
        if (
            self.req.finish_status.is_finished()
            and self.req.candetoken_out_len == len(self.output_ids)
            and self.req.finish_token_index == self.input_len + len(self.output_ids) - 1
        ):
            return True
        return False
