from collections import deque

from lightllm.server.reasoning_parser import ReasoningStopState


class StopSequenceBuffer:
    """Match stops and hold complete tokens until their text is safe to emit.

    普通 HTTP server 和 PD Master 共用；PD Master 跨分段保留缓冲。
    文本截断为空的 token 仍携带原始 payload，以保留 token ID、logprob 和计数。
    """

    def __init__(self, sampling_params, tokenizer, prompt_ids, reasoning_parser=None):
        self.stop_strings = sampling_params.stop_sequences.to_strings()
        self.include_stop = sampling_params.include_stop_str_in_output
        self.match_token_stops = not sampling_params.stop_match_mode.is_backend_all()
        self.stop_prefixes = {stop[:length] for stop in self.stop_strings for length in range(1, len(stop))}
        self.max_prefix_length = max((len(prefix) for prefix in self.stop_prefixes), default=0)
        self.stop_token_sequences = sampling_params.stop_sequences.to_list(include_strings=False)
        self.token_tail = deque(maxlen=max((len(ids) for ids in self.stop_token_sequences), default=1))
        self.reasoning_state = None
        if sampling_params.stop_sequences.size and reasoning_parser:
            self.reasoning_state = ReasoningStopState(
                reasoning_parser,
                tokenizer,
                sampling_params._initial_reasoning_state,
                prompt_ids,
            )
        self.output_strs = deque()
        self.pending_text = ""
        self.ready_tokens = deque()
        self.matched = False

    def append(self, token_id, text, payload, simulated=False):
        """Queue a token; return whether it matches a string or scoped token stop."""
        self.output_strs.append((text, payload))
        self.pending_text += text
        can_match = not simulated
        if can_match and self.reasoning_state is not None:
            can_match = self.reasoning_state.update(token_id)
        if not can_match:
            # reasoning / 模拟结束 token 是匹配边界，不能与边界前的内容拼成 stop。
            self.token_tail.clear()
            self.flush()
            return False

        visible_length = self._match_stop(token_id)
        if visible_length is not None:
            self.flush(visible_length)
            self.matched = True
            return True

        self._release_safe_tokens()
        return False

    def _match_stop(self, token_id):
        """返回匹配后的可见字符数；None 表示未匹配，0 表示全部截断。"""
        stop_index = len(self.pending_text)
        matched_stop = None
        for stop in self.stop_strings:
            index = self.pending_text.find(stop)
            if index != -1 and index < stop_index:
                stop_index, matched_stop = index, stop
        if matched_stop is not None:
            return stop_index + (len(matched_stop) if self.include_stop else 0)

        if self.match_token_stops and self.stop_token_sequences:
            self.token_tail.append(int(token_id))
            tail = list(self.token_tail)
            if any(tail[-len(ids) :] == ids for ids in self.stop_token_sequences):
                return len(self.pending_text)
        return None

    def _release_safe_tokens(self):
        # 末尾仍可能拼成 stop 时暂存覆盖该前缀的完整 token，其余 token 可以发出。
        prefix_length = next(
            (
                length
                for length in range(min(len(self.pending_text), self.max_prefix_length), 0, -1)
                if self.pending_text[-length:] in self.stop_prefixes
            ),
            0,
        )
        safe_length = len(self.pending_text) - prefix_length
        while self.output_strs and len(self.output_strs[0][0]) <= safe_length:
            token_text, payload = self.output_strs.popleft()
            self.ready_tokens.append((token_text, payload))
            safe_length -= len(token_text)
            self.pending_text = self.pending_text[len(token_text) :]

    def flush(self, visible_length=None):
        """Release an unmatched tail, or trim at a confirmed string stop."""
        if visible_length is None:
            visible_length = len(self.pending_text)
        while self.output_strs:
            text, payload = self.output_strs.popleft()
            self.ready_tokens.append((text[: max(0, visible_length)], payload))
            visible_length -= len(text)
        self.pending_text = ""
