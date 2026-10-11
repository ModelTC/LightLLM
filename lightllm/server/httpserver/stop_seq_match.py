from collections import deque
from contextlib import aclosing

from lightllm.server.core.objs import FinishStatus

from lightllm.server.reasoning_parser import ReasoningStopState


class StopSequenceBuffer:
    """Match stops and hold complete tokens until their text is safe to emit.

    应用侧共用；缓冲覆盖 PD Master 的所有生成分段。
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


async def match_stop_sequence_stream(
    generator, sampling_params, tokenizer, prompt, reasoning_parser=None, *, on_stop=None
):
    """应用侧按请求 ID 匹配 stop，独立保留各 choice 的前缀和 reasoning 状态。

    命中后停止对应请求，立即返回 stop 结束结果，忽略后续输出。
    缓冲覆盖整个输出流，不受 PD 内部容量分段影响。
    """
    async with aclosing(generator):
        if not sampling_params.stop_sequences.size:
            async for result in generator:
                yield result
            return

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False) if isinstance(prompt, str) else prompt
        parser = reasoning_parser if sampling_params.stop_match_mode.is_output_content_only() else None
        buffers = {}
        finished_ids = set()
        async for req_id, text, metadata, raw_finish in generator:
            if req_id in finished_ids:
                continue
            if req_id not in buffers:
                buffers[req_id] = StopSequenceBuffer(sampling_params, tokenizer, prompt_ids, parser)
            buffer = buffers[req_id]
            finish = FinishStatus(raw_finish.status)
            matched = buffer.append(
                metadata.get("id"),
                text,
                (req_id, metadata, finish),
                simulated=finish.is_error_finished() or metadata.get("id") is None,
            )
            if matched:
                finish.status = FinishStatus.FINISHED_STOP
                if not raw_finish.is_finished() and on_stop is not None:
                    await on_stop(req_id)
            if finish.is_finished():
                buffer.flush()
                finished_ids.add(req_id)
            while buffer.ready_tokens:
                output_text, (output_id, output_metadata, output_finish) = buffer.ready_tokens.popleft()
                yield output_id, output_text, output_metadata, output_finish
            if req_id in finished_ids:
                del buffers[req_id]
                if len(finished_ids) == sampling_params.n:
                    break
