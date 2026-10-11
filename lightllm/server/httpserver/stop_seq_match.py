from collections import deque
from contextlib import aclosing
from enum import Enum, auto
from typing import TYPE_CHECKING, AsyncGenerator

from lightllm.server.core.objs import FinishStatus, SamplingParams

from lightllm.server.reasoning_parser import ReasoningParser

if TYPE_CHECKING:
    from lightllm.server.httpserver.manager import HttpServerManager
    from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster


class ReasoningStopState:
    """按 token ID 识别思维链 start/end 标记，判断当前 token 是否可以匹配 stop。

    handle(token_id) 按当前状态分派，每个 _handle_* 函数直接处理 token_id。
    CONTENT：正文阶段，普通 token 可以参与 stop 匹配。
    REASONING：正在输出思维链内容，不匹配 stop。

    当前状态     | 当前 token 与最近 token 的匹配情况            | 下一状态     | 允许匹配 stop
    -------------|---------------------------------------------|--------------|--------------
    CONTENT      | 完整命中思维链 start 标记                    | REASONING    | 否
    CONTENT      | 未完整命中思维链 start 标记                  | CONTENT      | 是
    REASONING    | 完整命中思维链 end 标记                      | CONTENT      | 否
    REASONING    | 未完整命中思维链 end 标记                    | REASONING    | 否

    仅完整命中标记才改变状态；尚未完整的标记按当前阶段处理。
    最近 token 包含 prompt 尾部，支持跨 token、prompt 和 P/D 分段的完整标记匹配。
    prompt 尾部已完整命中思维链 start 标记时，初始化就进入 REASONING。
    """

    class State(Enum):
        CONTENT = auto()
        REASONING = auto()

    def __init__(self, reasoning_parser, tokenizer, initial_reasoning_state, tail_prompt_ids):
        detector_class = ReasoningParser.DetectorMap[reasoning_parser.lower()]
        kwargs = {} if initial_reasoning_state == -1 else {"force_reasoning": bool(initial_reasoning_state)}
        detector = detector_class(**kwargs)
        self.state = self.State.CONTENT
        if detector._in_reasoning:
            self.state = self.State.REASONING
        self._start_ids = tokenizer.encode(detector.think_start_token, add_special_tokens=False)
        self._end_ids = tokenizer.encode(detector.think_end_token, add_special_tokens=False)
        assert self._start_ids and self._end_ids, "Reasoning delimiters must encode to non-empty token sequences"
        window_size = max(len(self._start_ids), len(self._end_ids))
        self._recent_ids = deque(tail_prompt_ids[-window_size:], maxlen=window_size)
        if self.state is self.State.CONTENT and self._recent_ids:
            self._handle_content(self._recent_ids[-1])

    @property
    def in_reasoning(self):
        return self.state is self.State.REASONING

    def handle(self, token_id: int) -> bool:
        """处理 token，返回当前 token 是否允许参与 stop 匹配。"""
        token_id = int(token_id)
        self._recent_ids.append(token_id)
        match self.state:
            case self.State.CONTENT:
                return self._handle_content(token_id)
            case self.State.REASONING:
                return self._handle_reasoning(token_id)

    def _handle_content(self, token_id):
        if self._matches_marker(token_id, self._start_ids):
            self.state = self.State.REASONING
            return False
        return True

    def _handle_reasoning(self, token_id):
        if self._matches_marker(token_id, self._end_ids):
            self.state = self.State.CONTENT
        return False

    def _matches_marker(self, token_id, marker_ids):
        return token_id == marker_ids[-1] and list(self._recent_ids)[-len(marker_ids) :] == marker_ids


class StopSequenceBuffer:
    """通过 handle() 匹配字符串 stop，将确定可输出的 token 放入 ready_token_list。

    每个 choice 独立维护状态和缓冲，覆盖 PD Master 的所有生成分段。
    SCANNING：没有待确认的 stop 前缀，从当前 token 开始匹配。
    WAITING：尾部可能组成 stop，需要后续 token 确认，失败后也可能留下新的前缀。
    STOPPED：已命中 stop 或后端已结束，忽略后续 token。
    每个状态对应一个 _handle_* 函数，负责状态跳转和结束处理；handle() 只按当前状态分派。

    当前状态 | 输入条件                         | 下一状态 | 输出处理
    ---------|----------------------------------|----------|----------------------------
    SCANNING | 完整命中 stop                    | STOPPED  | 按 include_stop 截断并释放
    SCANNING | 未结束，尾部命中 stop 前缀        | WAITING  | 保留可能组成 stop 的文本
    SCANNING | 未结束，尾部无 stop 前缀          | SCANNING | 原样释放
    WAITING  | 完整命中 stop                    | STOPPED  | 按 include_stop 截断并释放
    WAITING  | 未结束，尾部仍有 stop 前缀        | WAITING  | 释放安全 token，继续等待
    WAITING  | 未结束，尾部不再有 stop 前缀      | SCANNING | 原样释放全部缓冲
    SCANNING | 未结束，reasoning 或模拟边界      | SCANNING | 清空匹配历史，原样释放
    WAITING  | 未结束，reasoning 或模拟边界      | SCANNING | 清空匹配历史，原样释放
    SCANNING | 未命中 stop，后端已结束          | STOPPED  | 释放全部尾部 token
    WAITING  | 未命中 stop，后端已结束          | STOPPED  | 释放全部尾部 token
    STOPPED  | 任意输入                         | STOPPED  | 不添加输出

    先匹配当前 token，再处理后端结束状态；最后一个 token 也可能命中 stop。
    未结束时，只暂存可能组成 stop 的尾部；遇到思维链或模拟 token 时释放缓冲并重置匹配历史。
    显式 token ID stop 始终由推理层匹配，本缓冲只处理字符串，不维护 token ID stop 历史。
    输入和 ready_token_list 均使用生成器的四元组结构，文本截断为空时也保留原始 metadata。
    """

    class State(Enum):
        SCANNING = auto()
        WAITING = auto()
        STOPPED = auto()

    def __init__(
        self,
        httpserver_manager: "HttpServerManager | HttpServerManagerForPDMaster",
        sampling_params: SamplingParams,
        tail_prompt_ids: list[int],
    ):
        self._httpserver_manager = httpserver_manager
        self.state = self.State.SCANNING
        self._stop_strings = sampling_params.stop_sequences.to_stop_str_list()
        self._include_stop = sampling_params.include_stop_str_in_output
        self._stop_prefixes = {stop[:length] for stop in self._stop_strings for length in range(1, len(stop))}
        self._max_prefix_length = max((len(stop) - 1 for stop in self._stop_strings), default=0)
        self._reasoning_state = None
        reasoning_parser = httpserver_manager.args.reasoning_parser
        if sampling_params.str_stop_match_mode.is_output_content_only() and reasoning_parser:
            self._reasoning_state = ReasoningStopState(
                reasoning_parser,
                httpserver_manager.tokenizer,
                sampling_params._initial_reasoning_state,
                tail_prompt_ids,
            )
        self._pending_tokens = deque()
        self._pending_text = ""
        # 已确定可输出的 (sub_req_id, request_output, metadata, finish_status)，调用方按顺序 popleft()。
        self.ready_token_list: deque[tuple[int, str, dict, FinishStatus]] = deque()

    async def handle(self, sub_req_id: int, request_output: str, metadata: dict, finish_status: FinishStatus) -> bool:
        """接收生成器的一次输出，返回本次是否命中 stop。

        错误结束或 metadata 中没有 token ID 的输出不匹配 stop，也不推进 reasoning 状态。
        确定可输出的 token 追加到 ready_token_list，调用方用 popleft() 消费。
        命中 stop 且后端未结束时，进入 STOPPED 后通知 manager 停止该子请求。
        """
        match self.state:
            case self.State.SCANNING:
                return await self._handle_scanning(sub_req_id, request_output, metadata, finish_status)
            case self.State.WAITING:
                return await self._handle_waiting(sub_req_id, request_output, metadata, finish_status)
            case self.State.STOPPED:
                return await self._handle_stopped(sub_req_id, request_output, metadata, finish_status)

    async def _handle_scanning(self, sub_req_id, request_output, metadata, finish_status):
        """没有可延续的前缀，从当前 token 开始匹配。"""
        self._pending_text = request_output
        # 只修改输出的结束状态，保留后端传入的原始状态。
        output_finish = FinishStatus(finish_status.status)
        if self._match_and_release_token(sub_req_id, request_output, metadata, output_finish):
            output_finish.status = FinishStatus.FINISHED_STOP
            self.state = self.State.STOPPED
            if not finish_status.is_finished():
                await self._httpserver_manager.stop_request(sub_req_id)
            return True
        if finish_status.is_finished():
            self._release_pending_tokens()
            self.state = self.State.STOPPED
        elif self._pending_tokens:
            self.state = self.State.WAITING
        return False

    async def _handle_waiting(self, sub_req_id, request_output, metadata, finish_status):
        """接续待确认的文本，尝试补全字符串 stop。"""
        self._pending_text += request_output
        output_finish = FinishStatus(finish_status.status)
        if self._match_and_release_token(sub_req_id, request_output, metadata, output_finish):
            output_finish.status = FinishStatus.FINISHED_STOP
            self.state = self.State.STOPPED
            if not finish_status.is_finished():
                await self._httpserver_manager.stop_request(sub_req_id)
            return True
        if finish_status.is_finished():
            self._release_pending_tokens()
            self.state = self.State.STOPPED
        elif not self._pending_tokens:
            self.state = self.State.SCANNING
        return False

    async def _handle_stopped(self, sub_req_id, request_output, metadata, finish_status):
        """请求已结束，忽略所有迟到输出。"""
        return False

    def _match_and_release_token(self, sub_req_id, request_output, metadata, finish_status):
        """匹配并释放 token，返回是否命中 stop；状态跳转由调用方处理。"""
        self._pending_tokens.append((sub_req_id, request_output, metadata, finish_status))
        token_id = metadata.get("id")
        can_match = token_id is not None and not finish_status.is_error_finished()
        if can_match and self._reasoning_state is not None:
            can_match = self._reasoning_state.handle(token_id)
        if not can_match:
            # reasoning / 模拟结束 token 是匹配边界，不能与边界前的内容拼成 stop。
            self._release_pending_tokens()
            return False

        visible_length = self._match_stop()
        if visible_length is not None:
            self._release_pending_tokens(visible_length)
            return True
        self._release_safe_tokens()
        return False

    def _match_stop(self):
        """返回匹配后的可见字符数；None 表示未匹配，0 表示全部截断。"""
        stop_index = len(self._pending_text)
        matched_stop = None
        # stop 字符串已按长度降序排列；同一位置同时命中时优先最长的 stop。
        for stop in self._stop_strings:
            index = self._pending_text.find(stop)
            if index != -1 and index < stop_index:
                stop_index, matched_stop = index, stop
        if matched_stop is not None:
            return stop_index + (len(matched_stop) if self._include_stop else 0)

        return None

    def _release_safe_tokens(self):
        """暂存可能组成 stop 的尾部，按完整 token 释放其余输出。"""
        safe_length = len(self._pending_text)
        for length in range(min(safe_length, self._max_prefix_length), 0, -1):
            if self._pending_text[-length:] in self._stop_prefixes:
                safe_length -= length
                break
        if safe_length == len(self._pending_text):
            self._release_pending_tokens()
            return
        released_length = 0
        while self._pending_tokens and released_length + len(self._pending_tokens[0][1]) <= safe_length:
            output = self._pending_tokens.popleft()
            self.ready_token_list.append(output)
            released_length += len(output[1])
        self._pending_text = self._pending_text[released_length:]

    def _release_pending_tokens(self, visible_length=None):
        """原样批量释放或按 visible_length 截断文本，保留 ID、metadata 和结束状态。"""
        if visible_length is None or visible_length >= len(self._pending_text):
            self.ready_token_list.extend(self._pending_tokens)
            self._pending_tokens.clear()
        else:
            while self._pending_tokens:
                sub_req_id, text, metadata, finish_status = self._pending_tokens.popleft()
                self.ready_token_list.append((sub_req_id, text[: max(0, visible_length)], metadata, finish_status))
                visible_length -= len(text)
        self._pending_text = ""


async def match_stop_sequence_stream(
    httpserver_manager: "HttpServerManager | HttpServerManagerForPDMaster",
    generator: AsyncGenerator[tuple[int, str, dict, FinishStatus], None],
    sampling_params: SamplingParams,
    prompt: str | list[int],
) -> AsyncGenerator[tuple[int, str, dict, FinishStatus], None]:
    """按子请求 ID 分派 token，每个 ID 对应一个独立的 StopSequenceBuffer。

    命中 stop 时通知后端停止该子请求，再输出截断结果；其他子请求继续生成。
    已结束的 buffer 保留，由 STOPPED 状态吞掉迟到输出，直到原始流自然结束。
    每个缓冲覆盖整个输出流，不受 PD 内部容量分段影响。
    是否启用匹配由外层 wrap_stop_sequence_stream() 判断。
    """
    async with aclosing(generator):
        # reasoning 只需要 prompt 尾部；各 choice 共用一次编码结果。
        tail_prompt_ids = (
            httpserver_manager.tokenizer.encode(prompt[-2048:], add_special_tokens=False)
            if isinstance(prompt, str)
            else prompt[-2048:]
        )
        buffers = {}
        async for req_id, text, metadata, raw_finish in generator:
            if req_id not in buffers:
                buffers[req_id] = StopSequenceBuffer(httpserver_manager, sampling_params, tail_prompt_ids)
            buffer = buffers[req_id]

            await buffer.handle(req_id, text, metadata, raw_finish)

            while buffer.ready_token_list:
                yield buffer.ready_token_list.popleft()


def wrap_stop_sequence_stream(
    httpserver_manager: "HttpServerManager | HttpServerManagerForPDMaster",
    generator: AsyncGenerator[tuple[int, str, dict, FinishStatus], None],
    sampling_params: SamplingParams,
    prompt: str | list[int],
) -> AsyncGenerator[tuple[int, str, dict, FinishStatus], None]:
    """公开 generate 的统一包裹入口；私有生成流程只负责原始输出。"""
    # RL 和最终 metadata 路径保留后端原始 token，不做应用侧截断或提前结束。
    needs_final_metadata = (
        httpserver_manager.args.enable_rl
        or httpserver_manager.args.enable_return_routed_experts
        or sampling_params.prompt_logprobs >= 0
    )
    if needs_final_metadata:
        assert (
            sampling_params.str_stop_match_mode.is_detoken_all()
        ), "RL and final token metadata require str_stop_match_mode=DETOKEN_ALL"
    # 只有字符串 stop 需要输出缓冲；纯 token ID stop 由推理层直接结束请求。
    if needs_final_metadata or not sampling_params.stop_sequences.has_stop_strs():
        return generator
    return match_stop_sequence_stream(httpserver_manager, generator, sampling_params, prompt)
