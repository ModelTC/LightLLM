from collections import deque
from contextlib import aclosing
from enum import Enum, auto
from typing import TYPE_CHECKING, AsyncGenerator

from lightllm.server.core.objs import FinishStatus, SamplingParams

from lightllm.server.reasoning_parser import ReasoningParser

if TYPE_CHECKING:
    from lightllm.server.httpserver.manager import HttpServerManager
    from lightllm.server.httpserver_for_pd_master.manager import HttpServerManagerForPDMaster


class ReasoningStateMachine:
    """思维链阶段识别状态机：按 token ID 识别 start/end 标记，区分 CONTENT 和 REASONING 阶段。

    handle(token_id) 按当前状态分派，每个 _handle_* 函数直接处理 token_id。
    CONTENT：正文阶段，普通 token 可以参与字符串 stop 匹配。
    REASONING：正在输出思维链内容，不匹配字符串 stop；显式 token ID stop 不受影响。

    当前状态     | 当前 token 与最近 token 的匹配情况            | 下一状态     | 允许匹配字符串 stop
    -------------|---------------------------------------------|--------------|--------------
    CONTENT      | 完整命中思维链 start 标记                    | REASONING    | 否
    CONTENT      | 未完整命中思维链 start 标记                  | CONTENT      | 是
    REASONING    | 完整命中思维链 end 标记                      | CONTENT      | 否
    REASONING    | 未完整命中思维链 end 标记                    | REASONING    | 否

    仅完整命中标记才改变状态；尚未完整的标记按当前阶段处理。
    最近 token 包含 prompt 尾部，支持跨 token、prompt 和 P/D 分段的完整标记匹配。
    初始化仅保留 prompt 最后一个标记窗口；初始为 CONTENT 时，末尾完整 start 标记触发预跳转。
    初始为 REASONING 时保持初始状态，后续生成的 token 才推进状态机。
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
        """处理 token，返回当前 token 是否允许参与字符串 stop 匹配。"""
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


class StopTokenIdsMatchBuffer:
    """匹配真实输出的 token ID 序列，命中时只修正当前输出的 finish_status。

    每个子请求独立保留有限的 ID 尾部，覆盖所有 P/D 容量分段，不读取用户 prompt。
    所有 token 立即输出，包括文本为空的特殊 token；不处理 reasoning 或 include_stop_str_in_output。
    推理层仍负责分段内的 token ID stop，本状态机补充跨分段匹配，并吞掉结束后的迟到输出。
    handle() 只分派状态；MATCHING 按匹配结果和后端结束状态处理输出，STOPPED 忽略后续输入。

    当前状态 | 输入条件                  | 下一状态 | 输出处理
    ---------|---------------------------|----------|----------------------------
    MATCHING | 错误结束                  | STOPPED  | 原样输出，保留错误状态
    MATCHING | 完整命中 token ID stop    | STOPPED  | 原样输出 token，结束状态改为 stop
    MATCHING | 未命中，后端已结束        | STOPPED  | 原样输出并保留后端结束状态
    MATCHING | 未命中，后端未结束        | MATCHING | 原样输出，保留 ID 尾部
    STOPPED  | 任意输入                  | STOPPED  | 忽略迟到输出

    错误或没有 token ID 的输出不更新匹配历史；正常结束 token 先匹配再结束，错误状态优先保留。
    """

    class State(Enum):
        MATCHING = auto()
        STOPPED = auto()

    def __init__(
        self,
        httpserver_manager: "HttpServerManager | HttpServerManagerForPDMaster",
        sampling_params: SamplingParams,
    ):
        self._httpserver_manager = httpserver_manager
        self.state = self.State.MATCHING
        self._stop_token_ids_list = sampling_params.stop_sequences.to_token_ids_list()
        self._tail_stop_token_ids: set[int] = {ids[-1] for ids in self._stop_token_ids_list}
        max_stop_length = max((len(ids) for ids in self._stop_token_ids_list), default=0)
        self._recent_ids: deque[int] = deque(maxlen=max_stop_length)
        # 每次输入立即准备一份输出，调用方按顺序 popleft()；STOPPED 后不再添加。
        self.ready_token_list: deque[tuple[int, str, dict, FinishStatus]] = deque()

    async def handle(self, sub_req_id: int, request_output: str, metadata: dict, finish_status: FinishStatus) -> None:
        """统一接收生成器四元组，准备好的输出追加到 ready_token_list。"""
        match self.state:
            case self.State.MATCHING:
                await self._handle_matching(sub_req_id, request_output, metadata, finish_status)
            case self.State.STOPPED:
                await self._handle_stopped(sub_req_id, request_output, metadata, finish_status)

    async def _handle_matching(self, sub_req_id, request_output, metadata, finish_status) -> None:
        """按“命中 stop、后端已结束、继续生成”三种情况处理。"""
        token_id = metadata.get("id")
        matched = False
        if token_id is not None and not finish_status.is_error_finished():
            self._recent_ids.append(token_id)
            matched = self._matches_stop()

        match matched, finish_status.is_finished():
            case True, backend_finished:
                self.state = self.State.STOPPED
                self.ready_token_list.append(
                    (sub_req_id, request_output, metadata, FinishStatus(FinishStatus.FINISHED_STOP))
                )
                if not backend_finished:
                    await self._httpserver_manager.stop_request(sub_req_id)
            case False, True:
                self.state = self.State.STOPPED
                self.ready_token_list.append((sub_req_id, request_output, metadata, finish_status))
            case False, False:
                self.ready_token_list.append((sub_req_id, request_output, metadata, finish_status))

    async def _handle_stopped(self, sub_req_id, request_output, metadata, finish_status) -> None:
        """请求已结束，忽略迟到输出。"""
        return

    def _matches_stop(self) -> bool:
        """先检查结束 ID 集合，再逐个检查长度、末尾 ID 和完整尾部序列。"""
        if not self._recent_ids or self._recent_ids[-1] not in self._tail_stop_token_ids:
            return False
        for stop_ids in self._stop_token_ids_list:
            stop_length = len(stop_ids)
            if len(self._recent_ids) < stop_length:
                continue
            if self._recent_ids[-1] != stop_ids[-1]:
                continue
            if list(self._recent_ids)[-stop_length:] == stop_ids:
                return True
        return False


class StopStrMatchBuffer:
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
    本缓冲只处理字符串；token ID stop 在内层 StopTokenIdsMatchBuffer 和推理层中独立匹配。
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
        if sampling_params.stop_str_match_mode.is_output_content_only() and reasoning_parser:
            self._reasoning_state = ReasoningStateMachine(
                reasoning_parser,
                httpserver_manager.tokenizer,
                sampling_params._initial_reasoning_state,
                tail_prompt_ids,
            )
        self._pending_tokens = deque()
        self._pending_text = ""
        # 已确定可输出的 (sub_req_id, request_output, metadata, finish_status)，调用方按顺序 popleft()。
        self.ready_token_list: deque[tuple[int, str, dict, FinishStatus]] = deque()

    async def handle(self, sub_req_id: int, request_output: str, metadata: dict, finish_status: FinishStatus) -> None:
        """接收生成器的一次输出，更新状态并准备可输出的 token。

        错误结束或 metadata 中没有 token ID 的输出不匹配 stop，也不推进 reasoning 状态。
        确定可输出的 token 追加到 ready_token_list，调用方用 popleft() 消费。
        命中 stop 且后端未结束时，进入 STOPPED 后通知 manager 停止该子请求。
        """
        match self.state:
            case self.State.SCANNING:
                await self._handle_scanning(sub_req_id, request_output, metadata, finish_status)
            case self.State.WAITING:
                await self._handle_waiting(sub_req_id, request_output, metadata, finish_status)
            case self.State.STOPPED:
                await self._handle_stopped(sub_req_id, request_output, metadata, finish_status)

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
            return
        if finish_status.is_finished():
            self._release_pending_tokens()
            self.state = self.State.STOPPED
        elif self._pending_tokens:
            self.state = self.State.WAITING
        return

    async def _handle_waiting(self, sub_req_id, request_output, metadata, finish_status):
        """接续待确认的文本，尝试补全字符串 stop。"""
        self._pending_text += request_output
        output_finish = FinishStatus(finish_status.status)
        if self._match_and_release_token(sub_req_id, request_output, metadata, output_finish):
            output_finish.status = FinishStatus.FINISHED_STOP
            self.state = self.State.STOPPED
            if not finish_status.is_finished():
                await self._httpserver_manager.stop_request(sub_req_id)
            return
        if finish_status.is_finished():
            self._release_pending_tokens()
            self.state = self.State.STOPPED
        elif not self._pending_tokens:
            self.state = self.State.SCANNING
        return

    async def _handle_stopped(self, sub_req_id, request_output, metadata, finish_status):
        """请求已结束，忽略所有迟到输出。"""
        return

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


async def match_stop_token_ids_stream(
    httpserver_manager: "HttpServerManager | HttpServerManagerForPDMaster",
    generator: AsyncGenerator[tuple[int, str, dict, FinishStatus], None],
    sampling_params: SamplingParams,
) -> AsyncGenerator[tuple[int, str, dict, FinishStatus], None]:
    """按子请求 ID 独立匹配 token ID stop，先于字符串匹配包裹原始流。"""
    async with aclosing(generator):
        buffers = {}
        async for req_id, text, metadata, finish_status in generator:
            if req_id not in buffers:
                buffers[req_id] = StopTokenIdsMatchBuffer(httpserver_manager, sampling_params)
            buffer = buffers[req_id]
            await buffer.handle(req_id, text, metadata, finish_status)
            while buffer.ready_token_list:
                yield buffer.ready_token_list.popleft()


async def match_stop_str_stream(
    httpserver_manager: "HttpServerManager | HttpServerManagerForPDMaster",
    generator: AsyncGenerator[tuple[int, str, dict, FinishStatus], None],
    sampling_params: SamplingParams,
    prompt: str | list[int],
) -> AsyncGenerator[tuple[int, str, dict, FinishStatus], None]:
    """按子请求 ID 分派 token，每个 ID 对应一个独立的 StopStrMatchBuffer。

    命中 stop 时通知后端停止该子请求，再输出截断结果；其他子请求继续生成。
    已结束的 buffer 保留，由 STOPPED 状态吞掉迟到输出，直到原始流自然结束。
    每个缓冲覆盖整个输出流，不受 PD 内部容量分段影响。
    是否启用匹配由外层 wrap_stop_match_stream() 判断。
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
                buffers[req_id] = StopStrMatchBuffer(httpserver_manager, sampling_params, tail_prompt_ids)
            buffer = buffers[req_id]

            await buffer.handle(req_id, text, metadata, raw_finish)

            while buffer.ready_token_list:
                yield buffer.ready_token_list.popleft()


def wrap_stop_match_stream(
    httpserver_manager: "HttpServerManager | HttpServerManagerForPDMaster",
    generator: AsyncGenerator[tuple[int, str, dict, FinishStatus], None],
    sampling_params: SamplingParams,
    prompt: str | list[int],
) -> AsyncGenerator[tuple[int, str, dict, FinishStatus], None]:
    """公开 generate 的统一入口：原始流 -> token ID stop -> 字符串 stop。"""
    # 最终 metadata 由后端结束输出携带，RL 等路径跳过两种应用侧 stop 包装。
    needs_final_metadata = (
        httpserver_manager.args.enable_rl
        or httpserver_manager.args.enable_return_routed_experts
        or sampling_params.prompt_logprobs >= 0
    )
    if needs_final_metadata:
        assert (
            sampling_params.stop_str_match_mode.is_detoken_all()
        ), "RL and final token metadata require stop_str_match_mode=DETOKEN_ALL"
        return generator
    # 先匹配真实 token IDs，再由外层处理字符串，避免文本截断影响 token ID 匹配。
    if sampling_params.stop_sequences.to_token_ids_list():
        generator = match_stop_token_ids_stream(httpserver_manager, generator, sampling_params)
    if sampling_params.stop_sequences.has_stop_strs():
        generator = match_stop_str_stream(httpserver_manager, generator, sampling_params, prompt)
    return generator
