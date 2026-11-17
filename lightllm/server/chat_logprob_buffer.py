from __future__ import annotations

from typing import Any, NamedTuple

from lightllm.server.core.objs import FinishStatus
from lightllm.server.core.objs.logprob_utils import chat_token_logprob
from lightllm.server.function_call_parser import FunctionCallParser
from lightllm.server.reasoning_parser import ReasoningParser


class ChatLogprobBuffer:
    """按 token 序列识别分隔符，把可见文本的 logprob 放进正文或思维链。

    分隔符可以是一个或多个 token。凑齐才切换通道，分隔符本身不输出。
    每个采样位置单独一条 logprob。未完成的 UTF-8 字节文本为空，字符文本落在补齐的那个
    token 上，分数不相加。特殊 token 的空文本不输出。
    """

    def __init__(
        self,
        tokenizer: Any,
        model_type: str | None,
        force_reasoning: bool,
        separate_reasoning: bool,
        tools: list | None = None,
        tool_call_parser: str | None = None,
    ):
        self.tokenizer = tokenizer
        self._separate_reasoning = separate_reasoning
        self._channel = "content"  # 当前输出通道：content、reasoning，或 commentary 的 drop。
        self._in_tool = False  # 是否处在工具调用开始和结束分隔符之间。
        self._reasoning: list[dict[str, Any]] = []  # 分开思考时的思维链 logprob。
        self._content: list[dict[str, Any]] = []  # 正文 logprob；不分开思考时思维链也放这里。
        self._sequences: list[tuple[tuple[int, ...], str]] = []  # 分隔符 token id 序列和切换动作。
        self._held: list[_Held] = []  # 可能是分隔符前缀、尚未判定的 token。
        self._mode = "body"  # gpt-oss 解析位置：body 正文、header 角色头、channel 通道名。
        self._channel_name = ""  # <|channel|> 和 <|message|> 之间拼出来的通道名。
        if model_type:
            self._init_reasoning(tokenizer=tokenizer, model_type=model_type, force_reasoning=force_reasoning)
        if tools:
            self._init_tools(tokenizer=tokenizer, tool_call_parser=tool_call_parser)

    def record(self, text: str, metadata: dict[str, Any], finish_status: FinishStatus) -> None:
        token_id, score = metadata.get("id"), metadata.get("logprob")
        if token_id is not None and score is not None:
            self._held.append(_Held(token_id=token_id, text=text, score=score, metadata=metadata))
            self._resolve_held(closing=False)
        if finish_status.is_finished():
            self._resolve_held(closing=True)

    def reasoning_logprobs(self) -> dict[str, Any] | None:
        tokens = self._drain(items=self._reasoning)
        return {"content": None, "reasoning_content": tokens} if tokens else None

    def content_logprobs(self) -> dict[str, Any] | None:
        tokens = self._drain(items=self._content)
        return {"content": tokens, "reasoning_content": None} if tokens else None

    def message_logprobs(self) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None]:
        return self._drain(items=self._content) or None, self._drain(items=self._reasoning) or None

    def _init_reasoning(self, tokenizer: Any, model_type: str, force_reasoning: bool) -> None:
        key = model_type.lower()
        detector_class = ReasoningParser.DetectorMap.get(key)
        if detector_class is None:
            raise ValueError(f"logprobs does not support reasoning parser {model_type!r}")
        if key == "gpt-oss":
            for text, action in (
                ("<|start|>", "harmony_start"),
                ("<|channel|>", "harmony_channel"),
                ("<|message|>", "harmony_message"),
                ("<|end|>", "harmony_end"),
                ("<|return|>", "harmony_end"),
                ("<|call|>", "harmony_end"),
            ):
                # <|call|> 不是每个词表都有，缺了就跳过。
                self._add_sequence(tokenizer=tokenizer, text=text, action=action, required=text != "<|call|>")
            if force_reasoning:
                self._channel = "reasoning"
            return
        detector = detector_class(stream_reasoning=True, force_reasoning=force_reasoning)
        self._add_sequence(tokenizer=tokenizer, text=detector.think_start_token, action="reasoning")
        self._add_sequence(tokenizer=tokenizer, text=detector.think_end_token, action="content")
        # 以解析器为准。Gemma 4 会忽略请求里的 force_reasoning，生成不是从已打开的思考通道开始。
        if detector._in_reasoning:
            self._channel = "reasoning"

    def _init_tools(self, tokenizer: Any, tool_call_parser: str | None) -> None:
        if not tool_call_parser:
            raise ValueError("logprobs requires tool_call_parser when tools are set")
        if tool_call_parser == "mistral":
            raise ValueError("logprobs does not support mistral tool_call_parser")
        parser = FunctionCallParser(tools=[], tool_call_parser=tool_call_parser)
        self._add_sequence(tokenizer=tokenizer, text=parser.detector.bot_token, action="tool_on")
        if parser.detector.eot_token:
            self._add_sequence(tokenizer=tokenizer, text=parser.detector.eot_token, action="tool_off")

    def _add_sequence(self, tokenizer: Any, text: str, action: str, required: bool = True) -> None:
        token_ids = _delimiter_ids(tokenizer=tokenizer, text=text)
        if token_ids is None:
            if required:
                raise ValueError(f"logprobs requires {text!r} to encode to the same token sequence")
            return
        sequence = tuple(token_ids)
        for existing, existing_action in self._sequences:
            if existing == sequence and existing_action != action:
                raise ValueError(f"logprobs delimiter {text!r} maps to conflicting actions")
        if any(existing == sequence for existing, _ in self._sequences):
            return
        self._sequences.append((sequence, action))

    def _resolve_held(self, closing: bool) -> None:
        # 还是更长分隔符的前缀时先留着。生成结束则按已经凑齐的短分隔符结算，否则当正文。
        while self._held:
            token_ids = tuple(item.token_id for item in self._held)
            exact, waiting, completed = self._lookup(token_ids=token_ids)
            if waiting and not closing:
                return
            if exact is not None:
                self._apply(action=exact)
                self._held.clear()
                return
            if completed is not None:
                action, size = completed
                self._apply(action=action)
                del self._held[:size]
                continue
            self._release(item=self._held.pop(0))

    def _lookup(self, token_ids: tuple[int, ...]) -> tuple[str | None, bool, tuple[str, int] | None]:
        exact = None
        waiting = False
        completed = None
        for sequence, action in self._sequences:
            if sequence == token_ids:
                exact = action
            elif sequence[: len(token_ids)] == token_ids:
                waiting = True
            elif token_ids[: len(sequence)] == sequence and (completed is None or len(sequence) > completed[1]):
                completed = (action, len(sequence))
        return exact, waiting, completed

    def _apply(self, action: str) -> None:
        match action:
            case "reasoning":
                # 思维链开始。这组 token 是分隔符，不输出。
                self._channel = "reasoning"
            case "content":
                # 思维链结束，后面的文本回到正文。
                self._channel = "content"
            case "tool_on":
                # 工具调用开始，参数不进入正文。
                self._in_tool = True
            case "tool_off":
                # 工具调用结束，后面的文本回到当前通道。
                self._in_tool = False
            case "harmony_start":
                # <|start|> 到下一个结构标记之间是角色头。
                self._mode = "header"
            case "harmony_channel":
                # <|channel|> 后面开始收集通道名。
                self._mode = "channel"
                self._channel_name = ""
            case "harmony_message":
                # <|message|>：analysis 进入思维链，commentary 整段丢弃，其余进入正文。
                name = self._channel_name.strip().lower()
                if name.startswith("analysis"):
                    self._channel = "reasoning"
                elif name.startswith("commentary"):
                    self._channel = "drop"
                else:
                    self._channel = "content"
                self._mode = "body"
                self._channel_name = ""
            case "harmony_end":
                # 这段消息结束。通道保持不变，下一次 <|message|> 再切换。
                self._mode = "body"

    def _release(self, item: _Held) -> None:
        token_id, text, score, metadata = item
        match self._mode:
            case "channel":
                # 通道名本身，如 analysis，不输出。
                self._channel_name += text
            case "header":
                # 头部里的角色名，如 assistant，不输出。
                return
            case _:
                if self._in_tool or self._channel == "drop":
                    # 工具参数或 commentary 正文不输出。
                    return
                self._save(text=text, score=score, metadata=metadata, token_id=token_id)

    def _save(self, text: str, score: float, metadata: dict[str, Any], token_id: int) -> None:
        if not text and self._is_special(metadata=metadata, token_id=token_id):
            return
        token = {
            "token": text,
            "logprob": score,
            "bytes": None if "\ufffd" in text else list(text.encode("utf-8")),
            "top_logprobs": [
                _candidate_logprob(tokenizer=self.tokenizer, token_id=candidate, logprob=value)
                for candidate, value in metadata.get("output_top_logprobs", ())
            ],
        }
        # 分开思考时思维链单独列出；否则和正文共用 content。
        if self._channel == "reasoning" and self._separate_reasoning:
            self._reasoning.append(token)
        else:
            self._content.append(token)

    def _is_special(self, metadata: dict[str, Any], token_id: int) -> bool:
        if "special" in metadata:
            return bool(metadata["special"])
        return token_id in getattr(self.tokenizer, "all_special_ids", ())

    @staticmethod
    def _drain(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        tokens = list(items)
        items.clear()
        return tokens


class _Held(NamedTuple):
    token_id: int
    text: str
    score: float
    metadata: dict[str, Any]


def _candidate_logprob(tokenizer: Any, token_id: int, logprob: float) -> dict[str, Any]:
    return chat_token_logprob(tokenizer=tokenizer, token_id=token_id, logprob=logprob)


def _delimiter_ids(tokenizer: Any, text: str) -> list[int] | None:
    if not text:
        return None
    encode = getattr(tokenizer, "encode", None)
    if encode is not None:
        token_ids = list(encode(text, add_special_tokens=False))
        if token_ids and tokenizer.decode(token_ids, skip_special_tokens=False) == text:
            return token_ids
    return _longest_token_ids(tokenizer=tokenizer, text=text)


def _longest_token_ids(tokenizer: Any, text: str) -> list[int] | None:
    vocab = tokenizer.get_vocab()
    pieces = sorted((piece for piece in vocab if piece), key=len, reverse=True)
    token_ids = []
    index = 0
    while index < len(text):
        piece = next((item for item in pieces if text.startswith(item, index)), None)
        if piece is None:
            return None
        token_ids.append(vocab[piece])
        index += len(piece)
    if tokenizer.decode(token_ids, skip_special_tokens=False) != text:
        return None
    return token_ids
