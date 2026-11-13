from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from lightllm.server.core.objs import FinishStatus
from lightllm.server.core.objs.logprob_utils import chat_token_logprob
from lightllm.server.reasoning_parser import ReasoningParser


class PendingLogprob(NamedTuple):
    """一条还没发出去的采样结果。

    ``text`` 是这次解码出的增量，可以是 ``</`` 这种半截标记。
    """

    text: str
    token: Optional[Dict[str, Any]]
    candidates: Any


class ChatLogprobBuffer:
    """一个子请求的输出 logprobs。自己持有一份解析器，不读响应用的那份。

    每次 ``record`` 只进来一个增量。解析器还没收齐标记时，token 留在 ``_hold``；
    收齐后整批进入 ``_reasoning`` 或 ``_content``。``<think>``、``</think>`` 丢掉。
    一个 token 写成 ``先想</think>好`` 时，解析器先返回思维链，整条概率进 ``_reasoning``。

    思维链帧用 ``reasoning_logprobs``，正文帧用 ``content_logprobs``，
    非流式整段消息用 ``message_logprobs``。是否分开思考在创建时定好。
    """

    def __init__(
        self,
        tokenizer: Any,
        model_type: Optional[str],
        force_reasoning: bool,
        separate_reasoning: bool,
    ):
        self.tokenizer = tokenizer
        self._separate_reasoning = separate_reasoning
        self._hold: List[PendingLogprob] = []
        self._reasoning: List[PendingLogprob] = []
        self._content: List[PendingLogprob] = []
        self._parser = None
        if model_type:
            self._parser = ReasoningParser(
                model_type=model_type,
                stream_reasoning=True,
                force_reasoning=force_reasoning,
            )

    def record(self, text: str, metadata: Dict[str, Any], finish_status: FinishStatus) -> None:
        """记下这次原始输出，并按自己的解析器把它放进思维链或正文。

        ``text`` 是模型刚吐出的增量。结束通知没有正文时不记概率。
        一个汉字拆成多个 token 时，前几个增量的 ``text`` 为空、解码含 ``\\ufffd``，
        这些字节的概率也要留下。

        例如 ``record(text="<think>", ...)`` 后，这条概率被认出是纯标记，直接丢掉。
        """
        self._remember(text=text, metadata=metadata, finish_status=finish_status)
        self._accept(text=text)

    def reasoning_logprobs(self) -> Optional[Dict[str, Any]]:
        """取出一帧思维链 logprobs。没有 token 时返回 None。

        只在分开思考时取出。不分开时 token 留着，由 ``content_logprobs`` 接到正文前面。

        例如 ``_reasoning`` 里是 ``先想一下``，分开思考时返回
        ``{"content": None, "reasoning_content": [先想一下的 token]}``。
        """
        self._close_partial_marker()
        if not self._separate_reasoning:
            return None
        tokens = self._drain(items=self._reasoning)
        if not tokens:
            return None
        return {"content": None, "reasoning_content": tokens}

    def content_logprobs(self) -> Optional[Dict[str, Any]]:
        """取出一帧正文 logprobs。没有 token 时返回 None。

        不分开思考时，把还没发出的思维链 token 接在正文前面。

        例如 ``_content`` 里是 ``好的``，返回
        ``{"content": [好的对应的 token], "reasoning_content": None}``。
        """
        self._close_partial_marker()
        tokens = self._take(include_reasoning=not self._separate_reasoning, include_content=True)
        if not tokens:
            return None
        return {"content": tokens, "reasoning_content": None}

    def message_logprobs(self) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
        """取出整段消息的 ``(content, reasoning_content)``。

        不分开思考时思维链 token 在 content 里，reasoning_content 是 None。

        例如分开思考且缓冲区是思维链 ``先想``、正文 ``好的``，返回 ``([好的], [先想])``。
        """
        self._close_partial_marker()
        if self._separate_reasoning:
            content_tokens = self._drain(items=self._content)
            reasoning_tokens = self._drain(items=self._reasoning)
            return content_tokens or None, reasoning_tokens or None
        tokens = self._take(include_reasoning=True, include_content=True)
        return tokens or None, None

    def _remember(self, *, text: str, metadata: Dict[str, Any], finish_status: FinishStatus) -> None:
        token_id = metadata.get("id")
        score = metadata.get("logprob")
        sampled = token_id is not None and score is not None
        finish_without_text = finish_status.is_finished() and not text
        token = None
        if sampled and not finish_without_text:
            token = chat_token_logprob(tokenizer=self.tokenizer, token_id=token_id, logprob=score)
        incomplete_character = token is not None and bool(token["bytes"]) and "\ufffd" in token["token"]
        if text or incomplete_character:
            self._hold.append(
                PendingLogprob(text=text, token=token, candidates=metadata.get("output_top_logprobs", ()))
            )

    def _accept(self, *, text: str) -> None:
        if self._parser is None:
            self._commit(
                reasoning_text="",
                content_text=text,
                waiting=False,
                in_reasoning=False,
                start_tag="",
                end_tag="",
            )
            return
        reasoning_text, content_text = self._parser.parse_stream_chunk(chunk_text=text)
        self._commit_parser_result(reasoning_text=reasoning_text or "", content_text=content_text or "")

    def _close_partial_marker(self) -> None:
        # 普通帧在 record 时已经放行，这里看到的缓冲是空的，不会提前丢掉半截标记。
        if self._parser is None or not self._parser.detector._buffer:
            return
        reasoning_text, content_text = self._parser.flush()
        self._commit_parser_result(reasoning_text=reasoning_text or "", content_text=content_text or "")

    def _commit_parser_result(self, *, reasoning_text: str, content_text: str) -> None:
        detector = self._parser.detector
        self._commit(
            reasoning_text=reasoning_text,
            content_text=content_text,
            waiting=bool(detector._buffer),
            in_reasoning=bool(detector._in_reasoning),
            start_tag=detector.think_start_token or "",
            end_tag=detector.think_end_token or "",
        )

    def _commit(
        self,
        *,
        reasoning_text: str,
        content_text: str,
        waiting: bool,
        in_reasoning: bool,
        start_tag: str,
        end_tag: str,
    ) -> None:
        # 只看到 ``</`` 这类前缀时，token 继续留在 ``_hold``。
        if waiting or not self._hold:
            return
        pending = self._hold
        self._hold = []
        # 返回了思维链就整批进思维链。
        # 两边都没有可见文字时，空字节跟解析器当前所在的一侧。
        if reasoning_text or (not content_text and in_reasoning):
            side = self._reasoning
        else:
            side = self._content
        visible = f"{reasoning_text}{content_text}"
        for item in pending:
            if item.token is None:
                continue
            # ``</``、``<think>`` 不在解析器返回的文字里，丢掉。
            # ``think`` 作为正文时会留在返回值里。
            if item.text and item.text not in visible and (item.text in start_tag or item.text in end_tag):
                continue
            side.append(item)

    def _take(self, *, include_reasoning: bool, include_content: bool) -> List[Dict[str, Any]]:
        tokens = []
        if include_reasoning:
            tokens.extend(self._drain(items=self._reasoning))
        if include_content:
            tokens.extend(self._drain(items=self._content))
        return tokens

    def _drain(self, *, items: List[PendingLogprob]) -> List[Dict[str, Any]]:
        tokens = []
        for item in items:
            if item.token is None:
                continue
            item.token["top_logprobs"] = [
                chat_token_logprob(tokenizer=self.tokenizer, token_id=token_id, logprob=score)
                for token_id, score in item.candidates
            ]
            tokens.append(item.token)
        items.clear()
        return tokens
