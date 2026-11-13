from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from lightllm.server.core.objs import FinishStatus
from lightllm.server.core.objs.logprob_utils import ChatTokenDecoder
from lightllm.server.text_spans import TextSpan


@dataclass
class PendingLogprob:
    start: int
    end: int
    token: Dict[str, Any]


class ChatLogprobBuffer:
    """Attribute generated tokens to the visible source ranges returned by parsers.

    A token spanning two channels belongs to the first visible channel. Empty
    detokenization increments stay pending until the character they complete is
    classified; recovering raw bytes is independent of retaining that token.
    """

    def __init__(
        self,
        tokenizer: Any,
        separate_reasoning: bool,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
    ):
        self.tokenizer = tokenizer
        self._separate_reasoning = separate_reasoning
        self._hold: List[PendingLogprob] = []
        self._reasoning: List[PendingLogprob] = []
        self._content: List[PendingLogprob] = []
        self._source_length = 0
        self._content_length = 0
        self._decoder = ChatTokenDecoder(tokenizer, skip_special_tokens, spaces_between_special_tokens)

    def record(self, text: str, metadata: Dict[str, Any], finish_status: FinishStatus) -> TextSpan:
        start = self._source_length
        self._source_length += len(text)
        token_id, score = metadata.get("id"), metadata.get("logprob")
        if "logprob_prompt_ids" in metadata:
            self._decoder.set_prompt(metadata["logprob_prompt_ids"])
        special = metadata.get("special", token_id in getattr(self.tokenizer, "all_special_ids", ()))
        if token_id is not None and score is not None and (text or not special):
            rank = metadata.get("logprobs", {}).get(token_id, {}).get("rank")
            token = self._decoder.token_logprob(token_id, -9999.0 if rank is not None and rank > 20 else score)
            token["top_logprobs"] = [
                self._decoder.token_logprob(candidate, value)
                for candidate, value in metadata.get("output_top_logprobs", ())
            ]
            self._hold.append(PendingLogprob(start, self._source_length, token))
        if token_id is not None:
            self._decoder.advance(token_id)
        return TextSpan(start, self._source_length)

    @staticmethod
    def _overlaps(item, span):
        if item.start == item.end:
            return span.start <= item.start < span.end
        return item.start < span.end and item.end > span.start

    def assign(self, spans: List[TextSpan], content_length: int) -> List[TextSpan]:
        spans = [span for span in spans if span.start < span.end]
        visible = [TextSpan(self._content_length, self._content_length + content_length)] if content_length else []
        content_items = []
        content_offset = self._content_length
        for span in spans:
            selected = [item for item in self._hold if self._overlaps(item, span)]
            self._hold = [item for item in self._hold if not self._overlaps(item, span)]
            if span.channel == "reasoning" and self._separate_reasoning:
                self._reasoning.extend(selected)
            else:
                for item in selected:
                    content_items.append(
                        PendingLogprob(
                            content_offset + max(item.start - span.start, 0),
                            content_offset + min(item.end, span.end) - span.start,
                            item.token,
                        )
                    )
                content_offset += span.end - span.start
        if spans:
            consumed = max(span.end for span in spans)
            self._hold = [item for item in self._hold if item.end > consumed]
        self._content_length += content_length
        self._content.extend(content_items)
        return visible

    def _take_content(self, spans=None):
        if spans is None:
            return self._drain(self._content)
        selected = [item for item in self._content if any(self._overlaps(item, span) for span in spans)]
        if spans:
            consumed = max(span.end for span in spans)
            self._content = [item for item in self._content if item.end > consumed and item not in selected]
        return [item.token for item in selected]

    def discard_content(self, spans: List[TextSpan]) -> None:
        self._take_content(spans)

    @staticmethod
    def _drain(items):
        tokens = [item.token for item in items]
        items.clear()
        return tokens

    def reasoning_logprobs(self) -> Optional[Dict[str, Any]]:
        tokens = self._drain(self._reasoning)
        return {"content": None, "reasoning_content": tokens} if tokens else None

    def content_logprobs(self, spans=None) -> Optional[Dict[str, Any]]:
        tokens = self._take_content(spans)
        return {"content": tokens, "reasoning_content": None} if tokens else None

    def message_logprobs(
        self, content_spans=None
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[List[Dict[str, Any]]]]:
        return self._take_content(content_spans) or None, self._drain(self._reasoning) or None
