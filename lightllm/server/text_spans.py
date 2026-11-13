from dataclasses import dataclass
from typing import List


@dataclass
class TextSpan:
    start: int
    end: int
    channel: str = "content"


def select_text_spans(source: str, text: str, channel: str = "content") -> List[TextSpan]:
    spans = []
    cursor = 0
    while text:
        start = source.find(text, cursor)
        if start >= 0:
            spans.append(TextSpan(start, start + len(text), channel))
            break
        start = source.find(text[0], cursor)
        if start < 0:
            break
        length = 1
        while length < len(text) and start + length < len(source) and source[start + length] == text[length]:
            length += 1
        spans.append(TextSpan(start, start + length, channel))
        cursor = start + length
        text = text[length:]
    return spans
