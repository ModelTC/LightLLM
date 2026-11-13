from dataclasses import dataclass
from typing import List


@dataclass
class TextSpan:
    start: int
    end: int
    channel: str = "content"


def slice_spans(spans: List[TextSpan], start: int, end: int) -> List[TextSpan]:
    selected = []
    offset = 0
    for span in spans:
        size = span.end - span.start
        left, right = max(start - offset, 0), min(end - offset, size)
        if left < right:
            selected.append(TextSpan(span.start + left, span.start + right, span.channel))
        offset += size
    return selected


def extract_text(text: str, offset=0, channel="content", strip=False, remove=()):
    spans = [TextSpan(offset, offset + len(text), channel)] if text else []
    for marker in remove:
        if not marker:
            continue
        start = text.find(marker)
        while start >= 0:
            end = start + len(marker)
            spans = slice_spans(spans, 0, start) + slice_spans(spans, end, len(text))
            text = text[:start] + text[end:]
            start = text.find(marker, start)
    if strip:
        start = len(text) - len(text.lstrip())
        end = len(text.rstrip())
        spans = slice_spans(spans, start, end)
        text = text.strip()
    return text, spans
