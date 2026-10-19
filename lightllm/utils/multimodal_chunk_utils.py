"""CPU-side token boundaries for indivisible multimodal inputs."""

from bisect import bisect_right
from typing import List, Mapping, Optional, Sequence, Tuple


def get_mm_input_spans(
    input_ids: Sequence[int], multimodal_params: Mapping, max_tokens: Optional[int] = None
) -> List[Tuple[int, int]]:
    """Locate image/audio blocks, including models without explicit positions.

    Tokenizers may provide a complete block range or an embedding start position.
    Otherwise, locate the consecutive virtual token IDs assigned by embed cache.
    Keep adjacent items separate, even when they reuse the same cached embedding.
    """
    spans = []
    missing = set()
    for modality in ("images", "audios"):
        for item in multimodal_params.get(modality, ()):
            start, end = item.get("block_start_idx"), item.get("block_end_idx")
            if start is None and end is None:
                start = item.get("start_idx")
                if start is not None:
                    end = start + item["token_num"]
                else:
                    token_id, token_num = item.get("token_id"), item.get("token_num")
                    if token_id is None or token_num is None or token_num <= 0:
                        raise ValueError("Multimodal input is missing its token range")
                    missing.add((token_id, token_id + token_num))
                    continue
            if start is None or end is None:
                raise ValueError("Multimodal input has an incomplete block range")
            spans.append((int(start), int(end)))

    if missing:
        ranges = sorted(missing)
        starts = [start for start, _ in ranges]
        found = set()
        position = 0
        while position < len(input_ids):
            token = int(input_ids[position])
            index = bisect_right(starts, token) - 1
            if index < 0 or token >= ranges[index][1]:
                position += 1
                continue
            first, last = ranges[index]
            end = position + last - first
            if end > len(input_ids) or any(
                int(input_ids[position + offset]) != expected for offset, expected in enumerate(range(first, last))
            ):
                raise ValueError("Multimodal input contains an incomplete token block")
            spans.append((position, end))
            found.add((first, last))
            position = end
        if found != missing:
            raise ValueError("Multimodal input contains a missing token block")

    spans = sorted(set(spans))
    previous_end = 0
    for start, end in spans:
        if not 0 <= start < end <= len(input_ids) or start < previous_end:
            raise ValueError("Multimodal input has invalid or overlapping block ranges")
        if max_tokens is not None and end - start > max_tokens:
            raise ValueError(
                f"Multimodal block of {end - start} tokens exceeds --batch_max_tokens={max_tokens} "
                "with --disable_chunked_mm_input; increase --batch_max_tokens"
            )
        previous_end = end
    return spans


def adjust_mm_chunk_end(start: int, end: int, spans: Sequence[Tuple[int, int]], max_tokens: int) -> int:
    """Stop before a partial item, or finish an item starting this chunk."""
    for block_start, block_end in spans:
        if block_start < start < block_end:
            raise ValueError("Cached prefix ends inside an indivisible multimodal block")
        if block_start < end < block_end:
            end = block_start if start < block_start else block_end
            break
    if end - start > max_tokens:
        raise ValueError(f"Multimodal chunk exceeds --batch_max_tokens={max_tokens}")
    return end


def clamp_mm_prefix_len(length: int, spans: Sequence[Tuple[int, int]], alignment: int = 1) -> int:
    """Retreat a cache endpoint until it is aligned and outside every item."""
    length = length // alignment * alignment
    for start, end in reversed(spans):
        if start < length < end:
            length = start // alignment * alignment
    return length
