"""CPU-side token boundaries for indivisible multimodal inputs."""

from typing import List, Mapping, Sequence, Tuple


def get_mm_input_spans(
    multimodal_params: Mapping, *, input_len: int = None, max_tokens: int = None
) -> List[Tuple[int, int]]:
    """Read whole image/audio blocks; V4 embedding offsets exclude alignment tokens."""
    spans = []
    for modality in ("images", "audios"):
        for item in multimodal_params[modality]:
            start, end = item.get("block_start_idx"), item.get("block_end_idx")
            if start is None and end is None:
                start = item["start_idx"]
                end = start + item["token_num"]
            if start is None or end is None or not 0 <= start < end:
                raise ValueError("Multimodal input has an invalid block range")
            if input_len is not None and end > input_len:
                raise ValueError("Multimodal block exceeds input length")
            if max_tokens is not None and end - start > max_tokens:
                raise ValueError("Multimodal block exceeds batch_max_tokens")
            spans.append((start, end))
    return sorted(spans)


def clamp_mm_cache_len(cache_len: int, mm_spans: Sequence[Tuple[int, int]]) -> int:
    """缓存终点落在多模态块内时退到块首；调用方页对齐后需再次检查。"""
    for block_start, block_end in mm_spans:
        if block_start < cache_len < block_end:
            return block_start
    return cache_len


def adjust_mm_chunk_end(chunk_start: int, chunk_end: int, mm_spans: Sequence[Tuple[int, int]]) -> int:
    """避免 chunk 终点落在多模态块内部；调用方保证起点位于块外或边界上。"""
    for block_start, block_end in mm_spans:
        if chunk_end <= block_start or chunk_end >= block_end:
            continue

        if chunk_start < block_start:
            # 先处理前面的文本，将完整多模态块留给下一个 chunk。
            return block_start
        else:
            # chunk 从块首开始，扩展到块尾，一次处理完整块。
            return block_end
    return chunk_end
