"""CPU-side token boundaries for indivisible multimodal inputs."""

from typing import List, Mapping, Sequence, Tuple


def get_mm_input_spans(multimodal_params: Mapping) -> List[Tuple[int, int]]:
    """Read image/audio ranges recorded by the tokenizer."""
    spans = []
    for modality in ("images", "audios"):
        for item in multimodal_params[modality]:
            start = item["start_idx"]
            spans.append((start, start + item["token_num"]))
    return spans


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
