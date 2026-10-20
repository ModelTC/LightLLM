"""CPU-side token boundaries for indivisible multimodal inputs."""

from typing import List, Mapping, Sequence, Tuple


def get_mm_input_spans(input_ids: Sequence[int], multimodal_params: Mapping) -> List[Tuple[int, int]]:
    """Locate image/audio blocks, including requests using a suffix of cached embeddings."""
    spans = []
    for modality in ("images", "audios"):
        for item in multimodal_params[modality]:
            start = item["start_idx"]
            # 缓存中的完整块使用连续虚拟 ID：[token_id, token_id + token_num)。
            # start_idx 指向当前输入中该块的首个虚拟 token；请求可能只引用缓存后缀，
            # 如 DeepSeek-V4 按位置对齐时会跳过开头的 padding token。
            # 首个实际 ID 与缓存起始 ID 的差就是跳过的槽数，完整引用时为 0。
            cache_offset = int(input_ids[start]) - item["token_id"]
            # 当前块长度为 token_num - cache_offset，返回左闭右开的输入区间。
            # 例如缓存 ID 为 [100, 108)，实际从 102 开始，则块长为 6。
            spans.append((start, start + item["token_num"] - cache_offset))
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
