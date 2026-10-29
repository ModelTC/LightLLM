from typing import Union, List

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from lightllm.server.embed_cache.utils import MULTIMODAL_TOKEN_ID_START
from .decode_req import DecodeReq


def decode_token(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    decode_req: DecodeReq,
    new_token_id: int,
    eos_id: List[int],
) -> str:

    is_eos_id = new_token_id in eos_id
    if is_eos_id and not decode_req.req.sample_params.print_eos_token:
        return ""

    prefix_tokens, read_tokens = decode_req.get_decode_tokens()
    # 增量反分词会分别解码“已读前缀”和“前缀 + 新 token”，通过两者的文本差提取新增内容。
    # 初始解码窗口会保留 prompt 末尾的上下文（默认最多 5 个 token，由 LIGHTLLM_DECODE_PREFIX_LENGTH 配置），
    # 以正确处理分词边界和未完整的 UTF-8 字符。
    # 多模态 prompt 的末尾可能包含图片/音频缓存的虚拟 ID；这些 ID 从 MULTIMODAL_TOKEN_ID_START
    # 起分配，不属于文本词表，且可能超过底层 tokenizer 接受的整数范围，直接解码可能报错。
    # 因此，prefix_tokens 和 read_tokens 都要按同一规则过滤，保证两次解码使用一致的文本上下文。
    # 注意：
    # 1. 只过滤本次传给 tokenizer 的局部列表，不修改共享内存、prompt_ids 或 output_ids。
    #    prefix_offset/read_offset 和 input_len 都基于原始 token 序列（包含虚拟 ID）；
    #    不能用过滤后的 token 数推进 offset，否则后续会读取错误的位置，导致输出丢失或重复。
    # 2. 只移除预留范围内的虚拟 ID，普通文本、added token 和 special token 应保留，
    #    由 tokenizer 及 skip_special_tokens 等参数决定如何解码；不能按 vocab_size 过滤，
    #    因为 added token 的合法 ID 可能超出基础词表大小。
    # 3. 这里的边界必须与图片/音频虚拟 ID 分配器一致，统一使用共享常量。
    prefix_tokens = [token_id for token_id in prefix_tokens if token_id < MULTIMODAL_TOKEN_ID_START]
    read_tokens = [token_id for token_id in read_tokens if token_id < MULTIMODAL_TOKEN_ID_START]
    prefix_text = tokenizer.decode(
        prefix_tokens,
        skip_special_tokens=decode_req.req.sample_params.skip_special_tokens,
        spaces_between_special_tokens=decode_req.req.sample_params.add_spaces_between_special_tokens,
    )
    new_text = tokenizer.decode(
        read_tokens,
        skip_special_tokens=decode_req.req.sample_params.skip_special_tokens,
        spaces_between_special_tokens=decode_req.req.sample_params.add_spaces_between_special_tokens,
    )
    if len(new_text) > len(prefix_text) and not new_text.endswith("\ufffd"):
        new_text = new_text[len(prefix_text) :]
        decode_req.prefix_offset = decode_req.read_offset
        decode_req.read_offset = len(decode_req.output_ids) + decode_req.input_len
        return new_text
    else:
        return ""
