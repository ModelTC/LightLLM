from typing import Union, List

from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
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
