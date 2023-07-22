from typing import List, Tuple, Union

from transformers import (AutoTokenizer, PreTrainedTokenizer,
                          PreTrainedTokenizerFast)
from transformers.convert_slow_tokenizer import convert_slow_tokenizer
from transformers import LlamaTokenizer
from ..io_struct import ReqDetokenizationState


def decode_token(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    req: ReqDetokenizationState,
    new_token_id: int,
    skip_special_tokens: bool,
) -> str:
    
    new_token = tokenizer.convert_ids_to_tokens(
        new_token_id, skip_special_tokens=skip_special_tokens)
    
    req.output_tokens.append(new_token)

    if not getattr(tokenizer, "added_tokens_encoder", {}):
        output_text = tokenizer.convert_tokens_to_string(req.output_tokens)
        return output_text

    if skip_special_tokens and new_token in tokenizer.all_special_ids:
        return req.output_str
    
    if new_token in tokenizer.added_tokens_encoder:
        if req.current_sub_text:
            sub_text = tokenizer.convert_tokens_to_string(req.current_sub_text)
            req.sub_texts.append(sub_text)
            req.current_sub_text = []
        req.sub_texts.append(new_token)
        return " ".join(req.sub_texts)
    else:
        req.current_sub_text.append(new_token)
        new_sub_text = tokenizer.convert_tokens_to_string(req.current_sub_text)
        return " ".join(req.sub_texts + [new_sub_text])