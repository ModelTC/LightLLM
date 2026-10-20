from functools import lru_cache


def create_tokenizer_info(tokenizer, vocab_size, eos_ids):
    """Use the same output vocabulary and stop tokens on HTTP and inference workers."""
    import xgrammar as xgr
    from transformers import PreTrainedTokenizerBase

    # Multimodal wrappers customize prompt encoding, not the output vocabulary.
    while not isinstance(tokenizer, PreTrainedTokenizerBase) and hasattr(tokenizer, "tokenizer"):
        tokenizer = tokenizer.tokenizer
    return xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size, stop_token_ids=eos_ids)


@lru_cache(maxsize=256)
def validate_grammar(kind: str, value: str) -> None:
    """Validate request syntax without initializing an inference backend or tokenizer."""
    if not value:
        return
    import xgrammar as xgr

    if kind == "json":
        xgr.Grammar.from_json_schema(value)
    elif kind == "regex":
        xgr.Grammar.from_regex(value)
    elif kind == "grammar" and value != "json":
        xgr.Grammar.from_ebnf(value)
