from transformers import PreTrainedTokenizerFast
from transformers.models.auto.tokenization_auto import get_tokenizer_config


def get_glm5_next_tokenizer(tokenizer_name, *args, **kwargs):
    """Load the checkpoint's Rust tokenizer with the existing transformers 4 image.

    GLM-5.3 exports the transformers 5 TokenizersBackend name and a list of
    extra_special_tokens. Transformers 4 calls that list additional_special_tokens.
    The tokenizer.json and checkpoint chat template remain authoritative.
    """
    config = get_tokenizer_config(tokenizer_name, **kwargs)
    extra_tokens = config.get("extra_special_tokens", {})
    if isinstance(extra_tokens, list):
        kwargs.setdefault("additional_special_tokens", extra_tokens)
        kwargs["extra_special_tokens"] = {}
    return PreTrainedTokenizerFast.from_pretrained(tokenizer_name, *args, **kwargs)
