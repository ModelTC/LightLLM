import json
import re

from typing import Any, Dict


def logprob_info(tokenizer, token_id: int, logprob: float, rank: int):
    decoded_token = None
    if tokenizer is not None:
        decoded_token = tokenizer.decode([token_id], skip_special_tokens=False)
    return {
        "logprob": float(logprob),
        "rank": rank,
        "decoded_token": decoded_token,
    }


# Hugging Face ByteLevel encodes raw bytes as printable Unicode characters.
_PRINTABLE_BYTES = [*range(33, 127), *range(161, 173), *range(174, 256)]
_BYTE_LEVEL_BYTES = {chr(byte): byte for byte in _PRINTABLE_BYTES}
_BYTE_LEVEL_BYTES.update(
    (chr(256 + offset), byte) for offset, byte in enumerate(byte for byte in range(256) if byte not in _PRINTABLE_BYTES)
)


def _decoder_types(decoder):
    if decoder is None:
        return set()
    config = json.loads(decoder.__getstate__())

    def types(node):
        result = {node.get("type")}
        for child in node.get("decoders", []):
            result.update(types(child))
        return result

    return types(config)


def chat_token_logprob(
    tokenizer: Any, token_id: int, logprob: float, prefix_ids=(), pending_ids=(), decode_options=None
) -> Dict[str, Any]:
    token = tokenizer.decode([token_id], skip_special_tokens=False)
    standalone_token = token
    special = token_id in getattr(tokenizer, "all_special_ids", ())
    if (prefix_ids or pending_ids) and "\ufffd" not in standalone_token and not special:
        options = decode_options or {"skip_special_tokens": False}
        prefix = tokenizer.decode([*prefix_ids, *pending_ids], **options)
        combined = tokenizer.decode([*prefix_ids, *pending_ids, token_id], **options)
        if combined.startswith(prefix):
            token = combined[len(prefix) :]
    raw_bytes = list(token.encode("utf-8")) if "\ufffd" not in token else None
    decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
    decoder_types = _decoder_types(decoder)
    if token_id not in getattr(tokenizer, "added_tokens_decoder", {}):
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if "ByteLevel" in decoder_types and piece and all(char in _BYTE_LEVEL_BYTES for char in piece):
            raw_bytes = [_BYTE_LEVEL_BYTES[char] for char in piece]
            token = standalone_token
        elif "ByteFallback" in decoder_types and piece and re.fullmatch(r"<0x[0-9a-fA-F]{2}>", piece):
            raw_bytes = [int(piece[3:5], 16)]
            token = standalone_token
    return {"token": token, "logprob": logprob, "bytes": raw_bytes}


class ChatTokenDecoder:
    def __init__(self, tokenizer, skip_special_tokens=True, spaces_between_special_tokens=True):
        self.tokenizer = tokenizer
        self.prefix_ids = []
        self.pending_ids = []
        self.options = {
            "skip_special_tokens": skip_special_tokens,
            "spaces_between_special_tokens": spaces_between_special_tokens,
        }

    def set_prompt(self, token_ids):
        from lightllm.server.embed_cache.utils import MULTIMODAL_TOKEN_ID_START

        self.prefix_ids = [token_id for token_id in token_ids if token_id < MULTIMODAL_TOKEN_ID_START]
        self.pending_ids = []

    def token_logprob(self, token_id, score):
        return chat_token_logprob(self.tokenizer, token_id, score, self.prefix_ids, self.pending_ids, self.options)

    def advance(self, token_id):
        prefix = self.tokenizer.decode(self.prefix_ids, **self.options)
        self.pending_ids.append(token_id)
        combined = self.tokenizer.decode(self.prefix_ids + self.pending_ids, **self.options)
        if len(combined) > len(prefix) and not combined.endswith("\ufffd"):
            self.prefix_ids = self.pending_ids
            self.pending_ids = []
