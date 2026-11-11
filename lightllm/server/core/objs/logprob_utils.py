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


def chat_token_logprob(tokenizer, token_id: int, logprob: float):
    token = tokenizer.decode([token_id], skip_special_tokens=False)
    raw_bytes = list(token.encode("utf-8")) if "\ufffd" not in token else None
    decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
    if type(decoder).__name__ == "ByteLevel" and token_id not in tokenizer.added_tokens_decoder:
        piece = tokenizer.convert_ids_to_tokens(token_id)
        if piece and all(char in _BYTE_LEVEL_BYTES for char in piece):
            raw_bytes = [_BYTE_LEVEL_BYTES[char] for char in piece]
    return {"token": token, "logprob": logprob, "bytes": raw_bytes}
