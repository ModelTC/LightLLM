def logprob_info(tokenizer, token_id: int, logprob: float, rank: int):
    decoded_token = None
    if tokenizer is not None:
        decoded_token = tokenizer.decode([token_id], skip_special_tokens=False)
    return {
        "logprob": float(logprob),
        "rank": rank,
        "decoded_token": decoded_token,
    }
