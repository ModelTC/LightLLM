def validate_prompt_text_length(prompt, max_req_total_len):
    if not isinstance(prompt, str):
        return

    max_prompt_chars = max_req_total_len * 8
    if len(prompt) > max_prompt_chars:
        raise ValueError(
            f"prompt text length {len(prompt)} exceeds the character limit {max_prompt_chars}, "
            f"the request is rejected before tokenization."
        )
