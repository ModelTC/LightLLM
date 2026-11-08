def completion_request(framework, model, prompt_ids, limit):
    if not prompt_ids or any(type(token) is not int or token < 0 for token in prompt_ids):
        raise ValueError("Require nonnegative integer prompt IDs")
    if type(limit) is not int or limit < 1:
        raise ValueError("Require a positive integer output limit")
    if framework == "lightllm":
        return "/generate", {
            "inputs": prompt_ids,
            "parameters": dict(
                do_sample=False,
                ignore_eos=False,
                max_new_tokens=limit,
                return_details=True,
                print_eos_token=True,
            ),
        }
    if framework in ("vllm", "sglang"):
        return "/v1/completions", dict(
            model=model,
            prompt=prompt_ids,
            max_tokens=limit,
            temperature=0,
            seed=3107,
            ignore_eos=False,
            return_token_ids=True,
        )
    raise ValueError("Unsupported framework")


def completion_tokens(body, framework, prompt_ids, limit, eos_ids):
    if framework == "lightllm":
        ids = [token["id"] for token in body["tokens"]]
        prompt_count, output_count = body["prompt_tokens"], body["count_output_tokens"]
        reason = body["finish_reason"]
    elif framework in ("vllm", "sglang"):
        if len(body["choices"]) != 1 or body["choices"][0].get("index") != 0:
            raise ValueError("Require exactly one completion choice")
        choice = body["choices"][0]
        if choice.get("prompt_token_ids", prompt_ids) != prompt_ids:
            raise ValueError("Service changed the exact prompt IDs")
        ids, reason = choice["token_ids"], choice["finish_reason"]
        prompt_count, output_count = body["usage"]["prompt_tokens"], body["usage"]["completion_tokens"]
    else:
        raise ValueError("Unsupported framework")
    if not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids):
        raise ValueError("Require actual nonnegative output token IDs")
    if (
        type(prompt_count) is not int
        or type(output_count) is not int
        or prompt_count != len(prompt_ids)
        or output_count != len(ids)
        or len(ids) > limit
    ):
        raise ValueError("Service counts differ from exact prompt/output IDs")
    if reason not in ("stop", "length") or any(token in eos_ids for token in ids[:-1]):
        raise ValueError("Invalid termination reason or output beyond first EOS")
    if reason == "length" and (len(ids) != limit or ids[-1] in eos_ids):
        raise ValueError("Length termination must exhaust the budget without EOS")
    if framework == "lightllm" and reason == "stop" and ids[-1] not in eos_ids:
        raise ValueError("Native natural termination must expose its declared EOS")
    return dict(token_ids=ids, finish_reason=reason, truncated=reason == "length")
