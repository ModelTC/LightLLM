import sys
import argparse
import requests
import base64
import numpy as np


def _check_prompt_logprobs(res, topk: int) -> bool:
    prompt_token_ids = res.get("prompt_token_ids")
    prompt_logprobs = res.get("prompt_logprobs")
    if not isinstance(prompt_token_ids, list) or not isinstance(prompt_logprobs, list):
        return False
    if len(prompt_token_ids) != len(prompt_logprobs) or not prompt_logprobs or prompt_logprobs[0] is not None:
        return False

    for position_logprobs in prompt_logprobs[1:]:
        if len(position_logprobs) != topk:
            return False
        if any(not np.isfinite(item["logprob"]) for item in position_logprobs.values()):
            return False
    return True


def test_routing_export(url: str = "http://localhost:8000", prompt_logprobs: int = 32):
    print(f"Testing routing export at {url}")
    print("-" * 50)

    try:
        response = requests.post(
            f"{url}/generate",
            json={
                "inputs": "What is the capital of France? What is the capital of France?",
                "parameters": {
                    "max_new_tokens": 50,
                    "prompt_logprobs": prompt_logprobs,
                    "return_routed_experts": True,
                    # "repetition_penalty": 1.0,
                },
            },
            timeout=60,
        )
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to server at {url}")
        print(
            "Make sure the LightLLM server is running with " "--enable_return_routed_experts --enable_prompt_logprobs"
        )
        return False
    except requests.exceptions.Timeout:
        print("ERROR: Request timed out")
        return False

    print(f"Status: {response.status_code}")

    if response.status_code != 200:
        print(f"ERROR: Request failed with status {response.status_code}")
        print(f"Response: {response.text}")
        return False

    res = response.json()
    print(res["prompt_logprobs"])
    print(res["prompt_token_ids"])
    prompt_logprobs_ok = _check_prompt_logprobs(res, prompt_logprobs)

    if "routed_experts" not in res or not res["routed_experts"]:
        print("\nWARNING: No routed_experts in response.")
        print("This could mean:")
        print("  - The model is not a MoE model")
        print("  - The server was not started with --enable_return_routed_experts")
        print("  - The routing capture manager was not initialized")
        return False
    routing_info = res["routed_experts"]
    shape = routing_info["shape"]
    dtype_str = routing_info["dtype"]
    dtype = np.dtype(dtype_str)
    data = base64.b64decode(routing_info["data"])
    routing_array = np.frombuffer(data, dtype=dtype).reshape(shape)

    print(f"\n{'=' * 50}")
    print("ROUTING CAPTURE SUCCESS!")
    print(f"{'=' * 50}")
    print(f"Shape: {shape}")
    print(f"Dtype: {dtype}")
    print(f"Num tokens: {shape[0]}")
    print(f"Num MoE layers: {shape[1]}")
    print(f"Top-K: {shape[2]}")

    # Compute payload size savings
    int32_size = np.prod(shape) * 4
    actual_size = len(data)
    savings = (1 - actual_size / int32_size) * 100
    print(f"Payload: {actual_size} bytes (vs {int32_size} bytes with int32, {savings:.0f}% smaller)")

    print(f"\nSample routing (first layer, first 5 tokens):")
    num_tokens_to_show = min(shape[0], 5)
    for i in range(num_tokens_to_show):
        print(f"  Token {i}: experts {routing_array[i, 0, :].tolist()}")

    if np.all(routing_array == 0):
        print("\nWARNING: All routing data is zeros. Capture may not be working correctly.")
        return False

    if not prompt_logprobs_ok:
        return False

    print("\nTest PASSED!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test R3 routing export feature")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--prompt-logprobs", type=int, default=32, help="prompt_logprobs value to request")
    args = parser.parse_args()

    success = test_routing_export(args.url, prompt_logprobs=args.prompt_logprobs)
    sys.exit(0 if success else 1)
