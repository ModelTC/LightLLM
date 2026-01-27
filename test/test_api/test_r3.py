import sys
import argparse
import requests
import base64
import numpy as np


def test_routing_export(url: str = "http://localhost:8000"):
    print(f"Testing routing export at {url}")
    print("-" * 50)

    try:
        response = requests.post(
            f"{url}/generate",
            json={
                "inputs": "What is the capital of France?",
                "parameters": {
                    "max_new_tokens": 50,
                    "return_routed_experts": True,
                },
            },
            timeout=60,
        )
    except requests.exceptions.ConnectionError:
        print(f"ERROR: Cannot connect to server at {url}")
        print("Make sure the LightLLM server is running with --enable_return_routed_experts")
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
    print(f"Generated text: {res.get('generated_text', 'N/A')[:100]}...")

    if "routed_experts" not in res or not res["routed_experts"]:
        print("\nWARNING: No routed_experts in response.")
        print("This could mean:")
        print("  - The model is not a MoE model")
        print("  - The server was not started with --enable_return_routed_experts")
        print("  - The routing capture manager was not initialized")
        return False

    routing_info = res["routed_experts"]
    shape = routing_info["shape"]
    dtype = np.dtype(routing_info["dtype"])
    data = base64.b64decode(routing_info["data"])
    routing_array = np.frombuffer(data, dtype=dtype).reshape(shape)

    print(f"\n{'=' * 50}")
    print("ROUTING CAPTURE SUCCESS!")
    print(f"{'=' * 50}")
    print(f"Shape: {shape}")
    print(f"Dtype: {dtype}")
    print(f"Num MoE layers: {shape[0]}")
    print(f"Num tokens: {shape[1]}")
    print(f"Top-K: {shape[2]}")

    print(f"\nSample routing (first layer, first 5 tokens):")
    num_tokens_to_show = min(5, shape[1])
    for i in range(num_tokens_to_show):
        print(f"  Token {i}: experts {routing_array[0, i, :].tolist()}")

    if np.all(routing_array == 0):
        print("\nWARNING: All routing data is zeros. Capture may not be working correctly.")
        return False

    print("\nTest PASSED!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test R3 routing export feature")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    args = parser.parse_args()

    success = test_routing_export(args.url)
    sys.exit(0 if success else 1)
