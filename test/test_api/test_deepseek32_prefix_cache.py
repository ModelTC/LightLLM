"""
Integration tests for DeepSeek v3.2 prefix cache fix.

Tests that prefix cache produces consistent results across multiple requests.
"""
import pytest
import time
import json


def test_prefix_cache_consistency():
    """Test that prefix cache produces consistent results"""
    base_url = "http://localhost:8088"

    prompt = "Explain quantum computing in simple terms."

    print("=" * 80)
    print("Testing prefix cache consistency")
    print("=" * 80)

    # First request (no cache)
    print("Sending first request (no cache)...")
    start_time = time.time()
    response1 = pytest.helpers.request(
        "POST",
        f"{base_url}/generate",
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 50,
                "temperature": 0.0,
            },
        },
    )
    elapsed1 = time.time() - start_time

    assert response1.status_code == 200, f"First request failed: {response1.text}"
    result1 = response1.json()

    if "generated_text" not in result1 and "text" in result1:
        result1["generated_text"] = result1["text"]

    print(f"  First request completed in {elapsed1:.3f}s")
    print(f"  Output length: {len(result1.get('generated_text', ''))} chars")

    # Second request (should hit cache)
    print("Sending second request (should hit cache)...")
    start_time = time.time()
    response2 = pytest.helpers.request(
        "POST",
        f"{base_url}/generate",
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 50,
                "temperature": 0.0,
            },
        },
    )
    elapsed2 = time.time() - start_time

    assert response2.status_code == 200, f"Second request failed: {response2.text}"
    result2 = response2.json()

    if "generated_text" not in result2 and "text" in result2:
        result2["generated_text"] = result2["text"]

    print(f"  Second request completed in {elapsed2:.3f}s")
    print(f"  Output length: {len(result2.get('generated_text', ''))} chars")

    # Verify identical results
    text1 = result1.get("generated_text", "")
    text2 = result2.get("generated_text", "")

    assert text1 == text2, f"Outputs differ!\n  First:  {text1[:100]}...\n  Second: {text2[:100]}..."

    print("  ✓ Outputs are identical")
    print(f"  Speedup: {elapsed1 / elapsed2:.2f}x")
    print("✓ Prefix cache consistency test passed!")
    print()


def test_prefix_cache_multiple_requests():
    """Test multiple identical requests"""
    base_url = "http://localhost:8088"

    prompt = "What is the capital of France?"

    print("=" * 80)
    print("Testing multiple identical requests")
    print("=" * 80)

    results = []
    times = []

    for i in range(5):
        print(f"Sending request {i + 1}/5...")
        start_time = time.time()
        response = pytest.helpers.request(
            "POST",
            f"{base_url}/generate",
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 20,
                    "temperature": 0.0,
                },
            },
        )
        elapsed = time.time() - start_time
        times.append(elapsed)

        assert response.status_code == 200, f"Request {i + 1} failed: {response.text}"
        result = response.json()

        if "generated_text" not in result and "text" in result:
            result["generated_text"] = result["text"]

        results.append(result.get("generated_text", ""))
        print(f"  Request {i + 1} completed in {elapsed:.3f}s")

    # All results should be identical
    for i in range(1, len(results)):
        assert results[i] == results[0], f"Request {i + 1} output differs from request 1!"

    print(f"  ✓ All {len(results)} requests produced identical output")
    print(f"  Average time: {sum(times) / len(times):.3f}s")
    print(f"  First request: {times[0]:.3f}s")
    print(f"  Last request: {times[-1]:.3f}s")
    print("✓ Multiple requests test passed!")
    print()


def test_prefix_cache_partial_hit():
    """Test partial prefix cache hits"""
    base_url = "http://localhost:8088"

    prompts = [
        "The sky is",
        "The sky is blue",
        "The sky is blue and",
        "The sky is blue and the",
    ]

    print("=" * 80)
    print("Testing partial prefix cache hits")
    print("=" * 80)

    results = []

    for i, prompt in enumerate(prompts):
        print(f"Sending request {i + 1}/4: '{prompt}'")
        start_time = time.time()
        response = pytest.helpers.request(
            "POST",
            f"{base_url}/generate",
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 10,
                    "temperature": 0.0,
                },
            },
        )
        elapsed = time.time() - start_time

        assert response.status_code == 200, f"Request {i + 1} failed: {response.text}"
        result = response.json()

        if "generated_text" not in result and "text" in result:
            result["generated_text"] = result["text"]

        results.append(result.get("generated_text", ""))
        print(f"  Completed in {elapsed:.3f}s, output: {results[-1][:50]}...")

    # Verify no crashes and reasonable outputs
    assert len(results) == 4, "Not all requests completed!"
    for i, result in enumerate(results):
        assert len(result) > 0, f"Request {i + 1} produced empty output!"

    print(f"  ✓ All {len(results)} requests completed successfully")
    print("✓ Partial cache hit test passed!")
    print()


def test_prefix_cache_different_prompts():
    """Test that different prompts don't incorrectly hit cache"""
    base_url = "http://localhost:8088"

    prompts = [
        "What is AI?",
        "What is ML?",
        "What is DL?",
    ]

    print("=" * 80)
    print("Testing different prompts (should not share cache)")
    print("=" * 80)

    results = []

    for i, prompt in enumerate(prompts):
        print(f"Sending request {i + 1}/3: '{prompt}'")
        response = pytest.helpers.request(
            "POST",
            f"{base_url}/generate",
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 30,
                    "temperature": 0.0,
                },
            },
        )

        assert response.status_code == 200, f"Request {i + 1} failed: {response.text}"
        result = response.json()

        if "generated_text" not in result and "text" in result:
            result["generated_text"] = result["text"]

        results.append(result.get("generated_text", ""))
        print(f"  Output: {results[-1][:50]}...")

    # Outputs should be different (different prompts)
    assert results[0] != results[1], "Prompt 1 and 2 should have different outputs!"
    assert results[1] != results[2], "Prompt 2 and 3 should have different outputs!"

    print(f"  ✓ All {len(results)} requests produced different outputs as expected")
    print("✓ Different prompts test passed!")
    print()


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
