"""
Example script demonstrating weight verification via SHA256 checksums.
This feature is similar to sglang's weight verification for ensuring bit-for-bit identical weights.

Usage:
    python examples/weight_verification_example.py

Features:
    - Compute SHA256 checksums for model weights
    - Verify weights against expected checksums
    - Support for tensor-parallel distributed weights
"""

import torch
from lightllm.utils.weight_checker import WeightChecker, create_weight_checker


def example_basic_checksum():
    """
    Example 1: Basic checksum computation for a single tensor.
    """
    print("=" * 80)
    print("Example 1: Basic Checksum Computation")
    print("=" * 80)

    # Create a weight checker
    checker = create_weight_checker()

    # Create a sample weight tensor
    weight = torch.randn(512, 256, dtype=torch.float32)
    param_name = "model.layer.weight"

    # Compute checksum
    checksum = checker.compute_checksum(param_name, weight, force_bfloat16=True)
    print(f"Computed checksum for '{param_name}': {checksum}")

    # Verify checksum
    is_valid = checker.verify_checksum(param_name, weight, checksum, force_bfloat16=True)
    print(f"Checksum verification: {'PASSED' if is_valid else 'FAILED'}")
    print()


def example_multiple_weights():
    """
    Example 2: Computing and verifying checksums for multiple weights.
    """
    print("=" * 80)
    print("Example 2: Multiple Weight Verification")
    print("=" * 80)

    checker = create_weight_checker()

    # Create sample weights
    weights = {
        "model.embed.weight": torch.randn(50000, 768, dtype=torch.bfloat16),
        "model.layer0.qkv.weight": torch.randn(2304, 768, dtype=torch.bfloat16),
        "model.layer0.out.weight": torch.randn(768, 768, dtype=torch.bfloat16),
    }

    # Compute checksums for all weights
    expected_checksums = {}
    for param_name, weight in weights.items():
        checksum = checker.compute_checksum(param_name, weight)
        expected_checksums[param_name] = checksum
        print(f"Computed checksum for '{param_name}': {checksum}")

    print("\nVerifying checksums...")
    # Verify all checksums
    all_passed, failed_params = checker.compare_checksums(
        expected_checksums,
        weights,
        force_bfloat16=True
    )

    if all_passed:
        print("✓ All checksums verified successfully!")
    else:
        print(f"✗ Verification failed for: {failed_params}")
    print()


def example_tensor_parallel_checksum():
    """
    Example 3: Checksum computation for tensor-parallel sharded weights.
    """
    print("=" * 80)
    print("Example 3: Tensor Parallel Weight Checksum")
    print("=" * 80)

    # Simulate TP rank 0 and TP world size 2
    tp_rank = 0
    tp_world_size = 2

    checker = WeightChecker(tp_rank=tp_rank, tp_world_size=tp_world_size)

    # Create a full weight and shard it
    full_weight = torch.randn(1024, 768, dtype=torch.bfloat16)

    # Shard along dimension 0 (column parallel)
    shard_size = full_weight.shape[0] // tp_world_size
    shard = full_weight[tp_rank * shard_size:(tp_rank + 1) * shard_size, :]

    print(f"Full weight shape: {full_weight.shape}")
    print(f"Shard shape (rank {tp_rank}): {shard.shape}")

    # Compute checksum for the shard
    param_name = "model.layer.weight"
    shard_checksum = checker.compute_sharded_checksum(
        param_name,
        shard,
        shard_dim=0,
        force_bfloat16=True
    )

    print(f"Shard checksum (rank {tp_rank}): {shard_checksum}")

    # Get all local checksums
    local_checksums = checker.get_local_checksums()
    print(f"\nAll local checksums: {local_checksums}")
    print()


def example_model_weight_verification():
    """
    Example 4: How to integrate with TpPartBaseModel.
    """
    print("=" * 80)
    print("Example 4: Model Weight Verification Integration")
    print("=" * 80)

    print("""
To use weight verification with TpPartBaseModel:

1. After loading model weights:
   ```python
   # Enable checksum verification
   model.enable_weight_checksum_verification()
   ```

2. Compute checksums for all model weights:
   ```python
   result = model.compute_weight_checksums(force_bfloat16=True)
   print(f"Computed checksums: {result.checksums}")
   ```

3. Verify weights against expected checksums:
   ```python
   expected_checksums = {
       'pre_post': {...},
       'layer_0': {...},
       'layer_1': {...},
       ...
   }
   result = model.verify_weight_checksums(expected_checksums, force_bfloat16=True)
   if result.success:
       print("All weights verified!")
   else:
       print(f"Verification failed: {result.failed_params}")
   ```

4. Use CheckWeightsReqInput for API requests:
   ```python
   from lightllm.server.pd_io_struct import CheckWeightsReqInput

   # Create a verification request
   req = CheckWeightsReqInput(
       checksums=expected_checksums,
       force_bfloat16=True,
       verify_only=True
   )
   ```
    """)
    print()


if __name__ == "__main__":
    print("\n")
    print("=" * 80)
    print("LightLLM Weight Verification Examples")
    print("Similar to sglang's weight verification feature")
    print("=" * 80)
    print("\n")

    # Run all examples
    example_basic_checksum()
    example_multiple_weights()
    example_tensor_parallel_checksum()
    example_model_weight_verification()

    print("=" * 80)
    print("All examples completed!")
    print("=" * 80)
