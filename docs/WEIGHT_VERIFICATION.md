# Weight Verification via SHA256 Checksums

This feature provides bitwise weight verification through SHA256 checksums, similar to [sglang's weight verification feature](https://github.com/sgl-project/sglang/pull/17009). It enables integrity checking for model weights, especially useful in distributed training/inference scenarios with tensor parallelism.

## Overview

The weight verification system allows you to:
- Compute SHA256 checksums for model weights
- Verify that loaded weights are bit-for-bit identical to expected values
- Support tensor-parallel (TP) sharded weight verification
- Detect corruption or mismatch in weight loading

## Key Components

### 1. WeightChecker Utility (`lightllm/utils/weight_checker.py`)

Core utility class for computing and verifying checksums.

```python
from lightllm.utils.weight_checker import WeightChecker, create_weight_checker

# Create a weight checker
checker = create_weight_checker(tp_rank=0, tp_world_size=1)

# Compute checksum for a tensor
checksum = checker.compute_checksum("param_name", tensor, force_bfloat16=True)

# Verify checksum
is_valid = checker.verify_checksum("param_name", tensor, expected_checksum)
```

### 2. Base Weight Classes

Extended `BaseWeightTpl` and `MMWeightTpl` to support checksum computation:

```python
# Enable checksum verification
weight.enable_checksum_verification()

# Compute checksums for all weights in the layer
checksums = weight.compute_checksums(force_bfloat16=True)

# Verify against expected checksums
passed = weight.verify_checksums(expected_checksums, force_bfloat16=True)
```

### 3. TpPartBaseModel Integration

Model-level checksum verification API:

```python
# Enable checksum verification for all weights
model.enable_weight_checksum_verification()

# Compute checksums for entire model
result = model.compute_weight_checksums(force_bfloat16=True)
print(f"Checksums: {result.checksums}")

# Verify against expected checksums
result = model.verify_weight_checksums(expected_checksums, force_bfloat16=True)
if result.success:
    print("All weights verified!")
else:
    print(f"Failed: {result.failed_params}")
```

### 4. Request Structures (`lightllm/server/pd_io_struct.py`)

Data structures for API integration:

```python
from lightllm.server.pd_io_struct import CheckWeightsReqInput, CheckWeightsResult

# Create verification request
req = CheckWeightsReqInput(
    checksums=expected_checksums,
    force_bfloat16=True,
    verify_only=True  # Only verify, don't compute
)

# Process result
result = CheckWeightsResult(
    success=True,
    checksums={...},
    failed_params=[],
    message="Verification successful"
)
```

## Usage Examples

### Example 1: Basic Checksum Computation

```python
import torch
from lightllm.utils.weight_checker import create_weight_checker

# Create checker
checker = create_weight_checker()

# Create a weight tensor
weight = torch.randn(512, 256, dtype=torch.float32)

# Compute checksum (with bfloat16 casting for consistency)
checksum = checker.compute_checksum("model.layer.weight", weight, force_bfloat16=True)
print(f"Checksum: {checksum}")

# Verify
is_valid = checker.verify_checksum("model.layer.weight", weight, checksum)
print(f"Valid: {is_valid}")
```

### Example 2: Model-Level Verification

```python
# After model initialization
model = TpPartBaseModel(kvargs)

# Enable checksum verification
model.enable_weight_checksum_verification()

# Compute all checksums
result = model.compute_weight_checksums(force_bfloat16=True)

# Save checksums to file for later verification
import json
with open("model_checksums.json", "w") as f:
    json.dump(result.checksums, f, indent=2)

# Later: verify against saved checksums
with open("model_checksums.json", "r") as f:
    expected_checksums = json.load(f)

result = model.verify_weight_checksums(expected_checksums)
if not result.success:
    print(f"Verification failed for: {result.failed_params}")
```

### Example 3: Tensor Parallel Sharded Weights

```python
# For TP distributed weights
checker = WeightChecker(tp_rank=0, tp_world_size=4)

# Compute checksum for a sharded weight
# The shard_dim indicates which dimension the tensor is sharded along
shard_checksum = checker.compute_sharded_checksum(
    param_name="model.layer.weight",
    tensor=weight_shard,
    shard_dim=0,  # Sharded along dimension 0 (Column Parallel)
    force_bfloat16=True
)

# Get all local checksums
local_checksums = checker.get_local_checksums()
```

## Technical Details

### Why bfloat16 Casting?

The `force_bfloat16` parameter ensures bit-perfect alignment across different systems and loading methods. By casting to bfloat16 before hashing:
- Eliminates floating-point precision differences
- Ensures consistent checksums regardless of original dtype
- Matches the approach used in sglang

### Checksum Computation Process

1. **Tensor to CPU**: Move tensor from GPU to CPU via PCIe
2. **Type Casting**: Optionally cast to bfloat16 for consistency
3. **SHA256 Hashing**: Compute SHA256 hash of the tensor bytes
4. **Hex Digest**: Return hexadecimal string representation

### Tensor Parallel Support

For sharded weights across TP ranks:
- Each rank computes checksum for its local shard
- Shard dimension information is stored (0 for Column Parallel, 1 for Row Parallel)
- Checksums can be gathered across ranks for full parameter reconstruction verification

## API Reference

### WeightChecker Methods

- `compute_checksum(param_name, tensor, force_bfloat16=True)`: Compute SHA256 for a tensor
- `compute_sharded_checksum(param_name, tensor, shard_dim, force_bfloat16=True)`: Compute checksum for TP shard
- `verify_checksum(param_name, tensor, expected_checksum, force_bfloat16=True)`: Verify single checksum
- `compare_checksums(checksums, weights, force_bfloat16=True)`: Verify multiple checksums
- `get_local_checksums()`: Get all computed checksums
- `clear_checksums()`: Clear stored checksums

### TpPartBaseModel Methods

- `enable_weight_checksum_verification()`: Enable checksum support for all weights
- `compute_weight_checksums(force_bfloat16=True)`: Compute checksums for entire model
- `verify_weight_checksums(expected_checksums, force_bfloat16=True)`: Verify model weights

### Data Structures

**CheckWeightsReqInput**:
- `checksums`: Dict[str, str] - Expected checksums
- `force_bfloat16`: bool - Whether to cast to bfloat16
- `verify_only`: bool - Only verify, don't compute new checksums

**CheckWeightsResult**:
- `success`: bool - Whether verification passed
- `checksums`: Dict[str, str] - Computed checksums
- `failed_params`: List[str] - Parameters that failed verification
- `message`: str - Result message

## Use Cases

1. **Distributed Training/Inference**: Verify weights are correctly loaded across all TP ranks
2. **Model Checkpointing**: Ensure checkpointed weights match original weights
3. **Weight Corruption Detection**: Detect corruption during loading or transfer
4. **Reproducibility**: Verify bit-identical weights for reproducible experiments
5. **Model Integrity**: Ensure model weights haven't been tampered with

## Performance Considerations

- **CPU Transfer**: Checksum computation requires moving tensors to CPU
- **Memory Overhead**: Temporary CPU memory allocation for weight copies
- **Computation Time**: SHA256 hashing takes ~O(n) time per weight
- **Recommendation**: Use checksum verification during model initialization, not during inference

## Comparison with sglang

This implementation is inspired by and similar to [sglang PR #17009](https://github.com/sgl-project/sglang/pull/17009):

**Similarities**:
- SHA256 checksum-based verification
- bfloat16 casting for consistency
- Support for tensor-parallel sharded weights
- Request/response structures for API integration

**Differences**:
- Adapted to LightLLM's architecture (TpPartBaseModel, MMWeightTpl)
- Integrated with LightLLM's existing weight loading system
- Simplified API for ease of use

## Example Script

See [`examples/weight_verification_example.py`](../examples/weight_verification_example.py) for complete working examples.

## Future Enhancements

Potential improvements:
- [ ] Distributed checksum gathering across all TP ranks
- [ ] Automatic checksum computation during weight loading
- [ ] API endpoint for remote checksum verification
- [ ] Checksum caching for faster repeated verification
- [ ] Support for other hash algorithms (MD5, SHA512)

## References

- [sglang PR #17009: Bitwise Weight Verification](https://github.com/sgl-project/sglang/pull/17009)
- [SHA256 Hashing in Python](https://docs.python.org/3/library/hashlib.html)
- [Tensor Parallelism in LightLLM](https://github.com/ModelTC/lightllm)
