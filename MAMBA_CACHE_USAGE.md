# Mamba Cache Ratio-Based Allocation

## Parameters

- `--mamba_cache_ratio <float>` (default: 0.5) - Percentage of cache memory for mamba
- `--mamba_cache_size <int>` (default: None) - Explicit buffer count (backward compatible)

## Ratio Meaning

`mamba_cache_ratio = mamba_memory / total_cache_memory`

Examples:
- `0.3` → 30% mamba, 70% KV
- `0.5` → 50% mamba, 50% KV (default)
- `0.7` → 70% mamba, 30% KV

## Usage Examples

### Automatic (recommended)
```bash
python -m lightllm.server.api_server \
    --model_dir /path/to/qwen3next \
    --mem_fraction 0.9
# Uses default ratio 0.5 → 50% mamba, 50% KV
```

### Custom ratio
```bash
# For long-context workloads (more KV cache)
python -m lightllm.server.api_server \
    --model_dir /path/to/qwen3next \
    --mamba_cache_ratio 0.3  # 30% mamba, 70% KV

# For high-concurrency workloads (more mamba cache)
python -m lightllm.server.api_server \
    --model_dir /path/to/qwen3next \
    --mamba_cache_ratio 0.7  # 70% mamba, 30% KV
```

### Explicit size (backward compatible)
```bash
python -m lightllm.server.api_server \
    --model_dir /path/to/qwen3next \
    --mamba_cache_size 3000
```

## Troubleshooting

### Error: "Insufficient memory for mamba cache allocation!"

**Solution 1**: Reduce `--running_max_req_size` to calculated value or lower
**Solution 2**: Increase `--mamba_cache_ratio` to give more memory to mamba
**Solution 3**: Increase `--mem_fraction` to leave more memory for caches
