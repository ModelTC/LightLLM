"""Verify pinned allocation rounding and cache reuse in a standalone CUDA process.

Example:
    CUDA_VISIBLE_DEVICES=4 python test/benchmark/benchmark_pinned_allocation_rounding.py \
        --output /tmp/pinned_allocation_rounding.json
"""

import argparse
import json
import os
from pathlib import Path
import weakref

import torch


MIB = 1024 * 1024
STAT_KEYS = ("allocated_bytes.current", "active_bytes.current", "num_host_alloc", "num_host_free")


def snapshot():
    stats = torch.cuda.memory.host_memory_stats()
    return {key: stats[key] for key in STAT_KEYS}


def measure(label, numel, dtype):
    # 仅清理本独立进程的 host 缓存，确保首次申请无法命中已有尺寸桶。
    torch._C._host_emptyCache()
    before = snapshot()
    tensor = torch.empty(numel, dtype=dtype, device="cpu", pin_memory=True, requires_grad=False)
    allocated = snapshot()
    requested_bytes = tensor.nbytes
    expected_block_bytes = 1 << (requested_bytes - 1).bit_length()
    block_bytes = allocated["allocated_bytes.current"] - before["allocated_bytes.current"]
    pointer = tensor.data_ptr()
    storage_bytes = tensor.untyped_storage().nbytes()
    assert tensor.is_pinned()
    assert block_bytes == expected_block_bytes, (label, block_bytes, expected_block_bytes)
    assert allocated["active_bytes.current"] - before["active_bytes.current"] == block_bytes
    assert allocated["num_host_alloc"] - before["num_host_alloc"] == 1
    assert storage_bytes == requested_bytes

    # 没有 H2D 或共享视图；删除最后一个引用后，整块内存应进入空闲缓存。
    tensor_ref = weakref.ref(tensor)
    del tensor
    assert tensor_ref() is None
    released = snapshot()
    # 此安装版本的无事件 free 路径未扣减 active 计数；用地址复用和实际分配次数验证回收。
    assert released["allocated_bytes.current"] == allocated["allocated_bytes.current"]

    # 按桶的完整大小再申请，验证与原申请复用同一块内存，不再调用底层申请。
    reused = torch.empty(block_bytes, dtype=torch.uint8, device="cpu", pin_memory=True, requires_grad=False)
    reuse_stats = snapshot()
    same_pointer = reused.data_ptr() == pointer
    assert same_pointer
    assert reuse_stats["num_host_alloc"] == allocated["num_host_alloc"]
    del reused
    torch._C._host_emptyCache()
    cleared = snapshot()
    assert cleared["allocated_bytes.current"] == before["allocated_bytes.current"]
    assert cleared["num_host_free"] - before["num_host_free"] == 1
    return {
        "case": label,
        "dtype": str(dtype),
        "numel": numel,
        "tensor_bytes": requested_bytes,
        "storage_bytes": storage_bytes,
        "allocator_block_bytes": block_bytes,
        "tensor_mib": requested_bytes / MIB,
        "allocator_block_mib": block_bytes / MIB,
        "reuse_request_bytes": block_bytes,
        "reuse_same_pointer": same_pointer,
        "reuse_new_host_allocations": reuse_stats["num_host_alloc"] - allocated["num_host_alloc"],
        "stats": {
            "before": before,
            "allocated": allocated,
            "released": released,
            "reused": reuse_stats,
            "cache_cleared": cleared,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # reserve segment 会改变缓存清理行为，本原型验证默认 host allocator 的独立分配路径。
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        assert not os.environ.get(name), f"Run this standalone prototype with {name} unset"
    torch.cuda.init()
    warmup = torch.empty(1, dtype=torch.uint8, pin_memory=True)
    del warmup
    cases = [
        ("64 MiB uint8", 64 * MIB, torch.uint8),
        ("64 MiB + 1 byte uint8", 64 * MIB + 1, torch.uint8),
        ("65 MiB uint8", 65 * MIB, torch.uint8),
        ("128 MiB uint8", 128 * MIB, torch.uint8),
        ("65 million bytes (decimal)", 65_000_000, torch.uint8),
        ("65M int32 elements (binary M)", 65 * MIB, torch.int32),
    ]
    results = [measure(*case) for case in cases]
    output = {
        "torch": torch.__version__,
        "torch_commit": torch.version.git_version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "allocator_environment": {
            name: os.environ.get(name) for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        },
        "active_counter_note": "This build does not decrement active counters on event-free release; "
        "reuse is verified by pointer equality and unchanged num_host_alloc.",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    for result in results:
        print(
            f"{result['case']}: tensor={result['tensor_mib']:.6f} MiB, "
            f"allocator={result['allocator_block_mib']:.0f} MiB, "
            f"reuse_same_pointer={result['reuse_same_pointer']}, "
            f"reuse_new_host_allocations={result['reuse_new_host_allocations']}"
        )
    print(f"All assertions passed. Results: {args.output}")


if __name__ == "__main__":
    main()
