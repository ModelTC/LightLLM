"""
Test Triton decode backend with MTP diverse attention integration (single token mode).

在静态 MTP 模式下，每个请求有 1 个 Q token，但第 i 个请求只能看到前 i+1 个 KV。
输出形状和输入 Q 形状相同：[batch_size, num_heads, head_dim]
"""
import torch
import os
import sys

os.environ['PYTHONPATH'] = '/data/nvme0/chenjunyi/project/lightllm'
sys.path.insert(0, '/data/nvme0/chenjunyi/project/lightllm')

# Manually set mtp_step before importing
mtp_step = int(os.environ.get('MTP_STEP', '0'))

# Mock the environment setup
import json
os.environ['LIGHTLLM_START_ARGS'] = json.dumps({
    'mtp_step': mtp_step,
    'model_dir': '/tmp/test_model',
    'max_total_token_num': 10000,
})

from lightllm.common.basemodel.attention.triton.fp import TritonAttBackend, TritonDecodeAttState
from lightllm.common.basemodel.attention.base_att import AttControl
from lightllm.utils.envs_utils import get_env_start_args


class MockReqManager:
    """Mock request manager for testing"""
    def __init__(self, req_to_token_indexs):
        self.req_to_token_indexs = req_to_token_indexs


class MockInferState:
    """Mock infer state for testing MTP mode"""
    def __init__(
        self,
        batch_size,
        max_kv_seq_len,
        req_to_tokens,
        b_req_idx,
        b_seq_len,
        input_ids,
    ):
        self.batch_size = batch_size
        self.max_kv_seq_len = max_kv_seq_len
        self.req_manager = MockReqManager(req_to_tokens)
        self.b_req_idx = b_req_idx
        self.b_seq_len = b_seq_len
        self.input_ids = input_ids


def setup_mtp_test_data(kv_len, group_size, batch_groups, test_dtype=torch.bfloat16, device="cuda", seed=42):
    """
    Setup MTP test data for single token mode.

    在静态 MTP 模式下：
    - 每个请求有 1 个 Q token
    - 组内第 i 个请求只能看到前 i+1 个 KV
    - 组内请求共享相同的 KV 前缀
    """
    torch.manual_seed(seed)

    num_heads = 32
    kv_head_num = 4  # gqa_group_size = 8
    head_dim = 128

    batch_size = batch_groups * group_size

    # KV pool
    kv_pool_size = batch_groups * kv_len
    k = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)
    v = torch.randn(size=(kv_pool_size, kv_head_num, head_dim), dtype=test_dtype, device=device)

    # req_to_tokens: [batch, max_kv_len]
    max_kv_len = group_size
    req_to_tokens = torch.zeros((batch_size, max_kv_len), dtype=torch.int32, device=device)

    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device=device)
    b_seq_len = torch.zeros(batch_size, dtype=torch.int32, device=device)
    input_ids = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # Q: [batch_size, num_heads, head_dim] - 每个请求 1 个 Q token
    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=test_dtype, device=device)

    # Setup KV indices
    for group_idx in range(batch_groups):
        group_start = group_idx * group_size
        kv_base = group_idx * kv_len

        for member_idx in range(group_size):
            batch_idx = group_start + member_idx
            # 第 member_idx 个请求可以看到前 member_idx+1 个 KV
            b_seq_len[batch_idx] = member_idx + 1

            # Setup KV indices - 组内请求共享相同的 KV 前缀
            for kv_pos in range(member_idx + 1):
                req_to_tokens[batch_idx, kv_pos] = kv_base + kv_pos

    return q, k, v, req_to_tokens, b_req_idx, b_seq_len, input_ids


def test_mtp_decode():
    """Test Triton MTP decode attention (single token mode)"""
    print("=" * 80)
    print("Testing Triton MTP Decode Attention (Single Token Mode)")
    print("=" * 80)
    print(f"MTP step: {mtp_step}")

    kv_len = 100
    group_size = 2  # With mtp_step=1, group_size=2
    batch_groups = 2

    q, k, v, req_to_tokens, b_req_idx, b_seq_len, input_ids = setup_mtp_test_data(
        kv_len, group_size, batch_groups
    )

    print(f"kv_len: {kv_len}, group_size: {group_size}, batch_groups: {batch_groups}")
    print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")
    print(f"b_seq_len: {b_seq_len}")

    # Calculate actual batch size
    actual_batch_size = b_seq_len.shape[0]

    # Create mock infer state
    infer_state = MockInferState(
        batch_size=actual_batch_size,
        max_kv_seq_len=kv_len,
        req_to_tokens=req_to_tokens,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        input_ids=input_ids,
    )

    print(f"Actual batch size: {actual_batch_size}")

    # Create Triton backend and decode state
    backend = TritonAttBackend(model=None)
    decode_state = backend.create_att_decode_state(infer_state)

    # Initialize state (this should setup MTP-related tensors)
    decode_state.init_state()

    print(f"MTP size: {decode_state.mtp_size}")
    print(f"b_mark_shared_group: {decode_state.b_mark_shared_group}")

    # Call decode attention
    att_control = AttControl()

    try:
        output = decode_state.decode_att(
            q=q,
            k=k,
            v=v,
            att_control=att_control,
            alloc_func=lambda shape, dtype, device='cuda': torch.empty(shape, dtype=dtype, device=device)
        )
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0, 0, :4]}")

        # Verify output shape matches input Q shape
        assert output.shape == q.shape, f"Output shape {output.shape} should match input Q shape {q.shape}"
        print("SUCCESS: Output shape matches input Q shape!")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_non_mtp_decode():
    """Test Triton non-MTP decode attention (baseline)"""
    print("=" * 80)
    print("Testing Triton Non-MTP Decode Attention (Baseline)")
    print("=" * 80)
    print(f"MTP step: {mtp_step}")

    kv_len = 100
    batch_size = 4
    num_heads = 32
    kv_head_num = 4
    head_dim = 128

    torch.manual_seed(42)

    # Standard GQA decode: one Q token per request
    q = torch.randn(size=(batch_size, num_heads, head_dim), dtype=torch.bfloat16, device="cuda")
    k = torch.randn(size=(batch_size * kv_len, kv_head_num, head_dim), dtype=torch.bfloat16, device="cuda")
    v = torch.randn(size=(batch_size * kv_len, kv_head_num, head_dim), dtype=torch.bfloat16, device="cuda")

    req_to_tokens = torch.arange(batch_size * kv_len, dtype=torch.int32, device="cuda").view(batch_size, kv_len)
    b_req_idx = torch.arange(batch_size, dtype=torch.int32, device="cuda")
    b_seq_len = torch.full((batch_size,), kv_len, dtype=torch.int32, device="cuda")
    input_ids = torch.zeros(batch_size, dtype=torch.int32, device="cuda")

    print(f"Q shape: {q.shape}, K shape: {k.shape}, V shape: {v.shape}")

    # Create mock infer state
    infer_state = MockInferState(
        batch_size=batch_size,
        max_kv_seq_len=kv_len,
        req_to_tokens=req_to_tokens,
        b_req_idx=b_req_idx,
        b_seq_len=b_seq_len,
        input_ids=input_ids,
    )

    # Create Triton backend and decode state
    backend = TritonAttBackend(model=None)
    decode_state = backend.create_att_decode_state(infer_state)

    # Initialize state
    decode_state.init_state()

    print(f"MTP size: {decode_state.mtp_size}")
    print(f"b_mark_shared_group: {decode_state.b_mark_shared_group}")

    # Call decode attention
    att_control = AttControl()

    try:
        output = decode_state.decode_att(
            q=q,
            k=k,
            v=v,
            att_control=att_control,
            alloc_func=lambda shape, dtype, device='cuda': torch.empty(shape, dtype=dtype, device=device)
        )
        print(f"Output shape: {output.shape}")
        print(f"Output sample: {output[0, 0, :4]}")
        print("SUCCESS: Non-MTP decode attention completed!")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'mtp':
        # Test MTP mode
        test_mtp_decode()
    else:
        # Test non-MTP first (baseline)
        test_non_mtp_decode()
