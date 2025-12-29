# DeepSeek v3.2 Prefix Cache Bug 修复文档

## 问题描述

### 现象
当使用 DeepSeek v3.2 模型并启用 prefix cache（radix cache）时，**同样的请求第一次输出正常，第二次输出结果错误**。

### 复现步骤
1. 第一次发送请求（无 prefix cache 命中）→ 输出正确 ✓
2. 第二次发送相同请求（命中 prefix cache）→ 输出错误 ✗

## 问题根因分析

### DeepSeek v3.2 的 NSA 机制

DeepSeek v3.2 使用了 **NSA (Neighbor-aware Sparse Attention)** 机制，这是其与 DeepSeek v2 的主要区别：

```
标准 Attention:  每个 token 关注所有历史 token
NSA Attention:   每个 token 只关注通过 top-k 算法选出的部分历史 token
```

为了实现 NSA，DeepSeek v3.2 需要维护额外的数据结构：

| 数据结构 | 用途 | 内存管理器 |
|---------|------|-----------|
| `kv_buffer` | 存储 main KV cache | `mem_manager.kv_buffer` |
| `indexer_ks` | 存储 indexer key（用于 top-k 选择） | `indexer_ks_mem_manager.kv_buffer` |
| `topk_indices` | 存储 top-k 选择结果 | - |

### 代码执行流程对比

#### 场景 1：第一次请求（无 prefix cache 命中）

```
请求: "你好，世界" (3 个 token，假设)

1. prefix_cache.match_prefix() → 未命中
   - b_ready_cache_len = [0] (所有 token 都是新的)
   - mem_index = [100, 101, 102] (分配 3 个新位置)

2. context_forward() → 逐层处理

3. 每层的 _get_qkv():
   input = [token0, token1, token2]  # 3 个 token
   q = [q0, q1, q2]                  # 3 个 token 的 query

   # 调用 indexer.get_indices(input, q, ...)
   └─> _get_q_k_bf16() 生成 k_fp8 (3 个 token)
   └─> deep_gemm.fp8_mqa_logits() 计算 topk (3 个结果)
   └─> destindex_copy_indexer_ks(
          k_fp8,          # 长度 = 3
          k_scale,        # 长度 = 3
          mem_index,      # 长度 = 3 ✓ 匹配
          ...
      )
   └─> 复制 3 个 token 的 indexer_ks 到位置 [100, 101, 102]

   结果: topk_indices = [idx0, idx1, idx2]  (3 个)
        indexer_ks_mem_manager 位置 [100, 101, 102] 有数据 ✓

4. _post_cache_kv():
   └─> 复制 KV cache 到位置 [100, 101, 102]

5. _init_nsa_indexing_structures():
   └─> 从 indexer_ks_mem_manager 提取数据 → 正确 ✓
   └─> 计算 topk → 正确 ✓

输出: 正确 ✓
```

#### 场景 2：第二次请求（命中 prefix cache）

```
请求: "你好，世界" (3 个 token)

1. prefix_cache.match_prefix() → 命中前 2 个 token
   - b_ready_cache_len = [2] (前 2 个 token 来自缓存)
   - mem_index = [103] (只有第 3 个 token 是新的)

2. copy_kv_index_to_req():
   └─> 更新 req_to_token_indexs
   └─> 将所有 3 个 token 的 KV cache 复制到新位置 [103, 104, 105]

   重要: req_to_token_indexs 现在指向连续的新位置 [103, 104, 105]

3. context_forward() → 逐层处理

4. 每层的 _get_qkv():
   input = [token0, token1, token2]  # 仍然是 3 个 token！
   q = [q0, q1, q2]                  # 仍然是 3 个 token！

   # 调用 indexer.get_indices(input, q, ...)
   └─> _get_q_k_bf16() 生成 k_fp8 (3 个 token) ⚠️
   └─> deep_gemm.fp8_mqa_logits() 计算 topk (3 个结果) ⚠️
   └─> destindex_copy_indexer_ks(
          k_fp8,          # 长度 = 3 ⚠️
          k_scale,        # 长度 = 3 ⚠️
          mem_index,      # 长度 = 1 ⚠️ 不匹配！
          ...
      )
   └─> Triton kernel: seq_len = DestLoc.shape[0] = 1
   └─> 只复制前 1 个 token 的 indexer_ks 到位置 [103]

   问题:
   - token0, token1 的 indexer_ks 没有被复制到新位置 [103, 104]
   - token2 的 indexer_ks 被复制到位置 [103]（错误的位置）

   当前状态:
   - req_to_token_indexs = [103, 104, 105]
   - indexer_ks_mem_manager:
     * 位置 [103] 有数据（但这是 token2 的，不是 token0 的）✗
     * 位置 [104] 无数据（应该是 token1 的）✗
     * 位置 [105] 无数据（应该是 token2 的）✗

5. _init_nsa_indexing_structures():
   └─> extract_indexer_ks(kv_buffer, req_all_mem_index=[103, 104, 105])
   └─> 从位置 [103, 104, 105] 提取数据
   └─> 位置 [103] 的数据错误（token2 的数据，不是 token0 的）
   └─> 位置 [104, 105] 无数据（可能是零或垃圾值）

   结果: topk_indices 计算错误 ✗

输出: 错误 ✗
```

### 问题本质

```
┌─────────────────────────────────────────────────────────────┐
│                    长度不匹配问题                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  _get_qkv(input, q, ...)                                     │
│       │                                                      │
│       ├─ input:  [token0, token1, token2]  (len=3)          │
│       ├─ q:       [q0, q1, q2]                (len=3)        │
│       │                                                      │
│       └─> indexer.get_indices(input, q, ...)                │
│              │                                               │
│              ├─ k_fp8:    [k0, k1, k2]        (len=3)       │
│              ├─ k_scale:  [s0, s1, s2]        (len=3)       │
│              │                                               │
│              └─> destindex_copy_indexer_ks(                  │
│                     k_fp8,                    (len=3)        │
│                     k_scale,                  (len=3)        │
│                     mem_index,                (len=1)  ✗    │
│                     ...                                      │
│                 )                                            │
│                                                              │
│  Triton kernel 行为:                                         │
│  for i in range(DestLoc.shape[0]):  # 只循环 1 次            │
│      kv_buffer[DestLoc[i]] = k_fp8[i]                        │
│                                                              │
│  结果:                                                       │
│  - 只复制了 k_fp8[0] (token0 的 indexer_ks)                   │
│  - k_fp8[1], k_fp8[2] (token1, token2) 没有被复制            │
│  - token1, token2 的 indexer_ks 仍然在旧位置                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## 修复方案

### 核心思想

**确保 `destindex_copy_indexer_ks` 的输入参数长度一致**

修改 `_get_qkv` 方法，当检测到 prefix cache 命中时：
1. 识别哪些 token 是新 token（不在 cache 中）
2. **只选择新 token** 的 `input` 和 `q` 来调用 `indexer.get_indices`
3. 这样 `k_fp8`, `k_scale` 的长度与 `mem_index` 的长度一致

### 修复后的代码流程

```
第二次请求（命中 prefix cache）

1. prefix_cache.match_prefix() → 命中前 2 个
   - b_ready_cache_len = [2]
   - mem_index = [103]

2. copy_kv_index_to_req() → 复制 KV cache

3. _get_qkv():
   input = [token0, token1, token2]  (len=3)
   q = [q0, q1, q2]                  (len=3)

   检测: b_ready_cache_len = [2] > 0

   创建 new_token_mask:
     token0: cached (skip)
     token1: cached (skip)
     token2: new    (✓)
   → new_token_mask = [False, False, True]

   选择新 token:
     input_new = [token2]  (len=1)
     q_new = [q2]           (len=1)

   调用 indexer.get_indices(input_new, q_new, ...)
   └─> _get_q_k_bf16() 生成 k_fp8 (len=1) ✓
   └─> deep_gemm.fp8_mqa_logits() 计算 topk (len=1) ✓
   └─> destindex_copy_indexer_ks(
          k_fp8,          # len=1 ✓
          k_scale,        # len=1 ✓
          mem_index,      # len=1 ✓ 匹配！
          ...
      )
   └─> 复制 token2 的 indexer_ks 到位置 [103] ✓

   对于缓存 token (token0, token1):
     - 不需要复制 indexer_ks（因为已经在之前计算过）
     - 但 KV cache 已经通过 copy_kv_index_to_req 复制到新位置
     - topk_indices 会在 _init_nsa_indexing_structures 中重新计算

4. _init_nsa_indexing_structures():
   └─> extract_indexer_ks 从正确的位置提取数据
   └─> 对于缓存 token，从之前的位置提取
   └─> 对于新 token，从刚写入的位置 [103] 提取

   结果: 所有 token 的 topk_indices 都正确 ✓

输出: 正确 ✓
```

### 关键代码变更

**文件**: `lightllm/models/deepseek3_2/layer_infer/transformer_layer_infer.py`

**位置**: 第 30-106 行，`_get_qkv` 方法

```python
@override
def _get_qkv(
    self,
    input: torch.Tensor,
    infer_state: Deepseek3_2FlashAttentionStateInfo,
    layer_weight: Deepseek3_2TransformerLayerWeight,
) -> torch.Tensor:
    input = input.view(-1, self.embed_dim_)

    q, cache_kv = layer_weight.qkv_a_proj_with_mqa_.mm(input).split(
        [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim], dim=-1
    )
    q = rmsnorm_forward(q, weight=layer_weight.q_a_layernorm_.weight, eps=self.eps_)

    # ===== 新增代码开始 =====
    # Only process new tokens (not from prefix cache) for indexer.get_indices
    # When prefix cache is hit, b_ready_cache_len > 0 for cached tokens
    # We need to select only the new tokens to avoid mismatch between
    # input length and mem_index length
    if infer_state.b_ready_cache_len is not None and infer_state.b_ready_cache_len.sum() > 0:
        # Create mask for new tokens
        new_token_mask = torch.zeros(input.shape[0], dtype=torch.bool, device=input.device)
        offset = 0
        for i in range(infer_state.b_ready_cache_len.shape[0]):
            seq_len = infer_state.b_seq_len[i].item()
            ready_len = infer_state.b_ready_cache_len[i].item()
            new_len = seq_len - ready_len
            if new_len > 0:
                new_token_start = offset + ready_len
                new_token_end = new_token_start + new_len
                new_token_mask[new_token_start:new_token_end] = True
            offset += seq_len

        # Select only new tokens for indexer processing
        input_new = input[new_token_mask]
        q_new = q[new_token_mask]

        # Get topk indices only for new tokens
        new_topk_indices = self.indexer.get_indices(
            input_new, q_new, infer_state, layer_weight.indexer_layer_weight
        )

        # Merge with placeholder for cached tokens
        total_tokens = input.shape[0]
        self.topk_indices = torch.zeros(
            total_tokens, dtype=new_topk_indices.dtype, device=new_topk_indices.device
        )
        self.topk_indices[new_token_mask] = new_topk_indices
    else:
        # No prefix cache hit, process all tokens
        self.topk_indices = self.indexer.get_indices(
            input, q, infer_state, layer_weight.indexer_layer_weight
        )
    # ===== 新增代码结束 =====

    q = layer_weight.q_b_proj_.mm(q)
    # ... 后续处理不变 ...
```

### 为什么这样修复有效

#### 1. 长度匹配得到保证

```
修复前:
  input   (len=3) → k_fp8 (len=3) → mem_index (len=1) ✗

修复后:
  input_new (len=1) → k_fp8 (len=1) → mem_index (len=1) ✓
```

#### 2. 数据一致性得到保证

```
修复后:
  - 新 token 的 indexer_ks 被正确复制到新位置
  - 缓存 token 的 indexer_ks 从之前的位置正确读取
  - req_to_token_indexs 指向的位置与实际数据位置一致
```

#### 3. 不影响其他功能

```
- 对于没有 prefix cache 命中的请求：
  └─ b_ready_cache_len = 0
  └─ 走 else 分支，行为与之前完全一致

- 对于全部命中 cache 的请求：
  └─ b_ready_cache_len = b_seq_len
  └─ num_new_tokens = 0
  └─ topk_indices = zeros (会被后续 _init_nsa_indexing_structures 覆盖)
```

## 验证方法

### 单元测试

```python
def test_deepseek32_prefix_cache():
    # 第一次请求
    output1 = model.generate("你好，世界")

    # 第二次请求（应该命中 prefix cache）
    output2 = model.generate("你好，世界")

    # 验证输出一致
    assert output1 == output2
```

### 集成测试

```bash
# 启动服务
python -m lightllm.server.api_server \
  --model_dir /path/to/deepseek32_model \
  --max_total_token_num 10000

# 第一次请求
curl http://localhost:8088/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "你好，世界", "parameters": {"max_new_tokens": 50}}'

# 第二次请求（应该命中 prefix cache）
curl http://localhost:8088/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "你好，世界", "parameters": {"max_new_tokens": 50}}'
```

## 相关文件

| 文件 | 说明 |
|------|------|
| `lightllm/models/deepseek3_2/layer_infer/transformer_layer_infer.py` | 修改的主文件 |
| `lightllm/models/deepseek3_2/layer_infer/nsa_indexer_layer_inder.py` | NSA indexer 实现 |
| `lightllm/models/deepseek3_2/triton_kernel/destindex_copy_indexer_ks.py` | 复制 indexer_ks 的 Triton kernel |
| `lightllm/models/deepseek3_2/infer_struct.py` | 推理状态管理 |
| `lightllm/server/router/dynamic_prompt/radix_cache.py` | Radix cache 实现 |

## 参考文档

- [DeepSeek-V3 Technical Report](https://github.com/deepseek-ai/DeepSeek-V3)
- [LightLLM Architecture](../CLAUDE.md#architecture)
- [Prefix Cache Design](../CLAUDE.md#prefix-cache)
