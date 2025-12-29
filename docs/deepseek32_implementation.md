# DeepSeek V3.2 实现详解

## 一、背景：为什么需要 NSA？

### 1.1 传统 Attention 的问题

在大语言模型中，每个新生成的 token 需要关注之前所有的 token。假设序列长度是 n：

```
传统 Attention 计算量 = n × n = n²
```

当 n 很大时（比如 128K），计算量会非常大。

### 1.2 DeepSeek V2 的优化 (MLA)

DeepSeek V2 使用了 **MLA (Multi-head Latent Attention)**：
- 将 Key 和 Value 压缩到低维空间
- 减少了内存占用，但仍然需要计算与所有历史 token 的 attention

### 1.3 DeepSeek V3.2 的进一步优化 (NSA)

**NSA (Neighbor-aware Sparse Attention)** 的核心思想：
> 每个 query token 不需要关注所有历史 token，只需要关注 top-k 个最相关的

```
NSA 计算量 = n × k  (其中 k 是固定的，比如 2048)
```

当 n = 128K，k = 2048 时：
- 传统: 128K × 128K = 16B
- NSA:   128K × 2K = 256M
- **减少 98%+ 的计算量**

## 二、NSA 如何选出 top-k？

### 2.1 核心思想：用 "Indexer" 来选择

NSA 不是简单地选择最近的 k 个 token，而是用一个小的神经网络来选择最相关的 k 个。

```
输入：当前 token 的隐藏状态
      ↓
  [Indexer 模块]
      ↓
输出：最相关的 k 个历史 token 的索引
```

### 2.2 Indexer 的工作流程

```python
# 简化的代码流程
def get_topk_indices(current_hidden_state, all_history_tokens):
    # 1. 计算 indexer 的 query 和 key
    q_indexer = project_to_indexer_q(current_hidden_state)
    k_indexer = project_to_indexer_k(all_history_tokens)

    # 2. 计算相关性分数
    scores = compute_similarity(q_indexer, k_indexer)

    # 3. 选择 top-k
    topk_indices = select_topk(scores, k=2048)

    return topk_indices
```

### 2.3 需要存储什么？

为了计算 top-k，NSA 需要额外存储：
- **indexer_ks**: 每个 token 的 indexer key（用于计算相关性）
- 格式：每个 token 132 字节（128 字节 FP8 + 4 字节 scale）

## 三、DeepSeek V3.2 的内存架构

### 3.1 双缓冲区设计

```
┌─────────────────────────────────────────────────────────┐
│              DeepSeek V3.2 内存布局                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  KV Cache (主注意力缓存)                        │    │
│  │  - 每个 token: (kv_lora_rank + rope_dim) × 2   │    │
│  │  - 用途: 存储 attention 计算用的 K 和 V         │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │  Indexer KS Cache (NSA 专用)                   │    │
│  │  - 每个 token: 132 字节                        │    │
│  │  - 用途: 存储 indexer 的 key，用于选 top-k    │    │
│  └────────────────────────────────────────────────┘    │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### 3.2 内存管理器的实现

```python
# lightllm/models/deepseek3_2/mem_manager.py
class Deepseek3_2MemoryManager(Deepseek2MemoryManager):
    def __init__(self, ...):
        super().__init__(...)  # 初始化 KV cache

        # 额外初始化 Indexer KS cache
        self.indexer_ks_mem_manager = Deepseek2MemoryManager(
            size=self.size,
            dtype=torch.uint8,   # 用 uint8 存储 FP8+scale
            head_num=1,
            head_dim=132,         # 128B FP8 + 4B scale
            layer_num=layer_num,
        )
```

## 四、前缀缓存的问题与解决

### 4.1 什么是前缀缓存？

前缀缓存（Radix Cache）是一种优化技术：
- 相同的 prompt 前缀只需要计算一次
- 后续请求可以直接复用

```
请求 1: "解释什么是人工智能"
         ↓ 计算整个 prompt
         ↓ 缓存结果

请求 2: "解释什么是人工智能，它有哪些应用？"
         ↓ 复用缓存的 "解释什么是人工智能"
         ↓ 只计算新增的部分
```

### 4.2 问题：缓存命中时的数据不一致

当使用前缀缓存时，系统会：
1. 将缓存的 token 的数据复制到新的连续内存位置
2. 只计算新 token 的数据

**在 DeepSeek V3.2 中，需要复制两份数据：**
- KV Cache：系统自动处理
- **Indexer KS：需要额外处理**

### 4.3 问题的具体表现

假设请求 3 个 token，前 2 个命中缓存：

```
Token 0: 缓存命中
Token 1: 缓存命中
Token 2: 新 token

问题：
- KV Cache 被正确复制到新位置 [100, 101, 102]
- 但 Indexer KS 没有被复制，还在旧位置 [50, 51, ...]
- 导致 Indexer 从错误的位置读取数据 → 结果错误
```

### 4.4 解决方案：同步复制 Indexer KS

**修改的文件：**
1. `lightllm/common/basemodel/basemodel.py` - 捕获旧位置
2. `lightllm/common/infer_utils.py` - 执行复制
3. `lightllm/models/deepseek3_2/triton_kernel/copy_indexer_ks.py` - 复制内核

**核心流程：**

```python
# 步骤 1: 在更新前，记录旧的 Indexer KS 位置
old_positions = [
    req_to_token_indexs[req_id, 0:cached_len]  # 旧位置
    for req_id in batch
]

# 步骤 2: 更新 req_to_token_indexs（系统自动完成）
# 这会分配新的连续位置

# 步骤 3: 复制 Indexer KS 到新位置（新增的逻辑）
for layer_idx in range(num_layers):
    for req_id in batch:
        new_positions = req_to_token_indexs[req_id, 0:cached_len]
        copy_indexer_ks(
            buffer=indexer_ks_buffer[layer_idx],
            src_loc=old_positions[req_id],
            dest_loc=new_positions
        )
```

## 五、代码结构详解

### 5.1 目录结构

```
lightllm/models/deepseek3_2/
│
├── model.py                           # 模型定义
│   └── Deepseek3_2TpPartModel         # 继承自 Deepseek V2
│
├── mem_manager.py                     # 内存管理
│   ├── Deepseek3_2MemoryManager       # 主管理器（包含两个子管理器）
│   └── Deepseek3_2FP8KVMemoryManager  # FP8 量化版本
│
├── infer_struct.py                    # 推理状态
│   └── Deepseek3_2FlashAttentionStateInfo  # 包含 NSA 特定状态
│
├── layer_infer/                       # 推理层实现
│   ├── transformer_layer_infer.py    # Transformer 层推理
│   │   └── _get_qkv()                # 调用 NSA indexer
│   │
│   └── nsa_indexer_layer_inder.py    # NSA Indexer 实现
│       └── get_indices()              # 计算 top-k 索引
│
├── layer_weights/                     # 权重加载
│   ├── transformer_layer_weight.py
│   └── nsa_indexer_layer_weight.py
│
└── triton_kernel/                     # 自定义 CUDA 内核
    ├── destindex_copy_indexer_ks.py  # 写入新 token 的 indexer key
    ├── extract_indexer_ks.py         # 提取 indexer key
    ├── copy_indexer_ks.py            # 复制 indexer key（前缀缓存用）
    ├── act_quant.py                  # FP8 量化
    └── token_group_quant.py          # 分组量化
```

### 5.2 关键数据流

```
输入: hidden_states [batch_size, seq_len, hidden_dim]
            │
            ▼
    ┌───────────────┐
    │  _get_qkv()   │
    └───────┬───────┘
            │
            ├─────────────────┐
            ▼                 ▼
    ┌──────────┐      ┌────────────────┐
    │ 计算 Q, K │      │ NSA Indexer   │
    │ (MLA)    │      │               │
    └────┬─────┘      │ 1. 计算索引 Q,K│
         │            │ 2. 量化到 FP8   │
         │            │ 3. 写入缓存    │
         │            │ 4. 提取历史 K   │
         │            │ 5. 计算 topk   │
         │            └───────┬────────┘
         │                    │
         │                    ▼
         │            topk_indices [seq_len, 2048]
         │                    │
         └────────────────────┘
                  │
                  ▼
         ┌────────────────┐
         │  Attention    │
         │  (使用 topk)   │
         └────────────────┘
```

### 5.3 关键函数说明

#### `destindex_copy_indexer_ks`
- **作用**: 将新计算的 indexer key 写入内存
- **输入**:
  - `K_fp8`: 所有 token 的量化 key
  - `K_scale`: 所有 token 的量化 scale
  - `mem_index`: **仅新 token** 的内存位置
- **关键**: 只写入 `mem_index` 指定的位置（新 token）

#### `extract_indexer_ks`
- **作用**: 从内存读取 indexer key
- **输入**: `req_all_mem_index` (所有 token 的位置)
- **输出**: 所有 token 的 indexer key

#### `copy_indexer_ks`
- **作用**: 在内存位置之间复制 indexer key
- **用途**: 前缀缓存命中时，将缓存的 key 复制到新位置

## 六、使用示例

### 6.1 启动服务

```bash
# 单 GPU
python -m lightllm.server.api_server \
  --model_dir /path/to/deepseek_v32_model \
  --tp 1 \
  --max_total_token_num 20000

# 多 GPU (8 卡)
LOADWORKER=18 python -m lightllm.server.api_server \
  --model_dir /path/to/deepseek_v32_model \
  --tp 8 \
  --max_total_token_num 100000
```

### 6.2 验证前缀缓存

```bash
# 第一次请求（无缓存）
curl http://localhost:8088/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "What is AI?", "parameters": {"max_new_tokens": 50}}'

# 第二次请求（命中缓存，应该更快）
curl http://localhost:8088/generate \
  -H "Content-Type: application/json" \
  -d '{"inputs": "What is AI?", "parameters": {"max_new_tokens": 50}}'
```

## 七、配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `index_topk` | 每个 query 选择的 token 数 | 2048 |
| `index_head_dim` | Indexer 隐藏层维度 | 128 |
| `index_n_heads` | Indexer 注意力头数 | 16 |
| `kv_lora_rank` | KV 低秩维度 | 512 |
| `qk_rope_head_dim` | Rotary embedding 维度 | 64 |

## 八、常见问题

### Q1: 为什么需要两个内存管理器？
- `kv_buffer`: 存储 attention 计算用的 KV
- `indexer_ks_buffer`: 存储 NSA 选 top-k 用的 key

### Q2: 前缀缓存不工作的症状？
- 相同的 prompt，第二次请求输出结果不同
- 可能是 indexer_ks 没有正确复制

### Q3: 如何调试？
```bash
# 检查内存分配
python -c "
from lightllm.models.deepseek3_2.mem_manager import Deepseek3_2MemoryManager
mm = Deepseek3_2MemoryManager(...)
print(f'KV buffer size: {mm.size}')
print(f'Indexer KS size: {mm.indexer_ks_mem_manager.size}')
"
```

## 九、参考链接

- [DeepSeek V3 技术报告](https://github.com/deepseek-ai/DeepSeek-V3)
- [DeepSeek V2 技术报告](https://github.com/deepseek-ai/DeepSeek-V2)
- [前缀缓存修复文档](./deepseek32_prefix_cache_fix.md)
