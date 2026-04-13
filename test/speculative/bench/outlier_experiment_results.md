# VSM 在 Outlier 场景下的优势验证

## Blog 关键观点回顾

> 原始的 kernel 基于 attention heads 和 batch size 分配 thread blocks。然而，这种设计在处理不均匀长度的 batch 时会导致 thread blocks 负载不均衡，影响性能。

> VSM 重新设计：固定 thread blocks 数量，将 context 分成固定大小的块。每个 thread block 迭代所有块，动态变化长度转为固定迭代次数。固定大小块确保负载平衡。

## 实验设计

### 测试配置
- **GPU**: NVIDIA H200
- **模型配置**: num_heads=32, kv_head_num=4, head_dim=128, dtype=bfloat16
- **测试 kernel**:
  - MTP Diverse Original (原始版本)
  - MTP Diverse VSM (VSM 优化版本)

### Outlier 场景设置
- **Uniform**: 所有组内请求长度均匀 (1, 2, 3, ..., group_size)
- **Outlier**: 指定比例的组包含异常长的请求 (max_seq_len)

## 实验结果

| Config | Uniform VSM Speedup | Outlier VSM Speedup |
|--------|---------------------|---------------------|
| gs=8, bs=128, seq=8192, outlier=10% | 0.82x (VSM 更慢) | 2.38x (VSM 更快) |
| gs=8, bs=128, seq=8192, outlier=25% | 0.81x (VSM 更慢) | 3.63x (VSM 更快) |
| gs=8, bs=128, seq=8192, outlier=50% | 0.82x (VSM 更慢) | 6.34x (VSM 更快) |
| gs=4, bs=128, seq=8192, outlier=10% | 0.90x (VSM 更慢) | 2.82x (VSM 更快) |
| gs=4, bs=128, seq=8192, outlier=25% | 0.89x (VSM 更慢) | 5.15x (VSM 更快) |
| gs=2, bs=128, seq=8192, outlier=10% | 0.98x (VSM 更慢) | 3.35x (VSM 更快) |
| gs=8, bs=128, seq=16384, outlier=10% | 0.86x (VSM 更慢) | 2.45x (VSM 更快) |

## 关键发现

### 1. Uniform 场景 (所有组长度均匀)
- VSM 与 Original 性能相当或略慢 (0.81x - 0.98x)
- Original 的 thread block 分配在这种情况下已经足够高效

### 2. Outlier 场景 (存在异常长请求)
- **VSM 显著优于 Original！**
- 10% outlier: VSM 快 **2.38x - 3.35x**
- 25% outlier: VSM 快 **3.63x - 5.15x**
- 50% outlier: VSM 快 **6.34x**

### 3. outlier_ratio 影响
- outlier 比例越高，VSM 优势越明显
- **原因分析**:
  - Original kernel 的 thread blocks 基于最大长度分配，需要等待长序列完成，造成空闲
  - VSM 的固定块大小确保负载均衡，不受 outlier 影响

### 4. 与 Blog 描述一致
> Blog: "new kernel performed better overall, with minimal impact from outlier requests"

**实验验证**：VSM 对 outlier 不敏感，性能稳定。

## 结论

VSM 优化在以下场景效果显著：
1. **存在 outlier 请求** - 性能提升 2x - 6x
2. **负载不均匀** - 组内/组间长度差异大
3. **长序列场景** - seq_len 越大，优势越明显

对于均匀的短序列负载，Original 和 VSM 性能相当。
