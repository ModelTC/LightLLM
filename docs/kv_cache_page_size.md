# KV Cache 多 Token 页设计

## 目标与不变量

启动参数 `--page_size N` 控制注意力 KV Cache 的物理页大小，默认值为 `1`。`N > 1` 时保持 token 级的
KV 存储和请求表，但资源的申请、缓存和释放以完整物理页为单位。

实现依赖以下不变量：

1. `cur_kv_len` 是请求已经写入有效 KV 的逻辑 token 数；`hold_kv_len` 是请求拥有的物理容量，始终满足
   `cur_kv_len <= hold_kv_len` 且 `hold_kv_len % page_size == 0`。
2. 一个物理页内的 KV 槽连续，页首索引可被 `page_size` 整除。请求表保存已持有页的全部 token 索引，
   包括尚未使用的尾部槽位。
3. `req_to_token_indexs` 保存请求拥有的完整页；创建 `InferState` 时根据请求索引和序列位置直接聚合本轮
   真实参与计算的 token，`ModelInput` 不携带物理 KV 索引，`InferState.mem_index` 不包含预留尾部。
4. Radix Cache 只插入、拆分、命中和淘汰完整页；不足一页的请求尾部在请求结束或暂停时整页回收。
5. `page_size=1` 与多 token 页使用相同的调度期预留和模型执行期索引选择路径。

页容量计算和整页申请由调度层在请求获准进入 Prefill/Decode batch 时完成；输入构造层只构造本轮输入，
真实推理索引由模型执行层从请求表选取。
`ReqManager` 只保留请求 ID、请求表及其原有生命周期职责，不提供 page allocator 接口。

## 生命周期

- Prefill：请求通过调度后，按目标 KV 长度将容量向上对齐，新页一次申请并完整写入请求表。模型执行时再选择
  `[cur_kv_len, target_kv_len)` 对应的真实索引。
- Decode：请求通过调度后，若尾页仍有预留槽位则不再访问 allocator；容量不足时先补齐完整页再执行。
- Prefix Cache 命中：命中长度天然是整页，`cur_kv_len` 与 `hold_kv_len` 同时初始化为命中长度。
- 完成/暂停：完整逻辑页可进入 Radix Cache；重复页和未完成尾页按物理页展开后释放。
- Attention：FA3/FlashInfer 页表每项由物理 token 页首索引除以 `page_size` 得到，KV buffer 视为
  `[page_count, page_size, ...]`。

## 当前兼容范围

首版支持普通与分块 Prefill、Decode、动态 Prompt Cache，以及非量化 FA3/FlashInfer（含 MLA Decode）。
MTP、PD 分离、CPU KV Cache、DP Prompt Cache 拉取、DP Prefill Balance、diverse mode 和混合线性注意力
拥有额外的 KV 申请或迁移语义；这些组合在启动阶段明确报错，后续应在各自模块接入统一的页所有权接口后再开放。

## 边界处理

- KV 总容量和请求表宽度分别向下、向上对齐，防止物理页越界。
- CUDA Graph 的 HOLD request 映射到额外保留的一整个物理页；请求表按
  `[HOLD, HOLD+1, ..., HOLD+page_size-1]` 循环填充，而不是重复同一个 token 索引。
- Radix 子节点用“首个完整 token 页”作为键，避免不同序列仅首 token 相同造成页级分支冲突。
- 非法的 `page_size < 1` 以及尚未支持的功能组合在模型加载前失败。
