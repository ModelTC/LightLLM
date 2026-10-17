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

当前支持普通与分块 Prefill、Decode、动态 Prompt Cache、FA3/FlashInfer（含 MLA Decode）、MTP、
PD 分离、CPU KV Cache、DP Prompt Cache 拉取、DP Prefill Balance 和混合线性注意力。
diverse mode 仍只支持 `page_size=1`，启动阶段会对其他取值明确报错。

## 边界处理

- KV 总容量和请求表宽度分别向下、向上对齐，防止物理页越界。
- CUDA Graph 的 HOLD request 映射到额外保留的一整个物理页；请求表按
  `[HOLD, HOLD+1, ..., HOLD+page_size-1]` 循环填充，而不是重复同一个 token 索引。
- Radix 子节点用“首个完整 token 页”作为键，避免不同序列仅首 token 相同造成页级分支冲突。
- 非法的 `page_size < 1` 以及尚未支持的功能组合在模型加载前失败。

## DeepSeek-V4

V4 的分配页 `--page_size` 支持 256 的正整数倍，例如 256、512、768、1024。
沿用默认值 `1` 时自动选择 256；显式配置的合法页大小会保留。
小页缓存边界 `--linear_att_hash_page_size` 必须是分配页大小的整数倍，例如：

```bash
--page_size 256 --linear_att_hash_page_size 256
# 或
--page_size 768 --linear_att_hash_page_size 768
```

大页 checkpoint 间隔是小页大小乘以 `--linear_att_page_block_num`。
例如 256-token 小页配合 `--linear_att_page_block_num 8`，每 2048 token 保存一次 checkpoint。
省略该参数时沿用主线默认值，关闭大页 checkpoint。PD Decode 节点仍按主线规则关闭大页。
启用大页和 CPU cache 时，`--cpu_cache_token_page_size` 必须等于大页间隔；未指定时自动设置。
PD 传输页 `--pd_kv_page_size` 和 CPU cache 页也必须是分配页大小的整数倍。
例如 768-token 分配页可以配合 `--pd_kv_page_size 1536`。

- 分配页按 256-token packed block 持有压缩历史。闭合分组的 C4/C128 槽分别由组末 full slot 除以 4/128 得到；
  不再维护独立压缩槽映射或 allocator。预留容量不改变 attention 的逻辑可见长度。
  每个压缩层持有一个 token buffer：C4 层包含 latent KV 和 indexer K，C128 层包含 latent KV；
  推理时创建 packed 视图，管理层保持相同的 token 容量和分配接口。
- packed 格式保持独立：SWA 每页 128 槽，C4 每页 64 槽，C128 每页 2 槽。
  这些物理块大小来自算子布局，与分配页大小独立。历史复制遍历分配页内的所有 packed block；
  HOLD 请求也预留整个分配页对应的压缩容量。
- SWA 与 linear state 一样属于请求运行态，长期缓存通过现有大小页 CPU checkpoint 保存和恢复。
  每个请求预留固定窗口，使用请求号和绝对位置直接计算环形槽位，不维护独立 SWA 页表或 allocator。
  窗口容量由滑窗和 MTP 前探余量确定，与 checkpoint 间隔无关。
- Prefill 的长 chunk 为每个请求使用连续的「旧窗口前缀 + 本次 chunk KV」。工作区容量按单次 forward 的 token
  上限加窗口大小乘以最大请求数预留，并计入每个请求首尾的存储对齐；两个 overlap microbatch 使用不同区域。
  短 chunk 的完整读取区间能放入固定窗口时直接写窗口；其余情况使用工作区。
  请求侧只记录本轮 token 范围和工作区偏移；不为每个请求重复预留完整长 chunk 空间。
  为供 forward 后的中间 checkpoint 读取，各层保留各自的工作区。KV 写入时同时保存必要的运行窗口尾部，
  无需通用 prepare/finish 生命周期接口、借页回收或延迟收尾。
- MemoryManager 只分配压缩历史和 CPU 大页 checkpoint。C4、indexer 和 C128 的容量均为同一个
  `mem_manager.size`，使用相同的 allocator 和 token 索引，buffer 仅以不同的 packed 宽度解释。
  全局层号通过简单映射定位对应历史和压缩器状态。
- C4、C4 indexer 和 C128 的压缩运行态由 ReqManager 按请求保存短环。
  C4 在压缩块边界将上一组候选向量与分数保存到独立的 prefill checkpoint 缓冲区，
  不与运行态混存，也不依赖 SWA 物理页定位；C128 分组在该边界已经闭合。
  两个请求命中同一前缀时共享压缩历史，分别恢复自己的窗口和运行态。
- CPU cache 完全沿用通用分配、异步卸载、加载和磁盘缓存流程。模型 operator 复制原始 packed history
  和已有 CPU checkpoint，无专用 cache module、session、lease 或增量 staging。
  启用 CPU cache 时，CPU 页与大页 checkpoint 使用相同边界；未配置大页时自动建立对应边界。
- 大小页 CPU checkpoint 按时间顺序保存必要的 SWA token、C4 和 indexer 尾部，
  不包含物理页位置，也不重复保存压缩历史。恢复沿用主线接口，从请求的 `cur_kv_len` 确定 token 位置；
  跨 DP 复用同一恢复接口。保存接口显式接收目标边界，以支持长 prefill 内的多个 checkpoint。
- PD 复用现有 `kv` / `att_state` 任务：前者传输压缩历史，后者传输窗口和未闭合压缩组，支持任意传输终点。
  P/D 可以使用不同的 MTP 环大小，传输时按绝对位置恢复运行态；推理 backend 无模型专用 checkpoint 回调。
  PD continuation 布局有变化，P/D 节点需要使用相同代码版本。

验证覆盖分页边界、长 prefill 的窗口前缀和中间 checkpoint、大小页命中和分叉、推测拒绝后重试、
CUDA Graph 索引、通用 CPU cache 异步卸载与恢复、PD 传输及跨进程 CUDA IPC/Gloo checkpoint 恢复。
