# support_ds4 拆分时排除的提交

本分支从 `support_ds4@7369626cc3487cacaaccad611943d61f9558635c` 派生，并用一个汇总清理提交移除不准备合入的内容。清理后的工作树只保留 DeepSeek-V4 模型支持相关能力（基础模型、MTP、DSpark、Vision、PD、多级缓存、DP/DPEP、硬件与量化适配、协议与性能优化）。下列 `support_ds4` 提交属于明确排除的专家能力或通用稳定性/运维改动。

## 专家能力

| 原提交 | 标题 | 处理 |
| --- | --- | --- |
| `c899b318b29786a39dbb9dd2bc63e186e3779687` | `add triton ep backend` | 排除 Triton EP 后端 |
| `33cd0c9d13f56cee332b533058c19685c923ecc9` | `feat: support EPLB for DeepSeek v4` | 排除 DeepSeek-V4 EPLB |
| `7722b82eb210136bd69cd5116ced688e3b8b6424` | `feat: add EPLB` | 排除通用 EPLB 基础设施 |
| `c8f53bebe1bb1bea92174b4bda6e3f12c34612de` | `support fp8xfp8 mega moe` | 排除 Mega-MoE 扩展 |
| `a3160f25b30bdad7856e0e77b7daff1254de9717` | `feat: imbalance statistics` | 排除专家负载不均衡监控 |
| `1a2f29d05ec2ae90fe608ff69e97995df40d6ba6` | `mega_moe support clamp_limit` | 排除 Mega-MoE clamp 扩展 |

`origin/main` 已有的 Mega-MoE 基础路径不回退；这里只排除 `support_ds4` 新增的 Mega-MoE 能力。

## 通用稳定性与运维

| 原提交 | 标题 |
| --- | --- |
| `9bb68ebabfb26a0579b8764155bcb5ddcd26380b` | `fix abort` |
| `9a3f99dcf3c492797105362d229479e29db71211` | `fix(pd): clean up requests correctly after node disconnect` |
| `33da96f290e6d58aba60feb658c8aecd84d61384` | `fix(pd): harden control sends and reconnect cleanup` |
| `2a7e6bd0f4ec6d2edda947c43c4d2d032ea97e09` | `PD nodes unavailable:  503` |
| `3d910fe7a6001f1494b9cb52f880ff68d6ee4db5` | `avoid premature KV transfer aborts and surface generation failures` |
| `ef0df4317f6e9db3e19c3366c8e2b2624bc5fb73` | `fix(pd): wait for KV transfer modules to become ready` |
| `aa9b2a34dc9ff310861dc586eae215ebb64dda69` | `fix(api): return 400 for oversized guided JSON schemas` |
| `8ed3da6e5c36ee4c7a722cce11c1eeb0f1d08f3d` | `fix(pd): measure decode KV timeout from request progress` |
| `7bc0955b20816d407748dcc927f461661609c155` | `fix(pd): handle deep prompt-cache keys iteratively` |
| `7229e567526c9428070e03bd025c138137d4390a` | `clean disk cache on shutdown` |
| `811230e0755f2922cf18c63dcfedfe75453da758` | `fix ZMQ crash when cache workers finish concurrently` |
| `7fad48a98421ccb326d529029bc5fa496c3512ce` | `fix(pd): prevent PD master hang on prefill timeout race` |
| `5e491fb5db02d01a5476dad4a35be9f722d7c1fb` | `feat(metrics): add PD pipeline latency metrics` |
| `095c043d4781c8cc8f8b515d4e4e618958ff5a6f` | `perf(openai): reduce chat streaming serialization overhead` |
| `7c65cd00ac66f0998f64709945de18c7139bb3d2` | `fix(pd): preserve aborts received before request registration` |

若同类修复已经存在于 `origin/main`，本次只去掉 `support_ds4` 的额外增量，不回退主分支已有行为。

## 混合提交的拆分处理

| 原提交 | 保留 | 排除 |
| --- | --- | --- |
| `baadd7ea316861c089956d46021e0b216a51bcf8` (`perf(server): reduce PD streaming and parser overhead`) | DeepSeek-V4 DSML 流式解析所需逻辑 | 通用 PD streaming/parser 性能改动 |
| `ca4f075860e00649ea73554780b33a417ead21bc` (`synchronize PDL top-k and fail fast on model thread errors`) | DeepSeek-V4 PDL top-k 同步 | 通用 fatal thread excepthook |

整理阶段产生的临时 revert 提交已经压平，因此最终分支相对 `7369626` 只多一个汇总清理提交；本文件记录被排除的原始提交，便于后续追溯。

## 2026-09-28 二次模型边界清理

本节是在 `support_dsv4_model@c370e9213f1632423f4c9478258bdf377b18715c` 上追加的第二次清理记录，不改写上面的首次拆分历史。本轮仍保留 DeepSeek-V4 基础模型、MTP、DSpark、Vision、PD、多级缓存及其必要的共享层契约；专家能力和通用稳定性/运维继续按上文排除。

| 类别 | 回退内容 | 对应原提交或路径 |
| --- | --- | --- |
| 通用 DP 调度与容量架构 | 恢复既有 NCCL 调度控制、统一请求容量和通用 DP 状态；仅为 DSV4 跨 DP CPU checkpoint 保留节点内 Gloo group | `94fdfdeca05f3f94619a0c183088114c42e65124`、`f2e8f1429dccd199750bc5d9334dc4a30622dd15`、`c211f485aa0c5dcd392a22104f5b37cf44056c2e` |
| DP cache-aware 调度 | 删除新增的 router DP cache-aware balancer，并恢复 `req_queue`、router manager 和控制状态 | `fbe49a2227d714ce373a0fde9d5701c77368ed85` |
| 通用 HTTP/PD 传输优化 | 回退 compact token payload、streaming/serialization、async queue 和 PD-master 增量；保留 DSV4 packed cache 所需的变长字节传输 | `cfce08726b09e44182ff63df2978ae95cfea1337`、`baadd7ea316861c089956d46021e0b216a51bcf8`、`095c043d4781c8cc8f8b515d4e4e618958ff5a6f` |
| MXFP4 Marlin 运行时后端 | 删除 `mxfp4w4a16-b32-marlin` 注册、实现、权重 finalize 和 CLI 暴露；保留 DSV4 MTP checkpoint 转 BF16 工具 | `e8009cb3e053ffe7dbe465c027e4fe6a676181c8` 中的可选 Marlin 路径 |
| xgrammar 通用 tokenizer 兼容 | 删除 `get_xgrammar_tokenizer` 及底层 HF tokenizer 旁路；DSV4 tokenizer 回退只对 DSV4 生效 | `b8073f843c71dcd3f837b3b17522383dc3688b9d` |
| structured-output 通用降级 | 恢复原有约束输出行为，不在 xgrammar 缺失时静默关闭 | `f14dd40738daf817d66ea8d025a93fe154fa8a8b`、`8ee409a5610d5a4a1023942a07ead1f020d269ec` |
| Disk cache 实例目录隔离 | 恢复原有 disk worker/目录语义 | `0ed251199d7df166be65974e9846599409de94f8` |
| Benchmark 与 autotune 产物 | 删除 static benchmark 增量及本 PR 新增的 H100/H200 autotune JSON | `388dd29de8a000500279ad273c06a6d7874f7cf1`、`4ac53e82fd3ecabef8ff0bbd4100a5fec3202821`、`0a48e27bcbfa8fec91f52bebb874616e1a2a95a0` |
| DeepEP 环境自动调参 | 回退 decode dispatch capacity、NVSHMEM QP depth 等通用自动派生，恢复原有默认值 | `9a071ca5e0e53bab4bca63586f78e92b493cfb68`、`97ae2d12acdd77bcf87a786a710c0f1f049d8196`、`69c260f4a5f193fcd9170d06d0c77187a2ee18c1` |
| B300 通用对齐 | 恢复设备判断；UE8M0 只在 SM100 上启用，不再无条件应用到所有设备 | `f5f3ed2cc74857ef8821840d5ca0d42c4f2a3e67` |
| 通用采样默认值优化 | 恢复 `core/objs` 直接通过 Transformers 加载完整 generation config，不再使用为改变默认 `top_k` 引入的共享 helper | `c19b8537134c66040a8dc1468c0b848650155967` (`default topk from huggingface's 50 to -1, if_inverse 70.6 -> 73`) |
| DSV4 DP 结束 barrier review | 不纳入无条件 DSV4 barrier；保留此前 CPU-cache 场景已有的条件 barrier | `a2a7052d1a9ffdf765e81f7c43bf59807d484f08` |
| mHC TileLang 启动预热 | 回退通用 `_kernel_warmup` hook、DSV4 mHC 预热流程及 split-K token 枚举；保留 MTP hidden 准备所需的 `hc_post` 引用 | `fc8b7ec411774fab269e1e3799efff4ac15f826e` (`warmup tilelang`) |

清理后，`lightllm/server/httpserver_for_pd_master/`、`lightllm/server/router/req_queue/`、`lightllm/server/multi_level_kv_cache/disk_cache_worker.py` 和 `lightllm/utils/device_utils.py` 相对拆分基线无额外差异；`communication_op.py` 只保留 DSV4 `experts_` 字段兼容，HTTP server 只保留 Vision 图像块不可跨 prefill 切分的校验。此前确认的 MTP CUDA Graph hidden 输入修复继续保留。

## 2026-09-30 多级缓存架构收敛

删除独立的 `Dsv4MultiLevelKvCacheModule`，统一通过 main 的 `MultiLevelKvCacheModule` 管理 CPU 页面引用、异步任务及磁盘发布。保留 DSV4 必需的 prefill 增量 checkpoint 保存；staging 与 pack/unpack 回归 `DeepseekV4MemOperator`，SWA 和 radix checkpoint 恢复回归 `DeepseekV4ReqManager`。保留原有 CPU 页布局及两阶段完成事件，不改变 PD 传输协议。

prompt-logprobs 请求的过滤、命中长度清零和匹配页引用释放统一放在公共加载入口；普通模型和 DSV4 仅在实际缓存加载阶段分派，不再分别实现过滤策略。

普通模型的正常匹配页也在公共入口统一释放，加载子流程保留结束 barrier；DSV4 的正常匹配页继续由异步 session 持有，公共入口只释放其跳过加载的页面。

## 2026-09-30 MTP CPU 元数据收敛

EAGLE 复用已有 `AsyncPinnedCpuTensor` 传递接受长度和 verify 完成事件，DP overlap 传递一个合并缓冲区，在 proposer 内按 microbatch 请求数拆分视图；不新增 D2H 拷贝或 CUDA event。CPU mirror 的 accepted-tail 选行与序列推进统一由 `ModelInput` 管理，保留 DSV4 host-owned SWA 分配所需的元数据。删除 DSpark 未使用的 CPU 接受长度参数；请求统计仍复用原来的 CPU 缓冲区。
