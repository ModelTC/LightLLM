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
