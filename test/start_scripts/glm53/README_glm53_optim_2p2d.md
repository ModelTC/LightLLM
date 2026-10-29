# glm53_optim：Replay8 / FP8KV / 四个独立 PD Master

`glm53_optim` 基于 `support_ds4`。部署由 ConfigServer、四个独立 PD Master、
两个 P 节点和两个 D 节点组成；每台 GPU 机器一个 Master，每个 Master 一个 HTTP worker。
P0/P1 是 TP16/DP16 组，D0/D1 是另一个 TP16/DP16 组，每台 8 张 H100。

完整镜像启动说明和组件表见 [docker/README.glm53_optim.md](../../../../docker/README.glm53_optim.md)。
脚本入口为 `glm53-pd config|master|p0|p1|d0|d1`，无需 HAProxy。
P/D 通过 ConfigServer 的 DNS/服务地址发现节点；角色名填写组内 `node_rank`，不需要机器 IP 列表。
保持 1M 上限、P128/D256、BF16 SSM 和 FP8KV。当前脚本使用经过全长度回归的参数：

| 参数 | P | D |
|---|---|---|
| MTP / SSM | MTP1 / native；仍加载 draft KV | MTP3 / Replay8、precompute、raw |
| batch / chunk | 8192 / 8192 | 1024 |
| mem_fraction | 0.73 | 0.85 |
| indexer logits 临时预算 | 512 MiB | 不适用 |
| PD 传输页大小 / 页数 | 32768 / 2 | 32768 / 2 |
| 每卡 KV token 上限 | 按可用显存计算 | 2800000 |

这些参数在 32 张 H100 80GB 的本模型上验证，不代表其他模型或 GPU 的容量保证。
如需恢复旧的启动参数，可统一设置 `PD_KV_PAGE_SIZE=16384 PD_KV_PAGE_NUM=1`，
P 另设 `P_MTP_STEP=3 P_SSM_STATE_MODE=replay P_BATCH_MAX_TOKENS=4096
P_CHUNKED_PREFILL_SIZE=2048 P_MEM_FRACTION=0.75 LIGHTLLM_GLM53_PREFILL_LOGITS_MIB=64`。
这只恢复参数，精度基线的原版源码需使用对应 Git 版本。

四个 Master 的 `/metrics` 各自统计；不要叠加 P/D 的同请求记录。
metadata 负载快照每 250ms 发布，超过 3 秒视为未知；这是调度估计，不是全局原子预占。

## Cache-aware 策略和测量

- 每 2048 个连续字符生成一个覆盖整块的摘要；前缀匹配超过 50% 才作为候选，并校验 P/DP 的当前进程代次；已重启的同地址节点不复用旧提示。
- 匹配字符数至少 65536，且匹配比例至少 80%，优先保留原 P/DP，即使另一个 P 暂时空闲。
- 其余请求比较已有预计未缓存工作量加本次冷计算成本。预占、共享负载发布和释放使用同一字符估算值。
- 首 token、异常、取消都会释放预占；metadata 暂不可用时，长前缀仍保留在线节点亲和，其余按 GPU 上报负载回退。
- namespace `contiguous_block_digest_v3_uncached_prefill` 隔离旧采样 key；升级时同时更新四个 Master。

这是历史位置亲和，不是 KV 驻留证明。实际 token 缓存命中由 P/D 回传元数据判断，日志包含：

```text
pd_cache_selection origin_req_id:... p_node:... candidate:... reason:... estimated_hit_rate:... reserved_prefill_chars:...
pd_dispatch origin_req_id:... block_req_id:... p_node:... d_node:... segment:...
pd_cache_result origin_req_id:... p_node:... prompt_tokens:... actual_cache_tokens:... actual_hit_rate:... ttft_ms:...
```

新增测量只记录数字、时间、原因以及请求/节点 ID，不记录 prompt、采样 key 或生成内容。
按时间窗口汇总时，以 `sum(actual_cache_tokens) / sum(prompt_tokens)` 计算 token 加权实际命中，
按 `p_node` 保留 DP 维度，用 `origin_req_id` 去重。不把关闭流的 segment timing 自动当作完整成功。
无请求样本时报告无样本；业务失败、取消、维护期错误分别统计。

纯文本请求在 Master 完成准入长度校验时保留精确 token IDs，首个 P 分段直接复用；
原字符串仍用于路由、停止词及计数。多模态、过长原文和后续续写分段保留原编码路径，
不会通过拼接两段 token IDs 来近似完整 BPE 编码。

## 性能调优范围

按 4K、32K、128K、512K、1M 分别比较冷请求、实际缓存命中、连续 Agent 续写和并发。
短输入高并发的首 token 延迟也必须验证，不能只依据长输入吞吐调整全局参数。
缓存检查点是离散的：例如输入 4092 与 4099 token 可能分别只能复用 2048 与 4096 token，
应同时检查实际输入长度和 `cached_tokens`，不能单凭命中率差异认定路由失效。

脚本支持 P 的 batch/chunk/workspace、MTP/native 状态以及 P/D 传输页大小覆盖；
所有 P/D 必须采用相同传输页参数。P 使用 MTP1/native 时仍加载 draft 模型并填充其 KV，
D 可继续使用 MTP3/Replay8。修改这些配置后应成组重启 P/D；单独滚动重启 D 的传输恢复
不应仅凭 readiness 成功判定通过，需验证完整请求流。

## 验证

控制面测试使用临时端口、微型 tokenizer 和假 P/D，启动四个真实独立 Master，验证注册、
请求 ID 唯一、各入口转发与退出；不启动真实模型。所有实例必须使用同一 Schema 长度：

```bash
CUDA_VISIBLE_DEVICES= LIGHTLLM_JSON_SCHEMA_MAX_LENGTH=65536 \
  python -m pytest -q unit_tests/server/test_glm53_independent_masters.py
```

真实 GPU 的通信、模型启动、流式完整性和实际缓存命中仍需在目标集群逐项验证。
