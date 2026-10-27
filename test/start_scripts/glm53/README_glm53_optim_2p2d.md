# glm53_optim：Replay8 / FP8KV / 四个独立 PD Master

`glm53_optim` 基于 `support_ds4`。部署由 ConfigServer、四个独立 PD Master、
两个 P 节点和两个 D 节点组成；每台 GPU 机器一个 Master，每个 Master 一个 HTTP worker。
P0/P1 是 TP16/DP16 组，D0/D1 是另一个 TP16/DP16 组，每台 8 张 H100。

完整镜像启动说明和组件表见 [docker/README.glm53_optim.md](../../../../docker/README.glm53_optim.md)。
脚本入口为 `glm53-pd config|master|p0|p1|d0|d1`，无需 HAProxy。
P/D 通过 ConfigServer 的 DNS/服务地址发现节点；角色名填写组内 `node_rank`，不需要机器 IP 列表。
保留之前压测的 1M、P128/D256、MTP3、BF16 SSM、Replay8/precompute/raw 和 FP8KV 参数。

四个 Master 的 `/metrics` 各自统计；不要叠加 P/D 的同请求记录。
metadata 负载快照每 250ms 发布，超过 3 秒视为未知；这是调度估计，不是全局原子预占。

## Cache-aware 策略和测量

- 抽样前缀匹配超过 50% 才作为候选，并校验 P/DP 的当前进程代次；已重启的同地址节点不复用旧提示。
- 匹配字符数至少 65536，且匹配比例至少 80%，优先保留原 P/DP，即使另一个 P 暂时空闲。
- 其余请求比较已有预计未缓存工作量加本次冷计算成本。预占、共享负载发布和释放使用同一字符估算值。
- 首 token、异常、取消都会释放预占；metadata 暂不可用时，长前缀仍保留在线节点亲和，其余按 GPU 上报负载回退。
- 新 namespace `sampled_chars_v2_uncached_prefill` 防止新旧估算单位混用。

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

## 验证

控制面测试使用临时端口、微型 tokenizer 和假 P/D，启动四个真实独立 Master，验证注册、
请求 ID 唯一、各入口转发与退出；不启动真实模型。所有实例必须使用同一 Schema 长度：

```bash
CUDA_VISIBLE_DEVICES= LIGHTLLM_JSON_SCHEMA_MAX_LENGTH=65536 \
  python -m pytest -q unit_tests/server/test_glm53_independent_masters.py
```

真实 GPU 的通信、模型启动、流式完整性和实际缓存命中仍需在目标集群逐项验证。
