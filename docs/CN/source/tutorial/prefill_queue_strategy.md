# Prefill 排队策略

LightLLM 支持在推理进程中选择 prefill 请求的排队策略。该功能用于调整请求进入本轮资源分配的顺序，便于根据业务流量优化 TTFT、尾延迟和长短请求之间的公平性。

启动参数为：

```bash
--prefill_queue_strategy default
```

当前支持三种策略：

| 策略 | 行为 |
| --- | --- |
| `default` | 负优先级 prefill 请求在前；相同优先级内保持 FCFS 顺序。 |
| `promote_shortest` | 负优先级请求按到达顺序排在前面，只把一个最短的普通 prefill 请求提升到普通请求队头。 |
| `hrrn` | 负优先级请求保持在前，普通请求按基于 token aging 的最高响应比排序。 |

## 调度阶段

排队策略在 `ModeBackend._get_classed_reqs()` 中执行。此时后端已经取得当前可处理的 `ready_reqs`，但还没有为本轮请求分配计算 token 和 KV cache。

整体流程如下：

```text
ready_reqs
    │
    ├─ 按本轮运行条件识别 decode 和 prefill 请求
    │
    ├─ decode 请求稳定放到最左侧，不参与 prefill 策略排序
    │
    ├─ 对右侧 prefill 请求应用选定策略
    │
    └─ 按新顺序检查请求状态并分配计算与 KV cache 预算
```

最终队列始终具有以下结构：

```text
[decode 请求，保持原始相对顺序] + [经过策略排序的 prefill 请求]
```

排序发生在资源检查之前，因此队列靠前的请求会更早尝试获得本轮计算 token 和 KV cache。该机制只改变尚未执行请求的检查顺序，不会抢占正在 GPU 上运行的请求。

## Decode 与 prefill 的识别

策略通过 `ModeBackend._is_decode_req()` 使用与后续请求分类完全相同的规则：

1. `no_decode=True` 时，所有请求都按 prefill 处理。
2. 通常情况下，当 `cur_kv_len + 1 == get_cur_total_len()` 时，请求进入 decode 阶段。
3. `strict_prefill=True` 时，如果请求位于 prompt 边界，即 `cur_kv_len + 1 == input_len`，它仍按 prefill 处理。

`no_decode` 主要用于只执行 prefill 的后端。`strict_prefill` 用于需要严格区分 prompt 阶段的运行模式。

识别出的 decode 请求会稳定地移动到队列最左侧。decode 请求之间保持到达顺序，并且不会读取 `infer_high_priority`，也不会按剩余 prefill token 数排序。

## 内部推理优先级

`infer_high_priority` 是共享内存 `SamplingParams` 中的内部整数字段：

```text
0 表示普通请求
负数表示高优先级请求
```

该字段不属于公开 API，外部请求不能设置它。普通请求初始化为 `0`。在 PD 分离模式中，PD Master 仅为第二段及后续续跑分段设置 `-1`，使这些请求在推理进程的 prefill 队列中优先处理。首段即使满足高 cache 命中条件，也只会通过 `high_priority_request` 提升 HTTP 资源申请和 Router 等待队列的优先级，不会改变推理进程中的 `infer_high_priority`。

`infer_high_priority` 与 `high_priority_request` 的职责不同：

| 字段 | 类型 | 使用位置 | 作用 |
| --- | --- | --- | --- |
| `high_priority_request` | `bool` | HTTP server 与 Router 等待队列 | 调整共享内存资源重试和进入 Router 的等待顺序。 |
| `infer_high_priority` | `int` | 推理进程的 prefill 排队策略 | 调整 prefill 请求参与计算和 KV cache 资源检查的顺序。 |

两者都是内部字段。`high_priority_request` 不会被推理侧的 prefill 策略读取。

## `default` 策略

`default` 策略对 prefill 请求执行以下稳定排序：

```python
sorted(prefill_reqs, key=infer_priority)
```

Python 的 `sorted()` 是稳定排序。因此：

- 负优先级请求排在普通请求之前。
- 如果以后出现多个负优先级等级，数值更小的请求排在前面。
- `infer_high_priority` 相同时保持原始相对顺序，也就是 FCFS。
- 当所有请求的优先级都是默认值 `0` 时，队列顺序完全不变。

该策略适合希望保持简单、公平排队，同时确保 PD 内部高优先级分段能够尽快继续执行的场景。

## `promote_shortest` 策略

`promote_shortest` 不会把所有普通请求按长度排序。它只提升一个请求，具体步骤如下：

1. 按 `infer_high_priority < 0` 和 `infer_high_priority >= 0` 将请求分为高优先级组和普通组，各组保持到达顺序。
2. 高优先级组保持在普通组之前。
3. 只在非负优先级的普通请求中查找最短请求。
4. 按下式计算每个普通请求的剩余 prefill token 数：

   ```python
   max(0, req.shm_req.input_len - req.cur_kv_len)
   ```

5. 选择剩余 token 数最少的一个普通请求。
6. 将它移动到普通请求队头。
7. 其他普通请求保持原始相对顺序。

如果多个普通请求的剩余 token 数相同，会选择其中原本最靠前的请求。如果不存在普通请求，则高优先级请求的到达顺序保持不变。

这种设计只允许一个短请求越过其他普通请求，可以改善短请求的 TTFT，同时避免完整 SJF 排序持续改变整个队列并显著增加长请求的等待时间。

该模式主要适合在 PD 分离架构的 Prefill 节点上使用。Prefill 节点同时排队处理多个长短不一的 prompt 时，优先调度一个剩余 prefill token 最少的普通请求，可以让部分短请求更早完成 Prefill 并进入 Decode，从而改善其 TTFT 和首字体验。续跑分段仍位于普通请求之前，不会被短请求越过。

## `hrrn` 策略

`hrrn`（Highest Response Ratio Next，最高响应比优先）在偏向短请求的同时引入 aging，避免持续到来的短请求使长请求一直无法获得调度。LightLLM 使用与墙上时间无关的 token 形式：

该策略的设计与实现参考了 [SGLang PR #32911](https://github.com/sgl-project/sglang/pull/32911)，感谢原作者及 SGLang 社区的工作与分享。

```text
priority_score = waited_prefill_tokens / uncached_prefill_tokens
waited_prefill_tokens = processed_prefill_tokens - arrival_processed_prefill_tokens
uncached_prefill_tokens = max(1, input_len - cur_kv_len_at_first_hrrn_sort)
```

- `processed_prefill_tokens` 是当前推理后端累计提交给 prefill forward 执行的 token 数。
- 请求进入推理等待队列时，会将当前计数保存为 `arrival_processed_prefill_tokens`。
- 请求第一次参与 HRRN 排序时，策略通过动态属性保存 `uncached_prefill_tokens`。因此此前完成的 Prefix/CPU Cache 初始化会反映在服务成本中；属性绑定后保持不变，后续 chunked prefill 不会持续缩小分母。
- 分数越高越先调度；服务成本最小按一个 token 计算，避免零分母分支。
- 分数相同时使用请求 ID 作为确定性的次级排序键，确保各 TP rank 得到相同顺序。
- `infer_high_priority < 0` 的内部高优先级请求始终位于 HRRN 普通请求之前。

如果吞吐量在一段时间内近似稳定，处理过的 prefill token 数与等待时间成正比，首次 HRRN 排序时的未缓存 prefill token 数与预计服务时间成正比。因此该公式与经典 HRRN 的 `1 + wait_time / service_time` 具有相同的排序效果，同时不依赖各 TP rank 可能略有差异的墙上时钟，也不需要配置预计 TPS。

该模式主要适合 PD 分离架构的 Prefill 节点，尤其是 prompt 长度和 Prefix Cache 命中长度差异较大的流量。短请求通常能更早完成 Prefill，从而改善 TTFT 和首字体验；长请求的分数会随等待期间处理的 token 数持续增长，从而降低饥饿和尾延迟风险。

## 排序示例

假设原始队列如下：

| 到达顺序 | 请求 | 阶段 | `infer_high_priority` | 首次 HRRN 排序时的 prefill 工作量 |
| --- | --- | --- | ---: | ---: |
| 1 | `normal-a` | prefill | 0 | 100 |
| 2 | `decode-a` | decode | 0 | 不参与 |
| 3 | `pd-high` | prefill | -1 | 800 |
| 4 | `normal-b` | prefill | 0 | 50 |
| 5 | `decode-b` | decode | -1 | 不参与 |
| 6 | `normal-short` | prefill | 0 | 20 |

`default` 的结果为：

```text
decode-a, decode-b, pd-high, normal-a, normal-b, normal-short
```

`promote_shortest` 的结果为：

```text
decode-a, decode-b, pd-high, normal-short, normal-a, normal-b
```

假设三个普通请求进入队列时保存的计数相同，且之后已经处理了一批 prefill token，`hrrn` 的结果为：

```text
decode-a, decode-b, pd-high, normal-short, normal-b, normal-a
```

可以看到：

- decode 请求始终在最左侧，并保持 `decode-a, decode-b` 的相对顺序。
- `pd-high` 不会被普通短请求越过。
- `promote_shortest` 只提升 `normal-short`，`normal-a` 和 `normal-b` 的相对顺序不变。
- `hrrn` 会按响应比排列全部普通请求；随着等待量变化，长请求也能逐步前移。

## 选择策略

使用默认策略：

```bash
python -m lightllm.server.api_server \
  --model_dir /path/to/model \
  --prefill_queue_strategy default
```

提升一个最短普通请求：

```bash
python -m lightllm.server.api_server \
  --model_dir /path/to/model \
  --prefill_queue_strategy promote_shortest
```

使用 HRRN 调度普通 Prefill 请求：

```bash
python -m lightllm.server.api_server \
  --model_dir /path/to/model \
  --prefill_queue_strategy hrrn
```

建议使用与生产环境一致的请求长度分布、并发度和 prefix cache 命中率进行压测，重点观察：

- TTFT 及其 P95/P99。
- TPOT 和端到端延迟。
- 长 prompt 请求的等待时间。
- 计算 token 与 KV cache 容量紧张时的尾延迟。
- PD 续跑分段能否及时获得推理资源。
- 长请求是否能通过 aging 在合理时间内得到调度。

`promote_shortest` 主要用于改善 PD 分离模式下 Prefill 节点中部分短请求的首字体验。它只提升一个普通请求，通常比完整 SJF 更温和，但具体 TTFT 和 SLA 收益仍取决于实际流量。

`hrrn` 更适合长短请求差异明显且需要兼顾平均 TTFT 与长请求公平性的场景。其服务成本使用请求首次参与 HRRN 排序时的未缓存 token 数近似；对于每 token 成本随上下文长度明显增长的模型，这个估算并不完全等价于真实 Prefill 时间，仍应通过生产流量压测确认收益。

## 扩展新策略

策略代码位于：

```text
lightllm/server/router/model_infer/mode_backend/prefill_queue_strategy
```

新增策略时，需要：

1. 继承 `PrefillQueueStrategy`。
2. 实现 `reorder_prefill(prefill_reqs)`。
3. 在 `PREFILL_QUEUE_STRATEGIES` 中注册名称与实现类。
4. 同步更新 CLI 和 `StartArgs` 的可选值。
5. 补充策略顺序、稳定性、空队列、单请求、decode 前置和输入列表不变等测试。

基类已经负责识别并前置 decode 请求。子类只能重排传入的 prefill 请求，并应遵守以下约束：

- 不丢弃或复制请求。
- 不修改请求状态。
- 不原地修改输入列表。
- 相同排序条件下保持稳定顺序。
- 在各 TP rank 上产生一致、确定的结果。

策略对象持有完整的 `ModeBackend`，可以通过 `self.backend` 读取模型、缓存、rank 和其他调度状态。读取这些状态时仍需保证所有 TP rank 得到一致的排序结果。
