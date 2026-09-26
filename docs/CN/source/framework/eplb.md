# EPLB 专家负载均衡实现

本文介绍 LightLLM 中 Expert Parallelism Load Balancer（EPLB）的完整实现，包括物理专家槽位、在线路由、负载采集、布局规划、权重迁移、运行时状态机、布局持久化，以及如何扩展新的规划算法。

EPLB 的目标是在不改变模型逻辑专家语义的前提下，利用额外的物理专家副本缓解热点专家造成的 EP rank 负载不均。它把“模型选择了哪个逻辑专家”和“本次由哪个物理副本执行”分成两个阶段，并允许服务运行期间重新安排物理副本。

## 1. 核心概念

设：

- `E`：模型每层的逻辑专家数；
- `W`：EP world size；
- `R`：每个 rank 配置的冗余专家数；
- `E / W`：每个 rank 原本持有的专家数；
- `E / W + R`：每个 rank 实际分配的物理专家槽位数；
- `E + W * R`：一个 MoE 层在整个 EP world 中的物理槽位总数。

逻辑专家 ID 来自模型路由器，范围固定为 `[0, E)`。物理专家 ID 标识实际执行权重所在的槽位：

```text
physical_expert_id = rank * num_physical_experts_per_rank + local_slot
```

一个逻辑专家可以拥有多个物理副本，但同一个 rank 上不会重复放置同一个逻辑专家。任意合法布局还必须满足：

1. 每个 rank 的物理槽位数相同；
2. 所有逻辑专家至少有一个物理副本；
3. 所有逻辑专家 ID 都在 `[0, E)` 范围内；
4. 同一 rank 内的逻辑专家 ID 不重复。

## 2. 启用方式

EPLB 通过冗余专家数量开启：

```bash
python -m lightllm.server.api_server \
    --model_dir /path/to/model \
    --enable_ep_moe \
    --eplb_num_redundant_experts_per_rank 2 \
    --eplb_plan_mode greedy \
    --eplb_rebalance_count 1 \
    --eplb_config_path /path/to/eplb-placement.json
```

主要参数如下：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--enable_ep_moe` | 关闭 | 启用专家并行；EPLB 的前置条件 |
| `--eplb_num_redundant_experts_per_rank` | `0` | 每个 rank 的额外物理专家槽位数；大于 0 时启用 EPLB |
| `--eplb_plan_mode` | `greedy` | 选择动态布局规划算法；当前支持 `greedy` |
| `--eplb_rebalance_count` | `1` | 最多完成的动态重排次数；`-1` 表示不限次数，`0` 表示不动态重排 |
| `--eplb_config_path` | `None` | 可选的布局加载与回写路径 |

完整命令行说明见 {doc}`../tutorial/api_server_args`。

在 PD 分离部署中，prefill 和 decode 进程各自拥有独立的 EPLB manager，可以分别设置 `--eplb_plan_mode`。同一个 EP 通信组内的所有 rank 必须使用相同配置。非 PD 部署只有一个 manager，它根据该进程采集到的全部路由负载生成统一布局。

## 3. 总体架构

```text
模型路由器
  │
  │ logical top-k IDs
  v
EPLB 路由 kernel
  ├── 把本次 prefill 的 logical expert 负载写入环形采样行
  ├── 查询 logical_to_physical_map
  └── 为每个 token 选择 physical expert ID
          │
          v
      MoE 执行 kernel

周期性控制面：

prefill_route_counter（最近 24 次 prefill 采样）
  -> 全局负载汇总
  -> placement planner
  -> target placement
  -> transfer planner
  -> 后台权重传输
  -> 安全边界提交权重和路由 metadata
```

主要实现位置：

| 模块 | 职责 |
| --- | --- |
| `fused_moe/impl/deepgemm_impl.py` | 初始化 EPLB 运行态，在 MoE 执行前修复 logical top-k IDs |
| `triton_kernel/fused_moe/eplb_topk_ids.py` | 统计逻辑专家负载，并把 logical ID 映射为 physical ID |
| `eplb/placement/initial.py` | 构建确定性的初始专家布局 |
| `eplb/placement/routing.py` | 根据完整布局构建紧凑路由表 |
| `eplb/placement/planner.py` | 布局规划器抽象接口 |
| `eplb/placement/factory.py` | 根据 `eplb_plan_mode` 创建具体规划器 |
| `eplb/placement/greedy.py` | 默认的贪心布局算法 |
| `eplb/async_transfer_planner.py` | 在后台生成跨层传输批次 |
| `eplb/expert_transfer.py` | 规划槽位依赖并执行专家权重传输 |
| `eplb/runtime_manager.py` | 驱动状态机，协调采集、规划、传输和提交 |

## 4. 初始化布局与权重加载

### 4.1 默认布局

启动时首先把逻辑专家连续划分到各 rank，然后从下一个 rank 的主专家区间开始循环选择冗余副本。例如 `E=8`、`W=4`、`R=2` 时：

```text
rank 0: [0, 1, 2, 3]
rank 1: [2, 3, 4, 5]
rank 2: [4, 5, 6, 7]
rank 3: [6, 7, 0, 1]
```

每行前两个槽位来自原始连续划分，后两个槽位是启动时已经加载完成的冗余副本。运行期允许重新分配所有物理槽位，不再区分不可移动的“主槽位”和只能替换的“冗余槽位”。

### 4.2 从历史布局启动

指定 `--eplb_config_path` 后，每个 MoE 层会尝试加载历史布局。配置必须同时匹配：

- 配置版本；
- 逻辑专家数；
- world size；
- 每个 rank 的冗余专家数；
- 模型层号；
- 每层布局形状、专家 ID 范围、rank 内唯一性和全专家覆盖关系。

任意校验失败都会记录 warning，并仅对受影响的层回退到默认布局。校验通过时，专家权重会直接按照历史布局加载，不需要服务启动后再执行一次恢复迁移。

### 4.3 本地运行态

每个 MoE 实现对象持有：

- `local_logics_expert_ids_list`：本 rank 每个物理槽对应的逻辑专家；
- `logical_to_physical_map`：logical ID 到可用 physical IDs 的设备路由表；
- `prefill_route_counter`：shape 为 `[24, E]` 的 `int64` GPU 环形采样缓冲区；
- `prefill_route_sample_index`：shape 为 `[2]` 的 `int64` GPU 状态，分别保存 sample index 和核内同步计数；
- `num_redundant_experts_per_rank`：本 rank 的额外槽位数。

目前启用 EP MoE 时使用 `FuseMoeDeepGEMM` 实现。EPLB manager 只收集启用了 EP 的 `layer.experts`，并保留模型中的层顺序。

## 5. 在线路由与负载采集

### 5.1 logical ID 与 physical ID 分离

MoE 路由器首先只在模型的逻辑专家空间中计算 top-k：

```text
_select_experts
  -> topk_weights + logical_topk_ids
  -> capture callback
  -> _prepare_expert_execution
       -> EPLB logical-to-physical 映射
  -> _fused_experts
```

逻辑 ID tensor 不会被原地修改。监控和 capture callback 始终看到模型语义上的 logical expert；只有实际执行 MoE kernel 前才生成新的 physical ID tensor。

### 5.2 路由表布局

每个 logical expert 对应一行固定宽度 metadata：

```text
[global_count, node_count, current_gpu_count,
 physical_ids..., -1 padding...]
```

- `global_count`：整个 EP world 中的有效副本数；
- `node_count`：当前节点内的有效副本数，包含本卡；
- `current_gpu_count`：当前 GPU 上的有效副本数；
- `physical_ids`：按“本卡、本节点其他卡、其他节点”的顺序稳定排列；
- `padding`：未使用槽位填 `-1`，kernel 不会读取。

路由槽位上限等于整个 world 的物理槽位总数，因此布局变化不会改变 tensor 的 shape。

### 5.3 副本分发模式

路由算子要求调用方显式指定分发模式：

| 模式 | 候选副本 |
| --- | --- |
| `current_gpu_first` | 本卡存在副本时只在本卡副本间选择，否则回退到全局副本 |
| `current_node_first` | 本节点存在副本时在节点内选择，否则回退到全局副本 |
| `global_first` | 直接在全局全部有效副本间选择 |

当前 DeepGEMM EPLB 路径使用 `global_first`。未来如果要支持“本卡 -> 本节点 -> 全局”的三级回退，需要布局规划算法同时具备节点拓扑感知能力。

### 5.4 副本哈希

同一候选集合内使用 `(token_index, logical_expert_id)` 生成 32 位哈希，再对有效副本数取模。实现先用 logical expert ID 给 token index 加盐，然后执行 32 位 avalanche finalizer。

该变换由奇数乘法和可逆的异或移位组成，可以显著降低规律性 token 间隔与副本数之间的低位相关性。例如同一专家每隔 4 个 token 出现且有 4 个副本时，简单线性哈希可能退化到单一副本，avalanche mix 能将流量重新打散。

### 5.5 prefill 环形采样

负载采样只在调用方明确传入 `is_prefill=True` 时启用。decode 仍然执行 logical-to-physical 映射，但不会更新采样缓冲区，也不会推进 sample index。这样可以只使用吞吐量较大、统计稳定性更好的 prefill 路由结果，同时避免给高频 decode 路径增加原子操作。

每层使用一个 `[24, E]` 的 `prefill_route_counter`。其中每一行表示一次 prefill 路由 kernel 调用的 logical expert 直方图，24 表示最多保留最近 24 次采样，而不是 24 个 token 或 24 个 manager step。一次 manager step 内如果发生多次 prefill dispatch，它们会分别占用不同的采样行；第 25 次采样开始按环形方式覆盖最旧的数据：

```text
sample_row = sample_index % 24

prefill_route_counter
  row 0  -> 一次完整 prefill dispatch 的 [expert_0, ..., expert_E-1] 计数
  row 1  -> 下一次完整 prefill dispatch 的计数
  ...
  row 23 -> 最近 24 次采样中的一行
```

固定 24 行可以限制设备内存和 CPU 快照成本，并让规划器观察最近一段时间的流量，而不是让很早以前的流量永久影响当前布局。当前 manager 在评估时沿 sample 维求和，将 `[24, E]` 聚合回 `[E]`，因此现有 planner 接口无需感知环形缓冲区。

计数始终使用 logical expert ID，而不是最终选中的 physical expert ID。同一逻辑专家即使拥有多个物理副本，规划器看到的仍然是一份完整需求量，不会因为副本分发而被拆散。

### 5.6 无额外清零 kernel 的采样事务

环形行在复用前必须清零，否则新旧两次采样会叠加。为避免每次 prefill 额外发射一个清零 kernel，清零、路由计数和 sample index 提交都融合在 `_eplb_repair_topk_ids_kernel` 内；其中 `_record_prefill_route_sample` 是 Triton 子 JIT 函数，不会形成独立的 kernel launch。

`prefill_route_sample_index` 的两个元素含义如下：

```text
[0] sample index：单调递增；对 24 取余得到当前环形行
[1] sync state ：0                     表示目标行尚未清零
                 1                     表示清零完成，采样可以开始
                 1 + completed_programs 表示已经完成的 program 数
```

一次采样事务按以下顺序执行：

1. 所有 program 读取同一个 sample index，并计算本次目标行。sample index 只由最后完成者推进，因此在本次 kernel 生命周期内保持不变。
2. `program_id == 0` 清空目标行，然后通过带 `release` 语义的原子加一把 sync state 从 0 发布为 1。
3. 其他 program 使用带 `acquire` 语义的原子读等待 sync state 达到 1，确保不会在清零完成前向目标行累加。
4. 屏障通过后，每个 program 根据自己处理的有效 top-k 元素，对对应 logical expert 执行 `atomic_add(1)`。
5. 每个 program 完成 physical ID 写回和负载计数后，再对 sync state 原子加一，提交一个完成信号。
6. Triton 的 `atomic_add` 返回加法前的旧值，因此用 `old_value + 1 == num_programs + 1` 判断唯一的最后完成者。额外的 1 是步骤 2 发布的 ready 标记。
7. 最后完成者先把 sample index 加一，再把 sync state 交换为 0，使下一次 kernel 可以复用后续环形行。

完整状态变化如下：

```text
sync=0
  -> program 0 清零目标行
  -> sync=1（ready）
  -> 所有 program 统计 logical expert 并分别提交完成信号
  -> sync=1+num_programs
  -> 唯一最后完成者推进 sample index，并复位 sync=0
```

ready 发布使用 `release`、等待方使用 `acquire`，最后完成信号使用 `acq_rel`，从而约束目标行清零和后续原子计数的可见顺序。所有调用还必须在同一 CUDA stream 上串行复用同一组 counter 和同步状态；当前 MoE forward 与采样都位于 overlap stream，满足这一约束。

### 5.7 manager 聚合与重置

manager 在安全推理边界把各层 `[24, E]` 环形缓冲区沿第 0 维求和并复制到 CPU，得到 planner 使用的 `[layer, logical_expert]` 负载。如果样本量不足或规划结果未改变布局，不主动清空缓冲区；后续 prefill 会继续写入，并在容量用满后滚动覆盖最旧行。

初始化、成功切换到新布局，以及达到重排次数上限后开始下一轮指标窗口时，manager 会同时清零 `prefill_route_counter` 和 `prefill_route_sample_index`。清零提交到 overlap stream，自然排在此前 forward 之后、后续 forward 之前，不需要额外的全设备同步。

## 6. EPLB 状态机

`EPLBManager.step()` 在安全的推理边界被调用，每次最多处理一个状态：

```text
[COLLECTING]
  将 prefill logical route 写入 24 行环形采样，等待评估周期
        |
        v
[EVALUATING]
  CPU 快照、指标上报、样本量与重排次数检查
        |
        v
[PLAN_PLACEMENT]
  汇集全局负载，rank 0 启动后台布局规划
        |
        v
[WAIT_PLAN_PLACEMENT_FINISHED]
  轮询规划结果，并广播目标布局
        |
        v
[PLAN_TRANSFER]
  各 rank 根据相同布局启动后台传输规划
        |
        v
[WAIT_PLAN_TRANSFER_FINISHED]
  等待所有 rank 生成一致的传输批次
        |
        v
[TRANSFERRING]
  分批启动/轮询权重传输，在安全边界统一提交
        |
        `-------------------------------> COLLECTING
```

提前返回 `COLLECTING` 的分支：

```text
EVALUATING
  |-- 平均 token 数不足 ----------> 保留环形窗口，继续滚动采样
  `-- 达到重排次数上限 ----------> 清空采样，只做周期性指标上报

WAIT_PLAN_PLACEMENT_FINISHED
  `-- 目标布局与当前布局相同 -----> 保留环形窗口，等待下次评估
```

默认每 20 个采样 step 评估一次，可以通过环境变量 `LIGHTLLM_EPLB_STEP_INTERVAL` 调整。该值必须大于 0。

只有当整个 world 的平均样本量达到每个“层 × 逻辑专家”128 个 token 时才开始规划。样本不足不会清空环形缓冲区，低流量服务可以跨多个评估周期继续采样；缓冲区写满后只保留最近 24 次 prefill dispatch。

## 7. 专家分布分析

基于 DeepSeek-R1（EP8 + DP8，单机 8xH200）加载 ShareGPT 语料（6000 条，input 512-2048 token）
实测的路由负载分布，用于校准布局规划（第 8 节）的设计假设。

### 7.1 各 rank 分布相似性（实测）

在 `EPLBManager` 全局汇总负载（`_step_plan_placement` 的 `all_gather` 之后）时，
把各 rank 的 `[layer][logical_expert]` 负载逐层归一化为概率分布，以 rank0 的分布为基准，
与其余 rank 逐层计算 cosine 相似度。观测窗口为 warmup 阶段一次完整采样
（58 个 MoE 层 × 7 对，共 406 对）：

| 指标 | 数值 |
| --- | --- |
| cosine 全体 min / 中位 / max | 0.9432 / 0.9728 / 0.9972 |
| 各 rank 对 rank0 的均值 | 0.971 ~ 0.977 |
| 最不相似层（层均值） | layer 43 (0.954)、32 (0.958)、41 (0.959) |
| 最相似层（层均值） | layer 0 (0.997)、1 (0.994)、4 (0.994) |

探针日志示例：

```text
eplb load prob cosine layer=0 vs_rank1..7: 0.9967 0.9965 0.9969 0.9962 0.9961 0.9972 0.9963
eplb load prob cosine summary mean_by_rank(0..7): 1.0000 0.9773 0.9711 0.9745 0.9710 0.9762 0.9749 0.9735
```

结论：**所有层上各 rank 的专家负载分布形状高度一致**。0.94~0.99 之间的小幅差异
主要来自每 rank 仅承载约 1/8 流量的采样噪声，而非分布本身存在 rank 间异质性。
DP 随机分流下每个 rank 的路由统计都是对全局路由分布的无偏采样，
分布形状（倾斜度、热点名单、长尾形态）在 rank 之间同源。

### 7.2 设计依据：分布规划只使用 rank0 的数据

上述相似性是后续设计中**分布规划只使用 rank0 负载统计**的合理性依据：

1. **代表性**：各 rank 分布形状同源且高度一致（cosine 中位 0.97，浅层几乎重合），
   rank0 的逐层概率分布与全局聚合分布在形状上等价，
   以 rank0 为样本做倾斜度分档、热点估计、副本预算分配（注水）不会产生系统性偏差。
2. **成本**：规划无需等待全 rank 负载汇聚即可获得分布形状，
   采集与规划的关键路径缩短到单 rank 统计，状态机的全局同步点相应减少。
3. **误差边界**：单 rank 估计相对全局的偏差上界为观测到的采样噪声
   （cosine ≥ 0.94），配合负载估计的平滑处理与迁移增益门槛，
   不会因单 rank 采样波动触发错误的副本迁移决策。

因此布局规划中所有"分布形状"相关的决策（概率分布、倾斜度、热点排序）
均以 rank0 的 logical expert 负载统计为准。

## 8. 布局规划

### 8.1 规划器接口与选择

所有布局算法实现统一的 `EPLBPlanner.plan(logical_expert_load, current_placement)` 接口，返回：

```text
[layer][rank][local physical slot] -> logical expert ID
```

`--eplb_plan_mode` 只负责选择布局算法。`create_eplb_planner` 将字符串模式转换成具体实例，使状态机不依赖某个算法类。当前唯一模式为 `greedy`。

规划只在 rank 0 的后台线程执行。完成后，目标布局通过控制通信组广播给所有 rank。相同输入必须产生确定结果，便于所有 rank 生成一致的传输计划。

### 8.2 Greedy 规划算法

默认算法按层独立规划，主要步骤如下：

1. **选择全卡冗余专家**：选取负载最高的 `R` 个逻辑专家，在每个 rank 上各放置一份；
2. **确定额外副本数**：其余专家先各保留一个副本，再把剩余 `R` 个副本逐次分给当前 `load / replica_count` 最大的专家；
3. **平铺多副本专家**：使用循环 rank 游标，把同一专家的副本放到不同 rank；
4. **放置单副本专家**：按专家负载从高到低处理，每次放到当前估算负载最低且仍有空槽的 rank；
5. **复用当前布局**：先把候选 rank 行匹配到共同专家最多的当前 rank，再让共同专家尽量保留原物理槽位，以减少跨 rank 传输和 rank 内覆盖。

规划负载按 128 token 对齐，降低很小的计数波动对布局的影响。专家 ID 和 rank ID 用作稳定的平局规则，因此结果是确定性的。

## 9. 权重迁移与安全提交

### 9.1 传输计划

传输规划器逐层比较当前布局和目标布局，为每个变化的目标槽绑定一个确定的源槽。选择源槽时优先使用不会被覆盖的稳定副本；没有稳定副本时，循环使用当前已有副本，避免把读取集中在同一个 rank。

随后根据“目标槽是否仍是其他任务的源槽”建立覆盖依赖：

- **安全任务**：目标槽不再承担待处理任务的源，可以先传输并提交；
- **依赖环**：所有目标槽同时也是源槽，必须先把整个环的权重读入 pinned memory，再统一覆盖；
- **rank 冲突拆批**：普通批次中每个 rank 最多参与一条任务，在限制 pinned memory 峰值的同时保留跨 rank 并行性。

每层单独生成批次，再按层顺序拼接。这样完成一层的提交后就能立即发布该层的新路由 metadata。

### 9.2 数据路径

远程专家的传输路径为：

```text
源 GPU 权重行
  -> 源 rank pinned CPU row
  -> Gloo point-to-point
  -> 目标 rank pinned CPU row
  -> 目标 GPU live 权重行
```

如果源和目标属于同一个 rank，则只执行 GPU 到 pinned CPU 的本地暂存，不经过网络。一次专家传输会覆盖实际推理需要的全部张量，包括量化权重及其 scale、zero point 等配套状态。

控制面和权重传输分别使用独立的 Gloo process group，避免两类通信相互干扰。后台线程只负责把数据传入 pinned memory，不直接修改 live 权重。

### 9.3 提交边界

只有当所有 rank 都确认当前批次传输完成后，主推理线程才会在 overlap stream 上统一：

1. 把目标 rank 的 pinned row 写入 live GPU 权重槽；
2. 更新 `current_placement` 和本地槽位的 logical expert ID；
3. 为发生变化的层重建 `logical_to_physical_map`；
4. 将新 metadata 异步复制到 GPU。

权重和路由 metadata 在同一条 stream 上更新，后续 forward 只能看到完整提交后的状态，不会观察到“新路由指向旧权重”或“旧路由指向新权重”的中间状态。

全部批次完成后，manager 发布目标布局、清空 prefill 路由采样及设备端 sample index、增加完成次数，并回到 `COLLECTING`。

## 10. 布局持久化

成功完成重排后，rank 0 会把最新完整布局写回 `--eplb_config_path`。配置内容包括：

```json
{
  "version": 1,
  "num_logical_experts": 8,
  "world_size": 4,
  "num_redundant_experts_per_rank": 2,
  "layers": {
    "3": [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 0, 1]]
  }
}
```

写入前会再次校验全部层。实现使用独占创建的 `.lock` 文件避免多个服务同时写同一路径，并在成功写入后清除读取缓存。保存失败只记录 warning，不会中断在线推理。

## 11. 指标与运行行为

rank 0 周期性上报：

```text
lightllm_eplb_topk_expert_imbalance_ratio
```

该指标先计算每层 `max(expert_load) / mean(expert_load)`，再对有效层求平均。值越接近 1，表示观测窗口内的逻辑专家负载越均衡。

`--eplb_rebalance_count` 的行为如下：

- `-1`：持续允许动态规划和重排；
- `0`：使用初始或配置文件布局，不进行动态重排；
- 正整数：只统计实际完成且发生布局变化的重排；样本不足和布局不变不计数。

达到次数上限后，manager 仍会周期性采集和上报负载指标，但不再执行全局负载汇总和布局规划。

## 12. 当前限制

启用 EPLB 时需要满足：

- 同时设置 `--enable_ep_moe`；
- `world_size > 1`；
- 逻辑专家数可以被 world size 整除；
- 冗余专家数大于 0，且不能超过本 rank 之外可复制的逻辑专家数；
- 同一 EP 通信组使用一致的 EPLB 参数；
- 不能与 `--enable_prefill_cudagraph` 同时使用；
- 当前不支持 `--enable_rl` 组合；
- 当前不支持 SM100 GPU；
- 当前动态 EPLB 执行路径依赖 EP DeepGEMM MoE 实现。

这些限制会在参数校验或 EPLB 初始化阶段尽早失败，避免服务带着不一致的布局进入推理。

## 13. 扩展新的规划算法

新增 planner 时建议遵循以下步骤：

1. 在 `eplb/placement/` 中实现 `EPLBPlanner`；
2. 接收逻辑专家负载和当前完整布局，返回同 shape 的合法目标布局；
3. 在 `placement/factory.py` 的 builder 表中注册新的 `plan_mode`；
4. 在 CLI 和 `StartArgs` 的 `eplb_plan_mode` choices 中加入新名称；
5. 更新中英文参数文档；
6. 增加布局合法性、确定性、热点负载和迁移量测试；
7. 分别评估 prefill、decode 和混合流量，不要假设一种算法对所有部署形态都最优。

规划器必须保持以下边界：

- 不直接修改 GPU 权重或路由 metadata；
- 不执行分布式通信；
- 不改变每个 rank 的物理槽位数；
- 不遗漏逻辑专家，不在同一 rank 重复放置同一专家；
- 相同输入返回确定结果；
- 尽量复用当前 rank 和物理槽位，避免均衡收益被迁移成本抵消。

`EPLBManager`、传输规划器和提交逻辑只依赖抽象的完整布局，因此新增算法不需要修改状态机。

## 14. 测试覆盖

EPLB 单元测试主要位于 `unit_tests/common/fused_moe/test_eplb.py`，覆盖：

- 初始布局、路由 metadata 和拓扑排序；
- planner 输入校验、布局合法性、负载均衡和槽位复用；
- 状态机各分支及后台任务轮询；
- 链式依赖、环形依赖和并发传输批次；
- 权重与 metadata 的安全提交；
- 配置文件加载、校验和持久化；
- logical-to-physical kernel 的精确映射；
- prefill 环形采样的逐行计数、循环覆盖、sample index 推进和同步状态复位；
- 非 prefill 路径不修改采样状态，以及空输入不推进 sample index；
- token index `0..4096`、expert ID `0..255`，以及 `2、3、4、5、101、127、128、251` 个副本时的哈希分布。

多 GPU pinned-memory 传输测试位于 `unit_tests/common/fused_moe/test_eplb_transfer_gpu.py`。
