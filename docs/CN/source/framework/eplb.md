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
    --eplb_run_mode prefill \
    --eplb_plan_mode topology_aware \
    --eplb_rebalance_count 1 \
    --eplb_config_path /path/to/eplb-placement.json
```

Decode 共现规划把其中两个模式参数替换为：

```bash
--eplb_run_mode decode \
--eplb_plan_mode decode_cooccurrence
```

主要参数如下：

| 参数 | 默认值 | 作用 |
| --- | --- | --- |
| `--enable_ep_moe` | 关闭 | 启用专家并行；EPLB 的前置条件 |
| `--eplb_num_redundant_experts_per_rank` | `0` | 每个 rank 的额外物理专家槽位数；大于 0 时启用 EPLB |
| `--eplb_run_mode` | `prefill` | 选择采集 prefill 负载或 decode 共现统计 |
| `--eplb_plan_mode` | `topology_aware` | 选择动态布局规划算法；prefill 支持 `topology_aware`、`global_balance`，decode 支持 `decode_cooccurrence` |
| `--eplb_rebalance_count` | `1` | 最多完成的动态重排次数；`-1` 表示不限次数，`0` 表示不动态重排 |
| `--eplb_config_path` | `None` | 可选的布局加载与回写路径 |

完整命令行说明见 {doc}`../tutorial/api_server_args`。

`eplb_run_mode` 决定采集哪一类路由统计。`topology_aware` 和 `global_balance` 是 prefill
专用算法；decode 使用 `decode_cooccurrence` 消费专家共现矩阵。plan mode 与 run mode 不匹配时
会在启动子进程前的参数校验阶段直接失败。
在 PD 分离部署中，prefill 和 decode 进程各自拥有独立的 EPLB manager。
同一个 EP 通信组内的所有 rank 必须使用相同配置。非 PD 部署只有一个 manager，由 `eplb_run_mode`
决定该 manager 采集 prefill 负载还是 decode 共现统计。
当 `eplb_run_mode=decode` 时，prefill 环形采样路径会在编译期关闭，decode 路径改为更新
logical expert 共现矩阵的上三角；反之，`prefill` 模式不会给高频 decode 路径增加统计原子操作。
decode 共现统计、全局汇集、共现规划和权重迁移复用同一套 manager 状态机。planner factory
统一校验 plan mode 与 run mode 是否匹配。

## 3. 总体架构

```text
模型路由器
  │
  │ logical top-k IDs
  v
EPLB 路由 kernel
  ├── prefill：把 logical expert 负载写入环形采样行
  ├── decode ：把同一 token 的无序 expert pair 写入共现矩阵上三角
  ├── 查询当前阶段的路由 metadata
  └── 为每个 token 选择 physical expert ID
          │
          v
      MoE 执行 kernel

周期性控制面：

prefill_route_counter（24 次 prefill 采样）或 decode_route_counter（E × E 上三角共现矩阵）
  -> 全局负载汇总
  -> 对应 run mode 的 placement planner
  -> target placement
  -> transfer planner
  -> 后台权重传输
  -> 安全边界提交权重和路由 metadata
```

主要实现位置：

| 模块 | 职责 |
| --- | --- |
| `fused_moe/impl/deepgemm_impl.py` | 初始化 EPLB 运行态，在 MoE 执行前修复 logical top-k IDs |
| `triton_kernel/fused_moe/prefill_eplb_topk_ids.py` | Prefill 统一入口，按输入 token 规模选择最小映射或副本哈希 |
| `triton_kernel/fused_moe/prefill_eplb_topk_ids_minimal.py` | Prefill 小批量直接查询最小映射，控制活跃物理专家规模 |
| `triton_kernel/fused_moe/prefill_eplb_topk_ids_replica_hash.py` | 原 prefill 副本哈希实现，保留本卡、本节点和全局三种候选策略 |
| `triton_kernel/fused_moe/prefill_eplb_route_statistics.py` | 两条 prefill 路由共用的逻辑专家环形采样与核内同步 |
| `triton_kernel/fused_moe/decode_eplb_topk_ids.py` | Decode 专家映射统一入口，通过环境变量选择算法 |
| `triton_kernel/fused_moe/decode_eplb_topk_ids_minimal.py` | 默认路由的小批量路径，直接查询最小映射并记录逻辑专家共现 |
| `triton_kernel/fused_moe/decode_eplb_topk_ids_hash_targets.py` | 本卡优先，token 级伪随机初选远端副本，再按共同 rank 随机优先级共址 |
| `triton_kernel/fused_moe/decode_eplb_topk_ids_rank_weighted.py` | 默认矩阵加权路由：按节点局部性和远端专家加权覆盖分评分，同分允许任选最高分 rank |
| `triton_kernel/fused_moe/decode_eplb_route_statistics.py` | Decode 路由共用的 logical 专家共现统计 |
| `eplb/placement/initial.py` | 构建确定性的初始专家布局 |
| `eplb/placement/routing.py` | 根据完整布局构建紧凑路由表 |
| `eplb/placement/planner.py` | 布局规划器抽象接口 |
| `eplb/placement/factory.py` | 根据 `eplb_plan_mode` 创建具体规划器 |
| `eplb/placement/topology_aware.py` | 固定主专家、仅规划冗余槽位的拓扑感知算法 |
| `eplb/placement/global_balance.py` | 全局均衡的样板布局算法 |
| `eplb/placement/decode_cooccurrence.py` | 固定规范主专家槽，按 decode 共现计数选择冗余专家并复用冗余槽 |
| `eplb/async_task.py` | 统一后台线程任务的启动、完成与异常处理 |
| `eplb/async_load_gather_task.py` | 在独立 Gloo 通信组中后台汇集逐 rank、逐 sample 的原始负载 |
| `eplb/async_placement_plan_task.py` | 在后台根据负载生成目标专家布局 |
| `eplb/async_transfer_planner.py` | 在后台生成跨层传输批次 |
| `eplb/async_expert_transfer.py` | 规划槽位依赖并在后台执行专家权重传输 |
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

每行前两个槽位来自原始连续划分，后两个槽位是启动时已经加载完成的冗余副本。运行期的
可移动范围由 planner 决定：`topology_aware` 和 `decode_cooccurrence` 固定前面的规范主专家槽，
只替换末尾冗余槽；`global_balance` 可以重新分配全部物理槽位。

### 4.2 从历史布局启动

指定 `--eplb_config_path` 后，每个 MoE 层会尝试加载历史布局。配置必须同时匹配：

- 配置版本；
- 逻辑专家数；
- world size；
- 每个 rank 的冗余专家数；
- 模型层号；
- 每层布局形状、专家 ID 范围、rank 内唯一性和全专家覆盖关系。

`eplb_run_mode=decode` 时还会调用 `validate_decode_layer_placement` 校验规范主槽：
rank `r` 的前 `E/W` 个槽必须按顺序保存 `[r*E/W, (r+1)*E/W)` 的逻辑专家。
例如 `E=4、W=2` 时，rank 0 的前两槽必须是 `[0, 1]`，rank 1 必须是 `[2, 3]`。
该检查在专家权重加载前执行，prefill 模式仍允许合法的全槽重排布局。

任意校验失败都会记录 warning，并仅对受影响的层回退到默认布局。校验通过时，专家权重会直接按照历史布局加载，不需要服务启动后再执行一次恢复迁移。

### 4.3 本地运行态

每个 MoE 实现对象持有：

- `local_logics_expert_ids_list`：本 rank 每个物理槽对应的逻辑专家；
- `logical_to_physical_map`：logical ID 到可用 physical IDs 的设备路由表；
- `node_world_size`：初始化时缓存的每节点 rank 数，与路由表使用相同拓扑；
- `logical_to_rank_physical_map`：shape 为 `[E, world_size]`，按逻辑专家和 rank 查询物理 ID，没有副本的位置为 `-1`；
- `minimal_logical_to_physical_map`：shape 为 `[E]` 的 `int32` 设备表，每个逻辑专家保留一个物理槽；prefill 和默认 decode 路由在小批量时使用此表，控制活跃物理专家规模；
- prefill 模式分配 `prefill_route_counter`：shape 为 `[24, E]` 的 `int64` GPU 环形采样缓冲区；
- prefill 模式分配 `prefill_route_sample_index`：shape 为 `[2]` 的 `int64` GPU 状态，分别保存 sample index 和核内同步计数；
- decode 模式分配 `decode_route_counter`：shape 为 `[E, E]` 的 `int64` GPU logical expert 共现计数矩阵，只写包含主对角线的上三角；
- decode 模式分配 `decode_route_recording_enabled`：shape 为 `[1]` 的 `int32` GPU 标志，在有限重排次数耗尽后关闭图内共现更新；
- `num_redundant_experts_per_rank`：本 rank 的额外槽位数。

prefill 和 decode 统计缓冲区互斥分配：未启用模式对应的属性显式为 `None`，不会同时占用两套显存。

decode 伪随机初选直接复用 `logical_to_physical_map` 的节点内或全局候选，按 rank 查询和共址聚合使用
`logical_to_rank_physical_map`。`_init_eplb_routing_maps` 从同一份布局初始化三张映射表，
动态重排原地刷新全部映射表，保持 CUDA Graph 中的设备地址不变。

`build_minimal_logical_to_physical_map` 优先沿用按 rank 顺序贪心分配的结果，
候选不足时保留已选槽位，只对 `-1` 的专家从真实副本中补选当前已分配专家数最少的 rank；同数量时选较小 rank。
不回溯、不重新匹配，保证全专家覆盖，但不保证严格均分或最小单卡专家数上限。
例如布局 `[[0,1,2], [0,1,3]]` 保留 rank 0 已选的 0/1，再补选 2/3，最终专家数量为 `3:1`。
该均衡目标是逻辑专家数量，不是运行时 token 负载或通信量。

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

prefill 的每个 logical expert 对应一行固定宽度 metadata：

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
decode 使用独立的紧凑映射。设 `T` 为当前 rank 的 token 数、`K` 为 top-k、`E` 为逻辑
专家数、`W` 为 rank 数、`R` 为每卡冗余数，每卡物理槽数为 `S = E/W + R`。算子接口如下：

| 输入 | shape / 类型 | 含义 |
| --- | --- | --- |
| `logical_topk_ids` | `[T, K]`，int32 或 int64 | 模型选中的逻辑专家 ID |
| `logical_to_physical_map` | `[E, 3 + W*S]`，int32 | 复用 prefill 表，第 0 列为全局副本数，第 3 列起为有效 physical ID |
| `logical_to_rank_physical_map` | `[E, W]`，int32 | 专家在各 rank 上的物理槽，包含主槽；没有副本时为 `-1` |
| `minimal_logical_to_physical_map` | `[E]`，int32 | 小批量时各源 rank 共用的唯一物理槽映射，不包含 `-1` |
| `current_rank` / `num_physical_experts_per_rank` | Python int | 当前源 rank / 每卡槽数 `S` |
| `route_counter` | 可选 `[E, E]`，int64 | logical expert 上三角共现计数 |
| `update_route_counter` | Python bool | 是否编译共现采样分支 |
| `route_recording_enabled` | 可选 `[1]`，int32 | 设备端采样开关 |

所有输入 tensor 位于同一 GPU 且连续存储。rank 内同一 logical expert 只有一个物理槽，
所以 `logical_to_rank_physical_map[e, r]` 能直接定位副本；prefill 表的候选前缀用于伪随机初选，
按 rank 的副本表用于本卡查询和共址聚合，不再单独构建 decode 候选表。
physical ID 编码为 `rank * S + local_slot`，目标 rank 通过 `physical_id // S` 恢复。
`build_logical_to_rank_physical_map` 从实际专家布局构建 `[logical_id][rank] → physical_id` 表，
返回 CPU list，由调用方转为设备 tensor；表本身不包含 prefill / decode 分发策略。
输出是同 shape、同 dtype 的新 physical ID tensor；logical ID 保持只读，top-k 权重
不进入映射算子。随后输出转成 int64，连同原权重和量化 hidden states 交给 DeepEP dispatch。

### 5.3 副本分发模式

prefill 统一调用 `eplb_repair_prefill_topk_ids`。`T*W < 512` 时直接查询
`minimal_logical_to_physical_map`，所有源 rank 对同一逻辑专家使用同一物理槽，
不再优先使用本卡或本节点的冗余副本，避免小批量激活过多物理专家。
`T*W >= 512` 时使用原有副本哈希，调用方显式指定以下分发模式：

| 模式 | 候选副本 |
| --- | --- |
| `current_gpu_first` | 本卡存在副本时只在本卡副本间选择，否则回退到全局副本 |
| `current_node_first` | 本节点存在副本时在节点内选择，否则回退到全局副本 |
| `global_first` | 直接在全局全部有效副本间选择 |

副本哈希路径与 planner 的负载模型保持一致：`topology_aware` 使用 `current_node_first`，
`global_balance` 使用 `global_first`。小批量路径覆盖上述候选策略，但仍检查 mode 合法性，
并继续按原始 logical ID 更新环形负载采样；两条路径不改变 top-k 权重或逻辑专家语义。

这里 `T` 是本 rank 输入 tensor 的 token 行数（包含补齐行），不是请求并发数；
`W` 是 EP rank 数，不是节点内 rank 数。`W=8` 时 `T<=63` 使用最小映射，`T>=64` 使用副本哈希。
`T*W` 不等于不均匀输入下实际全局 token 数。512 是待调优的启发式阈值，尚未测定 prefill 的性能交叉点；
最小映射可能增加通信或热点负载，需要结合专家数、top-k、冗余数和节点拓扑调优。

所有 EPLB 布局执行 decode 时统一调用 `eplb_repair_decode_topk_ids`。通过环境变量
`LIGHTLLM_EPLB_DECODE_ROUTING_MODE` 选择分发算法，默认值为 `default_targets`：`T*W < 512` 时直接使用最小映射，
否则本卡优先，远端副本按节点局部性和倒数加权覆盖分选择，具体实现见第 15.7 节。
两种算法都不需要在运行时读取共现概率；共现统计用于规划副本位置，分发只需要当前布局
和 token 的 top-k。

| 环境变量取值 | 分发规则 | 约束 |
| --- | --- | --- |
| `default_targets`（默认） | `T*W < 512` 时直接查最小映射；否则本卡优先，远端按节点局部性和倒数加权覆盖分选副本 | 小批量不优先使用本卡冗余槽；无位图的 `top-k <= 32` 限制；不保证最小通信目标数或计算均衡 |
| `hash_targets` | 本卡优先，token 级伪随机初选远端副本，再按临时目标 rank 的共同优先级共址 | 最终远端目标集合不超出临时初选集合 |

不设置环境变量即使用上述自适应路径；加权路由的节点分固定为 `same_node * 10000000`，默认使用整数倒数
加权覆盖分，没有可调权重系数。
算子参数 `normalize_replica_coverage=True` 为默认值，显式传 `False` 可使用原始覆盖数，
编译时通过 `tl.constexpr` 选择；统一服务入口的加权路由使用默认值，没有新增环境变量。
`default_targets` 在 `T*W < 512` 时调用 `eplb_minimal_topk_ids`，直接计算 `physical_id = minimal_logical_to_physical_map[logical_id]`；
`T*W >= 512` 时调用原来的 `eplb_rank_weighted_topk_ids`。例如 `W=8` 时 `T<=63` 使用最小映射，`T>=64` 使用加权路由；
`W=16` 的边界为 `T<=31` / `T>=32`。两条路径均要求非空输入，并在同一个 kernel 中按原始 logical ID 记录共现。
`T` 是当前输入 tensor 的行数，CUDA Graph 捕获时包括补齐行；`T*W` 不是跨 rank 实际 token 数求和，也不是服务请求并发数。
算法选择在 Graph 捕获时固定，重放只读取更新后的设备表和采样开关。
启动前设置 `export LIGHTLLM_EPLB_DECODE_ROUTING_MODE=hash_targets` 可选回伪随机临时目标共址。
原加权模式取值 `rank_weighted` 已改名为 `default_targets`；原哈希模式 `default_targets` 改名为
`hash_targets`，已有显式配置需按新的含义更新。
所有 worker 必须设置相同的模式。
原 `greedy_rank_cover` 已移除，旧配置会报错，不会静默切换成另一种算法。
环境变量在进程内首次读取后缓存，CUDA Graph 捕获时已确定算法；不支持运行中修改环境变量
切换已捕获的图，需要重启进程并重新捕获。此选择只影响已启用 EPLB 的 decode ID 映射，
不改变 prefill 算法、placement planner 或 DeepEP 通信模式。

两种布局统一维护以下路由表，不依赖共现统计或规范主槽：

- `logical_to_rank_physical_map[E, W]`：按 rank 查询唯一 physical ID，缺少副本为 -1。
- `minimal_logical_to_physical_map[E]`：默认路由的小批量分支直接查表，不做本卡优先或副本共址。
- `logical_to_physical_map[E, 3 + W*S]`：直接复用 prefill 表，第 0 列为全局副本数，
  第 3 列起保存全部副本，按本卡、本节点其他卡、其他节点排序，剩余位置填 -1。
  `hash_targets` 使用 global_count/node_count 选择初选范围；本卡副本由按 rank 查询表直接处理。
  默认加权路由只读取 `logical_to_rank_physical_map`，不需要 prefill 候选表。

以下初选与共址细节仅描述可选的 `hash_targets` 算法，不是默认加权路由。
本卡没有副本时，该算子优先在本节点副本前缀中初选；`node_count=0` 才回退到全局副本前缀。
路由表与算子必须使用相同的 `current_rank/node_world_size`，节点 rank 按连续区间分组。
算子使用 `tl.randint(seed=source_rank, offset=token_index, n_rounds=10)` 生成 uint32
伪随机数，对各专家的有效副本数取模，作为候选序号。source rank 是固定种子，
同 token 的专家共享随机数，不包含 logical ID 或 top-k 槽序号；副本数相同时共享候选序号，
但候选表不同仍可能选择不同 rank。top-k 重排不影响初选，相同输入与布局的 Graph 重放
产生相同路由。token index 是本次输入的行号，不是请求 ID；batch 重排可能改变初选，
不会在每个 decode step 主动推进随机状态。该方式提供统计分散，不保证每批负载均匀。

`hash_targets` 的规则与实现细节如下。
grouped-topk 始终输出 logical ID，随后进入 `_prepare_expert_execution`。decode 算子
按以下规则生成 physical ID，top-k 权重与 logical ID 输入保持不变：

1. 本卡存在副本时直接本地初选，否则优先在本节点候选中伪随机初选，没有才选全局候选。
2. 根据初选结果生成同 token 的临时目标 rank 集合；本地专家不会引入远端锚点。
3. 本卡没有副本时，查询临时目标集合中各 rank 是否持有该专家；初选槽也参与选择，
   按同 token 共用的固定种子 rank 随机优先级选择有效副本，没有更优副本则保留初选槽。
4. 所有专家同时依据原始集合判断，不把已重映射的专家当作新锚点。分数无碰撞时结果与
   top-k 顺序无关；同分时保留当前副本，允许重排后选择不同物理副本，但逻辑专家与权重不变。

节点优先仅改变初选范围，后续共址规则不变：如果同 token 包含节点外目标，节点内初选的
专家仍可能因 rank 分数更优而移到节点外，不保证最终结果都留在本节点。

共址分数也使用 10 轮 Philox：seed 为 `source_rank | (1 << 32)`，与初选随机流区分；
uint32 offset 为 `(token_index << rank_bits) | target_rank`，其中
`rank_bits = (world_size - 1).bit_length()`，低位预留给 rank，非二次幂 world size 也适用。
溢出按模 `2**32` 回绕，不包含 logical ID 或 top-k 槽序号。
同 token 的专家对同一 rank 使用相同的 uint32 随机分数，不同 token 可以偏好不同 rank。
直接比较 uint32 分数，越小越优先；同分保留当前副本，不额外比较 rank 或 physical 槽号。
同分只可能改变物理副本选择及其通信/计算负载，不改变逻辑专家选择。

副本表按 rank 直接索引。kernel 只查询本卡和 top-k 的临时目标，不扫描整个 world，
也不再把多副本布局截断成单个冗余候选。两种布局共用同一套规则，并且都严格保留本地执行。

因此对任意 token，最终远端目标集合始终是该 token 初选远端集合的子集。
该基准不再等价于规范主槽的 R=0 目标集合，不能据此保证相对于重排前或原始 R=0
布局也减少通信。规则只保证相对于临时初选结果的目标集合不扩大，
不保证每次副本切换都减少目标数，也不是逐 token 最小目标集合的全局优化。

例如源 rank 为 0，专家 a/b 的初选目标为 1/2，且二者在 rank 1/2 都有副本：
若该 token 对 rank 2 的分数更小，a/b 都选择 rank 2，远端集合从 `{1, 2}` 缩小为 `{2}`，
不会强制换到其他槽而造成 a/b 互换目标。这个选择只依赖布局与固定种子随机优先级，不需要历史共现概率；
不同 token 的 rank 偏好提供统计分散，不保证实际计算负载均匀。

#### decode 算子实现

Triton kernel 按 `[BLOCK_TOKENS, BLOCK_TOP_K]` 二维块处理。`BLOCK_TOP_K` 为不小于
`K` 的最小二次幂，`BLOCK_TOKENS = max(1, 256 // BLOCK_TOP_K)`，每个 program 使用
4 个 warps。例如 `K=8` 时一个 program 处理 32 个 token，`T=128` 时启动 4 个 program；
`K=6` 时列宽补到 8，额外列和末尾越界 token 均通过 mask 屏蔽。

每个 program 加载 logical ID 和本地槽；远端 lane 从 prefill 表第 0/1 列读取 global_count/node_count，
优先使用非零 node_count，否则使用 global_count。每节点只有一个 rank 时直接使用 global_count。
按 token 级伪随机数索引第 3 列起的候选 physical ID，把初选目标 rank 保存在寄存器，并预先计算
这些 rank 的共同优先级。随后遍历 `K` 个 top-k 位置，通过 `tl.gather` 取出同 token 的
临时目标 rank 和分数，查询 `logical_to_rank_physical_map[expert, target_rank]`。
以初选槽及其分数初始化最优结果，仅在远端 lane 存在更优有效副本时更新，因此本地专家
不会迁出，初选槽也无需额外回退逻辑。重复的临时目标 rank 可以重复查询，不影响结果。

每个 token 的 `K` 个专家分别检查 `K` 个临时目标，候选查询量为 `O(TK²)`，不扫描
全部 `W` 个 rank，也不随副本数量逐个遍历。Decode 只额外分配按 rank 查询表，
占用 `4EW` 字节；例如 `E=256、W=8` 时约 8 KiB，候选直接复用已有 prefill 表。
映射和可选统计在一次 kernel launch 内完成，实际通信
由后续 DeepEP dispatch 执行。

动态 decode 布局额外记录 logical expert 上三角共现计数。布局规划从目标 rank 的规范主专家
集合出发，每选中一个冗余专家，就将它的共现计数加入后续候选评分。prefill 布局执行 decode 和固定 decode
布局的统计分支由 constexpr 裁剪；有限重排耗尽后，动态 kernel 通过设备标志跳过整个
统计分支，包括 partner 查询、索引计算和原子计数，仅保留标志读取与分支判断。
CUDA Graph 无需重新捕获，也不会重新编译为纯映射 kernel。统计始终使用原始 logical ID：
例如 top-k 为 `[4, 7]` 时更新 `[4,4]`、`[4,7]`、`[7,7]`，不受最终副本位置影响。

布局更新时，manager 在权重整批提交的安全推理边界，同时原地覆盖 prefill 表和按 rank 查询表。
两张表的 shape 和设备地址保持不变，CUDA Graph 重放直接读取新布局。

### 5.4 副本哈希

prefill 在同一候选集合内使用 `(token_index, logical_expert_id)` 生成 32 位哈希，
再对有效副本数取模。实现先用 logical
expert ID 给 token index 加盐，然后执行 32 位 avalanche finalizer。

该变换减少规律性 token 间隔与副本数的低位相关性，避免简单取模造成副本集中。
decode 不调用 prefill kernel 的副本哈希分支，而是执行第 5.3 节选定的映射算法；
`hash_targets` 分别进行 token 级伪随机初选和同 token 的 rank 随机优先级选择，
默认 `default_targets` 不使用哈希或随机扰动。

### 5.5 prefill 环形采样

只有 `eplb_run_mode=prefill` 且调用方明确传入 `is_prefill=True` 时，才启用
prefill 负载采样。decode 仍然执行 logical-to-physical 映射，但不会更新该缓冲区，也不会推进
sample index。

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

### 5.6 decode 上三角专家共现矩阵

只有 `eplb_run_mode=decode` 且当前调用为 decode 时，才更新每层的 `[E, E]`
`decode_route_counter`。设一个 token 的 top-k logical expert 集合为 `S`，kernel 只为满足
`i <= j` 的无序 expert 对执行：

```text
decode_route_counter[i, j] += 1
```

使用 expert ID 的大小关系决定存储方向，而不是依赖 top-k 槽位顺序，因此同一个非对角 pair
始终落在 `[min(i, j), max(i, j)]`，只记录一次。`<=` 明确包含主对角线：当 `i == j` 时，
该位置记录 expert `i` 被路由选中的精确次数，不会因为上三角过滤而遗漏。top-k ID 唯一时，
每个 token 总共贡献 `top_k × (top_k + 1) / 2` 次计数；top-k=8 时从完整对称记录的
64 次降为 36 次。热点地址上的原子竞争仍可能主导统计成本。

实现以展平后的每个 top-k 元素作为矩阵行更新者，并在 Triton 静态循环中读取同一 token 的
其他 top-k 元素。所有位置使用原子加，不需要 prefill 环形行清零所需的 program 间同步，
也不会增加额外 kernel launch。manager 评估时完整汇集各 rank 的上三角矩阵；样本量判断和
负载指标使用主对角线。该统计不会传给 prefill 专用的 `topology_aware` 或 `global_balance`。
`decode_cooccurrence` 在 CPU 规划路径中先聚合源 rank，再恢复对称矩阵：

```text
full = upper + upper.T - diag(diagonal(upper))
```

减去一次主对角线是为了避免把 expert 自身负载重复两次。

### 5.7 prefill 采样事务

环形行清零、路由计数和 sample index 提交融合在同一个路由 kernel 内，不额外发射清零 kernel。
`prefill_route_sample_index[0]` 保存单调递增的 sample index；`[1]` 保存同步状态：
`0` 为未清零，`1` 为 ready，`1 + completed_programs` 为已完成的 program 数。

1. 所有 program 使用同一个 sample index，目标行为 `sample_index % 24`。
2. program 0 清零目标行并以 `release` 发布 ready；其他 program 以 `acquire` 等待。
3. 各 program 累加 logical expert 计数、写回 physical ID，再以 `acq_rel` 提交完成信号。
4. 唯一最后完成者推进 sample index，并将同步状态复位为 0。

同一组 counter 与同步状态必须在同一 CUDA stream 上串行复用；当前 forward 与采样使用
overlap stream，避免环形行清零与统计写入交错。

### 5.8 manager 聚合与重置

manager 在安全推理边界按运行模式堆叠各层统计，并复制为 CPU 快照：prefill 是
`[layer, sample, expert_num]`，decode 是 `[layer, expert_num, expert_num]`。rank 0 使用
prefill sample 总量或 decode 主对角线判断样本量；样本充足时，后台任务使用独立 Gloo
通信组执行 all-gather，分别得到 `[rank, layer, sample, expert_num]` 或
`[rank, layer, expert_num, expert_num]`，再把原始四维统计交给当前 run mode 对应的 planner。

样本量不足时不主动清空缓冲区；后续 prefill 滚动覆盖最旧行，decode 继续累积共现次数。
规划结果未改变布局时保留当前统计，成功切换布局后开始新的统计窗口。

初始化、成功切换到新布局，以及开始下一轮指标窗口时，manager 只清零当前 run mode 已分配的
统计缓冲区。清零提交到 overlap stream，自然排在此前 forward 之后、后续 forward 之前，不需要
额外的全设备同步。

## 6. EPLB 状态机

`EPLBManager.step()` 在安全的推理边界被调用，每次最多处理一个状态：

```text
[COLLECTING]
  按 run mode 写入 prefill 环形样本或 decode 上三角共现矩阵，等待评估周期
        |
        v
[EVALUATING]
  CPU 原始样本快照、指标上报；rank 0 判断样本量并广播结果
  样本充足时启动后台 load all-gather
        |
        v
[WAIT_LOAD_GATHER_FINISHED]
  等待各 rank 汇集完成，保留 planner 需要的原始四维输入
        |
        v
[PLAN_PLACEMENT]
  rank 0 启动后台布局规划
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
  |-- rank 0 平均 token 数不足 ----> 保留环形窗口，继续滚动采样
  `-- 达到重排次数上限 ----------> prefill 清空采样，只做周期性指标上报

step（decode）
  `-- 达到有限重排次数上限 ------> 关闭设备采集，停止统计状态机

WAIT_PLAN_PLACEMENT_FINISHED
  `-- 目标布局与当前布局相同 -----> 保留环形窗口，等待下次评估
```

默认每 128 个采样 step 评估一次，可以通过环境变量 `LIGHTLLM_EPLB_STEP_INTERVAL` 调整。该值必须大于 0。

专家权重迁移默认同时推进 16 个 MoE 层，可以通过环境变量
`LIGHTLLM_EPLB_TRANSFER_LAYER_PARALLELISM` 调整，该值必须大于 0。每层内部仍严格按照
槽位覆盖依赖顺序执行；并行只发生在相互独立的不同层之间。增大该值可以提高通信和
GPU/CPU 拷贝的重叠度，但也会按活动传输任务数增加 pinned memory 和 CUDA stream 的峰值占用。

规划门槛是 rank 0 每个“层 × 逻辑专家”的平均路由计数达到 128。rank 0 只负责触发判断，
不替代全 rank 的规划统计。样本不足时保留计数继续采样；prefill 环形窗口最多保留最近
24 次 dispatch，decode 矩阵继续累积。

## 7. 专家分布与统计汇集

DeepSeek-R1、EP8/DP8、8 张 H200 上，使用 6000 条 ShareGPT 请求（输入 512–2048 token）
观察一次 warmup 采样。58 个 MoE 层中，以 rank 0 为基准的 406 对负载分布 cosine 相似度
最小/中位/最大值为 `0.9432 / 0.9728 / 0.9972`。这组样本中的各 rank 分布较接近，
但不能据此假定所有业务流量都相同。

rank 0 只用于低成本判断是否达到规划样本量。实际规划仍在专用 Gloo 通信组中异步汇集
全部 rank 的原始统计，降低单 rank 采样噪声，并保留来源信息：

- Topology-Aware 保留 sample 维，只聚合同一源节点内的 rank。
- Global Balance 聚合 rank 和 sample 维，得到每层 logical expert 总负载。
- Decode Co-occurrence 聚合 source rank，使用每层专家共现矩阵。

主推理线程只轮询后台完成标记，统计通信不直接占用推理关键路径。

## 8. 布局规划

### 8.1 规划器接口与选择

所有布局算法实现统一的 `EPLBPlanner.plan(route_statistics, current_placement)` 接口，返回：

```text
route_statistics: CPU Tensor
  prefill: [rank, layer, sample, expert_num]
  decode:  [rank, layer, expert_num, expert_num]
[layer][rank][local physical slot] -> logical expert ID
```

`create_eplb_planner` 将 `--eplb_plan_mode` 转换成具体实例，使状态机不依赖某个算法类。
当前实现按运行阶段分为：

- `topology_aware`：固定规范主专家，只规划冗余槽位，匹配 prefill 的节点优先分发；
- `global_balance`：全局均衡的样板实现，规划完整物理布局，prefill 在全局全部副本间分发 token。
- `decode_cooccurrence`：固定规范主专家槽，按 decode 共现计数选择冗余专家；运行时分发算法由
  第 5.3 节的统一入口选择，默认使用 `default_targets`（加权路由）。

规划只在 rank 0 的后台线程执行。完成后，目标布局通过控制通信组广播给所有 rank。相同输入必须产生确定结果，便于所有 rank 生成一致的传输计划。

### 8.2 Topology-Aware 规划算法

Topology-Aware 模式以 prefill 源节点流量、副本局部性和关键 rank 计算量为规划目标。每个 rank 的前
`num_logical_experts / world_size` 个主专家槽保持不变，算法只填写末尾的冗余槽位，因此
单次规划最多改变 `layer × rank × redundant_slots_per_rank` 个槽位。主要过程如下：

1. 将 prefill 输入聚合为 `[sample, layer, source_node, expert]`，保留样本和流量来源节点；
2. 模拟节点本地优先分发：源节点存在目标专家副本时只使用节点内副本，否则回退到全局副本；
   每个 sample 的物理专家负载分别按 128 token 对齐；
3. 各层独立规划：每轮选择一个仍有空槽的低负载 rank，再同时比较该层的所有合法
   expert，选择能让各 sample 的关键 rank 计算量之和最小的副本；
4. 候选布局必须让每个被替换层的关键负载下降，并且模型级预计收益至少达到 5%，否则
   保留当前布局；最终把仍在同一 rank 的副本复用到原物理槽位。

该模式要求规范主槽布局。切换到 `topology_aware` 时，如果当前布局来自 `global_balance`
生成的全槽布局或旧配置，第一次规划会恢复规范主专家前缀，并重新生成冗余槽位；由于旧布局不满足
固定主专家假设，这一次无法与旧布局比较收益，也无法复用旧冗余槽位。后续规划只会修改
冗余槽位，并应用 5% 收益门槛。

各层独立规划以限制临时内存，并保留单层候选的向量化比较。候选生成只优化关键计算量；
收益门槛决定是否迁移，最后通过槽位复用降低传输量，不在候选目标中另加旧布局偏置。

### 8.3 Global Balance 样板规划算法

该样板算法按层独立规划，主要步骤如下：

1. **选择全卡冗余专家**：选取负载最高的 `R` 个逻辑专家，在每个 rank 上各放置一份；
2. **确定额外副本数**：其余专家先各保留一个副本，再把剩余 `R` 个副本逐次分给当前 `load / replica_count` 最大的专家；
3. **平铺多副本专家**：使用循环 rank 游标，把同一专家的副本放到不同 rank；
4. **放置单副本专家**：按专家负载从高到低处理，每次放到当前估算负载最低且仍有空槽的 rank；
5. **复用当前布局**：先把候选 rank 行匹配到共同专家最多的当前 rank，再让共同专家尽量保留原物理槽位，以减少跨 rank 传输和 rank 内覆盖。

规划负载按 128 token 对齐，专家 ID 和 rank ID 用作稳定的平局规则。
该算法只优化估算计算量，不考虑 expert group 或通信拓扑；用于样板和对照，不宜直接作为
具有强 group/节点局部性的生产布局。

### 8.4 Decode Co-occurrence 规划算法

Decode Co-occurrence 模式直接以“同一个 token 同时选择了哪些逻辑专家”为通信局部性信号。
每个 rank 固定前 `E / W` 个规范主专家槽，再选择 `R` 个冗余专家；
主专家作为评分起点并保证全局覆盖，规划只调整冗余槽。对每层执行：

1. 聚合所有 source rank 的上三角计数，恢复对称矩阵并清零评分矩阵的主对角线，
   不修改原始专家负载计数；直接使用共现计数评分，不做逐层归一化，因为公共分母不影响候选排序；
2. 每个 target rank 从本卡规范主专家集合开始，计算所有外部专家的初始共现分数：
   `score(e) = sum(count(e, x) for x in primary_experts_on_target_rank)`；
3. 每轮选出分数最高的未放置专家，相同 score 选择较小 logical expert ID；加入冗余槽后，
   对所有候选执行 `score(e) += count(e, newly_selected_expert)`。主专家及已选副本不再参与候选，
   直到填满 `R` 个槽，因此后续选择同时考虑与主专家和已选冗余专家的共现；
4. 各 rank 不共享候选占用标志。同一个 logical expert 可以在多个 rank 上获得冗余槽，
   不限制副本原属 rank，也不强制各 rank 的副本迁入、迁出数量相等；
5. 将主专家和新选冗余专家合成完整候选布局，与完整旧布局重排：共同专家保留原物理槽位，
   新选冗余专家按选择顺序填入空槽。主专家始终保留原规范槽，减少无意义的权重迁移。

输入和输出都必须满足规范主槽前缀、布局尺寸、单卡逻辑专家唯一和全局覆盖约束。
例如 rank 0 的主专家为 `[0, 1]` 时，`[0, 1, 2]` 合法，`[0, 2, 1]` 不合法。
加载历史布局时即执行 decode 主槽校验；不满足要求的层记录 warning 并回退默认布局，
不会把非规范布局交给后台 planner。仅冗余槽排列变化的布局仍可直接复用。

冗余数量只受 rank 内 logical expert 唯一性约束，容量上限
自然为 `R ≤ E − E/W`：本卡已有的 `E/W` 个主专家不能再占冗余槽。例如 `E=8、W=4` 时，
每卡有 2 个主专家，允许 `R=3、4、6`；`R=6` 时每卡恰好持有全部 8 个专家。
如果 expert 7 与前三个 rank 的主专家都有很强共现，它可以同时复制到这三个 rank；
其原属 rank 已有主槽，不再重复放置。该选择以共现局部性为目标，不保证实际计算负载均衡。

规划目标是通信局部性，不应用 prefill planner 的 128-token 对齐或 5% 计算收益门槛。
可选的 `hash_targets` 路由只在同 token 的伪随机初选目标 rank 或本卡使用副本；
默认加权路由则按完整副本布局的初始加权覆盖分评分，不要求目标 rank 已在临时集合中。
两种路由共用 logical 共现统计，但 planner 的共现评分并不等于分发后的实际收益。

空统计层按专家 ID 确定性填充；动态规划仍受 manager 最小样本量门槛保护。
有限重排次数耗尽后的统计关闭行为见第 11 节。

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
3. 为发生变化的层重建候选路由表 `logical_to_physical_map`、按 rank 查询表 `logical_to_rank_physical_map` 和最小映射表 `minimal_logical_to_physical_map`；
4. 原地异步覆盖 GPU metadata，保持 CUDA Graph 捕获的设备地址不变。

权重和路由 metadata 在同一条 stream 上更新，后续 forward 只能看到完整提交后的状态，不会观察到“新路由指向旧权重”或“旧路由指向新权重”的中间状态。

全部批次完成后，manager 发布目标布局、清空 prefill/decode 路由统计、增加完成次数，并回到
`COLLECTING`。

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

rank 0 周期性上报 logical expert 路由分布：

```text
lightllm_eplb_topk_expert_imbalance_ratio_p25
lightllm_eplb_topk_expert_imbalance_ratio_p50
lightllm_eplb_topk_expert_imbalance_ratio_p100
```

manager 先对每个有效层计算 `max(expert_load) / mean(expert_load)`，过滤没有采样负载的层，再对所有层的比值排序并使用 nearest-rank 位置取值：

- P25：较均衡的四分之一位置，可观察大多数浅层或稳定层的基线；
- P50：中位层，用于描述典型 MoE 层的路由倾斜程度；
- P100：最大值，即当前窗口中最不均衡的层。

这三个值都以 `1` 表示完全均衡。例如 P50 为 `1.8`，表示中位层最热 logical expert 的 token 数是该层专家平均值的 1.8 倍。

当 prefill 样本量达到规划阈值且 planner 产生目标布局后，rank 0 固定选取原始快照的第 0 个
sample 行并沿 rank 汇总，构造 `[layer, logical_expert]` 指标负载并上报：

```text
lightllm_prefill_ep_compute_critical_overhead_ratio_before_rebalance
lightllm_prefill_ep_compute_critical_overhead_ratio_after_rebalance
```

这些指标采用关键路径计算开销定义，但不维护独立的 compute counter、后台 monitor 线程和额外通信组。
manager 假设同一 logical expert 的流量由 hash 均匀分配给全部 physical 副本，并按 128 token 对
每个副本的估算负载向上对齐。`before_rebalance` 使用当前布局，`after_rebalance` 使用 planner
给出的目标布局；二者输入负载相同。不沿 sample 维累加，因为不同 sample 行来自不同
prefill 批次。`decode_cooccurrence` 优化的是通信局部性而不是该计算负载模型，因此 decode
不发布“重排前/重排后”的预计计算开销指标。

每层先计算最繁忙 rank 相对平均 rank 的额外负载，最后跨层汇总：

```text
overhead_ratio = sum(max_rank_load - mean_rank_load) / sum(mean_rank_load)
```

因此不同层的热点 rank 不会互相抵消。指标为 `0` 表示估算的 EP rank 负载完全均衡，`0.3` 表示最慢 rank 造成的关键路径计算量比理想均衡状态高约 30%。它是基于聚合 logical route 和均匀副本分发假设的布局质量估算值，不是 DeepEP 接收缓冲区的实测 compute load，也不表示每次 prefill 的瞬时开销。只有实际执行 placement 规划时这两个 gauge 才会更新；其余时间保留最近一次规划结果。

`--eplb_rebalance_count` 的行为如下：

- `-1`：持续允许动态规划和重排；
- `0`：使用初始或配置文件布局，不进行动态重排；
- 正整数：只统计实际完成且发生布局变化的重排；样本不足和布局不变不计数。

达到次数上限后，prefill manager 仍会周期性采集和上报负载指标，但不再执行全局负载汇总和
布局规划；decode manager 会关闭设备共现采集并停止统计状态机，因为这些计数只服务于规划。

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

主要测试位于 `unit_tests/common/fused_moe/`：

- `test_eplb.py`：布局、运行态、状态机、迁移依赖、安全提交、持久化及 prefill 环形采样。
- `test_eplb_prefill_routing.py`：prefill 统一入口、`T*W=512` 边界、三种副本模式、dtype、空输入、环形采样及 CUDA Graph 原地布局更新。
- `test_eplb_decode_cooccurrence.py`：规范主槽校验、共现评分、跨 rank 聚合、多副本、容量边界和冗余槽迁移。
- `test_eplb_decode_rank_weighted.py`：最高分副本集合校验、本地优先、top-k 重排、节点优先与全局回退及专家输出精度。
- `test_eplb_decode_routing.py`：统一入口、参数转发、`T*W=512` 切换边界、两种路由模式、CUDA Graph 布局更新与统计。
- `test_eplb_decode_minimal.py`：小批量查表映射、dtype、补齐、逻辑 ID 还原和共现计数。
- `test_eplb_decode_hash_targets.py`：双表结构、初选与共同 rank 优先级参考、本卡/节点优先、全局回退、副本共址、跨 rank 分散和 Graph 更新。
- `test_eplb_decode_rank_random.py`：`hash_targets` 的 rank 随机分数、uint32 offset 回绕、CPU/GPU 一致性、首选分布及同分时的副本选择；统计质量不等于实际路由负载均衡。
- `test_eplb_decode_statistics.py`：两种路由在同一 Graph 内反复切换统计开关，计数精确累加且映射结果不变。
- `test_eplb_transfer_gpu.py`：多 GPU pinned-memory 权重传输。

环境变量的默认值、合法性和缓存行为由 `unit_tests/utils/test_envs_utils.py` 覆盖。

## 15. 性能分析

本节区分整模型服务、通信时间轴和 routed-MoE 微基准。不同版本、输入分布与计时窗口的数据
不能混用；减少通信目标不等于同等幅度的吞吐提升。

### 15.1 评测口径

- 端到端主结果使用多样请求，规划集与评测集隔离；相同 prompt 只适合路由聚集的控制实验。
- 静态布局评测在重排完成、共现采集关闭后进行，避免把规划和迁移开销混入稳态 decode。
- 各配置保持模型、通信容量、CUDA Graph 覆盖范围和调度参数一致，预热后交替顺序重复测量。
- Profile 用于算子归因，吞吐由无 profiler 测试给出。通信 kernel 时长包含同步等待，
  不能直接当作有效网络传输时间。
- 通信与计算可重叠，各阶段的 rank 最大值也可能来自不同卡，不能简单相加或用
  `1 - 计算占比` 推导通信占比。

### 15.2 不同并发下的通信占比

以下为 2026-10-05、版本 `0e9ea6db9` 的历史测量：DeepSeek-R1、8 张 H200、
TP=8/DP=8、EP MoE、未启用 EPLB，`graph_max_batch_size=40`。每个请求生成 256 token，
全局并发 16/32/64/128/256 对应每个 DP rank 约 2/4/8/16/32 个活跃请求。

占比来自独立稳态 profiler 窗口，分母为首个 dispatch 到末个 combine epilogue 的 GPU 跨度，
不是请求全生命周期。通信包含 dispatch/combine 及其 copy/reduce epilogue；重叠区间取并集，
最后对 8 个 rank 的占比取平均。“活跃”指至少一个 DeepEP kernel 执行，“独占”指该时段
没有非 DeepEP CUDA kernel 同时执行，不等同于可获得的加速比。吞吐来自无 profiler 的
首个输出 token 到最后请求完成窗口。

| 全局并发 | 非 overlap 通信活跃 / 独占 | overlap 通信活跃 / 独占 | 非 overlap 吞吐（token/s） | overlap 吞吐（token/s） |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 22.00% / 21.90% | 41.28% / 13.49% | 814 | 424 |
| 32 | 22.22% / 22.09% | 35.40% / 9.14% | 1591 | 824 |
| 64 | 23.48% / 23.33% | 35.53% / 9.20% | 2864 | 1604 |
| 128 | 23.98% / 23.91% | 36.25% / 9.39% | 5267 | 3064 |
| 256 | 30.48% / 30.37% | 39.58% / 10.99% | 8792 | 5720 |

非 overlap 下通信基本串行，16–128 并发的通信占比约 22%–24%，256 并发升至约 30%。
核心算子的分项占比如下，不包含 epilogue：

| 全局并发 | dispatch | combine |
| ---: | ---: | ---: |
| 16 | 7.87% | 10.67% |
| 32 | 7.00% | 11.71% |
| 64 | 7.71% | 12.21% |
| 128 | 8.18% | 12.45% |
| 256 | 8.52% | 18.95% |

combine 是本组高并发通信占比增长的主要来源。Overlap 隐藏了部分通信，但没有转化为吞吐
提升；评估时还需考虑 microbatch 固定开销与资源竞争，不能只看通信覆盖率。
该结果只适用于本组配置，不能据此断言所有业务都应关闭 overlap，也不能把 kernel 活跃
时间增大理解为字节量同比增加。

### 15.3 冗余专家的端到端结果

以下数据来自规范主槽共址版本的历史评测，不是当前全部副本映射或 `rank_weighted`
的最新端到端结果。配置为 DeepSeek-R1、8 张 H200、TP=8/DP=8、EP MoE、非 overlap、
256 并发，每请求生成 256 token。输入为 256 个等长且不同的合成 prompt，规划集和评测集
无交集；关闭动态 prompt cache，重排和采集在正式计时前结束。

服务配置为 `max_total_token_num=160000`、`graph_max_batch_size=40`、
`batch_max_tokens=6000`、`max_req_total_len=56000`。预热后交替顺序、多次冷启动复测，
共 84 轮，统一剔除每个冷启动区块最初两轮，汇总稳态中位数。

| 每卡冗余数 R | 稳态样本数 | 吞吐中位数（token/s） | 相对 R=0 |
| ---: | ---: | ---: | ---: |
| 0 | 30 | 6663.76 | 基线 |
| 1 | 22 | 6631.56 | -0.48% |
| 2 | 10 | 6704.73 | +0.61% |
| 4 | 10 | 6628.51 | -0.53% |

各配置相对基线的复测置信区间均跨过 0，尚不能证明稳定的端到端正收益。
冗余数增加不保证吞吐增加：副本可能降低远端目标数，也可能增加活跃 physical segment、
最小计算 tile、权重工作集合和慢 rank 的尾部耗时。

### 15.4 128-token、八卡 DeepGEMM 计算分布

以下为 2026-10-08 的受控 routed-MoE 微基准：全局 128 个 decode token，每个源 rank
16 个；256 个 logical experts、top-k=8、FP8 权重，`hidden=7168`、
`intermediate=2048`，128-row alignment。EP 专家权重不按 TP=8 再切分；
每次调用始终有 1024 次有效 expert-token 计算。

使用真实尺寸合成权重、四套层权重和四批输入，单机 NVLink、非 overlap；通信使用 78 SM，
DeepGEMM 使用 118 SM，每源 rank 容量固定为 272。各场景累计 128 次调用，交替顺序测量
7 轮。引用结果未观察到热降频，不同算法只在相同 Graph 长度内比较。

“完整计算”为 W1＋SwiGLU/量化＋W2 的独立 Graph 耗时；“MoE 总时间”为无埋点流水线的
八卡最大平均耗时，均取轮次中位数。“计算占比”单独在同一埋点 Graph 内计算
`max_rank(完整计算) / max_rank(MoE 跨度)`，不能用下面两列独立计时值直接相除。
分母不含 attention、共享专家、router/top-k、ID 映射和 EPLB 统计/迁移，因此不是整模型占比。

#### 活跃专家数

各 rank 有效计算行数均为 128，平均远端目标数为 7，只改变每卡活跃专家数。耗时单位为 μs：

| 每卡活跃专家 | 每专家有效行 | 每卡对齐布局行 | 完整计算 | MoE 总时间 | 计算占比 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 128 | 52.71 | 104.17 | 49.46% |
| 2 | 64 | 256 | 49.41 | 102.17 | 48.18% |
| 4 | 32 | 512 | 69.46 | 123.13 | 54.91% |
| 8 | 16 | 1024 | 120.07 | 175.54 | 65.94% |
| 16 | 8 | 2048 | 212.12 | 273.24 | 75.96% |
| 32 | 4 | 4096 | 399.52 | 464.25 | 84.27% |

有效计算量相同，1→32 个活跃专家使完整计算增加约 7.58 倍。小 M decode 的权重复用、
最小 tile 和调度成本不能忽略；对齐布局行数也不能直接当作硬件实测执行 FLOPs。

#### 两个目标 rank 的计算分布

每个 token 仍选择八个专家，但只触达两个目标 rank，平均远端目标为 1.75。
`compact` 使用固定少量专家；`spread8` 轮换专家，使每卡均有八个活跃专家。耗时单位为 μs：

| 分布 | 每卡有效行（轻/重） | 每卡活跃专家（轻/重） | 完整计算 | MoE 总时间 | 计算占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| compact 4+4 | 128 / 128 | 4 / 4 | 69.85 | 110.84 | 59.54% |
| compact 6+2 | 64 / 192 | 2 / 6 | 106.10 | 147.71 | 67.43% |
| compact 7+1 | 32 / 224 | 1 / 7 | 115.53 | 158.38 | 69.09% |
| spread8 4+4 | 128 / 128 | 8 / 8 | 120.06 | 163.48 | 69.87% |
| spread8 6+2 | 64 / 192 | 8 / 8 | 120.29 | 163.41 | 69.83% |
| spread8 7+1 | 32 / 224 | 8 / 8 | 120.04 | 163.42 | 69.85% |

通信目标相同，`compact` 的重卡激活更多专家，形成计算尾部；`spread8` 的 tile 和权重集合
相近，224:32 的有效行数偏斜也几乎不改变耗时。因此不能只按 expert-token 总行数估计负载。

保持八活跃专家的计算分布、仅将平均远端目标从 7 降到 0.875 的控制组，MoE 从
175.54 μs 降至 163.05 μs，约减少 7.1%。通信目标的降幅不能直接换算为吞吐收益。

#### 冗余数量与历史路由对照

共现流量来自跨 rank 的 16-expert 群，是合成分布，不是实际模型 router trace。
本组历史布局使用仅按主专家评分的 planner，表中的“旧 default_targets”使用规范主槽默认表，
不是当前同名的加权路由，也不代表当前逐轮扩展评分算法和哈希临时目标分发的性能。
Planner 使用 4096 个校准 token，正式评测使用独立种子的未见 token；共现组统一采用
16 步小图累计 128 次调用。旧共址与历史贪心路由使用同一布局和副本权重，ID 在计时前生成，
下表不含路由 kernel 自身耗时。R 为每卡冗余数，耗时单位为 μs：

| 场景 / 路由 | 每卡活跃专家范围 | 平均远端目标 | 完整计算 | MoE 总时间 | 计算占比 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 随机 R0 | 30–32 | 4.629 | 397.21 | 459.37 | 84.91% |
| 共现 R0 | 24–31 | 4.729 | 366.10 | 441.11 | 84.23% |
| 共现 R1 / 旧 default_targets | 24–32 | 4.510 | 375.51 | 453.09 | 84.48% |
| 共现 R2 / 旧 default_targets | 25–33 | 4.215 | 391.13 | 472.71 | 84.98% |
| 共现 R4 / 旧 default_targets | 27–35 | 4.070 | 407.23 | 495.02 | 85.62% |
| 共现 R4 / 历史贪心覆盖 | 27–35 | 3.871 | 408.57 | 495.48 | 85.71% |
| 共现 R16 / 旧 default_targets | 35–46 | 2.549 | 533.20 | 628.14 | 88.00% |
| 共现 R16 / 历史贪心覆盖 | 34–46 | 2.348 | 549.73 | 630.09 | 88.08% |
| 共现 R64 / 旧 default_targets | 50–79 | 1.160 | 858.05 | 992.49 | 91.24% |
| 共现 R64 / 历史贪心覆盖 | 43–80 | 0.961 | 900.28 | 1031.65 | 91.47% |

多样流量激活更多专家，计算占比约 85% 或更高；本组中冗余减少的通信不足以覆盖计算代价。
已移除的历史贪心覆盖在 R4/R16 下分别减少约 4.9%/7.9% 的远端目标，MoE 耗时没有明确改善；
R64 减少约 17.2% 的远端目标，MoE 反而增加 3.95%，最忙 rank 的平均活跃专家从
69.75 增至 73.5。优化全局通信总量不保证缩短慢 rank 的关键路径。

#### 同步等待与 tile 成本

下表每列独立取 rank 最大值后取中位数，单位为 μs；不能跨列相加还原总时间：

| 场景 | dispatch＋等待 | combine＋等待 | 埋点 MoE 跨度 |
| --- | ---: | ---: | ---: |
| compact 4+4 | 23.04 | 20.70 | 140.16 |
| compact 6+2 | 23.81 | 79.20 | 177.18 |
| compact 7+1 | 24.06 | 94.18 | 187.22 |

目标数和有效计算总量相同，combine 的表观时间仍显著增加，支持轻卡等待重卡计算完成的
解释；不能据此认定传输字节量增长。

上述 R 值的两次 GEMM 配置均为 `BLOCK_M=128, BLOCK_N=192, BLOCK_K=128`、
4 stages、384 threads、118 SM，未因新增副本切换 tile。psum 调度按每个专家的实际行数
产生 `ceil(m_expert / BLOCK_M)` 个 M tile，空专家不产生有效任务。本组每个 physical expert
最多接收 128 行，因此一个专家只要被激活就至少占一个 M tile，活跃专家数比总有效行数
更能解释小 M 下的计算开销。

### 15.5 启动显存

历史版本 `0e9ea6db9` 与 `origin/main@6c9e03b14` 的对照均未启用 EPLB：
DeepSeek-R1、8 张 H200、TP=8/DP=8、EP MoE，`max_total_token_num=160000`、
`graph_max_batch_size=16`、`batch_max_tokens=6000`。每 100 ms 采样设备显存，
先求各卡启动峰值再取平均；就绪值在健康检查通过 5 秒后读取，单位为 MiB/GPU。

| 模式 | 版本 | 启动峰值均值 | 就绪常驻均值 |
| --- | --- | ---: | ---: |
| 非 overlap | origin/main | 125724 | 121187 |
| 非 overlap | 0e9ea6db9 | 118955 | 113632 |
| overlap | origin/main | 127237 | 122718 |
| overlap | 0e9ea6db9 | 119454 | 114180 |

两种模式的启动峰值分别下降约 6.61/7.60 GiB/GPU。该结果只描述固定配置的启动与 Graph
捕获显存，不包含请求运行期间的 KV cache 和业务峰值，也不是增加冗余专家后的显存结果。

### 15.6 使用与复现

路由收益应同时比较去重通信 payload、最忙 rank 的活跃专家/tile 数、同步等待和路由自身成本。
默认加权路由更适合共现明显、不会制造计算热点的布局；可显式选择 `hash_targets`
对照，避免仅凭路由算子耗时判断整体收益。
上述数据尚未证明 RDMA 或 overlap 下的加权路由收益。
评测脚本中的历史标签 `current` 表示 `hash_targets` 对照，`weighted` 表示当前
`default_targets` 加权路由，不要将 `current` 标签误认为当前服务默认值。

相关评测脚本：

- `test/benchmark/service/benchmark_fixed_decode.py`：固定并发的端到端 decode；使用
  `--prompt-mode indexed-binary` 并隔离规划/评测请求。
- `test/benchmark/kernel/benchmark_eplb_decode_rank_weighted.py`：hash 与加权路由的算子时间及通信目标对照。
- `test/benchmark/kernel/benchmark_eplb_decode_compute.py`：八卡 DeepGEMM、通信及计算分布微基准；
  使用 `--steps 128 --rounds 7`，大 R 可采用 `--graph-steps 16`，避免混用不同图长度的结果。

### 15.7 加权 rank 优先级路由

实现放在 `decode_eplb_topk_ids_rank_weighted.py`，统一入口通过 `default_targets` 模式在 `T*W>=512` 时调用，
未设置环境变量时默认使用该算法。分块和 warp 数由内部根据本 rank 的 token 数 `T`、top-k `K`
和 rank 数 `W` 确定，不作为公开参数。小 batch 默认每 program 一行；补齐后的
`K × W <= 128` 使用 1 个 warp，否则使用 4 个。H200 多轮 Graph 评测支持以下
`K=8` 的大 batch 规则（其他 K/W 保留默认配置）：

| W | 使用 2-token 分块的条件 | 使用 1 个 warp 的条件 |
| --- | --- | --- |
| 8 | T ≥ 512 | 始终 |
| 16 | T ≥ 1536；T ≥ 8192 且无统计时改用 4-token 分块 | 始终 |
| 32 | T ≥ 2048 | T ≥ 1024，否则 4 个 |

规则只依赖输入形状和是否编译统计分支，Graph 捕获后固定，不在推理过程中计时或调参；同时检查了原始覆盖数、
倒数加权及开启共现统计的情况。不保证其他 GPU 上最优，也不等同于端到端收益。
同分副本的选择可能随执行布局变化，但本卡/节点优先级及最高分约束不变。
W=16 专项复测覆盖 uniform/共现流量、不同种子、1/2/4/8 warps，每次 9 轮、每轮 Graph
内 128 次调用。默认倒数加权、无统计时，T=1536 从约 2.55 降到 2.37 μs；T=8192 从
原 2-token 分块的约 4.61 降到 4-token 分块的 4.00 μs。含统计时，T=8192 的 2-token/
4-token 分块约为 6.37/6.96 μs，因此保留 2-token。这里“无统计”指
`update_route_counter=False`，不是运行中设备统计开关暂时关闭；后者不改变捕获布局。
以下描述 `eplb_rank_weighted_topk_ids` 本身；统一入口在 `T*W < 512` 时改用最小映射，不执行该评分过程。
加权路由只使用 `logical_to_rank_physical_map[E,W]`：本卡有副本的专家直接本地执行，其余专家形成
`available[T,K,W]`；沿 K 求和得到每个 token、每个 rank 的远端覆盖数。所有专家共享评分：

```text
score(token, rank) = 10000000 * same_node + coverage_score
coverage_score = sum(该 rank 可执行的远端专家权重)
expert_weight = 1024 // global_replica_count  # 默认 normalize_replica_coverage=True
expert_weight = 1                            # normalize_replica_coverage=False
```

每个远端专家在有副本的 rank 中选最高分，不再构造临时目标集合，也不迭代删除专家。
评分不包含哈希或随机扰动，也不依赖 token 行号。通过 `argmax + gather` 选择单个副本，
使用 `tie_break_left=False`，同分允许选择任一最高分可用 rank，不会将多个同分槽号相加。
同分选择依赖设备归约布局，不保证较小或较大 rank，也不代表随机选择或负载均衡。
本地专家不贡献覆盖分，不提供可调权重系数。默认归一化模式按全局副本 rank 数计算整数权重，
例如三副本贡献 `341`、单副本贡献 `1024`；不使用浮点数，不再次缩小累计分数。
副本数超过 1024 时权重为 0。节点优先要求原模式 `K < 10000000`，归一化模式
`K * 1024 < 10000000`，同时保证总分不溢出 int32。归一化开关在 Graph 捕获时固定，
布局更新后的副本数在 kernel 中重新读取和计算，不需要另建权重表。
不提供跨 token 的负载打散，重复热专家仍可能集中到同一 rank。
有本节点副本时一定选本节点，否则回退全局；在这个范围内选择覆盖分最高的可用 rank。
这与 `hash_targets`“本节点优先初选、随后共址可能跨节点”的规则不同。
覆盖数不是接收端负载惩罚，也不保证最小通信目标数。倒数加权偏向副本较少的专家，
不再等价于原始覆盖数最多，也不能预先认定通信或端到端性能更好。
CPU 布局指标在同分时取最小 rank，仅是一种合法选择，不代表 GPU 实际负载；
路由耗时和八卡 MoE 评测使用真实 kernel，计算侧探针也读取真实 kernel 的物理路由结果。

移除哈希后、尚使用可调权重的历史双项评分与哈希加权版本同轮对照：单卡 H200、`E=256,K=8,R=4`，
两种输入种子，每种 31 轮、每轮 Graph 内 512 次调用，`W=8/16/32,T=16/128`。
新算子耗时增加约 0.2%–1.9%，未观察到路由算子加速；`W=8`、种子 2027 下，
`T=16` 为 1.40 → 1.43 μs，`T=128` 为 1.52 → 1.53 μs。
CPU 路由模拟使用全局 128 token（每源 rank 16）、两种评测种子各 4096 token；
共现流量、单节点 `W=8,R=16,coverage_weight=0.5` 下，平均远端目标数约 2.319 → 2.318，
但最忙 rank/平均负载的逐 batch 中位数再取两种子均值约 1.27 → 1.65。
因此去掉哈希主要简化评分语义，不保证整体性能收益；以上不是实际通信耗时或端到端测量。

本节历史对照的“现有路由”使用包含 logical ID 的独立哈希初选；可选的 `hash_targets`
现使用 token 共享 Philox 初选与 rank 随机优先级，默认则为加权路由，因此历史耗时
和收益不能直接视为当前默认路由的测量结果。
下面的历史加权路由数据也使用带哈希扰动及完整哈希同分决胜的旧版本，不代表当前固定节点优先评分的测量结果。

原型仅保留全 rank 矩阵归约，试验阶段与逐 rank 扫描的数值结果一致。H200 上以 `T=16,K=8,W=8`
进行两次独立测试，每次 9 轮、每轮 Graph 内 128 次调用：现有路由约 2.81 μs，
矩阵版本约 1.45 μs，扫描版本约 1.49 μs。选择矩阵版本、每 token 一个 program；
`next_power_of_2(K) * next_power_of_2(W) <= 128` 时默认 1 warp，否则 4 warps。
这些是单卡路由 kernel 时间，不是八卡 MoE 或完整模型加速比。

系数初筛使用种子 2026，验证使用独立种子 2027–2030；每个种子 4096 token，
拆成全局 128-token、每源 rank 16-token 的 batch，每批重置 token 行号，布局校准种子为 2025。
单节点下 `coverage_weight=0.5`、`node_weight=0` 的四种子均值如下。
“远端目标”指每 token 去重后的远端 rank 数，不包括本卡，也不代表通信协议的实际字节数。
负载比为每批最忙 rank 的有效专家行数除以八卡平均，再取各批中位数。

历史实验包含现已移除的 greedy 对照；下列 greedy 数据只保留用于解释负载差异，
不代表当前可用的配置，也不能当作 rank_weighted 的结果。

| 流量 / 每卡冗余数 | 远端目标：现有 → 加权 | 加权负载比 | 历史 greedy 负载比 |
| --- | --- | --- | --- |
| 均匀 / R4 | 4.491 → 4.421 | 1.132 | 1.137 |
| 均匀 / R16 | 4.069 → 3.870 | 1.148 | 1.183 |
| 均匀 / R64 | 2.778 → 2.396 | 1.237 | 1.435 |
| 共现 / R4 | 4.056 → 3.964 | 1.329 | 1.336 |
| 共现 / R16 | 2.450 → 2.305 | 1.270 | 1.619 |
| 共现 / R64 | 0.980 → 0.837 | 1.243 | 1.420 |

历史覆盖权重 0.5 不是通用最优值，当前已取消可调权重。均匀 R64 的最忙 rank 对齐行数仍会增加，
说明通信目标减少不能替代 DeepGEMM 计算成本评估。模拟 `node_world_size=4` 时，
均匀 R64 下节点权重 1 使跨节点目标增加约 9–10%，节点权重 2 才减少约 2%；
共现 R64 下节点权重 1 已减少约 8%。这些仅是布局指标，未实测双节点 RDMA。

计算侧进一步使用同一张空闲 H200 串行重放八个虚拟 rank 的真实路由计数，固定 118 个
计算 SM、四套真实尺寸权重、128 步、7 轮、每次 Graph 16 步；评测种子为 2031–2034。
每组算法采用相同接收容量，按有效结束位置构造 128 对齐 psum，并验证所有有效输出。
时间只含 W1、SwiGLU/quant、W2；取各虚拟 rank 最大值作为计算瓶颈估计，**不含通信或
跨卡同步**，接收容量也不是 DeepEP 的最坏情况容量，不能替代八卡 MoE 时间。

| 流量 / 每卡冗余数 | 现有计算时间 / μs | 加权 0.5 / μs | 加权变化 | 历史 greedy 变化 |
| --- | --- | --- | --- | --- |
| 均匀 / R4 | 443.20 | 441.15 | -0.46% | -0.07% |
| 均匀 / R16 | 541.13 | 550.19 | +1.67% | +2.86% |
| 均匀 / R64 | 853.49 | 888.77 | +4.13% | +12.16% |
| 共现 / R4 | 412.95 | 410.76 | -0.53% | -0.54% |
| 共现 / R16 | 551.04 | 541.89 | -1.66% | -1.58% |
| 共现 / R64 | 954.30 | 926.02 | -2.96% | +0.24% |

因此加权共址更适合作为共现布局的候选；均匀流量、大冗余下不能只看通信目标减少。
覆盖权重 0.25/1 也不能普遍消除计算增长，权重选择仍需看完整流水线的关键路径。

八卡 DeepEP + DeepGEMM 脚本将路由放进 Graph，使用四套相同 logical 专家内容的合成 FP8
权重轮转，各算法交替计时，并验证路由与 MoE 输出。分段占比使用同一次执行中最慢 rank
的完整阶段向量，不能将各阶段独立最大值相加。阶段占比来自插入 timing event 的图，
脚本同时报告插桩耗时与开销；不能将这些占比直接套到无插桩延迟上。无插桩结果才用于
判断加速。单节点 NVLink、全局 128 token、种子 2031–2034、128 步、7 轮的结果如下；
范围是包含路由的 MoE 层，不是完整模型或服务吞吐。

| 流量 / 每卡冗余数 | 现有 MoE / μs | 加权 0.5 / μs | 加权变化 | 历史 greedy 变化 |
| --- | --- | --- | --- | --- |
| 均匀 / R4 | 513.37 | 507.88 | -1.07% | -0.72% |
| 均匀 / R16 | 631.46 | 635.13 | +0.58% | +0.71% |
| 均匀 / R64 | 971.62 | 1011.76 | +4.13% | +6.64% |
| 共现 / R4 | 500.84 | 496.87 | -0.79% | -0.56% |
| 共现 / R16 | 635.26 | 628.23 | -1.11% | -2.47% |
| 共现 / R64 | 1068.17 | 1041.95 | -2.45% | -2.35% |

换用种子 2035–2038 独立复测，每组仍为 7 轮、128 步，加权 0.5 相对现有路由的变化为：
均匀 R4/R16/R64 分别 -0.41%/+1.08%/+0.98%，共现 R4/R16/R64 分别
-1.93%/-2.37%/-1.76%。历史八卡两组共 48 个对照案例（含已移除的 greedy）均通过输出与接收计数校验，
计时没有记录到热降频。收益方向一致，但幅度受请求分布影响；小于 1% 的改善应谨慎解释。
R64 共现另做完整插桩图复核，timing event 额外成本约 3–4%，进一步说明应以无插桩
结果判断性能，而不是简单相加各阶段最大时间。

上述历史评测并未证明加权覆盖在所有布局上更快；当前默认实现和配置见第 5.3 节。
不宜仅因路由 kernel 更快或远端目标更少而推断端到端收益。

当前实现评测入口（不能直接复现上面的旧权重版本表格）：

```bash
# 独立种子验证（布局指标，不测通信时间）
PYTHONPATH=. python test/benchmark/kernel/benchmark_eplb_decode_rank_weighted.py \
  --metrics-only --seeds 2027 2028 2029 2030 \
  --output /tmp/weighted_metrics.json

# 当前生产接口的路由耗时对照
PYTHONPATH=. python test/benchmark/kernel/benchmark_eplb_decode_rank_weighted.py \
  --launch-only --output /tmp/weighted_launch.json

# 需要八卡空闲；包含路由的 MoE 对照，不含 attention 或 shared expert
PYTHONPATH=. EP_DISABLE_GIN=1 python -m torch.distributed.run --standalone --nproc-per-node=8 \
  test/benchmark/kernel/benchmark_eplb_decode_rank_weighted_moe.py \
  --variants current weighted --output /tmp/weighted_moe.json
```

八卡独立输入复测使用 `--evaluation-seed 2035`；`--cases uniform_R64 cooccurrence_R16`
可只运行指定组合。脚本启动和每轮计时前检查其他进程 / 容器占用，不强制终止外部任务。

### 15.8 逐轮覆盖方案评估结论

逐轮选择最大覆盖 rank、移除已分配专家的方案未保留。单卡 H200、W8 布局元数据、K8，
均匀/共现流量、R0/R4/R16/R64、T16/32/64/128 的 64 个同轮案例中，路由算子慢约 33%–96%。
部分大冗余布局减少了远端目标数，多数共现场景没有改善；未验证实际通信或端到端收益，
因此保留现有一次性矩阵评分，删除实验实现及专用测试、评测脚本。

### 15.9 单 rank DeepGEMM 的 M64/M128 分块与有效行数对照

以下为 2026-10-09 在单张 H200、DeepGEMM 2.5.0 上进行的隔离 GEMM 微基准。
它补充第 15.4 节的八卡 MoE 测试，用于观察专家分组和 M 分块成本，不是 TP/EP=8
的并行执行，也不是服务吞吐或端到端 decode 延迟。

#### 测试配置与计时口径

- 固定声明 64 个物理专家槽位，只改变活跃专家数；未激活槽位的有效行数为 0。
  表中覆盖 15 个活跃专家数取值，并非对 0–64 逐点扫描。
- 使用真实尺寸的合成 FP8 E4M3 权重，W1 为 `[64, 4096, 7168]`，
  W2 为 `[64, 7168, 2048]`，输出为 BF16；不是原模型权重。
- 四套独立权重按调用轮转，避免反复访问同一套权重导致缓存偏差。W1、W2 使用独立的
  合成量化输入、相同的专家行数分布；不包含中间 SwiGLU，因此不是完整 FFN 输出测试。
- 所有场景固定 118 个计算 SM、`expected_m_for_psum_layout=4`。
  `expected_m` 只是配置选择的启发式输入，不是每个专家的实际有效行数；
  未针对 128/256 行场景另行调参。两种对齐的接收容量均固定为 24576 行，
  避免把缓冲区容量变化混入比较。
- 每个 CUDA Graph 包含 4 次 W1＋W2 调用，每轮重放 32 次，累计 128 次调用。
  每种配置测量 5 轮，轮内随机排列场景顺序；先求每轮平均单次耗时，再取五轮中位数。
  每次计时前等待 20 ms，被动冷却不计入 GPU Event 计时，不修改 GPU 时钟或功耗限制。
- 单位为 μs，计时只包含 W1＋W2，不含输入量化、SwiGLU/量化、路由、通信、
  attention、共享专家及 EPLB 统计/迁移，也不包含 profiler。
- 每次测试启动时验证稀疏 psum 的有效区间，与反量化 FP8 输入/权重的 FP32 matmul
  对照，相对 L2 误差约 0.17%；同时验证全空专家不写输出。这不是完整模型生成精度测试。
- 四组测试的 NVML 边界采样均为 1980 MHz，最高温度 41°C，未观察到热降频或功耗限频。
  M64、M128 分开运行；8/64 行两列在补测时均重新测量，128/256 行四列来自此前各自的
  完整测试。合并表是同口径的跨轮对照，不是同一轮内交替 M64/M128 的配对实验。

M64/M128 同时改变专家数据起始位置对齐和 GEMM 的 `BLOCK_M`。实际配置如下：

| 配置 | 专家起点对齐 / BLOCK_M | BLOCK_N（W1/W2） | BLOCK_K | stages（W1/W2） | threads | cluster_m / cluster_n |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| M128 | 128 / 128 | 192 / 192 | 128 | 4 / 4 | 384 | 1 / 2 |
| M64 | 64 / 64 | 192 / 192 | 128 | 6 / 6 | 256 | 2 / 1 |

因此，这是同一安装版本为两种 M 对齐实际选出的 kernel 配置的比较，并非保持其余调度参数不变、只将
M 块减半的实验。M64 也不会把 FP8 的 128 通道量化分组或 `BLOCK_K=128` 改成 64。

#### 完整对照表

设单 rank 的活跃专家数为 G，`m_e` 为专家 e 的有效 token-expert 行数。
“每专家 8/64 行”“全部 128/256 行”均指每个活跃专家的行数，不是 rank 的总行数。
均匀场景的总有效行数为 `G × m_e`；“一个 129/257 行”只让槽位 0 的专家多一行，
其余专家保持 128/256 行，因此总有效行数只增加 1。未激活专家保持为空。

每个 G 对应 M128、M64 和比值三行；比值按未舍入的耗时计算：

```text
比值 = M64 耗时 / M128 耗时 × 100%
```

比值小于 100% 表示 M64 耗时更低，大于 100% 表示更高。例如 95% 表示耗时减少 5%，
122.2% 表示耗时增加 22.2%；不是吞吐提升百分比。所有时间值保留一位小数，单位为 μs。

| 活跃专家数 | 配置 | 每专家8行 | 每专家64行 | 全部128行 | 一个129行，其余128行 | 全部256行 | 一个257行，其余256行 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | M128 | 15.3 | 15.2 | 15.4 | 15.6 | 15.5 | 15.6 |
|  | M64 | 13.8 | 13.8 | 13.7 | 13.8 | 13.6 | 13.8 |
|  | 比值 | 90.6% | 90.6% | 89.1% | 88.4% | 87.4% | 88.6% |
| 1 | M128 | 47.0 | 47.1 | 52.2 | 52.4 | 53.6 | 53.5 |
|  | M64 | 46.3 | 46.6 | 43.2 | 47.0 | 50.5 | 54.9 |
|  | 比值 | 98.5% | 98.9% | 82.7% | 89.6% | 94.1% | 102.6% |
| 2 | M128 | 51.0 | 50.9 | 55.6 | 57.0 | 66.3 | 66.5 |
|  | M64 | 49.4 | 48.9 | 52.1 | 55.7 | 79.5 | 80.1 |
|  | 比值 | 96.9% | 96.1% | 93.7% | 97.8% | 120.0% | 120.6% |
| 3 | M128 | 57.8 | 58.0 | 61.5 | 69.2 | 100.1 | 104.4 |
|  | M64 | 54.5 | 54.6 | 75.4 | 80.7 | 107.3 | 112.8 |
|  | 比值 | 94.2% | 94.1% | 122.5% | 116.6% | 107.2% | 108.0% |
| 4 | M128 | 71.4 | 71.5 | 74.9 | 75.4 | 109.4 | 110.0 |
|  | M64 | 69.4 | 69.5 | 83.6 | 84.3 | 121.9 | 141.8 |
|  | 比值 | 97.3% | 97.2% | 111.7% | 111.7% | 111.4% | 128.9% |
| 5 | M128 | 80.7 | 80.5 | 82.7 | 104.8 | 123.9 | 151.0 |
|  | M64 | 78.4 | 78.0 | 92.1 | 111.0 | 149.1 | 151.1 |
|  | 比值 | 97.1% | 96.9% | 111.3% | 105.9% | 120.4% | 100.1% |
| 8 | M128 | 121.0 | 121.3 | 126.8 | 126.9 | 180.7 | 206.5 |
|  | M64 | 116.4 | 116.3 | 131.2 | 149.0 | 217.4 | 238.3 |
|  | 比值 | 96.2% | 95.9% | 103.5% | 117.4% | 120.3% | 115.4% |
| 12 | M128 | 174.3 | 174.4 | 182.2 | 185.6 | 265.8 | 276.8 |
|  | M64 | 168.7 | 168.9 | 189.5 | 197.4 | 313.4 | 334.0 |
|  | 比值 | 96.8% | 96.9% | 104.0% | 106.3% | 117.9% | 120.7% |
| 16 | M128 | 212.8 | 212.9 | 218.7 | 236.3 | 336.1 | 363.4 |
|  | M64 | 204.4 | 204.0 | 235.3 | 252.6 | 409.1 | 430.8 |
|  | 比值 | 96.0% | 95.8% | 107.6% | 106.9% | 121.7% | 118.5% |
| 20 | M128 | 258.7 | 258.8 | 266.6 | 266.1 | 422.5 | 436.1 |
|  | M64 | 248.8 | 249.1 | 293.5 | 300.8 | 505.3 | 530.6 |
|  | 比值 | 96.2% | 96.3% | 110.1% | 113.0% | 119.6% | 121.6% |
| 24 | M128 | 309.3 | 309.2 | 318.7 | 321.5 | 493.3 | 521.0 |
|  | M64 | 299.0 | 299.1 | 338.9 | 356.1 | 602.4 | 627.6 |
|  | 比值 | 96.7% | 96.7% | 106.3% | 110.7% | 122.1% | 120.5% |
| 32 | M128 | 398.9 | 398.3 | 407.2 | 423.7 | 651.4 | 677.4 |
|  | M64 | 381.6 | 381.3 | 443.4 | 460.9 | 800.3 | 824.6 |
|  | 比值 | 95.7% | 95.7% | 108.9% | 108.8% | 122.9% | 121.7% |
| 36 | M128 | 445.9 | 445.7 | 455.8 | 453.7 | 747.1 | 751.4 |
|  | M64 | 428.5 | 428.4 | 509.0 | 510.6 | 899.4 | 921.3 |
|  | 比值 | 96.1% | 96.1% | 111.7% | 112.5% | 120.4% | 122.6% |
| 48 | M128 | 585.6 | 585.1 | 595.6 | 609.7 | 969.1 | 998.0 |
|  | M64 | 558.0 | 557.5 | 652.9 | 676.2 | 1191.6 | 1194.1 |
|  | 比值 | 95.3% | 95.3% | 109.6% | 110.9% | 123.0% | 119.7% |
| 64 | M128 | 775.0 | 774.5 | 784.0 | 797.2 | 1299.0 | 1315.1 |
|  | M64 | 735.4 | 735.6 | 870.5 | 887.5 | 1587.4 | 1587.9 |
|  | 比值 | 94.9% | 95.0% | 111.0% | 111.3% | 122.2% | 120.7% |

G=0 时没有活跃专家，也不存在“一个专家多一行”，六列只是空 GEMM kernel 的重复测量。
其开销约 14–16 μs，不能将空组比值理解为有意义的业务加速；接近 100% 的微小差异也不宜
直接认定收益或退化。

#### 分块粒度与单专家跨界的解释

当前 SM90 FP8 kernel 的硬件 WGMMA M 粒度为 64 行，不能与专家数据对齐或
GEMM 的 `BLOCK_M` 混为一谈。M128 使用两个 math warp-group 分别处理前、后 64 行，
各自判断是否包含有效行；完全无有效行的 warp-group 跳过矩阵计算，但仍参与同步。
因此，M128 的布局填充到 128 行，不等于始终执行 128 行完整有效计算。

8 行和 64 行在 M64 下都只有一个 M tile，在 M128 下都只激活前一个 64 行计算子块，
这与表中两列耗时近乎相同的结果一致。增大活跃专家数仍会增加专家权重访问和
二维 tile 任务；有效行少并不意味着专家组本身没有执行成本。本次没有测量 HBM
性能计数器，不能进一步把全部耗时归因为显存带宽。

psum 调度器按有效行产生 M tile，总数为：

```text
total_m_tiles = sum(ceil(m_e / BLOCK_M))
```

一个专家从 128 增到 129 行，只增加一个 M tile，而不是整个 grouped GEMM 工作量翻倍。
例如 G=64 时：

| 专家有效行分布 | M128 总 M tile | M64 总 M tile |
| --- | ---: | ---: |
| 全部128行 | 64 | 128 |
| 一个129行，其余128行 | 65 | 129 |
| 全部256行 | 128 | 256 |
| 一个257行，其余256行 | 129 | 257 |

GPU 的持久化 CTA 统一调度各专家的 `(M tile, N tile)` 任务，并不是逐专家串行执行，
也不会因一个专家多出一个块而额外启动一次 GEMM。尾块计算、加载和同步仍有成本，
但额外任务能否落在已有调度轮次中，会影响最终完成时间，所以增量不固定：

- G=5、M128、一个专家 128→129 行，W1 的任务数从 `5 × 22 = 110` 增到
  `6 × 22 = 132`，超过 118 个工作 CTA 的单轮容量；W1＋W2 从 82.7 增到 104.8 μs，
  增加约 26.7%。这是调度台阶影响的例子，不能据此认为每增加一个块都慢 26.7%。
- G=64、M128、同样只增加一个 129 行专家，耗时从 784.0 增到 797.2 μs，
  只增加约 1.7%；M64 从 870.5 增到 887.5 μs，增加约 2.0%。
- G=64、M64、一个专家 256→257 行，1587.4→1587.9 μs 的约 0.6 μs 增量接近
  本组测量波动，不应把它解释为额外 tile 完全免费。

上述“一个专家越界”不等于“所有专家都越界”：后者会让每个专家都增加 M tile。
也不能把分开测量的单专家耗时相加，来预测任意混合分布的 grouped GEMM 延迟。

#### 对 decode 路由与对齐选择的结论

- 每专家 8/64 行、G=8–64 时，M64 的 GEMM 耗时通常降低约 3%–5%；
  例如 G=64、8 行时为 775.0→735.4 μs，降低约 5.1%。这是隔离计算收益，
  不能直接换算为完整 MoE 或服务吞吐收益。
- 每专家 128/256 行时，多数测试点 M128 更快。G=64 时，全部128行采用 M64
  耗时增加约 11.0%，全部256行增加约 22.2%。这些结果仅适用于上述固定
  `expected_m=4` 和安装版本实际配置，不能推广到所有 DeepGEMM kernel。
- 当前继续统一使用 M128 更合理；小行数下的有限收益，尚不足以证明需要引入
  prefill/decode 两套对齐流程。M64 在当前版本的 psum 路径中可用，但切换时必须保证
  dispatch 布局、psum、SwiGLU 和 GEMM 解释一致；进程级设置与 CUDA Graph 捕获也要一致。
  本次评测没有修改服务的对齐策略。
- EPLB 路由的计算成本应同时观察活跃物理专家数、总 M tile 数和最慢 rank 的分布，
  不能只优化通信目标数或只平衡有效行总数。低行数专家的碎片化值得优先关注，
  但分组/tile 指标仍不能替代真实分布下的完整流水线测量。
