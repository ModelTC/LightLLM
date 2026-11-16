# Attention 执行设计

Attention 执行将持久存储与 provider 计算分开。模型 runner 负责层顺序、padding、集合通信和图 bucket；
`ModelAttention` 选择各阶段 backend，将 State 视图绑定到层，每次调用使用一个 `AttentionSession`。
请求和缓存管理方继续负责接受结果、checkpoint、传输及回收。

## 所有权

| 所有者 | 生命周期 | 职责 |
| --- | --- | --- |
| `AttentionStateManager` 和具体 State 类 | 模型组合 | 分配 KV、Conv、SSM 存储，提供执行视图及 checkpoint/传输操作 |
| `ReqManager` | main/draft 共享请求域 | 管理请求槽位、token 映射、槽位 generation 和 MTP 验证步骤 |
| `ModelAttention` | 模型实例 | 选择并分组阶段 backend，绑定层资源，预留 workspace，执行 recurrent epilogue |
| `ModelBinding` | 一次权重/存储绑定 | 保留权重和 State 视图，重新绑定时使旧图失效 |
| `AttentionBackend` | 一个静态 spec 和阶段 | 分配 context，准备 host 元数据，构造 device 元数据，执行 attention |
| `AttentionWorkspace` | 模型 lane 或 capture | 管理命名 scratch 和 provider 对象；capture 借用 scratch，但持有独立 provider 对象 |
| `ExecutionScope` | 模型实例 | 串行化 host 提交，为 device 复用建立依赖，保留输入直到完成 |
| `AttentionSession` | 一次调用的一个 lane | 将 batch 绑定到 context，验证层调用，跟踪 State 是否可能被修改 |
| `CapturedGraph` / `PrefillGraphExecution` | 一个图 bucket | 保留图的输入、输出、绑定和可执行状态 |

Backend 按完整 `LayerSpec` 分组，包括阶段、family、layout、dtype 和 features。相同 spec 允许复用
provider，不代表可以共享可变请求状态或并发使用同一 workspace。`LayerAttention` 绑定逻辑层、资源、
backend 和 family 输出约定。Provider 借用 tensor，不接收请求生命周期操作。

代码分布在 `lightllm/common/attention/` 的 provider 接口和 session、
`lightllm/common/basemodel/` 的模型编排和图，以及 `lightllm/common/attention_state/` 的存储和生命周期操作。

## 初始化与显存准入

服务、静态 benchmark 和缓存续跑工具都调用 `initialize_model_runtime(models)`，模型顺序为 main 后接
各 draft。初始化在发布缓存和 worker readiness 之前完成：

1. 建立共享请求域，注册全部模型的 State，选择 backend，预留命名 scratch，加载缓存校准参数。
2. 分配尚未发布的 bootstrap 存储，满足启动 capture 和 profiling 所需的容量。
3. 绑定层，执行 autotune，初始化 padding，捕获配置内的图 bucket。
4. 保留图内存池，在 eager 模式执行全部层。覆盖冷 prefill、长缓存 query、多请求缓存和最大物理 MTP
   decode 行数；配置的 overlap、保留输出和采样概率也参与测量。
5. 排空执行，关闭图，解除资源绑定，释放 bootstrap 存储。
6. 使用测得的运行时预留量、固定分配、传输预留，以及 provider 声明的容量相关开销，准入最终 KV 容量。
   各 rank 使用最小可用容量。
7. 在最终地址重新绑定和 capture，再次 profiling。自动容量可缩减并重试，最多进行四次准入；显式容量
   超出预算则报错。

Profile 将完整只读历史页映射到 HOLD 页。Query 会写入的所有页都使用独立存储，包括已缓存一部分的
首个写入页。这允许 bootstrap 测量长上下文形状，而不必为全部请求分配完整历史 KV。

Prefill scratch 预留考虑 token 预算、TP padding 和可能增加的 HOLD 请求。Decode 使用物理 MTP 行数
及 batch 对齐，capture 使用独立 bucket 范围。Sparse FP8 压缩和 GLM 行映射声明依赖 KV 容量的开销，
与命名 scratch 分开计算。`RuntimeMemory` 记录峰值、预留量、尝试过的容量和最终容量。

启动后封闭图准入。缺失、关闭或失效的 bucket 在 attention preparation 之前选择 eager 路径，服务期间
不按需捕获新 bucket。替换权重时持有模型提交域，排空执行，退役旧图并重新绑定资源。

这是基于测量的显存准入，并不证明 opaque provider 分配、allocator 碎片或所有数据相关 MoE 路由的
峰值上界。显存比例保留余量，新 provider 和 workload 范围仍需实测。Bootstrap profiling 和重新 capture
会增加启动时间。

## 模型调用

公开的 `forward`、`microbatch_overlap_prefill` 和 `microbatch_overlap_decode` 在输入传输、padding
和模型专用元数据准备之前获取 `ExecutionScope`。竞争的 host 提交在触碰输入前报错。Session、图操作和
两个 overlap lane 加入同一次提交。

Session 创建 batch context 并完成 host preparation。每次层回调进入 `ModelAttention.layer_pass`：
构造一次 device 元数据，执行模型层，并在层和 overlap hook 成功后执行 recurrent decode epilogue。
Capture warmup 可在同一 host scope 内多次执行这个 pass。Prefill 不运行 decode epilogue。
输出 head 在 attention pass 之后执行，仍然可能失败。

模型层保持相同调用形式：

```python
output = self.decode_attention.forward(state.attention, inputs, out=destination)
```

各 family adapter 提供逻辑输出形状。调用 backend 前，session 检查目标 tensor 的范围、dtype、device，
以及与输入和持久资源的重叠关系。调用后检查返回视图，并保留 recurrent gate 等辅助输出。
Provider 可以直接写入目标，也可以返回由 session 复制的 tensor。

Sparse indexer 使用有类型的元数据。DeepSeek3.2 通过
`LayerAttention.sparse_index_metadata(session)` 获取
`SparseIndexMetadata(ks, ke, lengths, ragged_mem_index)`，不再读取 provider 私有 context。
模型专用 adapter 留在对应模型目录。

提交退出时，scope 记录完成事件并保留借用输入。其他 stream 复用资源前等待该事件。
Provider 返回前必须将辅助 stream 工作汇合到调用 stream。默认最多保留两次未完成提交和两个外部 reader；
超过上限时等待最早完成事件。Host 返回仅表示工作已提交，不代表 device 已成功完成。

## 图捕获与回放

原生 decode 图保存 inference-state 绑定快照和独立的 captured-attention 记录。
复用原 session 不会替换图所持有的 batch 或 context。Capture workspace 保留独立的可变 provider 对象；
即使其他 bucket 扩大模型池，也继续持有原先借用的 scratch 地址。

`CaptureManifest` 为捕获值记录明确的更新策略：

| 策略 | 示例 | Replay 行为 |
| --- | --- | --- |
| `copy` | 请求长度、rotary 输入、draft hidden 输入 | 验证后复制到保留的目标地址 |
| `shared` | 请求映射、权重、State 视图、scratch | 保留所有者，直到图退役前维持地址稳定 |
| `generated` | Device 元数据、DeepSeek-V4 C4 索引、输出 | 由捕获的执行过程生成 |
| `constant` | 阶段、角色、容量、静态 spec | 拒绝变化 |
| `prepare` | Host planner 元数据和实际 KV 长度 | Replay 前更新内容，不替换捕获的 tensor |

模型通过 `graph_input_names`、`graph_generated_names` 和 `graph_shared_names` 声明额外 tensor。
原生 capture 拒绝未声明的 tensor 字段。绑定拥有的权重和 State 视图仅在绑定时枚举一次；replay 检查绑定
有效性，不重复扫描权重。Capture 本地地址和 layout 在 replay 前检查，必要时也在 host preparation 后检查。

`TensorCopyPlan` 在 capture 时编译目标之间的关系。Replay 在任何输入复制前检查所有 lane 的数量、存在性、
形状、dtype、device、目标身份和 alias。相同复制以及映射一致的连续前缀视图会合并；冲突的目标写入或
会破坏另一个源数据的复制在修改前被拒绝。同地址输入也参与 alias 验证。

Replay 随后加入模型提交，复制输入，准备 host 元数据并启动图。复制开始后的失败可能使图失效；
输入验证不提供对已部分提交 GPU 工作的回滚。

图输出是借用视图。保留 Python tensor 引用并不能阻止下次内存池复用覆盖其值。调用方必须在复用前消费
完输出、复制需要的值，或通过 `record_output_use(event)` 注册外部 stream 的完成事件。Consumer 也必须
等待生产该输出的 stream。Reader 注册会约束共享模型 scope 下所有 bucket 的后续复用。
Runner 之外共享同一图池的图同样必须共享 scope。

分段 prefill 用 `PrefillGraphExecution` 保留图段、输入、输出和 hidden collector。段间 attention 回调
使用当前 batch 的 session。Prefill 与 decode 共享复制计划验证，但保留独立执行顺序。
Prefill 图不支持 microbatch overlap，配置阶段即拒绝该组合。

Capture 对 bounded State 做 snapshot/restore，在第一次 live replay 前恢复，失败路径也在排空执行后
恢复。KV 写入使用预留 token 位置，这不构成完整 KV 回滚。图退役拒绝竞争提交，排空 device 工作，
销毁可执行对象并释放引用。重复关闭已关闭的图没有副作用。

## 接受结果与失败处理

Attention pass 完成、device 完成、投机验证接受和 checkpoint 发布是不同阶段。
ReplaySSM 在层 pass 后推进整个验证窗口；MTP 验证通过 `commit_bounded_att_state` 剪去被拒绝的尾部。

`RequestGenerationStamp` 将 target 输出绑定到角色、请求域、请求索引、MTP 索引、槽位 generation 和
验证步骤。验证在模型提交域内一次性消费该记录。一个 live 请求最多有一个尚未消费的 target 输出；消费前
再次执行该请求的 target decode 或 prefill 会报错。回收后复用的槽位、其他 manager、重复结果和过期步骤
不能提交当前 State。HOLD 行保持 padding 语义。

可能修改 State 的区间从进入模型层之前开始，因为层可能先写 KV 再调用 attention backend。
此后发生异常，包括输出 head 或完成事件失败，会附带 `ExecutionFailure` 并使共享的本地 State manager
失效。所有角色随后拒绝执行、commit、checkpoint 保存和本地导出，不提供重置失败标志的恢复操作。

修改前的错误可以修正后重试。输入复制或 provider preparation 后的失败会使相关图失效。
无法建立完成依赖时，scope 保留借用输入并进入隔离状态。未处理的推理线程异常触发 worker 的 fatal thread
hook，避免 RPC 进程在执行状态已不可用时继续存活。

恢复和分布式故障处理由 worker 与请求管理方负责。有效恢复点必须包含同一已接受前缀下相互兼容的 KV
及全部 bounded State。普通 scope 清理不会恢复这个前缀；该设计也不在每次 forward 前自动 snapshot，
或在 State 部分修改后透明重试。

## 扩展与验证

Provider 实现各阶段 capability 检查、资源需求、context 分配、host planning、device 元数据和计算。
依赖容量的分配及由 host 准备的 capture 字段必须明确声明。Family adapter 负责 tensor 解释和输出约定，
State 类负责存储表示、checkpoint 和传输。

回归覆盖位于 `unit_tests/common/attention/`、`unit_tests/common/basemodel/` 和
`unit_tests/common/att_resources/`。修改这些边界需要验证修改前拒绝、stream 复用、capture 生命周期、
State 中性和失败处理。Provider 修改还需要在支持的 device 上进行数值对拍。
`test/att_resources/validate_model_cache.py` 使用真实 checkpoint 验证恢复后的 main/draft 续跑。

接口或 kernel 测试通过，不等于 HTTP 服务、跨主机传输、分布式恢复或性能已验证。
这些路径需要使用实际模型、provider 和 workload 单独验证，并随结果保留源码版本和配置。
模型接入步骤见 [添加新模型](../models/add_new_model.md)。
