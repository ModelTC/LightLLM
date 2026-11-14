# Attention 执行设计

这次重构将 attention 的长期存储、静态计算配置和批次执行分开：State 管存储，Backend 实现单个阶段的计算，Session 统一保证执行约束。模型层只描述需求、提供输入并使用结果。

设计目标是让新增模型或后端时，修改集中在对应实现中；容量、资源复用、Graph 输入和输出回写等公共规则只实现一次。存储部分沿用 [Attention Resources 设计](cache_manager_design.md)。

## 1. 职责和数据归属

| 对象 | 持有的内容 | 职责 |
| --- | --- | --- |
| `State` / `AttentionStateManager` | KV、Conv、SSM 等长期存储及其布局 | 分配存储、提供层视图、checkpoint、恢复和传输；外部页索引由 allocator 和调用方管理 |
| `ModelAttention` | backend 分组、层路由、adapter、workspace 池和存储代次 | 初始化选型、资源预留、层绑定，将模型推理信息转换为执行输入 |
| `AttentionBackend` | 不可变 `LayerSpec` 和显式注入的运行依赖 | 实现一个 family、provider、phase 下的规划与计算 |
| `LayerAttention` | 层号、组号、backend 和资源视图 | 将层计算路由到 Session；借用 State 存储，不拥有其生命周期 |
| `AttentionSession` | 当前 `AttentionBatch` 和各组 `AttentionContext` | 校验 owner、phase、capacity，管理执行期间的 workspace lease 和输出回写 |
| `CapturedGraph` | decode 捕获的推理状态、输出、workspace 和 provider hooks | 保持捕获地址有效，协调重放输入、辅助任务及完成事件 |

三个数据类型各表达一种事实：

- `LayerSpec`：静态需求，包括算法、阶段、head 维度、特性、`CacheLayout` 和不可变的 family 配置。只由 backend 持有，层绑定不保存副本。
- `AttentionBatch`：本次请求索引、长度、位置等输入；Tensor 借用自 `InferStateInfo` / `ReqManager`。
- `AttentionContext`：backend 为本次执行或某个 Graph 捕获实例准备的 metadata、容量和 workspace 引用。同一 Session 内的 context 共享一份 batch。

`InferStateInfo` 单向持有 Session。Session 只接收 batch、分组、容量和 workspace 提供函数，不反向引用模型或 `InferStateInfo`。State 返回存储层定义的视图，固定字段绑定但允许修改 Tensor 内容，adapter 将其与权重组成计算资源。ReplaySSM 只暴露 Tensor 和配置，不把可执行生命周期操作的 State 对象交给 backend。Backend 可以通过资源视图更新 KV/Conv/SSM，但不接管存储分配、checkpoint 或请求回收。

## 2. 初始化：描述、选择、绑定

```text
模型层 describe_attention(phase) + State 布局
                    │
             family adapter
                    │
         (prefill_spec, decode_spec)
                    │
       ModelAttention 按完整 spec 对分组
                    │
      每组分别选择 prefill / decode backend
                    │
       预留 workspace → 分配 State 存储
                    │
       adapter 借用层资源 → LayerAttention
```

只有完整的 `(prefill_spec, decode_spec)` 相同，层才能共享 backend 和批次 context；权重、逻辑层号和存储视图仍留在各自的 `LayerAttention`。同一层的两个阶段必须使用相同的物理 `CacheLayout`，避免 prefill 写入的内容无法被 decode 读取。

Backend 构造统一为 `Backend(env, spec)`。`BackendEnv` 显式传入设备 runtime、allocator、TP 和 MTP 等运行信息，不传模型配置字典；量化组长等布局参数读取 `spec.cache`。GDN/KDA 的卷积和 gate 参数、MLA 的分解维度以及稀疏索引参数由 adapter 翻译成不可变配置，进入 `LayerSpec`，共同参与分组。能力声明放在 backend 类上，注册表只枚举实现和默认优先级。选择时检查环境、计算需求和服务约束，并执行对应探针；显式指定失败时报错，`auto` 才尝试其他候选。

每层选择一次 family adapter，描述和绑定复用它。Dense 与 MLA 是独立适配；GDN 与 KDA 只共享 Conv/SSM 布局描述。DeepSeekV4 等模型专有适配放在模型目录。

## 3. 执行：调用方统一保证约束

模型入口使用一个作用域管理执行准备与收尾：

```python
with model.attention.session(infer_state):
    output = model._token_forward(infer_state)
```

普通执行按以下顺序进行；decode 捕获时的 device metadata 和计算进入 Graph：

```text
ModelAttention.session
  → 检查存储代次，构造 batch 和 capacity
  → AttentionSession.prepare
      → 校验 batch，取得 workspace lease
      → backend.allocate_context
      → backend.plan_host                  # 提交 Graph 前的规划，可启动设备操作
  → 模型 forward
      → session.build_device_metadata      # 设备 metadata
      → 逐层 LayerAttention.forward
          → Session 检查 context 归属
          → backend.forward                # 阶段计算
          → Session 处理 out 回写
      → model.attention.finish_decode      # 所有层成功执行后统一推进 ReplaySSM 游标
  → 退出作用域，结束 lease
```

Backend 的执行接口只有 `allocate_context`、`plan_host`、`build_device_metadata` 和 `forward`；资源预算与捕获 hooks 按需提供。每个实例只实现 prefill 或 decode，阶段专属构造和 helper 放在对应类中，共享类只保留共同定义。

层执行及 metadata 构建必须处于活跃的 Session 作用域，退出后拒绝继续执行；捕获 context 可以保留供 Graph 重放使用。数值探针也使用同一作用域入口。Session 保证 context 属于当前 backend 和 phase，backend 必须保留请求的容量、workspace 和 capture 模式。`batch_size`、`token_num` 必须与容量契约一致，声明的序列最大长度不能超过上界；padding 和 Graph bucket 选择由模型执行路径负责。

提供 `out` 时，Session 统一校验 dtype、device 和结果范围：backend 可直接写入 `out`；共享存储的结果必须落在连续 `out` 内，或是 strided `out` 的同起点、同 stride 前缀切片，不能使用行间 padding；返回独立结果时由 Session 复制，并保留辅助输出。

Workspace 按 `(group_id, microbatch_index)` 隔离。同一槽位要求串行执行，lease 负责事件依赖，不是并发互斥锁。准备、规划或 forward 抛出异常时，`ExitStack` 仍结束已取得的 lease。结束 lease 会记录完成事件供后续复用等待，不释放池中的 Tensor，也不表示 GPU 已同步完成。

## 4. CUDA Graph：固定地址，先校验再写入

decode 捕获时，所有组的 context 使用同一份捕获 batch。长度容量显式采用 `graph_max_len_in_batch`，当前实际长度另行传给 host planner，避免用服务最大长度扩大捕获容量。捕获的 workspace 保留原 Tensor 引用并冻结其缓冲区绑定；provider 共享对象的内部地址稳定性仍由 provider 负责；池后续扩容不替换旧 Graph 使用的地址。

`CapturedGraph` 一次重放的顺序为：

1. 校验存储代次，以及所有 microbatch lane 的模型角色、backend 分组、容量和输入 Tensor 的存在性、形状、dtype、device，以及同地址视图的 stride；只生成复制计划。
2. 所有 lane 校验通过后，等待 provider 上次 metadata 使用结束，并取得 workspace lease。
3. 将新输入复制到捕获地址，调用各组 `plan_host` 更新主机规划，再提交 Graph 和 provider 辅助任务。
4. 将辅助任务完成事件纳入 lease 收尾，保证下一次复用的依赖完整。

因此，后一个 lane 的输入校验失败不会留下前一个 lane 已改写的输入。这个保证覆盖写入前的校验，不提供复制或 provider 执行失败后的事务回滚。

请求映射表是共享存储，重放时必须保持地址；State 存储代次改变后必须重新绑定层并重新捕获。Graph 销毁时等待相关设备工作，关闭 provider hooks，再释放捕获引用。

prefill 采用另一条路径：`PrefillGraphExecution` 持有分段 Graph 的输入、输出和 collector，attention 通过段间回调使用当前批次的 Session。`build_device_metadata` 在分段执行前运行，不属于上述 decode 捕获流程；多 lane 写入前校验的保证也仅针对 `CapturedGraph` 路径。

各模型的 InferState 类通过 `graph_input_names` 声明 `AttentionBatch` 之外的 RoPE、draft hidden 等重放输入。新增输入时必须扩展此清单；DeepSeek-V4 C4 索引等图内产生的中间结果不列入清单，由捕获的节点重新计算。

GLM 的请求展开信息由模型推理状态生成，以 `SparseQueryMetadata` 传入 indexer；它不再读取 provider context 的内部字段。decode 捕获输入复制包含这些 Tensor，prefill 回调使用当前批次的数据。

ReplaySSM 的 cursor 由模型在参与层全部成功返回后统一推进一次，普通和 overlap decode 使用同一操作，Graph 捕获包含该操作。层执行失败时不会执行这一步；此前已写入的状态不提供事务回滚。

DeepSeek3.2 仍从 sparse provider context 读取 ragged-index metadata；替换该 provider 时，需要先明确这部分家族数据契约。当前 provider 没有实现辅助 capture hooks，现有假 provider 测试验证调用顺序，不能证明真实 provider 提交辅助任务后的故障恢复。

## 5. 本次精简的取舍

| 原有复杂度 | 当前处理 |
| --- | --- |
| `AttentionBackend` / `BaseAttentionBackend` 两层基类及构造转发 | 合并为一个基类，删除 `create()`、`configure_spec()` |
| spec、causal、dynamic verify 等配置重复保存 | 从 backend 的 `spec` 读取，避免多个副本需要同步 |
| BatchBuilder 单方法类、`ContextSource` 间接接口 | 模型侧直接构造 batch，Session 接收明确的数据 |
| prepare / forward / finish 分散配对 | 一个 Session 作用域覆盖正常与异常收尾 |
| prefill / decode 的 helper 和依赖混在共享类 | 归入阶段实现；例如 GDN decode 构造不加载 prefill kernel |
| GDN 继承 Dense、KDA 继承 GDN | 以实际共享的 Conv/SSM 布局为基础组织 adapter |

保留 family adapter，是因为模型字段与 State 资源确实存在差异；保留 Session，是因为生命周期和校验需要被所有后端复用。State 已能提供布局和资源视图，不再增加一套平行的存储描述接口。

## 6. 修改入口与验证

新增模型时先实现层需求描述，优先复用 family adapter；只有资源绑定不同才增加模型专有 adapter。新增 provider 时实现对应阶段的 backend、声明能力并注册；新增存储编码时扩展 State 的布局和资源，交由 `CacheLayout` 做兼容性检查。具体接入见[添加新模型](../models/add_new_model.md)。

代码阅读入口：

| 内容 | 文件 |
| --- | --- |
| 分组、选型与模型适配 | `lightllm/common/basemodel/attention.py`、`lightllm/common/basemodel/attention_adapters.py` |
| 静态需求与后端接口 | `lightllm/common/attention/api/spec.py`、`lightllm/common/attention/api/backend.py` |
| 批次执行与 workspace | `lightllm/common/attention/session.py`、`lightllm/common/attention/api/workspace.py` |
| Graph 重放 | `lightllm/common/basemodel/graph.py`、`lightllm/common/basemodel/prefill_cuda_graph.py`、`lightllm/common/basemodel/infer_struct.py` |

回归重点是 context 归属和容量拒绝、异常 lease 收尾、输出回写、多 lane 写入前校验，以及捕获地址和存储代次。接口契约与 workspace 测试位于 `unit_tests/common/attention/`；模型绑定和后端数值测试位于 `unit_tests/common/basemodel/attention/`。模型验证还需覆盖 prefill/decode、Graph、MTP 和状态恢复；吞吐或跨机 PD 结论需要各自的运行验证。
