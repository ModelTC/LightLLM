# 七模块重构实施记录

基线：`9719c56c4ffbd9dfa64ab8610adc4979b14013a4`。

本文件记录实际实现与验收范围，架构图描述最终目标。完整七模块计划尚未完成。

数值验收仍未完成。Qwen3.5-MoE、Qwen3 RoPE 和 GPT-OSS 的既有数值修复保留；GDN verify 的中间舍入修复已在首轮扩大完整回归通过。原完整集合尚有七项 BF16 失败：五个 Loop 小模型、正式 Qwen3.5 与 SenseNova 各一项；本轮正式 DSpark 的 BF16 confidence 超限已在完整 2945 项复验中确认，合计八项失败。该候选另暴露的七项包装接口/参考代码失败已修复，155 项专项及最终完整复验均确认它们通过。没有放宽门槛。Qwen3.5 和 SenseNova 的独立对照显示 Transformers 自身 BF16 也超过原 FP32 参考的 1% 门槛，尚不能仅由超限判定重构回归。保留原输入和断言继续定位，详见 [H100 诊断记录](h100-validation.md)。

已接入 35 个 Adapter 实现类，主模型 27/29 个注册入口、草稿 28/30 个入口；这是装配声明覆盖，不是全模式验收比例。最新迁移为 GLM5 KDA/pooled verify 与 MTP 草稿；119 项 H100 扩大专项为 111 通过、8 项主模型 verify 数值失败，草稿与独立 kernel 检查通过，3029 项完整回归为 3012 通过、17 失败；原八项 BF16、新八项主模型 verify 和一项视觉草稿测试接口失败均未解决。此前 MLA verify、DeepSeek/GLM4 MTP 与 MoE 路由修复的 3000 项完整复验确认新增 44 项全部通过。此前 DFlash2 的 154 项专项及新增 11 项完整复验通过，正式 DSpark 的共享权重/normalized-head 修复在 FP16 三阶段通过，BF16 confidence 仍未通过。V4、GLM5 MTP、其余量化、完整任务/服务及性能仍未完成。

最近完整 H100 回归 `glm-mtp-phase-reference-full.xml` 为 **3012 passed、17 failed / 3029**，592.14 秒，无 error/skip/deselect；逐项 Counter 核对上一轮 3000 项全部保留，新增 29 项。原八项 BF16 失败仍在，新增八项 GLM5 主模型 verify 数值失败；另有一项原视觉草稿测试失败，其 norm 替身不接受新增的 alloc_func 参数，尚未修复复验。此前 MLA/MTP 完整结果为 2992 passed、8 failed / 3000。专项不重复累加，也不替代完整数值、模式、任务、生命周期和性能验收。

SenseNova Loop 正式 9B 权重：FP16 七阶段检查通过，包括 prefill/decode、decode Graph、prefill Graph 首次回放和 GPU Conv/SSM 页恢复。两条输入共 14 个 greedy token 与独立 FP32 参考相同，最大相对 RMS 0.30%。BF16 greedy 也相同，但最大相对 RMS 10.38%，超过 1% 门槛。追加诊断中，同状态 decode Graph/eager 逐值相同；Transformers BF16 自身最大 RMS 9.67%，原生与其最大差别 2.89%，仍需定位。该结果不代表服务、任务精度、多卡或性能验收，完整证据见 [H100 记录](h100-validation.md)。

本轮补齐 mHC `pre_scale` 和 `fixed_identity` 的声明、权重布局和执行语义：启动检查参数，恒等路由仅加载 pre/post 字段，缩放按模型的激活舍入顺序在 RMSNorm 前执行。新增 20 项配置、FP16/BF16 kernel/Graph 与完整 FP16 小模型检查均通过。此修复保持默认配置，不用于掩盖尚未解决的数值失败。

## 已确定的迁移边界

- 最终形成 Model、Planner、Runner、Compute、StateStore、Transfer、DeviceRuntime。
- ModelAdapter 与 AttentionBackend 独立扩展；按语义、格式、设备能力匹配。
- 允许修改内部与启动接口，保持服务协议；先完成正确性，再优化性能。
- 保留 Llama、Qwen2/3/3.5/Next、DeepSeek、GLM、GPT-OSS、SenseNova 及相应现有多模态/草稿变体。
- 保留 TP/DP/EP、PD、MTP、Graph、CPU cache 和现有量化能力。
- 清退模型注册不等于删除源目录：保留模型仍依赖部分旧模型中的公共实现。

2026-11-04 静态入口复核：当前 84 个注册项（含草稿模式）与 `refactor-inventory.json` 一致，59 项保留、25 项仍是清退候选。保留项中，主模型有 29 个注册项，其中 27 项显式声明 Adapter，对应 24 个不同模型类；草稿有 30 个注册项，其中 28 项显式声明 Adapter，对应 12 个模型类。未接入的主模型是 V4 普通及视觉入口；未接入的 2 个草稿入口为 V4 MTP/DSpark。GLM5 Next MTP 的四个注册别名现已声明 Adapter，但主模型 verify 数值验收仍未通过。DeepSeek3 / GLM4-MoE-Lite MTP 和 Qwen3 DFlash2 已声明独立 Adapter，专项范围见文末。注册数包含模式和 text 别名，不代表独立实现或验收数量。正式草稿 checkpoint 验证已包含 Qwen3.5 草稿及 raw Qwen3.8-27B + DSpark 的 FP16，未因此宣称其它草稿、Loop/GPT-OSS 的 MTP 或全部模式通过。

## 已实现的原生 Dense 链路

当前 Dense/混合入口是显式接入的 Llama、Qwen2、Qwen3、Qwen3Next、Qwen3.5、Qwen3-MoE、SenseNova Loop、Qwen2-VL、Qwen3-VL 和 Qwen3-VL-MoE；Dense/GDN/MLA 和 GLM5 的 MTP verify 迁移及未通过项见文末；Loop 的 BF16 独立数值验收尚未通过。普通 Dense KV（full/causal sliding）支持 Triton、FA3、FlashInfer 或 auto；分组 INT4/INT8 KV 支持 Triton 或 auto；静态 FP8 per-head/per-tensor 支持 Hopper 上的 FA3，per-tensor 也支持原生 FlashInfer。Qwen 混合模型的 Gated DeltaNet 链路见下。其它模型或未迁移组合保留原执行路径。

### 启动

1. `LlamaModelAdapter`、`Qwen2ModelAdapter`、`Qwen3ModelAdapter` 在权重和推理层创建前从配置声明逐层 `DenseAttentionSpec` 和 `StateSpec`，不依赖模型实例或已分配资源。它们选择权重映射类、创建模型计算 recipe、加载并校验权重；TP 切分和共享权重仍复用原权重实现。recipe 的实际 head 形状必须与声明一致。旧模型类上的工厂别名仅供未迁移子类使用。
2. `Planner.prepare()` 在 KV 分配前联合检查两个阶段的能力、格式、状态形状、消费者、静态参数和 workspace 预算。显式指定失败会报错。auto 按候选优先级匹配：普通精度先 FA3、Triton，再 FlashInfer；静态 FP8 per-tensor 可在 FA3 不兼容时选择 FlashInfer。两个阶段可以选择不同 backend，但必须支持同一状态格式。
3. `PreparedPlan.reserve()` 通过 `CudaRuntime` 分配 workspace。允许复用的顺序执行层共享 scratch；并发 microbatch 各有独立区间。Triton decode 的分块中间结果、FA3 的固定 stride 页表在这里预留。FlashInfer 额外按语义和格式隔离持久执行计划，不能只按 backend 名合并。
4. `DenseStateStore` 从声明创建 meta State，注册到现有共享 token 分配域。相同形状合并物理分配，只有明确共享的消费者才复用内容。异构形状有独立资源标识；普通同构模型保留原 CPU/PD 资源名称。
5. 原共享容量规划器读取剩余显存时，workspace 已实际占用，因此与 KV 使用同一个物理预算。随后分配 KV，Planner 绑定原生 K/V 视图、静态参数和 workspace。
6. 全部绑定成功后一次性安装到各层；活跃计划不能重新绑定。`DenseKVWriter` 直接写已绑定的 K/V 张量，存储组件不再负责选择这个计算实现。

这里的 workspace 数字只覆盖当前实现显式管理的固定 scratch/页表。FA3 库内部临时分配、模型 activation、Graph pool 等仍依赖原运行期预留与 warmup，不宣称完整峰值显存已由 Planner 精确计算。

### 推理

- `Runner` 从批次提取张量和长度，backend 只接收这些明确字段，构造独立 metadata；不接收整个模型、资源 manager 或旧 attention state。
- `BoundAttention.run()` 直接调用 Triton/FA3 kernel 或已完成 planning 的 FlashInfer wrapper；KV writer 直接消费原生视图。计算与写入不反查状态管理器。
- Decode Graph 通过 `DenseBatchMetadata.copy_for_graph()` 更新固定输入地址；页表保持固定行 stride。Prefill Graph 的 attention 回调使用当前批次 metadata，兼容前缀增长。
- 原生 FA3 显式声明传输前需要计算流同步，CPU cache 协调模块读取该能力；不依赖已经移除的旧 backend 实例。
- 已删除 `execution/bridge.py`。迁移路径不再创建旧 `BaseAttBackend`/`Base*AttState`。

## Linear Compute 与权重加载分离

- `LinearSpec` 声明本地矩阵输入/输出维度、输入 dtype 和可选批量矩阵数；`FormatSpec` 描述普通权重、FP8 per-tensor/per-channel/block128 的布局与 scale。权重仍由 ModelAdapter 选定的 loader 完成 checkpoint 映射、TP 切分和在线量化。
- 权重加载并验证后，`Planner.bind_linears()` 按格式和设备能力匹配 Compute，实现全部通过检查后才安装绑定。`LinearPlan` 记录原生绑定及尚未迁移的量化方法。已有 recipe 的 `mm`/`bmm` 入口直接指向 `BoundLinear.run`；执行对象仅持有原生张量、静态参数和分配函数，不持有权重 loader 或量化配置对象，也不调用 `QuantizationMethod.apply()`。
- 普通 Linear/BMM 直接使用 Torch 的设备矩阵算子，保留 bias、非连续矩阵视图、显式输出和 FP32 输出。FP8 per-tensor 使用 Triton 或 SGL，per-channel 权重配合 activation group64/128 使用 Triton，block128 使用 Triton。SGL 的设备、对齐和符号能力在启动检查，已绑定的 `run()` 不再选择 backend。
- FP8 的 activation 量化与 GEMM 消费明确绑定的 scale，独立于权重加载。SGL 自行分配结果时直接返回；只有调用者传入输出缓冲时才复制回该地址，保留 Graph 输入输出地址约定。
- 当前代码接入十六个主模型 ModelAdapter 实现类，另有 Qwen3.5 和 Qwen3.5-MoE 原生草稿适配器；具体模型和模式的验证边界见文末。AWQ/Marlin、vLLM、DeepGEMM 等未迁移的 Linear 组合继续使用原接口并记录在 `LinearPlan.legacy`；普通及 block128 FP8 MoE 的路由/专家计算，以及 Hopper FP8 EP 已接入原生绑定，其它格式/设备组合仍待完成，详见下节。FP8 activation 临时张量、库内部暂存和完整峰值预算仍未纳入启动精确预算，不能将“已绑定”理解为零临时分配或性能验收完成。

## MoE Compute 原生绑定

- ModelAdapter 通过 `moe_specs()` 声明实际 MoE 层及其路由、专家维度、共享专家、门控和激活截断语义。DeepSeek/GLM 与 Qwen3Next 分别解析自身配置；全 Dense 的模型即使带有专家配置字段也不产生 MoE 声明。
- `Planner.bind_moe()` 校验声明覆盖、recipe、已加载权重、TP 切片和格式，并将 `BoundMoe` 安装到 recipe。执行对象只保留原生 gate/up、down、scale、correction bias、per-expert scale；不持有 `FusedMoeWeight` 或量化 loader。原生 recipe 直接调用 `run(inputs, router_logits, shared_gate=..., capture=...)`，不调用旧权重执行入口。
- `triton.moe` 支持普通权重和 block128 FP8，分组 top-1/top-2 评分、softmax/sigmoid、renormalize、routed scale、共享专家以及动态 sigmoid 共享门控。GLM 的 clamp/alpha/add-one 语义在绑定时固定，kernel 显式接收 activation 选择，原生路径不再查询模型类型决定激活。
- 原生 FP8 activation 量化明确使用 LightLLM Triton 实现；旧路径默认行为保留。H100 诊断发现已安装 SGL 的量化在边界输入上与独立 Torch 参考产生不同 E4M3 字节，不能把依赖是否存在当作隐式的执行选择。该选择可能改变相对旧 SGL 路径的逐位输出，独立数值检查与模型任务精度需要分别评估。
- 路由记录回调由 recipe 在调用前准备，Compute 在追加共享专家前提交原始 routed ids，保持既有 routed-experts 布局。回调管理和批次元数据进一步移入 Runner 仍待完成。
- Hopper BF16/block128 FP8 的 EP dispatch/combine 已迁入 `ExpertTransfer`，专家计算和调度分别由 `BoundExpertCompute`、`ExpertRunner` 负责，验证范围见文末。EP 的 FP4、其它硬件/格式、Marlin 和 packed UE8M0 继续保留旧路径并记录在 `MoePlan.legacy`。专家对齐、activation、中间矩阵和 Graph scratch 仍有动态分配，未完成全峰值预算或性能验收。

## 已接入的 Transfer 与延迟回收

- `CopyPlan` 显式保存方向、资源布局和已绑定搬运操作。CPU cache 与 PD 的布局选择在提交前完成；PD 先构造本地 TP 整组计划，再提交任何字段复制。
- 计划构建阶段拒绝错误设备：PD 页与 token 索引必须在 CUDA，CPU cache 页必须在 CPU、token 索引必须在 CUDA。这类输入错误不进入异步提交，也不包装成需要等待完成信号的部分传输失败。
- `Transfer.submit()` 通过 `CudaRuntime` 提交、记录完成事件，分别返回 `source_reusable` 与 `destination_ready`。CPU offload 的目标完成点包含原有 TP 同步；本地复制的两个完成点可以相同。计划保留原生视图和索引的引用。
- 部分提交失败返回 `TransferFailure.quiesced`，不返回可使用的目标。`AttentionStateManager.retire()` 在完成信号后归还 KV/bounded 页，并复制可能被 allocator 重用的索引缓冲；容量查询收集已完成的延迟释放，不阻塞其它独立任务。
- CPU load 等完整 KV 与 bounded runtime 恢复完成后，才发布 `cur_kv_len` 和快照页映射。绑定/提交异常时撤销新增 hold 长度，把新增资源交给延迟回收；设备执行错误仍向上传播，不擅自复用资源。单 TP offload 失败同样等待后再解除 CPU 页引用；多 TP 的本地失败不能证明其它 rank 已停止写入，因此保留引用。
- PD 本地复制失败先等待搬运流停止访问，再上报失败。发送通知剥离本地完成对象。NCCL/NIXL 在提交可能失败前先记录句柄，失败路径仍能保留和跟踪在途资源。
- `PDPageReclaimer` 在本地复制完成、网络报告 DONE 且句柄释放成功后归还 staging 页。接收端 NIXL 无本地句柄时，等待对端完成通知；超时后迟到的完成通知仍可解除隔离。错误和超时本身不构成资源可复用的证据。

这仍是阶段 3 的部分迁移：各 State 内的具体 pack/unpack 实现尚未全部移出；普通请求、radix 和 DP 的全部 allocator 生命周期尚未统一。长期 ERR/失联页面保留至能够确认安全，主动取消、peer 重建与 TP 全组失败收敛尚待实现；可能占用 staging/CPU 页，不能宣称故障后的自动恢复已完成。

## Runner 的批次协调、Graph 执行与设备机制

- `BatchRunner` 接收固定的 `BatchPolicy`、`BatchProgram` 回调、请求资源和 `GraphRunner`，统一处理 token 收集、空 DP 批次、MTP/TP 对齐、padding 和输出裁剪；组件本身不持有模型或读取 model.config。普通和双 microbatch 的 prefill/decode 均已接入。输出按请求行裁剪 logits/token ids，按 token 行裁剪 prefill hidden/prompt logits，保持原 Graph 输出对象与输入不被裁剪修改。
- `GraphRunner` 持有 decode/prefill Graph 集合，负责 eager、capture、replay 的选择。普通 decode 与双 microbatch decode 使用相同的 `DecodeStep`，先选容量，再 padding，并在 attention metadata 初始化前设置执行阶段。模型保留计算 recipe；`model.graph`、`model.prefill_graph` 是到协调组件的访问入口。
- `BatchProgram` 显式保留模型特有的状态构造与初始化回调，EAGLE hidden 投影仍发生在状态创建前。DSpark 的特殊 padding 通过绑定回调保留，Runner 检查最终容量与选定计划一致；默认 padding 直接由 Runner 实现。旧模型上的 padding/输出辅助方法仅转到相同实现，供现有派生类与 warmup 使用。
- `BatchStateFactory` 绑定请求表、State 资源、角色、hidden collector 原型、lane 通信组和执行计划，构造每批次的 InferState。默认批次入口直接调用 factory；EAGLE 等模型特有的输入变换仍可先执行，再调用相同 factory。尚未迁移的 backend 只通过启动绑定的状态构造函数接入，不在每批次重新选择实现。
- `WarmupRunner` 负责 autotune 长度序列、HOLD 初始化、decode/prefill Graph 捕获输入和最大长度检查。Graph 对象不再接收整个模型来组织 warmup，也不再生成 ModelInput。模型提供专有输入和计算回调，并通过可恢复的 context 保留 autotune 的层选择；层数不会超过实际 recipe，随机 token 受实际词表约束。
- Qwen3Next/3.5 的 autotune 按实际 recipe 的 attention/FFN 类型选择覆盖前缀，不再依赖周期字段。显式 `layer_types` 的资源布局也允许省略 `full_attention_interval`，非周期模型继续以声明的层顺序为准。
- Graph warmup 仅使用 HOLD 行，不再重置共享分配器。需要临时 KV 的 autotune/最大长度检查，只归还本次申请的请求槽与整页 KV；先等待设备操作完成，再释放。普通 forward 异常也经过同一清理路径；同步失败则保留分配、停止启动，不继续下一轮 autotune。该机制不代表跨 TP rank 的失败收敛已经完成。
- prefill 的 KV 索引就绪事件保持原发布顺序：单批次在索引构造后、额外 metadata 初始化前记录；双 microbatch 在两份 metadata 初始化后记录。事件不表示模型计算完成。overlap 成功才清理 DeepEP 暂存；异常时不把可能在途的资源视为可回收。
- 原 `basemodel/cuda_graph.py`、`prefill_cuda_graph.py` 移到 `execution/decode_graph.py`、`prefill_graph.py`，内部导入同步迁移。Graph pool、创建、capture context、同步和计时事件由 `CudaRuntime` 提供；调优、执行顺序和批次策略留在 Runner 组件。
- `GraphProgram` 保存按顺序执行的 CUDA 段和动态函数，取代 `InferStateInfo` 的 Graph 列表与执行方法。`call(function, inputs, batch, output_specs)` 为动态输出保留固定地址，`defer(function, inputs)` 登记原地操作；回放总是传入当前请求。模型层不再手动退出/进入 CUDA capture context。
- 模板 attention、Qwen GDN、GLM sparse/KDA、DeepSeek V4、Qwen3-VL/VL-MoE deepstack 使用上述入口。GDN/KDA 声明输出形状，V4 声明原地输出，避免为探测形状而执行有状态或会同步的 kernel；普通 attention 保留隔离的形状探测。Qwen 图像特征在回放时读取当前批次的页与 token 映射。
- capture 异常会退出当前捕获，未完成程序拒绝回放；动态输出的数量、形状、dtype、device 改变时明确拒绝。eager prefill 用 `finally` 清理临时张量缓存，失败后可执行下一批。
- 模型位置/rotary 初始化已改为启动绑定的批次 metadata 函数，详见后文；启动流程装配、缓存策略和路由记录仍与旧模型/服务体系连接，这里尚未完成全部 Runner 迁移。prefill Graph 与 microbatch overlap 的原未实现组合也未新增支持。V4 FlashMLA 已有真实 kernel/Graph 验证，完整 Planner 迁移仍待完成；GLM5 Next 的无 MTP 原生路径见下。SGL 的 `only_qv` 和 dtype 支持由 GLM 原生 backend 在启动时检查，auto 可选兼容实现。

## 分组量化 Dense 扩展

- `FormatSpec` 明确 payload dtype、scale dtype 和分组大小；`DenseAttentionSpec` 独立声明 query dtype。Planner 按 backend 实现族筛选，再匹配格式，模型无需增加 INT4/INT8 专用分支。
- `GroupQuantizedAttentionBackend` 直接绑定 K/V payload 和 scale 四个原生字段。prefill 在 attention tile 中解码，不再分配整段历史的反量化临时 KV；full decode 使用已有分块 kernel，sliding decode 使用带窗口的量化 prefill kernel。两个并发 lane 的 decode scratch 隔离。
- 量化 writer 直接调用原生写入 kernel。逐字节测试发现近似除法在整数边界会少编码一档，已改为精确除法，处理零 scale 并限制整数范围。CUDA BF16 转换下溢的 scale 归零；FP16/BF16 零值、极小值和随机值已检查。
- 修复 Llama 及其派生模型的外部存储声明漏传 `llm_kv_quant_group_size`：CPU cache 服务的 meta 布局现在与原生 StateStore 的 group 8/32 格式一致。真实 CPU 和 PD CopyPlan 往返验证 payload 与 scale 完整恢复。
- 修复 Llama recipe 未读取显式 `head_dim` 的问题，与原权重映射及新语义声明保持一致。

## 静态 FP8 Dense 扩展

- `StaticFP8AttentionBackend` 接收独立格式声明，在 bind 时固定 E4M3 K/V 视图及 FP32 K/V、可选 Q 标定；计算路径不再查询 State 或 manager。writer 也直接使用绑定的视图和标定。
- FA3 使用单 token 推理页，物理 allocator 仍可使用多 token 页。Planner 为每个阶段和并发 lane 预留固定 stride 页表。静态 Q 标定和缺少 Q 标定时的动态量化均支持 Graph；结果转换回声明的 query dtype。
- per-tensor FP8 可通过新的 `FlashInferAttentionBackend` 绑定，不要求 Q 也量化。FA3 不兼容时 auto 可选此路径；显式不兼容选择会由 Planner 报告。
- `FormatSpec.calibration` 使用启动时 KV 标定表的 FP32 内容摘要，独立于路径、JSON 排列和 Q 标定。规划与 State 绑定共享一次文件读取的快照，避免两阶段读到不同内容；隐式单位 scale 与显式全 1 表使用相同身份。State 另按实际物理层范围和全局 KV heads 规范化标定，摘要在 TP 切分前计算。
- P/D 交接携带各静态 FP8 State 的标定摘要、逻辑层映射和全局形状。传输进程先检查本节点全部 TP/DP rank 的声明一致，再分别在 P 申请源页前、D 申请目标页前检查对端身份；不一致或缺少必需身份时进入原失败通知流程，不提交 KV 复制。相同格式允许不同 TP 切分；Q 标定属于计算参数，不参与 KV 搬运兼容判断。
- 修复显式 `replicated=True` 时仍按 TP rank 扩展、切分 scale 的旧错误：完整副本现在保留全局 K/V/Q scale，普通分片和 KV heads 少于 TP ranks 的副本映射保持原语义。
- 以上只补静态 FP8 的标定内容契约。普通/量化布局的完整跨节点格式协商、持久 CPU/磁盘缓存跨重启身份、正式 P/D 服务与多卡模型数值验收仍待完成。

## FlashInfer 原生计算适配

- `FlashInferAttentionBackend` 通过 probe 声明 Dense causal/full/sliding、FP16/BF16 query、普通或 per-tensor FP8 KV，以及 head 64/128/256 的支持条件。模型无需专用 FlashInfer 分支。计算时只消费绑定的 K/V 与静态 scale，直接调用 wrapper，不创建旧 attention backend/state。
- 批次准备生成 CSR 页索引、末页长度和库执行计划。metadata 按 backend、attention 语义、数据格式分组；不同窗口、head 形状或物理格式不会覆盖彼此的计划。相同规格的层可复用，两个 microbatch lane 保持独立。
- Graph 输入更新后，对捕获时的 wrapper 使用公开 `plan()` 重新规划，刷新长度增长及 split 调度。prefill/decode 的 backend 可以独立选择，并共用兼容的已绑定状态表示。
- Planner 为每组每 lane 预留 128 MiB float workspace、8 MiB integer plan workspace 及最大页索引区；整数计划不能在准备完 metadata 后被不同规格复用。目前使用 FA2 wrapper 路径。公开 planning 会读取 CPU 长度并产生同步，wrapper 构造仍有暂时的内部 GPU allocation 和 pinned host buffer；尚未消除这些逐批开销，也未完成库内部峰值预算或性能验收。

## Gated DeltaNet 与混合模型

- `GDNModelAdapter` 从纯配置逐层声明 `GatedDeltaNetSpec` 和 recurrent `StateSpec`；Qwen3Next、Qwen3.5 分别提供权重与 recipe。显式 `layer_types` 优先于周期排列，计算层、权重映射、状态声明保持一致。Qwen3.5 的原生单层 draft 保留全局层号，SenseNova 的原生入口分别映射物理权重层与展开执行层；其它草稿及相应模式仍待迁移和验收。
- `ConvSSMStateStore`（原 `GDNStateStore`）创建并注册 Conv/SSM 状态，向 Compute 绑定零拷贝视图；`FormatSpec.field_dtypes` 明确 Conv 与 SSM 的独立 dtype，支持 FP16/BF16 Conv 和 BF16/FP32 SSM。沿用已有状态快照和搬运格式，统一 request allocator 所有权。
- `GatedDeltaNetBackend` 提供 Triton prefill/decode、FlashQLA prefill。卷积权重、A_log、dt_bias 在启动时绑定；执行不接收 model/config/manager。prefill 明确区分无历史请求和续算请求，避免读入已回收槽位的旧 SSM。decode 融合卷积打包和 recurrent update，Graph 只更新批次索引与长度。
- `Runner` 同时装配 Dense 和 GDN，各自使用声明的语义、格式、参数与 backend；不同算法族保留各自输入接口。FlashQLA 当前仅接入 128 维 key/value 的 Hopper 实现，其他条件由 probe 拒绝。Dense/GDN 已接入固定及动态候选 verify 和 Qwen3.5 原生草稿，KDA/MLA/sparse 等 MTP 仍保留原路径，详见后续验收记录。
- 修复原 Qwen3Next Dense attention 将 Q 宽度重复乘 2，以及非 MoE 层缺失 MLP 权重名称的问题。Qwen 混合模型的外部存储工厂接入 INT4/INT8 并传递 group size；仅确认格式与已验证的 Dense kernel 兼容，完整混合量化模型任务精度待验收。
- Conv/SSM 运行态在容量规划前实际分配；vendored Triton/FlashQLA prefill 内部仍动态分配临时张量，当前 offer 中的零固定 workspace 不表示零临时显存。完整峰值预算、临时分配优化仍待实现。

## 普通 latent MLA

- `Deepseek2ModelAdapter` 为现有 DeepSeek V2/V3 入口声明 `MLAAttentionSpec`：逐层 Q head、latent/非旋转/旋转/value 维度与 YaRN 修正后的 scale。`RecipeModelAdapter` 复用权重加载和 recipe 装配，Dense/GDN/MLA 各自校验语义，无需把 MLA 编码为 Dense head。
- `MLAStateStore` 注册原有 MLA State，绑定单个 `token_latent` 原生字段；latent 在 TP rank 间保持完整副本，仅 Q heads 切分。保留原 CPU/PD 资源身份与搬运格式。writer 直接写绑定的 latent，prefill/decode 不查询 manager。
- `MLAAttentionBackend` 接入 Triton 和 Hopper FA3，`FlashInferMLABackend` 接入普通 FP16/BF16 FlashInfer。prefill 接收投影后的 Q，并使用启动时绑定的 KV 展开函数；decode 接收低秩 Q 与旋转 Q。prefill/decode 可以独立选择后端；FA3 decode 要求 latent 至少 256 维，由 probe 提前拒绝其他宽度，auto 可选择 Triton。
- Triton decode 固定 128 个持久 worker，Planner 预留 split 索引、FP32 中间结果和 logsumexp；顺序层共享，两个执行 lane 分离。每次执行根据当前长度更新 split，Graph 回放不沿用旧长度计划。FA3 页表使用固定 stride 的单 token 推理页，与物理 allocator 页大小独立。
- FlashInfer prefill 使用 ragged KV wrapper，decode 使用原生 MLA paged wrapper；均绑定原生视图，计划按语义和格式分组，每组每 lane 预算 128 MiB float、8 MiB integer 及 decode 页索引。decode Graph 回放前调用公开 `plan()` 刷新工作表，并检查捕获时的计划偏移和启动几何保持一致，变化时拒绝回放。当前使用库的 FA2 路径。
- 当前 FlashInfer MLA wrapper 没有公开的整数 workspace 替换接口，因此计算适配器在首轮 plan 前绑定 `_int_workspace_buffer`，并校验库要求的容量；Graph 布局检查读取 `_plan_info`。这两处库接口依赖需要随版本验证。公开 planning 仍读取 CPU 长度，wrapper 构造仍有临时 GPU/pinned 分配，尚未消除逐批开销或完成库内部峰值预算。
- `Glm4MoeLiteModelAdapter` 显式接入同一 MLA 契约，选择自己的权重和 recipe 类，保留模型自身 RoPE。新增模型适配不需要在 attention backend 内增加 GLM 分支。单卡路由专家和共享专家融合的完整 forward 已纳入验证；EP/多卡仍待验收。
- 修复 DeepSeek `rope_scaling=None` 时原启动代码引用未初始化局部变量的问题。普通 prefill、decode、前缀续算、Graph 已接入；DeepSeek V3.2 的 packed/sparse 路径见下，其余量化 MLA、MTP 和未显式接入的派生模型仍走旧路径。未创建 CC 展开权重的普通 MLA 配置也保留原路径。
- prefill 仍需按当前 KV 总长度采样并展开完整历史，临时显存未纳入精确峰值预算；固定 worker 配置尚未做性能调优。这里没有宣称完成 MLA 的全部实现和分布式模式迁移。

## DeepSeek V3.2 sparse MLA

- `Deepseek3_2ModelAdapter` 声明 `SparseMLAAttentionSpec`，包含 MLA 投影、indexer heads/维度、top-k 与 indexer 量化规则；模型 recipe 保留 indexer projection、norm、RoPE 和 Hadamard。`PackedMLAStateStore` 绑定 latent 与 indexer 两个原生字段，保留 TP 完整 KV 副本及既有 CPU/PD 布局。
- Triton 路径支持 BF16 与 `fp8kv_dsa`，后者直接读取 512 字节 E4M3 payload、4 个 FP32 scale 和 64 维 BF16 RoPE。writer 与 indexer writer 分别更新各自字段；支持非连续的 token 索引。FP8 prefill 对本批新增 token 使用未量化 KV，对历史使用缓存解码，decode 使用量化缓存。
- indexer 直接读取绑定的 FP8 key/scale，计算加权 ReLU 分数并筛选物理 token，不反查 manager。当前安装的 DeepGEMM 在压缩行宽小于批内 KV 总长度时出现跨行覆盖，多请求对照已复现；原生路径改用 Triton 逐行计算。没有修改第三方库或旧执行路径。
- 普通 packed BF16 可选择 `flashmla_sparse`：prefill 调用 SGL FlashMLA，要求 Q heads 为 64 的倍数、top-k 为 256 的倍数；decode 使用支持原生 stride 的 FA3 KV 接口。Planner 可以分别选择两个阶段。group128 FP8 现也接入 standalone FlashMLA，页内单元素维度使用显式 stride，保留 packed 行步长及同一行的 indexer；此前判定整行布局不兼容过于宽泛，H100 修正和验证见文末。
- split prefill Graph 显式保留 hidden/q-lora 输入视图，回放不再读取已删除的临时字典。decode Graph 使用固定形状 metadata 和 indexer logits 行宽，更新长度及请求映射后可回放。
- Planner 为每条并发 lane 预留 ragged 物理索引。logits、Q/K 量化和 top-k 的临时张量仍由运行期分配，完整峰值预算与性能调优未完成。此路径迁移显式 V3.2 入口；GLM5 Next 的 KDA/窗口压缩见下，V4、MTP、跨卡模型与正式 checkpoint 任务精度仍待验收。

## GLM5 Next：KDA 与 pooled NoPE

- `Glm5NextModelAdapter` 显式声明逐层 KDA 或 pooled NoPE 语义，保留权重映射、mHC、projection、norm、MLP/MoE recipe。NoPE 有独立 `PooledNoPEAttentionSpec`，不借用带旋转维度的 MLA 声明。当前仅原生接入无 MTP 组合。
- `AttentionRequest.auxiliary_states` 将历史 KV 和 indexer tail 分开声明；Planner 检查同一模型角色、消费者、格式和物理绑定，拒绝遗漏或错误的辅助视图。`PackedMLAStateStore` 提供历史 latent/indexer，`IndexerWindowStateStore` 提供独立窗口，`ConvSSMStateStore` 复用 GDN/KDA 的分配与快照机制。运行不查询 manager。
- `KimiDeltaAttentionBackend` 绑定卷积权重、gate 参数及 Conv/SSM，分别执行 chunk prefill 和 recurrent decode。新请求显式清除初始 SSM 的旧内容；续算使用已恢复状态。prefill 支持 head dim 32/64/128，现有 decode kernel 只支持 128，其他形状在 probe 拒绝；卷积依赖也在启动时检查。
- `PooledNoPEAttentionBackend` 绑定 history 和 tail，执行池化、FP8 indexer weighted-ReLU/top-k 和 latent attention。Triton 支持 FP16/BF16 普通 KV 及 BF16 的 group64/E8M0 packed KV；量化 prefill 对本批 token 使用未量化 KV。SGL prefill 仅 BF16，普通 SGL decode 需要 `only_qv`；阶段可独立选择。group64/E8M0 FP8 + FlashMLA 已接入原生绑定，固定 scratch 与验证范围见文末。
- Decode Graph 按计划的最大长度决定是否需要 indexer，避免在短序列捕获后增长超过 top-k 时仍走 dense 分支。HOLD 请求显式输出零；并发 lane 使用各自 ragged 索引区。prefill Graph 显式声明输出，避免为探测形状额外推进窗口状态。
- FP8 Triton attention 使用单 pipeline stage，避免 H100 shared-memory 超限。Hadamard 量化接受 FP16/BF16，保留转换到 BF16 后再量化的既有数值语义。mHC 关闭时 embedding 不再创建多流输出。
- 这仍未覆盖 GLM 正式 checkpoint 精度、完整多模态/MTP、分布式模型执行或性能。logits、top-k、KDA 内部张量及库暂存的完整峰值预算待完成；Linear 与普通/block128 FP8 MoE 已由 Planner 绑定原生执行；其它量化及 EP 仍保留旧路径。

## 验证结果

H200：历史回归使用 GPU 4/5，Sparse MLA 合并回归使用 GPU 7，peer IPC 使用 GPU 5/7；CUDA 12.8；独立 `/tmp/lightllm-refactor-flashinfer` 编译缓存。

- 相关组合：**161 passed**。覆盖新增执行测试、原 Attention 生命周期、共享状态启动、逐层选择、prefill Graph 和 FlashInfer MLA planning。
- GPU kernel：Triton/FA3 × prefill/decode，与 PyTorch SDPA 对照；FP16/BF16、MHA/GQA、page size 1/4、超过 512 token 的不等长序列、两个 CUDA stream 的独立 microbatch、Graph 回放时请求重排和长度增长。
- 完整 forward：两层随机权重的 Llama、Qwen2、Qwen3 × 两个 backend，另含 Qwen3 full/sliding 混合模型，共 8 项；真实权重映射、RoPE、norm、MLP、attention、lm-head，对照 Transformers logits。覆盖 prefill、decode、真实 `CudaGraph` capture/replay、缓存前缀续算、真实分段 `PrefillCudaGraph` capture/replay。
- 资源/CPU/PD 扩展回归：**164 passed, 1 skipped**（单卡配置）；跳过的跨 GPU peer IPC 项在双卡单独运行 **1 passed**。三个互不重复的测试集合合计 **326 passed**。
- 修复扩展回归发现的 GQA 搬运编译问题：未对齐/有 stride 的字段可能只能按 2 字节搬运，Triton 多级流水生成的 `cp.async` 要求至少 4 字节；这些窄字段使用单级流水，宽字段保留原配置。相关不同 stride、对齐、位宽、CPU/CUDA 页测试通过。
- CPU 契约：选择失败、整体预算、共享 workspace、并发 lane 隔离、异构状态、显式共享、TP 副本声明、绑定后禁止资源查询、FA3 cache 同步要求、重复绑定拒绝。
- 上述 Dense 证据是局部 GPU 和随机小模型验证；没有正式 checkpoint 精度、服务请求、真实多卡模型 TP/DP/EP、复杂 MTP 或性能验收。PD 驱动的新增双进程验证见下。
- 原共享 FlashInfer 编译缓存固定了失效的 CUDA 12.6 路径，因此使用独立缓存，没有删除共享缓存。先前更广的 sparse-MLA 测试要求 FlashInfer ≥ 0.6.18，环境为 0.6.8.post1；未升级依赖，该项仍未通过。

Transfer 接入后的当前验证（与上轮集合有重叠，不累加为总数）：

- GPU 4 上执行 native execution、State 创建/查询、CPU 页各编码、混合状态恢复、PD 复制与 DP 页分配：**343 passed, 2 skipped**。跳过项是需要双卡的 peer IPC 和缺少 `light_mem` 的磁盘页测试；peer IPC 随后在 GPU 5/7 补跑 **1 passed**。
- CPU cache 策略、多模态缓存边界、PD 通知/进度及当时的 Transfer 故障集合：**103 passed, 1 skipped**。随后增加的故障场景连同原故障集合单独验证：**19 passed**（这两个数字有重叠）。
- GPU 5/7 上独立进程运行实际 PD transporter，检查 packed KV、accepted linear state 和完整 tail 字节恢复：NCCL **1 passed**，NIXL **1 passed**。这是驱动与状态往返测试，不是完整 P/D 服务请求验收。
- NIXL 初次加载因 `libssl.so.3` 缺失失败；使用已有 `/tmp/lightllm-nixl-runtime/lib` 作为该测试进程的 `LD_LIBRARY_PATH` 后通过，没有安装或替换共享依赖。
- 新增/变更的传输与生命周期代码通过 Flake8、编译检查、diff 空白检查。GPU 扩展回归曾因旧测试仍拦截 manager 而直接执行 CPU 假状态导致非法访存；已将替身改为拦截绑定的 State 操作，并在新进程完成上述回归。失败记录保留于 `/tmp/lightllm-transfer-regression.log`，最终结果在 `/tmp/lightllm-transfer-regression-final.log`。

ModelAdapter 与 INT4/INT8 接入后的验证（与前述集合有重叠）：

- `unit_tests/common/execution`：GPU 4 上 **127 passed**。其中 24 个完整小模型案例覆盖 Llama/Qwen2/Qwen3、Qwen3 full/sliding、Llama/Qwen3 显式宽 head，分别使用普通 Triton、普通 FA3、INT4、INT8；禁用旧模型工厂仍能加载权重、执行 prefill/decode、前缀续算和两类 Graph。
- 分组 attention 对照独立解码的 SDPA：FP16/BF16、group 8/32、不规则 token 索引、不等长长序列、并发 lane、Graph 输入更新。写入编码与 scale 逐字节一致；另有 8 个 CPU 服务布局及 CPU/PD 实际往返案例。
- 完整模型普通精度保持原 logits 阈值。INT4/INT8 与独立 Torch 量化的 HF cache 对照，分别限制相对 RMS 误差 5%/1%、最大 logits 误差 0.075/0.015。FP16 RoPE 的微小差异会跨整数截断边界，因此该预算不能当作正式 checkpoint 的任务精度验收。初次严格比较及定位日志保留在 `/tmp/lightllm-group-quantized-model.log` 和 `/tmp/lightllm-debug-intkv.log`。
- 原 KV 编码/解码测试与新增搬运集合 **81 passed**；共享启动、模型存储、量化缓存、生命周期、Transfer 故障、CPU 恢复与 Graph 回归 **263 passed, 1 skipped**。执行集合与其中部分案例重叠，不累加。日志分别为 `/tmp/lightllm-group-quantized-transfer.log`、`/tmp/lightllm-adapter-regression.log`、`/tmp/lightllm-adapter-execution.log`。

FP8 扩展验证：

- FP8 独立集合与完整模型集合 **85 passed**。覆盖两种 KV 标定、静态/动态 Q、FP16/BF16 query、allocator page 1/4、full/sliding、Graph replay。完整小模型矩阵增至 36 项，包括 per-head/per-tensor FP8 的原生启动、标定文件加载和两类 Graph；仍是随机小模型证据。
- FP8 writer 与独立 Torch 编码逐字节一致。attention 对照独立量化并解码 Q/K/V 的 SDPA：限制相对 RMS 误差 3.5%，最大绝对误差小于参考峰值的 6.25%（半个 FP8 相对 ULP）。定位时直接捕获传给 FA3 的实际张量后反量化，单个案例仍有约 2% RMS 误差，因此不能沿用普通精度阈值。完整 logits 使用 5% RMS、0.075 最大误差预算，参考 HF 仅量化 KV；正式模型精度仍未验收。
- FP8 初次阈值失败、输入定位及通过记录保留于 `/tmp/lightllm-native-fp8.log`、`/tmp/lightllm-fp8-input-diagnostic.log`、`/tmp/lightllm-native-fp8-v3.log`。
- 最终同一进程合并执行 native execution、标定、共享启动、模型存储、量化缓存、生命周期、Transfer 故障、CPU 恢复、Graph 和原 INT KV kernel 检查：**533 passed, 1 skipped**。跳过项是 `test_cpu_resource_layout.py` 中缺少 `light_mem` 的磁盘缓存检查。日志 `/tmp/lightllm-native-quantization-final.log`；Black、Flake8、编译和 diff 空白检查通过。此集合替代本轮分批数字作为当前回归证据，不与前轮数字相加。

FlashInfer 接入后的当前回归（替代前轮组合数字，不累加）：

- GPU 5 上同一进程执行上述 execution、标定、共享启动、模型存储、量化缓存、生命周期、Transfer 故障、CPU 恢复、Graph 和 INT KV kernel 集合：**586 passed, 1 skipped**，耗时 41.40 秒。跳过项仍是缺少 `light_mem` 的磁盘缓存检查，日志 `/tmp/lightllm-native-flashinfer-final.log`。
- 完整随机小模型增至 **60 项**：6 种模型配置 × 10 种 prefill/decode/格式组合。新增普通与 per-tensor FP8 FlashInfer，以及 FA3-prefill/FlashInfer-decode 普通精度、FlashInfer-prefill/FA3-decode FP8 两种混合执行。均包含真实权重映射、前缀续算、decode Graph 和分段 prefill Graph。
- FlashInfer 长序列、请求重排、MHA/GQA、FP16/BF16、两条 stream/lane、物理页 1/4、full/sliding 的普通/FP8 对照通过。FP8 FlashInfer 保持 FP16/BF16 Q，对照解码 KV 的 SDPA 使用 `rtol=2e-2, atol=5e-3`；不同于 FA3 FP8 的 Q 量化与数值误差预算。
- 同一 Planner 同时绑定普通与 FP8 两种格式，先准备两份计划再执行，验证计划不会相互覆盖；非默认 score scale、softcap、窗口 mask 与独立 Torch 计算一致，长度增长后的 Graph replay 通过。CPU 预算检查确保不同格式的计划单独计入；非 Hopper 的 auto 选择只做了能力声明测试，未在该硬件上实测。
- 单独定位日志保留于 `/tmp/lightllm-native-flashinfer-model.log`、`/tmp/lightllm-native-flashinfer-kernel.log`、`/tmp/lightllm-native-flashinfer-fp8.log`、`/tmp/lightllm-native-flashinfer-mixed.log`、`/tmp/lightllm-native-flashinfer-format.log`。相关代码通过 Black、Flake8、编译和 diff 空白检查。没有正式 checkpoint 任务精度、完整服务、多卡模型或性能验收。

GDN 接入后的当前回归（替代前轮组合数字，不累加）：

- GPU 5 同一进程执行 native execution、原 GDN/状态/模型配置、SenseNova 层映射及 MTP draft 构造、Transfer/CPU 恢复、Graph 与 INT KV kernel 集合：**640 passed, 1 skipped**，54.10 秒。唯一跳过项是缺少 `light_mem` 的磁盘缓存测试；日志 `/tmp/lightllm-native-gdn-final-v2.log`。
- 12 个 GDN kernel 案例覆盖 Triton 32/128 维和 FlashQLA 128 维、FP16/BF16 Conv、BF16/FP32 SSM，对照独立 Torch 卷积与 recurrent 递推。覆盖跨 chunk 的不等长 prefill、带历史与脏槽位的新请求、两字段快照恢复、decode Graph 重排/增长、两条 CUDA stream 的独立 lane。
- 另有 6 个四层 Qwen3Next 完整随机模型对照 Transformers，包括周期/非周期排列，以及 Triton/FA3/FlashInfer Dense 与 Triton/FlashQLA GDN 组合。覆盖真实权重映射、prefill/decode、前缀续算及两类 Graph；绑定后禁止 manager 查询。完整随机模型集合从 60 增至 66 项。
- 本地 **Qwen3.5-0.8B 正式权重**新旧 LightLLM 链路比较通过：两个中文提示、连续 4 步 greedy decode、后缀 prefill、Conv/SSM 保存后破坏再恢复并继续 decode。比较 logits、greedy token、每层 Conv/SSM，使用 `rtol=0.003, atol=0.005`。这是迁移一致性证据；参考实现同为 LightLLM，尚未完成独立任务精度、图像输入、服务、多卡或性能验收。单独日志 `/tmp/lightllm-native-qwen35-checkpoint.log`；合并回归已设置 `LIGHTLLM_TEST_QWEN35_CHECKPOINT=/nvme/models/Qwen3.5-0.8B` 包含该项。
- 初次模型测试暴露的 MLP 权重名称、Q 宽度问题已修复；Graph 测试必须在重复 capture/warmup 后恢复全部 KV 与 recurrent 状态，不能只恢复 SSM。失败日志和较早的测试夹具错误记录保留于 `/tmp/lightllm-native-gdn-model-*.log`、`/tmp/lightllm-native-gdn-final.log`。

MLA 接入后的当前回归（替代前轮组合数字，不累加）：

- GPU 5 同一进程合并上述集合：**671 passed, 1 skipped**，59.18 秒。唯一跳过项仍是缺少 `light_mem` 的磁盘页测试；日志 `/tmp/lightllm-native-mla-final.log`。其中包含 Qwen3.5-0.8B 正式权重一致性检查和全部先前原生执行案例。
- MLA 新增 22 个实际 GPU 数值案例、2 个 FA3 小 latent 启动拒绝案例和 1 个 TP 副本声明检查。覆盖 latent 128/256/512、FP16/BF16、非整组 Q heads、不规则物理 token、长前缀、普通写入、双 lane/stream、decode Graph 请求重排及长度增长；数值参考由独立 Torch 展开/attention 构造，`rtol=0.02, atol=0.004`。
- 6 个 DeepSeek 两层完整随机模型与 Transformers 对照，包括普通 Q/低秩 Q，Triton、FA3、FA3-prefill/Triton-decode。沿用普通模型 logits 阈值 `rtol=0.01, atol=0.004`，覆盖权重映射、prefill/decode、前缀续算、decode Graph、分段 prefill Graph；绑定后禁止 manager 查询。完整随机模型集合现为 72 项；不等同于 DeepSeek 正式 checkpoint、MoE/EP 或服务验收。
- 随后复核 YaRN scale 的浮点运算顺序，保持声明与模型 recipe 一致，并加入 DeepSeek V3、factor=2、mscale_all_dim=1 的三种后端组合。专项最终 **9 passed**（包含前述 6 项，不能累加），日志 `/tmp/lightllm-native-mla-yarn.log`。当前完整随机模型矩阵为 75 项；新增的 3 个 YaRN 案例不在先前 671 项合并日志内。
- 初次 kernel 检查发现 FA3 `q_v` 要求 latent 至少 256，已纳入 probe 并覆盖边界。首次完整模型参考配置错误地把 MLA KV head 设成 GQA 形式，修正为完整 Q head 数后通过；没有放宽 logits 阈值。日志 `/tmp/lightllm-native-mla-kernel.log`、`/tmp/lightllm-native-mla-model.log`、`/tmp/lightllm-native-mla-model-v2.log` 保留定位过程。
- 当时新增执行模块、模型接入和测试通过 Black、Flake8、Python 编译与 diff 空白检查。后续 FlashInfer MLA 接入见下，剩余模式仍按阶段表继续推进。

FlashInfer MLA 与 GLM4-MoE-Lite 接入后的回归（替代前轮组合数字，不累加）：

- GPU 5 同一进程执行上述集合：**710 passed, 1 skipped**，68.94 秒；日志 `/tmp/lightllm-native-flashinfer-mla-final.log`。唯一跳过项仍为缺少 `light_mem` 的磁盘页测试，包含 Qwen3.5-0.8B 正式权重迁移一致性检查。
- 普通 MLA kernel 覆盖 Triton/FA3/FlashInfer、FP16/BF16、latent 128/256/512；FlashInfer decode Graph 增长后再缩短长度。新增 128 Q heads、latent 512、最长 4049 token 的多 CTA cluster Graph 回放，以及不同 head/score scale 的两层先同时 planning、再计算，检查持久计划不会互相覆盖。专项日志 `/tmp/lightllm-native-flashinfer-mla-model.log`。
- 完整随机模型矩阵增至 **96 项**：60 Dense、6 GDN、18 MLA、12 MLA/MoE。MLA 模型含普通 Q、低秩 Q、V3 YaRN；prefill/decode 覆盖同后端及 FA3/FlashInfer 交叉组合。
- 12 个单卡 MoE 案例覆盖 DeepSeek V3、GLM4-MoE-Lite × 三种 attention 后端 × 共享专家融合开关，包含真实权重映射、路由专家、共享专家、前缀续算及两类 Graph。GLM4-MoE-Lite 使用实际入口配置，数学参考为相同 MLA/MoE 语义的 Transformers DeepSeek V3 小模型；不是正式 GLM checkpoint 或独立 GLM 参考实现验收。普通 logits 阈值保持 `rtol=0.01, atol=0.004`，EP/多卡仍待验收。
- 相关代码通过 Black、Flake8、编译与 diff 空白检查。以上测试不证明完整服务、分布式能力或性能目标已达成。

标定内容身份与 P/D 兼容检查的回归：

- GPU 5 同一进程合并原执行集合、标定内容检查、P/D 通知/进度/索引/页面任务：**764 passed, 1 skipped**，118.41 秒。唯一跳过项仍为缺少 `light_mem` 的磁盘页测试；日志 `/tmp/lightllm-calibration-identity-final.log`。包含全部 96 个随机完整模型与 Qwen3.5-0.8B 正式权重迁移一致性检查；替代前轮组合数字，不累加。
- 新增内容检查覆盖 FP32 数值规范化、Q scale/路径无关、隐式与显式单位 scale、非法 scale、TP 1/2/8 的全局摘要、紧凑与重复 head 表等价、只修改非本地 head 时仍拒绝，以及规划后文件变化仍使用同一快照。协议测试实际执行 P/D 工作循环，检查错误或缺失身份不会申请 staging 页或排入复制队列。
- 随后补充完整副本配置，三个案例先复现旧错误：K/V scale 有一半数值不匹配；失败日志 `/tmp/lightllm-fp8-replicated-before.log`。修复后重跑标定、TP 配置及实际 FP8 CUDA 计算绑定，**43 passed**，7.04 秒，日志 `/tmp/lightllm-fp8-calibration-replicated-final.log`。这组与合并回归有重叠，新增三个副本案例不在先前 764 项日志内。
- Black、Flake8、编译和 diff 空白检查通过。此处是本地内容契约、协议工作循环和数值检查；未执行新的完整 P/D 服务或跨节点 TP 模型验收。

Sparse MLA 与搬运设备预检后的当前回归（替代前轮组合数字，不累加）：

- GPU 7 同一进程合并执行原生 execution、标定/资源、原 GDN、SenseNova/MTP 构造、CPU/PD/DP 各编码往返、Transfer 故障、CPU 恢复、P/D 协议工作循环、两类 Graph 和 INT KV kernel：**809 passed, 1 skipped**，130.40 秒。唯一跳过项是缺少 `light_mem` 的磁盘页测试；日志 `/tmp/lightllm-native-sparse-mla-final.log`。包含 Qwen3.5-0.8B 正式权重迁移一致性检查和标定阶段新增的三个完整副本案例。
- 完整随机模型矩阵增至 **101 项**。新增 5 个 V3.2 组合覆盖 BF16/FP8 Triton、普通 FlashMLA prefill/FA3 decode 以及交叉阶段，含前缀、decode Graph 和 split prefill Graph，并禁止旧工厂与执行期 manager 查询。数学参考使用 Transformers DeepSeek V3 权重与独立 Torch 低秩 attention；短序列入选全部 token，不代表正式 V3.2 checkpoint 或稀疏任务精度。BF16 普通 logits 要求 RMS 误差低于 1.5%、最大误差低于参考峰值的 1/64、greedy token 相同；FP8 保持 5% RMS、0.075 最大误差预算。
- sparse kernel 对照独立 Torch，覆盖普通/量化历史、未量化当前 prefill KV、不规则物理索引、5/64 Q heads、长短不等请求和 Graph 增长/缩短。FP8 writer 的 payload 与 scale 逐字节一致，indexer 字段不被覆盖。
- indexer 单独验证 64/6 heads、普通/幂次 FP32 scale、256/2048 top-k；最长 4063 token，多请求行宽小于 KV 合计长度。有效 logits 使用 `rtol=1e-4, atol=1e-5`，入选物理 token 集合与 Torch 完全一致。2048 top-k 的库输出顺序可变化，eager/Graph 比较仍要求集合完全一致，attention RMS 差异低于 1%、最大差异低于参考峰值的 1/128。
- 新增搬运集合首先复现 6 个错误设备异常被包装的问题；CPU/PD 计划提前拒绝后通过。纯参数转发测试使用显式设备替身，真实搬运测试继续执行 CUDA 数据往返。专项最终 **122 passed, 1 skipped**（与上述合并集合重叠），日志 `/tmp/lightllm-sparse-mla-long-and-transfer-v2.log`。
- 失败与定位日志保留于 `/tmp/lightllm-native-sparse-mla-model-final.log`、`/tmp/lightllm-native-sparse-mla-combined.log`、`/tmp/lightllm-sparse-mla-long-and-transfer.log`。35 个相关 Python 文件通过 Black 检查；相关目录与文件的 Flake8、编译、diff 空白检查通过。尚未执行 V3.2 正式 checkpoint、完整服务、多卡模型或性能验收。

Graph 协调迁移后的本地历史回归（与后续 H100 集合不累加）：

- GPU 7 同进程合并原集合与 Graph 调度、MTP/TP/SP 对齐、双 microbatch、GLM 层与 FP8 workspace 检查：**979 passed, 2 skipped, 1 deselected**，96.76 秒，日志 `/tmp/lightllm-runner-graph-combined.log`。包含全部 101 个完整随机模型与 Qwen3.5-0.8B 正式权重一致性检查。
- 两项 skip 分别为缺少 `light_mem` 的磁盘页检查和缺少 standalone `flash_mla` 的 V4 Graph/kernel 测试模块。一个 deselect 是已单独复现失败的 `test_sparse_layer_prefill_and_decode_match_dense_nope_attention`：现装 SGL 的 `flash_attn_with_kvcache` 没有 `only_qv` 参数，失败记录在 `/tmp/lightllm-runner-graph-policy.log`。这是待修复的环境/API 兼容缺口，不是通过或验收豁免。
- 新增真实 CUDA Graph 检查覆盖混合段顺序、动态输入/输出释放后的地址有效性、当前批次替换、形状改变拒绝、形状探测与普通段异常后退出捕获并建立新 Graph。Qwen3-VL 与 VL-MoE 的实际 deepstack kernel 从 pinned CPU 特征缓存读取当前图像；KDA 入口用包含主机同步的状态型替身验证不会在 capture 时试运行。它们不等同于完整多模态/KDA 模型精度。
- GLM sparse prefill 使用实际权重投影、SGL attention 与 Graph 回放，对照独立 Torch；请求更换后重新使用当前 top-k 映射及保留的 indexer 输入。单独验证 **1 passed**，日志 `/tmp/lightllm-runner-graph-glm-prefill-v2.log`。测试先执行实际 warmup 初始化 cuBLAS，首次缺少 warmup 的失败记录保留。
- Graph 策略专项 **147 passed**，日志 `/tmp/lightllm-runner-graph-policy-v2.log`；与合并集合重叠。最初单独运行对齐检查时缺少 rank 环境，已让测试显式提供 rank；调用参数兼容问题也在合并前修复。没有新增正式模型服务、多卡模型、吞吐或显存性能验收。
- 53 个相关 Python 文件通过 Black、Flake8 和编译检查，diff 空白检查通过。Black 的多文件 diff 模式受沙箱 socket 限制，改用同版本 `format_str` 顺序检查；没有为格式检查扩大权限或修改共享工具。

复核主要相关测试：

```sh
CUDA_VISIBLE_DEVICES=4 CUDA_HOME=/usr/local/cuda-12.8 \
FLASHINFER_WORKSPACE_BASE=/tmp/lightllm-refactor-flashinfer \
FLASHINFER_NVCC=/usr/local/cuda-12.8/bin/nvcc MAX_JOBS=2 \
python -m pytest unit_tests/common/execution \
  unit_tests/common/basemodel/attention/test_state_lifetime.py \
  unit_tests/common/basemodel/attention/flashinfer/test_mla_cuda_graph_plan.py \
  unit_tests/common/basemodel/test_prefill_cuda_graph_state.py \
  unit_tests/common/att_resources/test_backend_startup.py \
  unit_tests/common/att_resources/test_runtime_transport.py::test_cpu_pd_and_dp_model_layouts_cover_quantized_fields \
  unit_tests/common/att_resources/test_layer_state_selection.py -q
```

运行前确认 GPU 可用；GPU 编号与工具链路径属于本次环境，不是固定项目配置。

## H100 验证与兼容修复

用户已指定后续 GPU 验证使用 `ssh h100`。环境、快照、复现命令和日志见 [H100 验证记录](h100-validation.md)。使用现有 CUDA 13 镜像及隔离代码/缓存，没有改共享依赖或部署服务。

本轮修复 BF16 INT4/INT8 scale 下溢随编译器变化的问题，补新版 `rope_parameters` 配置归一化及默认 RoPE factor，并用 FP32 保存 Qwen/Gemma norm 的 `1 + weight`，避免小偏移在绑定前丢失。V4 FlashMLA 在启动检查 `out` 能力，旧接口额外复制到固定输出。

H100 同进程合并 **1485 passed, 1 failed**，260.34 秒，无 skip 或 deselect；包含 101 项完整随机模型、24 项 GLM5 Next 层、42 项 V4 FlashMLA、51 项 CPU 状态资源及 8 项双卡量化搬运。双卡 NCCL/NIXL 驱动专项另行执行，2 项通过。合并集合的唯一失败为下面的 BF16 独立数值检查；不能将此结果报告为全部通过。

正式 Qwen3.5-0.8B 增加独立 FP32 Transformers 参考，使用 eager attention 与 Torch recurrent 运算。FP16 的 prefill、四步 decode、后缀 prefill、Conv/SSM 保存/破坏/恢复后的 decode 通过；BF16 在 14 个位置的 greedy token 一致，但逐请求 RMS 最大约 1.7853%，超过新增检查的 1% 阈值，仍未通过独立数值验收。正式任务精度、服务与多卡模型验收继续保留为未完成。

进一步对同样输入运行 BF16 Transformers，自身相对 FP32 的 RMS 最大为 2.4124%；首轮 prefill 的逐层误差与 LightLLM 接近。暂未定位到单个异常算子，不能把 1% 门槛失败直接等同重构回归。检查仍保留失败，详见 H100 记录中的诊断证据。

### GLM5 Next 原生绑定的 H100 验证

本轮合并 `glm-final.xml` 共 1543 项：**1540 passed, 3 failed**，313.00 秒，无 skip 或 deselect。包含原 101 项完整随机模型、新增 4 项 GLM 完整小模型、37 项既有 GLM kernel 和 2 项实际 NCCL/NIXL 驱动检查，均通过。两项长上下文失败是测试夹具漏算底层存储的额外保留行；修正后 `glm-long-fixed.xml` 的全部 **18 项新增 GLM 检查通过**，18.46 秒。两轮之间执行代码没有变化，未重跑完整集合，不将结果合并成一次运行的数字。

新增检查覆盖 KDA 独立 Torch 递推与状态恢复，pooled NoPE 普通/FP8 KV、独立 weighted-ReLU/top-k、FP16/BF16 Hadamard 字节对照、短转长 Graph 和 HOLD，长上下文 64 heads/2048 top-k/最长 4093 token 的双 lane 与 Graph。完整 GLM 小模型覆盖真实 weights/recipe、mHC/MoE、前缀续算和两类 Graph；参考为旧 LightLLM，不能代替官方 checkpoint 任务精度。

唯一仍未解决的数值失败是 Qwen3.5 BF16 独立参考 RMS 1.785284% 大于 1%，阈值未放宽。合并日志另外保留 CUDA IPC 引用释放顺序警告，生命周期审计仍待继续。详见 [H100 验证记录](h100-validation.md) 的脚本、源码摘要与失败日志；相关 14 个 Python 文件通过格式、Flake8、编译和 diff 检查。H100 任务容器退出，八卡显存释放；完整七模块目标仍未完成。

### Linear 原生绑定的 H100 验证

当前完整合并 `linear-final.xml` 共 1572 项，**1571 passed, 1 failed**，295.56 秒，无 skip/deselect，替代上一轮分批结果作为当前证据。包含此前全部 1543 项、新增 Linear 24 项及原普通矩阵 5 项；两个 GLM 长上下文夹具修复已在完整集合通过。唯一失败仍为 Qwen3.5 BF16 RMS 1.785284% 大于 1%，未放宽阈值。

Linear 对照覆盖普通矩阵/BMM、bias、非连续视图、FP32 输出、空批次，FP8 per-tensor/per-channel/block128 的独立 Torch 参考与 Graph。TP=2/rank=1 的本地切片与融合权重分批加载、未完成加载拒绝绑定、绑定后禁止旧量化执行并替换 loader 引用也通过；这是单卡切片检查，不是完整 TP 模型验收。原 101 项完整随机模型在推理前禁止 `NoQuantization.apply()`，验证运行使用新绑定；加上 4 项 GLM 模型，共 105 项通过。

首轮 149 passed/1 failed 的失败为 TP 测试夹具权重名称漏掉 `.weight`，scale 名称错误映射到原权重；修正后 24 项专项通过，随后完成上述合并。六个 Python 文件格式、Flake8、编译和 diff 检查通过；源码摘要、脚本及日志见 [H100 验证记录](h100-validation.md)。本轮仍保留 CUDA IPC 引用释放警告为生命周期待查项，未声称完整服务、MoE/EP 或性能验收通过。

### MoE 原生绑定的 H100 验证

- `moe-final.xml` 确认 **1621 passed, 1 failed**，296.52 秒，1622 项，无 skip/deselect。包含 29 项新原生 MoE、108 项完整随机模型（新增 Qwen3Next MoE 三配置）、既有状态/Graph/CPU恢复/PD及双卡传输。唯一失败仍为 Qwen3.5 BF16 独立参考 RMS 1.785284% >1%，未放宽阈值。
- MoE 专项覆盖 FP16/BF16、plain/block128 FP8、分组 top-1/top-2、160 专家、shared/gated shared、clamp/add-one、路由捕获、空批次、输入更新和并发 Graph。独立 Torch 数学检查 RMS <1%、最大差 <参考峰值/32；route 权重与专家集合分别检查。TP=2 rank1 是单卡本地加载切片验证，非完整 TP 服务。
- 模型回归禁用旧权重 MoE 执行入口，确保真实使用原生绑定。独立 EP、正式 MoE checkpoint 精度、完整服务、性能和全峰值预算尚未验收；CUDA IPC 退出警告仍待生命周期审计。历史失败、数值诊断、修复复验和最终快照均见 `h100-validation.md`。

## Runner 批次入口的 H100 回归

通用批次收集、padding 与输出裁剪已从模型基类迁到 `BatchRunner`，普通和双 microbatch 入口共用执行协调逻辑。模型专有状态构造、位置初始化和计算 recipe 通过显式回调接入；InferState 组装、warmup、缓存策略与路由记录仍待继续迁移。

`batch-final.xml` 完整回归为 1666 passed、5 failed（1671 项）；新增草稿测试的 4 个失败来自手工构造缺少真实初始化必填的 draft 列表。修正测试后，`batch-fixed.xml` 的全部 49 项新增集合通过，执行代码未改变。去重覆盖 1671 项，1670 项已有通过结果；这不是一次完整运行的通过数。唯一数值失败仍为 Qwen3.5 BF16 RMS 1.785284% > 1%，CUDA IPC 退出警告也仍保留。源码摘要、脚本与证据范围见 [H100 验证记录](h100-validation.md)。

## 清单与未完成阶段

InferState 通用组装与 warmup 调度/资源归还已迁入 Runner。模型 metadata 和正常退出 IPC 引用问题的后续迁移见文末；当前完整回归结果见本文开头。启动多卡失败收敛、缓存策略和完整服务验收仍未完成。

`refactor-inventory.json` 固定当前 84 项 target/draft 注册：59 项保留，25 项为迁移后清退候选。静态依赖额外保护 internlm、internlm2、phi3、qwen、qwen_vl、vit、whisper 目录。尚未删除旧模型。

| 阶段 | 当前状态 | 剩余验收 |
| --- | --- | --- |
| 0：范围与基线 | 注册/依赖清单、小模型 logits 对照已落地 | 正式 checkpoint、服务与分布式能力基线 |
| 1：模型纵向链路 | 二十六个 ModelAdapter 实现类，含两类 Qwen3.5 草稿、Reward、InternVL、Tarsier 和 Omni Thinker；原生 attention、普通/BMM 与部分 FP8 Linear、普通/block128 FP8 MoE 及 Hopper FP8 EP 的启动绑定；bridge 已删除 | BF16 数值失败、其它模型适配、其余 Linear/MoE 量化与 EP 格式/设备、库内部峰值资源要求、正式模型任务验收 |
| 2：Runner/DeviceRuntime | 通用批次/InferState 组装、metadata、padding/输出裁剪、预热、Graph 与 route capture 提前绑定；限定 TP2/MTP 服务的正常和故障退出已实测 | 模型专有初始化、启动装配、缓存策略、完整路由服务、全峰值预算和完整分布式执行仍需收拢 |
| 3：StateStore/Transfer | Dense/ConvSSM/MLA/packed/window 原生视图、多状态绑定、CPU/PD CopyPlan、双完成点、局部失败延迟回收、PD staging 隔离 | 完整格式协商、持久缓存身份、全部 allocator 生命周期、State 内搬运实现分离、主动取消及 TP/peer 失败收敛 |
| 4：复杂语义与模式 | Dense full/sliding/sink、INT4/INT8、FA3 FP8、FlashInfer；GDN/MLA/sparse/KDA；Dense/GDN verify、Qwen3.5 草稿、Qwen VL 语言链路和 GPT-OSS 已接入 | 其它量化 MLA、V4、其它 MTP/草稿、完整 PD/多模态与 MoE/EP 矩阵迁移；vision encoder 与图片 HTTP 验收 |
| 5：清退与收敛 | Dense bridge 已删除 | 其它旧执行分支、模型注册清退与 CLI 迁移说明 |
| 6：性能 | 尚未开始 | 固定负载的 TTFT/TPOT/吞吐/显存比较 |


### 批次位置和模型元数据绑定

- 删除运行时 `InferStateInfo.init_some_extra_state(model)`，改为启动阶段 `bind_metadata(model)` 返回只接受当前 InferState 的函数。`init_runtime()` 在 autotune 前完成绑定；默认 `BatchProgram.initialize_state` 直接使用绑定函数。模型上的辅助入口委托同一函数，DSpark 的特殊 commit 路径也已接入。
- `execution/batch_metadata.py` 统一序列长度、累计长度与位置生成；`RotaryTables` 保留原生 cos/sin 张量，`RotaryMetadata` 处理普通/多组旋转表。Llama、Gemma3/4、Qwen 动态 NTK/logn、Qwen2/3-VL 三轴 MRoPE、NeoChat 时空位置分别绑定明确的表和静态参数，不在每批查询 model.config。图像位置与可见性仍按当前请求构造，Neo 的空图像边界直接在目标设备分配。
- V4 启动绑定实际拥有的层 ID、C4/C128 需求、两组旋转表、workspace 和 SWA 元数据函数。C128 容量通过 GraphRunner 的显式回调读取当前 Graph，允许 Graph 在元数据绑定后建立；DSpark 保留因果 commit 与整块可见 proposal 的区分。V4 的 compressor/compressed State、部分 SWA 和旧 backend workspace 路径仍有 manager 查询，本轮不能视为 V4 Compute 的完整原生迁移。
- 新增 34 项批次元数据检查：16 项 CPU 显式位置替身、16 项真实 CUDA 序列/旋转计算、1 项真实 Gemma4 图像前缀可见性、1 项绑定后安装 Graph 的容量检查。删除原模型并清空配置/替换表引用后仍可运行，且两批输出互不覆盖。CPU 替身检查不冒充 GPU 数学验收。
- 本地原批次/启动/输出/Qwen-VL 集合 92 passed；新增 metadata 与 Qwen-VL 集合 20 passed、17 skipped（本地无 CUDA）。H100 `metadata-focused.xml` 的 263 项全部通过，82.54 秒，无跳过；包含全部 112 项完整小模型，以及 V4 层子集/双 lane/Graph 更新、DSpark 提交与提议、图像缓存可见性和无 CUDA 标量读取检查。专项结果不与完整回归重复累加。

- 首次完整回归在资源启动测试夹具处停止：1 passed、15 failed。该夹具跳过模型构造，未提供新增启动绑定所需的 RoPE 表；补齐表并断言 warmup 前已绑定后，H100 全部 49 项启动检查通过。执行代码与 263 项专项快照相同，没有放松生产路径要求。

- 最终完整 `metadata-combined.xml` 为 **1770 passed、1 failed**，1771 项，304.62 秒，零 skip/deselect。集合为上一轮 1707 项加本轮 64 项；263 项专项及 49 项启动检查均已包含，不能累加。XML 核对 108 项模型文件、18 项 GLM（含 4 项完整小模型）、34 项新增 metadata、26 项 V4、49 项共享启动与 2 项网络驱动均通过，112 项完整随机小模型覆盖保持不变。
- 唯一失败仍为 Qwen3.5 BF16 对独立 FP32 参考，max RMS `0.017852840945124626 > 0.01`，与前轮完全一致；没有放宽阈值、skip 或 xfail。退出仍有 CUDA IPC producer 引用释放顺序警告，生命周期和正式任务/服务/多卡模型/性能验收继续列为未完成。
- 本轮 22 个 Python 文件通过 Black、Flake8、编译和 diff 检查；最终 GPU 快照之后只有两份设计文档变化。四轮日志/XML 已保存到 `/tmp/lightllm-a716ad58-h100-evidence/logs/metadata-*`。所有本任务临时容器退出，八张 H100 显存均为 0 MiB，既有 `lightllm-ci-agent` 未动；没有共享依赖安装、提交或部署。


### State 查询缓存与 IPC 引用生命周期

- 原 `AttentionStateManager.get_state/_get_states`、State CPU/PD 区域及 CPU/PD Operator 布局使用类级 `lru_cache`，缓存 key 强引用实例；manager 与三个 Operator 之间还存在双向强引用。查询结束后丢弃 manager 并不释放它及其张量，旧测试依赖全局 `cache_clear()` 清理。
- 新增实例所有的 `CachedQueries/cached_query`：key 不含实例，缓存随实例释放，`clear_query_cache()` 只失效当前资源。State 返回布局深拷贝的行为保留。复制和 pickle 排除本地查询缓存；manager 同时排除 Operator，并在导入后按新的 owner 重建，避免引用源进程的查询结果或不可序列化的本地函数。
- CPU/DP/PD Operator 使用弱引用指向 owner，manager 仍拥有这些辅助对象。搬运计划继续以绑定方法和原生 Tensor 保留 State；释放 manager 不会提前释放已提交计划使用的物理数据。这个边界不改变逐请求页回收策略，也不证明取消、peer 退出或跨 TP 故障已经收敛。
- 新增 6 项检查：5 项 CPU 覆盖关闭循环 GC 后即时释放、浅拷贝与 pickle 的资源身份、独立缓存失效、本地缓存不进入序列化；1 项 CUDA 在 manager 释放后执行真实 PD 打包/读回，核对数据和 TransferCompletion 所持张量的寿命。旧全局缓存清理夹具已移除，不能靠测试末尾强制清理掩盖引用问题。
- 原 H100 快照的分组诊断：GQA IPC 1 passed（15.07 秒），运行态 IPC 6 passed（54.69 秒）且复现 CUDA IPC producer 警告，网络驱动 2 passed（25.68 秒）无该警告。正确设置 CPU/PD 页参数后的本地回收检查在旧快照失败、修改后通过；最初夹具缺页参数的无效复现日志另行保留。
- 本地相关集合 149 passed、47 skipped（CUDA）；扩大集合发现 FP8 测试标定在 KV heads 少于 TP ranks 时重复过多，已在旧快照复现并按逻辑 head 数生成；meta 状态夹具也改为仅对 CUDA 申请 pinned memory。生产标定校验未放宽。
- H100 `query-focused.xml`：479 passed、2 failed，133.07 秒，无跳过。唯一两项失败为旧有界快照夹具使用 CPU 空 token 索引，现有搬运入口要求 CUDA 索引；修正两处后 `query-fixed.xml` 2 passed，3.06 秒。执行源码未改变，定向集合退出已无 CUDA IPC producer 警告。完整验收结果随后记录，不把多轮结果相加。
- 最终同进程 `query-final.xml`：**2147 passed、1 failed**，2148 项，325.72 秒，零 skip/deselect。合并前轮全部检查与新增状态/容量/传输集合；112 项完整随机小模型、6 项新引用生命周期、32 项运行态传输和 2 项网络驱动检查均通过。唯一失败仍为 Qwen3.5 BF16 对独立 FP32 参考，max RMS `0.017852840945124626 > 0.01`，没有放宽阈值或排除失败项。
- 完整进程退出后核对日志，未再出现 CUDA IPC producer 引用释放警告。这个结论限于已运行的 IPC 读写和正常退出；未启动读者、部分导入、主动取消、peer 异常退出及跨 TP 失败收敛仍未覆盖。正式模型任务、完整服务、多卡模型和性能验收也未完成。
- 完整运行复用修正快照 `915cf45253d5094c06ea08c9a00662e2d9e098588d416044570b888afaa1cbb2`（2113 文件），之后仅修改两份设计文档。31 个 Python 文件格式、Flake8、编译和 diff 检查通过。日志/XML 已取回本地证据目录；任务临时容器全部退出，八卡显存 0 MiB，既有 `lightllm-ci-agent` 未动。

### group128 FP8 FlashMLA 原生绑定

- 删除 V3.2 `fp8kv_dsa + flashmla_sparse` 的旧执行回退。Planner 在启动检查 standalone FlashMLA API、Hopper、group128/FP32 scale 字段、top-k 对齐与 head 数；prefill/decode 继续独立选择，复用原生 packed State、writer、indexer 和传输布局。
- prefill 以已绑定的 cache 和当前批次映射组合历史反量化 KV、本批未量化 KV，再调用库的 sparse kernel。复用既有 compact kernel，不反查 model/manager。动态 unique/压缩在 Runner 的 GraphProgram 步骤执行；模型明确声明输出规格，捕获时不再探测一个含动态压缩的函数。
- decode 在绑定时创建 `(token, 1, 1, 656)` 的零拷贝视图，保留 token stride，两个长度为 1 的维度使用 656 字节 stride。直接 `unsqueeze` 会保留包含 indexer 的 800 字节 stride，虽不改变可见元素却违反库的页内连续性校验；首轮 H100 36 passed/5 failed 均定位到此检查，修正后 43 项专项全部通过。
- 本地不足 64/128 的 query heads 补零到库 tile，输出裁回实际 head 数。metadata 按 backend、量化格式、补齐后的 head 数和 top-k 分组；每个批次/lane 单独拥有调度对象，decode Graph 保留已经捕获的调度 buffer。固定 top-k、动态 `-1` 索引表达有效长度，不传随长度变化的调度提示。
- 新增检查覆盖 16/64/128 heads 的 prefill/decode、历史前缀与本批精度、长短请求和行顺序切换、2048 top-k 长上下文 indexer 与 decode Graph、实际 Runner 的异构 head 分组及双 lane。新增三个完整小模型配置分别选择 Triton/FlashMLA、FlashMLA/Triton 和双 FlashMLA，禁止运行时 manager 查询，验证前缀、decode Graph 和分段 prefill Graph。
- GLM5 Next 的 group64/E8M0 + FlashMLA、V4、MTP 等仍需继续迁移。prefill 的动态压缩与库内临时分配尚未纳入完整峰值预算，也未做性能验收。
- 最终同进程 `flashmla-final.xml`：**2159 passed、1 failed**，2160 项，320.15 秒，零 skip/deselect。保持上轮完整集合，新增 9 项稀疏检查和 3 项完整模型；`test_model_cuda` 111 项、GLM 18 项（含 4 个完整模型）全部通过，完整随机小模型共 115 项。唯一失败仍为 Qwen3.5 BF16 RMS `0.017852840945124626 > 0.01`，阈值未改。完整退出无 CUDA IPC producer 警告；失败日志与所有专项结果保留，不累加通过数。
- 最终验证快照 `073a0ffa5761ef10a49505362a2899ecf8d734deaaff25af5ce0f6efaa59975d` 后只有两份设计文档变化。日志/XML 已归档；八卡显存 0 MiB，任务容器退出，原有 `lightllm-ci-agent` 未动；没有安装共享依赖、提交或部署。

### GLM5 Next group64/E8M0 FP8 FlashMLA

- 删除 pooled `fp8kv_dsa + flashmla_sparse` 的 legacy 回退。Planner 检查 SGL `only_qv`、standalone FlashMLA 的稀疏 decode API、设备与格式；原生执行绑定 584 字节 KV 视图，保留实际行 stride。decode 为 64/128 head tile 补齐查询并裁剪输出，调度对象按批次和执行通道独立管理。
- prefill 复用原有的直接读取、选中 KV gather 和通用分块 gather kernel；当前 token 保留 BF16，历史从 packed KV 读取。小批次 prefill 的固定归约缓冲、选中 KV 与元数据，以及小批次 decode Graph 的分段归约缓冲，均计入 offer 并在启动分配；并发通道不共享区间。通用 prefill 的物理行表、gather 与库内部临时张量仍动态分配，完整峰值预算和性能尚未验收。
- HOLD 请求的 CU_K 区间先填为保留的 HOLD 物理行，本批 HOLD KV 也映射到该行；最终 HOLD 输出清零。避免通用 gather 读取未填充索引，或 HOLD 的 fresh KV 覆盖有效请求的行表。
- `pooled-native.xml` 37 passed，24.96 秒：新增 FP8 prefix/decode Graph、长上下文 2048 top-k 与双通道 Graph，以及 mHC、有/无 MoE 的两个完整随机模型。`pooled-workspace.xml` 6 passed，6.31 秒：8/16/64 heads、1/3/4/5/12 query tokens、至少 65536 历史长度，覆盖直接读取/选中 KV/通用 gather、HOLD、预算不足时拒绝及固定 workspace 隔离。六项使用明确选中的索引与独立 Torch 数学参考，indexer 由前一组真实执行检查覆盖。
- 本地 10 passed、25 skipped，GPU 项由 H100 执行；五个 Python 文件通过 Black、Flake8、编译与 diff 检查。
- 最终同进程 `pooled-final.xml`：**2169 passed、1 failed**，2170 项，335.86 秒，零 skip/deselect。沿用上一轮完整清单，新增 4 个参数组合与 6 项 workspace 检查；`test_model_cuda` 111 项和 GLM 28 项全部通过，其中完整随机小模型为 111 + 6 = 117 项。唯一失败仍为 Qwen3.5 BF16 独立 FP32 参考 max RMS `0.017852840945124626 > 0.01`，未放宽或排除失败。没有 CUDA IPC producer 退出警告。
- 最终复用 `pooled-workspace` 快照 `d186801c0738d5693978fdaf1a32b71ed38bb1bfdea703c66f930f96cdcf9064`，执行源码未变；之后只修改两份设计文档。日志/XML 已归档，全部临时容器退出，八卡显存 0 MiB，原有 `lightllm-ci-agent` 未动。没有共享安装、提交或部署。正式任务精度、真实服务、多卡模型与性能仍待验收，完整七模块重构保持未完成。

本轮另外隔离诊断 Qwen3.5 BF16：仅把残差累加改为 FP32、仅把最终投影改为 FP32、两者同时修改，14 个位置最大 RMS 分别为 1.8521%、1.7776%、1.8450%，均未达到 1%。诊断只在独立脚本中运行，未合入这些精度改动；原失败及验收标准保留。


### Qwen3-MoE 与 SenseNova Loop 的原生模型入口

- 新增 `Qwen3MoEModelAdapter`，按 `mlp_only_layers` 和 `decoder_sparse_step` 声明实际专家层，保留显式 renormalize 选择与原默认值。模型 recipe 通过已绑定 `BoundMoe` 执行；沿用已有 Dense attention 和权重加载工具。兼容新版配置的 `num_local_experts`，避免把专家层误建为 Dense 层。
- 新增 `SenseNovaLoopModelAdapter`，四个物理层展开成六次执行时共享对应权重对象，加载和校验每个物理权重一次；每次执行仍有独立 KV/Conv/SSM State。`Planner.bind_moe()` 按执行 recipe 顺序匹配权重，允许同一物理专家权重绑定多个执行层，不能用物理层号替代执行层号。
- Qwen3.5/Loop 的显式 `rope_parameters` 优先于同一配置中序列化的旧默认字段。H100 首轮发现参考使用完整 RoPE、原实现却使用默认四分之一维度；修正配置归一化，并增加无输入修改与幂等检查，未放宽数值阈值。
- 完整小模型新增 Qwen3-MoE/混合 Dense 两种结构 × 三个 Dense backend，以及 Loop/二流 mHC/四流 mHC × 三种 Dense/GDN backend 组合 × FP16/BF16。独立参考用 Transformers 的 attention/MLP 和 Torch mHC 数学，按物理权重展开 reference；不使用会按物理层号共享状态的参考 cache。覆盖初始 prefill、decode、decode Graph、后缀与分段 prefill Graph，并禁止运行期 manager 查询和旧权重计算入口。
- `loop-fixed.xml` 专项为 **111 passed、5 failed**（116 项，无跳过），包括 Qwen3-MoE 六项、Loop FP16 九项全部通过，以及物理权重共享/执行状态隔离和 mHC kernel/Graph 检查。五个 BF16 失败分别为两个最大误差超限、三个 greedy token 不一致，不能视为 Loop 完成验收。本地 51 passed、32 skipped（无 CUDA）；11 个 Python 文件通过格式、Flake8、编译和 diff 检查。
- 隔离诊断只将同一份 BF16 权重的参考提升到 FP32，九个 BF16 用例为 6 passed、3 failed；剩余二流 mHC 失败均为最终候选同分造成 greedy 不一致，RMS/最大误差检查通过。进一步只将原生最终投影临时改为 FP32，结果为 7 passed、2 failed，FlashInfer/FlashQLA 二流与四流仍有 token 次序差异。两个实验都没有合入源码或修改验收条件，不能用诊断通过数替代正式专项结果。
- 新适配器的 EP、MTP、多模态、正式 checkpoint 任务精度、完整服务和性能均未验收。完整七模块目标和所有剩余迁移范围保持不变。

- 最终同进程 `loop-final.xml`：**2207 passed、6 failed**，2213 项，335.85 秒，零 skip/deselect。完整清单包含上轮全部 2170 项与新增 43 项；六项失败与上述专项和旧 checkpoint 失败一致，没有因诊断提升精度而替换正式实现或参考。新模型迁移不改变原 Qwen3.5 BF16 max RMS `0.017852840945124626 > 0.01`。
- XML 核对 `test_model_cuda` 130 passed、5 failed；GLM 28 项（含六个完整随机模型）、mHC kernel/Graph 九项、查询生命周期六项和网络驱动两项全部通过。因此完整随机模型为 136 passed、5 failed（141 项），不是正式 checkpoint 任务或多卡服务验收。完整退出无 CUDA IPC producer 警告。
- 日志/XML 已归档到本地证据目录，源码与 fixed 快照一致，仅两份设计文档后来更新。全部任务临时容器退出，八卡显存均为 0 MiB，仅原有 `lightllm-ci-agent` 在运行；未共享安装、提交或部署。


### Hopper FP8 专家并行的 Compute / Transfer / Runner 分离

- `ExpertComputeBackend` 按专家分片、BF16 输入、block128 FP8 权重和 FP32 scale 探测 Hopper 能力。`Planner.bind_experts()` 在启动绑定原生权重、路由参数、冗余专家映射、每个并发通道的 workspace 与 DeepEP buffer。计算对象不持有权重 loader 或通信 manager；路由复用 `MoeRouter`，共享专家仍由模型 recipe 计算。
- `ExpertTransfer` 管理 expanded prefill 与 low-latency decode 的 dispatch/combine、接收 hook 和两类完成点。`ExpertRunner` 安排路由、量化、通信、专家计算和合并；Qwen3-MoE、Qwen3Next、DeepSeek2、GLM5 Next 接入正常批次，Qwen3-MoE/DeepSeek 的双批次步骤也使用绑定执行。声明相同通信布局的层共用 Transfer，局部提交/接收/合并异常会隔离该 Transfer 并保留在途输入；这不是跨 rank 自动恢复。
- prefill 显式使用 Triton activation 量化，避免随 SGL 是否安装而改变 FP8 编码。decode 继续使用 DeepEP 融合量化。两个阶段共享权重表示，但量化的浮点运算顺序不同；独立 Torch 参考分别表达这些顺序，不放宽误差门槛。
- 双 H100 `ep-reference.xml` **52 passed**，54.40 秒，零 skip/deselect。其中两个真实双卡用例覆盖普通/幂次 FP32 scale、每 rank 不等 token 数、无本地命中专家、prefill/decode、Graph 输入更新、双批次和绑定后禁用 manager/替换 loader。其余为 14 项 EP CPU 契约/模型 recipe 检查，以及已有 MoE/activation 回归。独立数学门槛仍为 RMS <1.5%、最大误差 <参考峰值/32。
- 首次双批次测试未按实际 recipe 先消费 receive hook，再复用 low-latency dispatch，导致通信超时；已修正测试顺序。非幂次 decode 的原参考也没有表达 DeepEP 的乘法缩放顺序，按已安装源码校正后通过。全部初始失败、独立诊断和成功日志保留，详见 `h100-validation.md`。
- 当前证据是双卡专家子链路，尚非正式模型 EP checkpoint 或完整服务验收。冗余专家动态更新、空来源 rank、故障后的跨 rank 收敛、其它 EP 格式/硬件、全部路由记录、完整内存预算和性能仍需继续验证/迁移。本地 14 passed、2 skipped（无 CUDA），13 个 Python 文件通过 Black、Flake8、编译和 diff 检查。

- 同一源码快照的完整 `ep-full.xml`：**2233 passed、6 failed**，2239 项、350.69 秒，零 skip/deselect。沿用完整 2213 项，并增加 14 个 EP CPU 契约/recipe 检查和 12 个 DeepGEMM activation 组合；真实双卡 EP 的两个用例在具备 DeepEP 的独立镜像专项中执行。六个失败的用例身份与上一轮完全相同，Qwen3.5 BF16 max RMS 仍为 0.017852840945124626，没有放宽阈值或排除失败。专项与完整结果重叠，不累加。
- `test_model_cuda` 130 passed、5 failed，GLM 28 passed（含六个完整模型），完整随机模型仍为 136 passed、5 failed/141；正常退出无 CUDA IPC producer 警告。源码/测试与验证快照一致，随后仅更新两份设计文档。日志/XML 已归档，任务容器全部退出，八卡 0 MiB，仅原有 `lightllm-ci-agent`；没有共享安装、提交或部署。完整重构仍未完成。


## MoE 路由记录提前绑定

- `RouteCapture` 在启动时绑定 pinned CPU 存储、GPU 指针和执行层到记录槽位的映射；`BatchStateFactory` 为每批绑定 CUDA token 索引和记录回调。计算与旧 MoE 入口的 `get_moe_capture_callback()` 现在只读批次字段，不再查询全局 manager。绑定对象持有原生存储，回调不保留模型或批次对象。
- 修正启动顺序：路由 buffer 原先在模型 warmup / CUDA Graph 捕获之后初始化；现在在共享 KV 分配完成后、各模型 `init_runtime()` 之前初始化并绑定给主模型。非主 TP rank 和草稿模型不记录主模型路由。Graph 回放保持索引地址并更新内容；替换记录存储时拒绝复用旧图。
- Qwen3MoE、DeepSeek 的双 microbatch prefill/decode 补传记录回调；原生 `ExpertRunner` 和旧 DeepGEMM split API 都记录逻辑 expert ID，冗余专家物理映射不改变返回内容。普通 MoE/EP 路径继续消费同一批次契约。
- H100 `route-native.xml`、最终 `route-fixed.xml` 均 **77 passed**；后者 56.03 秒。验证包含真实双 rank EP/Graph/双 microbatch 的路由结果，uint8/int16 记录、无效 HOLD 槽位、非连续 routed 列、manager 释放后原生视图存活，以及 Graph 的 token 槽位更新。它仍不是 HTTP 服务的 routed_experts 端到端验收。
- 本地 **31 passed、15 skipped**（无 CUDA），本轮 15 个 Python 文件通过格式、Flake8、编译和 diff 检查。最终源码快照 `c1826ed1d93eeca0b14c59cce614ce69c7e946e641acde8279cc3ffc64ae4b0b`，2119 文件；完整回归使用同内容的独立只读目录。
- 两轮 BF16 隔离实验没有进入生产或正式测试：仅提高最终词表投影精度为 6 passed/3 failed；让 decode 舍入与 prefill 一致为 16 passed/6 failed，Qwen3.5 最大 RMS 升到 1.9415%。精度缺口仍保留，不通过改变参考或门槛消除失败。

- 最终 `route-final.xml`：**2258 passed、6 failed/2264**，347.75 秒，零 skip/deselect。原 2239 项清单增加 12 个路由绑定检查及 13 个既有 route manager 检查；49 个后端启动检查通过。六项失败与 `ep-full.xml` 的 classname/name 集合完全一致，Qwen3.5 BF16 max RMS 保持 0.017852840945124626，没有放宽或排除失败。
- 专项和完整 log/XML 已取回；完整退出无 CUDA IPC producer 警告。代码/测试与最终快照一致，仅两份设计文档后改。任务容器退出、八卡 0 MiB、既有 CI 容器保留；未安装共享依赖、提交或部署。其余模型/格式迁移、精度、真实服务、多卡完整模型与性能仍未完成。


## 真实服务验收与通信能力绑定

- Qwen3.5-0.8B FP16 单卡 HTTP 服务已完成实际生成、三请求并发、前缀复用、流式一致性、中断后的下一次请求、错误参数后的恢复；三个输入共 36 个生成 token 与独立 FP32 参考相同。CPU cache 模式经主动逐出后，指定请求的 GPU 命中 0、CPU 命中 224 token，恢复输出保持一致。它们是正式权重上的有限输入服务验证，不是模型任务准确率或全模式验收。
- TP=2 服务暴露 BF16 环境探测与模型 FP16 实际算子支持不一致。`SymmMemAllreduce` 已按真实 dtype、进程组和算法在启动探测并绑定；缺失能力关闭该候选，设备运行错误向上抛出。新增双卡 FP16/BF16/FP32 与 Graph 检查，连同启动回归共 19 passed。首轮测试在 NCCL 销毁时仍持有 Graph 而挂起，释放顺序修正后通过；完整失败证据保留。
- 具体源码快照、日志、检查范围见 [H100 验证记录](h100-validation.md)。完整回归仍为前一快照 2258 passed/6 failed，未声称本轮已全量重跑。服务退出尚有 semaphore 清理警告，完整重构、六项 BF16 精度、其余多卡模式及性能验证仍未完成。

- 修复后 TP=2 真实 HTTP 请求链路已通过：相同三组 prompt、并发、缓存复用、流式中断与错误请求恢复，36 个 token 与独立 FP32 参考一致。双 rank 的 FP16 two-shot 能力在启动被正确禁用，FlashInfer 保持启用。日志/JSON 已归档；任务容器退出、八张卡显存均已释放。
- TP=2 服务退出码 0，但仍有 `SharedMemory.__del__` 导出引用未释放的 `BufferError` 和 5 个 semaphore 清理告警。它们是新增确认的生命周期缺口，不能用请求成功或容器退出替代修复。没有提交或部署，本轮也没有重跑全部 2264 项。


## 后台任务关闭与退出资源持有关系

- `AtomicLockItem` 不再长期导出共享内存切片；原子操作结束即释放切片，锁仍持有底层存储。修复真实 HTTP 退出时资源锁的 BufferError。
- launcher 在 HTTP 排空后给内部 worker 发退出请求，先让 multiprocessing 执行资源 finalizer，再对超时进程强制回收。所测 TP=2 / CPU cache 服务已不再报告原有五个 semaphore 清理告警。异常和强制退出路径保留。
- HTTP 与 PD Master 通过 `HttpServerTasks` 拥有后台任务，关闭时设置状态并等待取消完成。解决真实服务在回收协程中等待、直到 launcher 强杀的问题。H100 容器整组 HTTP/语法/退出回归 **505 passed**；活动请求收到首 token 后触发退出，512 token 完整返回，HTTP 约五秒正常退出。
- **本阶段严格退出未通过：**当时模型后台推理线程和 NCCL 尚未协调停止，检查因 C++ 终止信息和未销毁通信组警告失败。后续修复与复验见下一节；原始失败日志继续保留，不能用 launcher 退出码 0 代替各工作进程正常回收。
- 其余模型/量化迁移、完整规划、跨进程故障收敛、六项 BF16 精度与完整多卡/性能验收仍未完成。无提交或部署。


## 推理线程、Graph 与通信组的退出顺序

- 模型进程关闭 RPC 入口后，等待初始化/RL 操作结束，再请求推理停止。正常 TP 在已有请求广播中传递停止标记，多节点 TP 通过原有 gather 传播，DP 在原有请求数量 gather 中统一停止决定；没有增加每批额外集合通信。两个推理线程完成最后一轮事件握手，避免遗留等待另一线程的 post-processing 阶段。
- 等待线程结束后同步设备，依次释放主/草稿模型的 prefill GraphProgram 和 decode Graph、FlashInfer/DeepEP/Mega MoE 通信存储，再显式销毁进程组。同步异常继续向上抛出，不能提前释放尚在执行的资源。模型加载失败时允许清理尚未完整装配的 backend。
- H100 第一轮相关回归 **90 passed**，覆盖线程停止、Graph 程序、真实双 rank collective 和后端启动。TP=2 + CPU cache 的 512-token 退出测试通过，原 NCCL/C++ 告警不再出现。DP=2 + prefill/decode Graph 的真实服务检查也通过，两个模型进程的 OS 退出码均为 0。
- TP=2 增加 prefill Graph 的严格检查发现父进程退出期限不足：两个模型进程退出码为 0，但 Router 在五秒期限被 SIGKILL，原失败保留。正常退出统一等待窗口调整为 30 秒，等待各层父子进程完成 finalizer；强制退出兜底保留。复验脚本进一步要求所有受管进程退出码为 0，结果见 H100 记录。
- 多节点真实服务、PD/EP/草稿/多模态整体退出与故障收敛尚未证明；CPU 传播测试和通用释放接口不替代这些验收。六项 BF16 数值失败和其余七模块迁移范围继续未完成。

- 最终复验：TP=2 + CPU cache + prefill/decode Graph 的 26 检查点通过，8 个受管进程实际退出码全部为 0；DP=2 + decode 双 microbatch + prefill Graph 的 16 检查点通过，7 个受管进程实际退出码全部为 0。两组 512-token 活动流完整返回，无强杀或退出告警；各自 36 个生成 token 与独立 FP32 参考一致，TP 组 CPU 恢复命中 224 token、GPU 命中 0。上述正常退出问题在这些模式下已修复，不扩展为全模式生命周期完成。源码快照和失败/成功记录见 [H100 验证记录](h100-validation.md)。

## Qwen VL 模型适配（2026-11-04）

- 新增 `Qwen2VLModelAdapter`、`Qwen3VLModelAdapter` 和 `Qwen3VLMoEModelAdapter`，各自拥有权重与计算 recipe，复用已有 Dense AttentionBackend；MoE 复用已有专家声明与 Compute。没有新增按模型名分支的 attention backend。
- VL 启动复用基础配置归一化，处理 flat/text_config、序列化 `rope_parameters` 和 MoE `num_local_experts`，保留原 VL 不填充文本模型滑窗默认值的行为。MRoPE/deepstack、VL 权重前缀和 packed expert 布局由已有 recipe/loader 保留。
- 新增完整 tiny VL 语言模型验证：CPU pinned 图像特征、两种网格/不同长度、图片中途分块与前缀续算、decode、decode Graph、prefill Graph 更新图片内容与文本；输出与独立 Transformers eager 数学对照，FP16 `rtol=1e-2, atol=4e-3` 且 greedy token 相同。禁用旧模型工厂、运行期 manager 查询、全局 embedding 客户端与旧 Linear/MoE 计算入口。
- H100 专项 `vl-adapters-final.xml` 86 passed、9.21 秒、零 skip/deselect；包括三模型 × 三后端的 9 项完整语言模型用例、配置、绑定、MRoPE/embedding/tokenizer 回归。本地相关 56 项通过。此处是注入图像特征的语言模型检查，未包含 vision tower、真实图片 HTTP、多卡或新 VL 模型的 BF16 验收。
- 初次 MoE 参考未遵循 `tie_word_embeddings`，修正为共享 embedding/head；第二次参考对未声明该字段的 Dense 配置直接取属性失败，按未声明即不共享处理后全过。测试阈值不变。完整回归使用同一生产/测试快照进行，结果补充在 H100 记录。

- 最终完整 `vl-adapters-full.xml`：2300 passed、6 failed/2306，344.12 秒（JUnit 344.106 秒），零 skip/deselect。上一轮 2288 项无遗漏，新增 18 项全通过；六个失败 classname/name 集合与 Qwen BF16 最大 RMS 均不变。生产和测试与 `e159ccaa5273...` 快照一致，之后仅改两份设计文档。H100 任务容器退出、八卡 0 MiB、既有 CI 保留，完整目标继续进行。

## Dense MTP verify 原生绑定（2026-11-04）

- Planner 显式选择 `verify` 阶段，模型 recipe 继续通过已绑定的执行对象调用。`ExecutionLimits.max_verify_tokens` 声明最大候选宽度，容量按展开后的 token 行数预留，不能沿用逻辑请求数。
- Triton 普通 Dense 复用已有 MTP shared-prefix stage1/stage2，顺序层共享预留 scratch、并发 lane 隔离；Graph 使用绑定容量选择 kernel 配置。滑窗保持逐行因果窗口。FA3/FlashInfer 将连续候选行压成 query 序列，HOLD 行逐行处理，零长度尾部使元数据形状固定。FlashInfer 的 verify 显式使用 paged prefill planning，在 replay 前更新 plan。
- 原生 INT4/INT8、FA3 静态 FP8 和 FlashInfer per-tensor FP8 也提供 verify offer。KV writer、量化参数和状态视图继续由启动绑定，计算不查询 manager。
- MTP 主模型只有全部声明 Dense attention 的已迁移 Adapter 才启用这条路径；GDN/KDA/MLA/sparse 的检查点或 indexer verify 绑定仍未完成，继续保留原执行路径。草稿模型仍未新增原生 Adapter；这不是解除全部 MTP 限制或完成 MTP 验收。
- H100 37 项完整模型专项首轮 35 passed/2 failed；扩展页式长序列和并发/Graph 后 71 passed/2 failed。FP16 固定/动态布局、滑窗、MoE、INT4/8、静态 FP8 通过；两个新增 BF16 用例在拒绝候选后的 suffix prefill 出现 greedy 不一致，原断言保留。
- 隔离诊断显示三种后端的续算 logits 与 fresh prefill 逐位相同，两失败在原生 BF16 中将候选 106/107 舍入成同分，而参考仍有差异。该输入未显示 KV 回退错误，但数值验收仍失败；不能据此忽略新增两项失败。完整回归和诊断数据见 H100 记录。
- 最终同一生产/测试快照 `784c6f86233c...` 的 `dense-verify-full.xml` 为 **2495 passed、8 failed/2503**，350.19 秒（JUnit 350.171 秒），零 skip/deselect。逐项对比上轮 2306 项无遗漏；扩展 197 项中仅上述两个新 BF16 用例失败，原六失败保留。完整目标仍进行中，尚未覆盖原生草稿/recurrent verify、完整 MTP 服务或性能。全部日志已归档，任务容器退出，八卡 0 MiB；此快照之后只改两份设计文档，未提交或部署。

## GDN MTP verify 与已接受状态恢复（2026-11-04）

- `FormatSpec.candidate_slots` 描述每请求 SSM 候选数量；`ConvSSMStateStore` 同时分配加宽 Conv 历史、SSM 候选和 `accepted_index`，三个字段直接绑定为原生视图。Planner 检查候选容量与执行容量一致，拒绝用普通 decode 操作多槽状态。元数据按游标存储身份隔离，不在运行查询 manager。
- 固定和动态候选统一按显式 `b_mtp_index` 分组，HOLD 行不参与 recurrent 更新，零长度尾部保持 Graph 形状。verify 调用现有 MTP Conv 和递推 kernel，读取上一批接受的位置并写出本批每步状态。
- 后缀 prefill 在批次准备时快照接受位置，各层读取相应 Conv 窗口和 SSM，完成后归一到基准槽。共享游标在某层归零不会改变其它层已准备的元数据。
- GDN 页缓存保存、DP 运行态复制、PD 页写出按已接受游标选择源状态；目标恢复到自身基准槽并归零游标，允许两端候选数量不同。传输不携带拒绝候选，原 source/destination 完成条件继续有效。
- 新增六种模型配置 × 三后端组合 × 固定/动态布局共 36 项 FP16 完整小模型检查，另有 8 项计算/SSM 精度组合及 9 项传输检查。覆盖部分接受、覆盖拒绝候选、decode/prefill Graph、接受后保存/清零/恢复。首轮测试仅恢复 recurrent 快照、漏掉 Graph 捕获改写的 Dense KV，造成重放失败；修正测试快照后，生产代码未再修改，专项 **64 passed、1 failed/65**（包含原 12 项 GDN 检查）。新增 BF16 计算 + BF16 SSM + Triton prefill 用例最大输出误差超限，保留原断言。
- 独立诊断还修复了 Graph 模块先于模型导入时的循环依赖：类型注解只在 TYPE_CHECKING 下导入模型类型。两个新进程导入检查和原 GraphRunner 检查本地通过；最终源码为 `f34226606ceb...`。
- 四组 BF16 诊断在相同状态下的 eager/Graph 输出逐位相同，失败组 RMS 与 greedy 通过，但最大误差 `0.01953125` 仍超限。该输入不是 Graph 重放差异，不降低验收门槛。首次完整 `179bf3d28dbd...` 为 2538 passed、9 failed/2547，原 2503 项无遗漏；最终源码的扩展完整回归结果在 H100 记录中补充。

- 最终 `gdn-verify-final-full.xml` 使用 `f34226606ceb...`：**2566 passed、9 failed/2575**，378.38 秒（JUnit 378.360 秒），零 skip/deselect；原 2503 项全部保留，新增 72 项中仅上述 BF16 SSM 用例失败。12 个 Python 文件格式/Flake8/AST 通过，之后只有两份设计文档修改。原始日志全部归档，任务容器退出、八卡 0 MiB，无提交或部署。完整目标仍未完成，下一阶段继续草稿 Adapter、KDA/MLA/sparse verify、其它模型/格式、规划/故障恢复和全模式验收。

## Qwen3.5 草稿 ModelAdapter 与扩展批次容量（2026-11-04）

`Qwen35MTPModelAdapter` 负责创建草稿权重和计算 recipe。checkpoint 使用 `mtp.layers.N` 的局部编号；执行层编号位于主模型与之前草稿层之后。`DenseModelAdapter` 允许明确声明执行编号，`layer_types` 等语义仍按模型内的位置解析。共享 embedding、输出头和主模型 norm 在创建 Adapter 时传入；Runner 按 recipe 与权重的对应关系装配静态参数，不把执行层号当列表下标。未迁移的 MoE 草稿子类保留原路径。

真实 TP2 MTP 服务暴露了额外容量需求：EAGLE 会先用完整 target verify 批次补写草稿 KV，然后按每请求一行生成后续候选。因此，启动预算按自身 decode 宽度与 target verify 宽度的最大值预留；草稿的 decode 语义和 Graph 对齐保持原有含义。三请求并发触发的原始容量异常保留在 `service-mtp-native-server.log`，不能用此前单请求成功作为服务通过证据。

专项 `draft-capacity.xml` 的 52 项通过（23.34 秒，无跳过/筛除），其中三项实际 Qwen3.5 草稿 checkpoint 分别覆盖 Triton、FA3、FlashInfer。参考独立使用 FP32 Torch 计算共享 norm、草稿 fusion 和 Transformers 单层模型；覆盖 prefix、普通 decode、decode Graph、suffix prefill、prefill Graph，以及两请求展开成六行的草稿补写。验证共享权重身份和主/草稿 KV 隔离；执行时禁止旧模型工厂、manager 查询和权重 loader 的计算入口。本地相关 74 项通过。

首次服务在并发错误后超时，退出还出现共享内存 `BufferError`，宽限期后强制清理导致 launcher exit -9，三个进程没有正常退出记录；异常传播和失败退出仍属于 C 类未完成项。完整服务与合并回归以随后记录的实际结果为准，不将专项结果累加为全量结果。

容量修复后的固定与动态 TP2 MTP 服务分别取得 30 个检查点通过，含 CPU 恢复（GPU=0、CPU=224）、并发、取消、独立 FP32 的 36 token 一致性和 512 token 退出流。两组 launcher 与 8 个受管工作进程实际退出码均为 0。动态模式确有压缩验证及候选接受，但多数步骤只验证一行，尚无性能收益结论。详见 [H100 原始证据](h100-validation.md)。

最终合并 `draft-capacity-full.xml` 为 2572 passed、9 failed / 2581，396.56 秒，无 skip/deselect；原 2575 项全保留，新增六项全过，九个数值失败身份不变。生产代码/测试匹配 `eacee72374e80cefd5f62d1d8206a5d1107c3762aa1ffba69a390ddf8126fd89` 快照。H100 临时任务均终结、八卡显存释放。完整 A–E 范围继续，下一步需收敛实际暴露的线程异常传播/退出问题及数值失败，并继续其余模型、量化、规划和传输迁移。

## 推理线程异常与服务故障收敛（2026-11-04）

上一轮真实 MTP 容量错误暴露了线程异常只写日志、模型进程仍存活的问题。`ModelRpcServer` 的进程入口现在在启动后台线程前安装仓库现有的致命线程异常处理：记录原始异常、以非零状态退出模型进程，交由 launcher 的进程监督结束故障服务。错误上下文不执行普通 Graph/collective 析构，避免在另一线程或 TP rank 已无法配合时无限等待。正常退出仍走原有的同步、Graph 与通信组释放路径。

首轮 H100 故障注入确认请求约 2 秒失败、launcher 约 4 秒退出，GPU 与模型工作进程已释放，但遗留五个 multiprocessing 信号量，未通过验收。故障清理随后改为先向 HTTP 管理进程发 SIGTERM，最多等待 10 秒，超时才强制终止；无论 HTTP 是否自行退出，其余服务进程仍会被清理。这样保留 HTTP 自身回收工作进程和信号量的机会。

相关 H100 生命周期检查 44 项通过；本地最初 41 通过、1 项受 Unix socket 沙箱权限限制，已由 H100 补验；后续本地 shutdown/supervisor 检查 27 项通过。本轮没有更改计算代码或重跑 2581 项数值回归，原九项 BF16 失败仍待解决。真实故障最终结果、异步 IPC 清理等待与剩余析构告警详见 [H100 验证记录](h100-validation.md)。本次不代表全部取消、PD/EP、多节点、GPU 致命错误或完整 C 类要求完成。

最终 `9730ed022321...` 快照另修复端口预检查误拒 TIME_WAIT：使用与 HTTP listener 一致的 SO_REUSEADDR，活动监听端口仍拒绝。H100 生命周期与真实 TCP 检查 **47 项全过**。TP2 分别在 rank 0、rank 1 注入草稿元数据异常，三个并发请求均约 5.2 秒返回 HTTP 500，launcher 分别在 11.480/11.429 秒退出 1；进程、GPU、POSIX（含信号量）、SysV 和 IPC socket 全部无残留。故障组存在强制结束的工作进程，日志仍有 BufferError 与 semaphore 告警，不能称正常退出或无告警。

两次故障后立即使用同端口启动的完整 TP2 MTP 文本服务 **30 检查点全过**：主/草稿原生绑定、并发、取消及非法请求后恢复、CPU 恢复 224 token（GPU=0）、36 个 token 与独立 FP32 参考一致，退出时完整返回 512-token 活动流。正常重启服务的 launcher 与 8 个工作进程实际退出码全为 0，无退出告警。证据已归档；任务容器退出、八卡显存释放。下一步继续析构告警、九项数值失败及完整 A–E 剩余迁移，未提交或部署。

## 故障退出时的本地视图与信号量所有权

在相同生产快照上增加隔离诊断后，确认 BufferError 来自 HTTP worker 的 `req_shm_total`。五个未执行 finalizer 的信号量分别属于 Router 的 PD 消息队列（三个）与两个模型进程的 tqdm 进度锁。诊断仍保留原异常和全部资源断言，实际故障回收通过，告警复现。

HTTP manager 现在先等后台任务停止，再分离请求数组、清除本进程的请求视图，最后关闭 ShmReqManager 的本地映射；不归还共享槽位、不修改跨进程引用计数、不 unlink 其他进程可能仍在使用的资源。PD Master 的关闭入口保持等待后台任务结束。新检查使用真实共享内存的独立映射验证：HTTP 关闭后，另一映射仍可读写，租约和分配位保持原值；有未释放的导出请求视图时，关闭必须显式失败。

普通模式不再创建未使用的 PD 消息队列。权重加载改用 ThreadPoolExecutor 与进程内线程进度锁，保留乱序任务完成、线程内 CUDA device 选择和加载异常传播，避免使用进程信号量。首个加载检查还发现旧 multiprocessing.ThreadPool 自带临时信号量，因此一并替换；修复后正常加载与损坏 shard 两项本地检查通过。初次本地 HTTP/共享内存集合 20 项通过、加载检查 1 项失败，修复后 21 项通过；随后增加了加载异常参数。完整 HTTP 本地收集缺 xgrammar，交由 H100 执行，不通过跳过依赖用例宣称完成。最终 H100 结果见验证记录；全部 A–E 剩余范围不变。

首轮 H100 340 项相关回归全过，但真实故障未通过新增无析构告警断言。进一步诊断确认：已取消 Task 的原始 CancelledError 仍保留两个后台协程的请求视图，warmup 还另建了两个 tqdm 信号量。现已在后台任务结束后清理取消 traceback 的 frame 局部引用，并让加载、warmup 共用线程进度锁。本地保留已取消 Task 的真实共享视图复现先失败，修复后相关 36 项通过；原 H100 失败与两轮诊断全部保留，后续严格复验不放宽条件。

最终还补齐 Python 3.10 取消异常的 context 链，以及 CPU KV 注册、autotuner、V4 MTP/DSpark 的进程内进度锁。H100 **354 项回归全过**，两个 TP rank 分别注入异常后均通过严格验证：请求约 5 秒返回错误，launcher 在 15 秒内失败退出；本服务新增资源零残留，每次 15 个信号量全部正常清理，无 BufferError/semaphore 析构告警。故障模型退出 1，其他内部工作进程仍有强杀和故障 peer 通信信息，不扩展为正常退出或完整分布式故障验收。

同端口重启后的 **30 个真实服务检查点通过**，含独立 FP32 的 36-token 一致性、CPU 恢复、三并发、取消恢复和 512-token 退出流；launcher 与 8 个工作进程退出码全为 0，无退出告警。详细快照、四次失败、诊断和最终证据见 [H100 验证记录](h100-validation.md)。完整数值回归继续保留原集合并新增 44 项检查，未修改或排除九项 BF16 失败；完整 A–E 目标继续进行。

最终完整 `worker-failure-final-full.xml` 为 **2616 passed、9 failed / 2625**，396.96 秒，零 error/skip/deselect。原 2581 项全保留，新增两项加载与 42 项 autotuner 检查全过；原九个 BF16 失败身份和 Qwen 最大 RMS 不变。完整退出无 BufferError、信号量泄漏或 CUDA IPC producer 告警。源码为 `b06f455707a5...`，与服务验证版本所有 Python AST 相同；之后仅更新文档，十六个 Python 文件格式/flake 检查通过。原始证据已归档，任务容器及显存已释放；下一步继续数值缺口和完整 A–E 剩余迁移，未提交或部署。

同端口恢复此前发现 `validate_ports` 未设置 SO_REUSEADDR，误把已关闭连接的 TIME_WAIT 当作监听服务。现已与 HTTP listener 的 bind 选项对齐；真实 TCP 回归同时要求活动监听端口继续被拒绝。该修复没有更换端口或靠等待 TIME_WAIT 绕开重启问题。此前 HTTP ctypes/共享内存析构告警已在上述限定 TP2 故障场景复验消失，其它模式和跨节点故障收敛仍未完成。

## BF16 数值基线的进一步区分（2026-11-04）

当前源码保持 `b06f455707a5...`。H100 上追加独立观测：同一 checkpoint、相同 14 个输入位置，Transformers 自身 BF16 对原 FP32 参考的最大 RMS 为 **2.4124%**，当前原生为 **1.7853%**；两者 greedy 均 14/14 一致。18 个 GDN norm 权重在 BF16 加载时会舍入；隔离实验仅让原生保留这些权重的 FP32 值，平均 RMS 降低但最大 RMS 升至 **1.9744%**，没有解决原检查，未合入。

两个实验均真实退出 1，日志、XML、JSON 与脚本已归档，原参考、输入和门槛未改。该证据说明需要区分低精度计算本身与重构回归，不能把超出 1% 直接认定为新架构错误，也不能据此宣称精度合格。最近完整回归仍为 **2616 passed、9 failed / 2625**；此次未修改生产代码或正式测试，全部 A–E 剩余迁移和验收继续进行。详细数值与复现信息见 [H100 验证记录](h100-validation.md)。

## GPT-OSS 的模型语义与原生专家视图（2026-11-04）

`GptOssModelAdapter` 声明交替 full/sliding attention、每个 query head 的 sink，以及 top-k 选择后按输入 dtype 做 softmax 的路由。Dense 计算的静态参数在启动时接收 sink；普通与静态 FP8 FA3 消费该视图，未实现 sink 的 backend 在 probe 拒绝。GPT-OSS 的 attention recipe 复用已绑定的 Dense 路径，FFN 直接调用 `BoundMoe`，不查询 State manager 或调用权重 loader 执行。

MoE 的 `FormatSpec` 增加交错 gate/up 布局和显式 `gate_up_bias`、`down_bias` 字段；Compute 绑定这些张量及 alpha=1.702、clamp=7、up+1 的激活语义。Planner 通过 loader 提供的 `native_moe_view()` 获取格式和视图，不按 GPT-OSS 模型名选择计算。GPT-OSS loader 只负责普通专家张量或 MXFP4/E8M0 checkpoint 的解码、TP 切分和 bias；down bias 的 TP 分摊在加载时完成，解码临时索引按 32768 行分块。blocks/scales 可从不同文件并发到达，完整加载后才允许绑定。这是加载时解码到计算 dtype，不是原生 packed FP4 专家 kernel。

YaRN 保留 GPT-OSS 的 `truncate=False`，并读取 beta 和 attention factor；可选空值仍使用默认值。完整随机模型覆盖 FP16/BF16、普通/MXFP4 权重、默认 RoPE/YaRN、prefill、decode、前缀与两类 Graph，拒绝旧工厂、执行时 manager 查询及量化 loader 的计算入口。CPU 检查覆盖 TP1/2/4 的字节映射和 bias 分摊，不能替代多 GPU GPT-OSS 模型验证。

首轮专项在两个 GPT-OSS BF16 数值失败之后出现 Graph capture 错误；相同源码的独立 MoE 31 项全过。小模型测试现在在所有退出路径同步并释放 Graph；同进程修订专项 203 passed、2 failed / 205，最终补两个 YaRN 可选参数对照后为 **205 passed、2 failed / 207**，没有再出现 capture 污染。独立 sink 检查的普通/FP8 KV、full/window、prefill/decode 和 Graph 输入更新八项全过。八个完整小模型有六个通过；默认 RoPE 的 BF16 普通权重后缀 greedy 不一致，BF16 MXFP4 权重 decode 最大误差超限。

最终完整回归 **2639 passed、11 failed / 2650**，404.02 秒；原用例逐一保留，原九项和新增两项失败身份均不变。代码与测试和最终快照一致，之后仅更新文档。全部验证容器退出，八卡显存为 0 MiB，既有 CI 保留。原始 XML/log 已本地归档；正式 GPT-OSS checkpoint、服务、多卡、任务精度、性能以及完整 A–E 范围继续推进。

## GPT-OSS 中间精度契约（2026-11-04）

逐层同输入对照确认，两项 GPT-OSS BF16 失败与专家计算中的舍入边界有关；观测到的 norm 与参考逐值一致。仅在隔离实验中替换专家计算后两项原测试通过。原融合实现与固定路由 FP32 数学对照的距离更小，但与模型的分步 BF16 行为不同，不能用这一点替代既有数值验收。

`MoeSpec.round_intermediates` 现在允许模型声明投影、bias 和激活运算之间必须保留的输入 dtype 舍入；GPT-OSS 显式启用，Planner 绑定后直接交给 Compute。grouped GEMM 与激活 kernel 保持融合，只在所需位置保留转换。其它模型维持默认行为；尚不支持该要求的 EP backend 在 probe 拒绝，不静默忽略精度声明。

H100 首次修复专项 **211 项全过**，包含原来失败的两项及新增 top-k=4 的四种 dtype/权重组合，原输入、参考和门槛保留。同一 kernel 还修复了 FP32 中间结果误转 FP16、造成合法大数输出 `inf` 的问题；新增两个精确值与 Graph 更新检查后，最终候选 771 的专项 **213 项全过**。

最终完整回归 **2647 passed、9 failed / 2656**，402.72 秒；按用例身份及次数确认原 2650 项全部保留，GPT-OSS 两项已通过，剩余失败与原九项完全相同。代码和测试与最终快照一致，之后只更新文档；全部任务容器退出、八卡显存释放，原始证据已归档。正式 GPT-OSS checkpoint、多卡服务和性能，以及其它模型、量化、预算与传输迁移范围不因此关闭。

## Qwen3.5-MoE 主模型与草稿 Adapter（2026-11-04）

`Qwen35MoEModelAdapter` 独立选择 packed gate/up、down 专家权重 loader，继承 Qwen3.5 的混合 Dense/GDN 语义与模型 recipe。`Qwen35MoeMTPModelAdapter` 声明草稿专家，并保留 checkpoint 层号与全局执行层号的区分；embedding、输出头和主模型 norm 在启动时共享。两者都由 Planner 绑定原生 Attention、Linear 和 MoE，不为模型名称新增计算分支。

主模型使用独立 Transformers 小模型及原始 packed 权重，覆盖三个 backend 组合、FP16/BF16、固定/动态 MTP verify、前缀和 Graph。`qwen35-moe.xml` 为 **78 passed、3 failed / 81**，新增 FP16 和 verify 用例通过，三个 BF16 用例在同一 decode 位置超出原 RMS 门槛。原参考、输入和断言不变，具体定位见 H100 记录。

草稿使用随机 packed MoE checkpoint 和独立 FP32 草稿计算参考，三个后端验证共享权重、主/草稿 KV 隔离、前缀、decode/prefill Graph、两请求展开六行的补写。执行时禁止旧模型工厂、manager 查询以及 loader 计算入口。包含原有 Qwen3.5 正式草稿 checkpoint 三项回归与 CPU 绑定检查的 `qwen35-moe-draft.xml` **35 项全过**，25.28 秒；本地绑定 29 项通过。该 MoE 草稿结果不是正式 MoE checkpoint、端到端 MTP 服务、多卡或视觉验收。

最终候选源码/测试 `dbd426326a38aebb2e95f0b735f83195cb8cf0c3cafb33e6fab9c38c6f093988`，2134 文件；五个 Python 文件通过 Black/flake，AST 与 diff 检查通过。没有把专项相加成新的完整回归结果；此前完整 2678 项及十项 BF16 失败仍保留，完整 A–E 范围继续。

同状态诊断确认新增 BF16 用例的 eager/Graph logits 最大差为零；最后一层第二请求的专家选择与参考不同，原生 `[3,1]`、参考 `[3,0]`，前三层 top-k 一致。上游数值偏差和精度契约仍待定位，原失败保留，不修改阈值。

## Qwen3.5-MoE 中间精度契约（2026-11-04）

同输入 MoE 对照观察到约 0.4%–0.52% 的 RMS 差异，进一步沿参考运算顺序隔离：路由权重需按输入 dtype 舍入，专家 down projection 在乘路由权重前舍入，routed 与 shared 专家分别形成舍入后的小计再相加。单独修复任意一处没有通过原完整小模型检查；同时恢复三处边界，以及独立 Torch MoE 替换实验均通过。原输入和断言保持，不以提高累积精度替代模型规定的中间表示。

`Qwen35MoEModelAdapter`（包括复用其专家声明的草稿）现在显式启用 `MoeSpec.round_intermediates`。Compute 按声明保存路由权重 dtype，复用已有 projection 舍入，并将 routed/shared 的求和边界传入融合 kernel；仍由已绑定的原生视图执行。普通模型默认路径保持原行为。该轮 EP 仍由 probe 拒绝此精度契约；后续支持与验证见下，不能用拒绝代替模式迁移完成。

初始候选 `10c5f42fd9b5...` 的 H100 专项 **126 项全过**，包含此前三个 BF16 失败、全部 12 个主模型参数、六个草稿参数、GPT-OSS、原有专家算子及新增四个可精确判定的舍入/Graph 检查。最终候选 `c40409060065...` 又加入 FP16/BF16 × 普通/FP8 的四项独立参考检查，四项通过。最终完整回归结果见 H100 记录；上述专项不累加为完整结果。

最终完整 H100 回归 **2695 passed、10 failed / 2705**，469.46 秒，零 error/skip/deselect。按用例身份及次数核实，上一轮 2678 项全保留，新增 27 项全部通过；三个原 MoE BF16 失败在相同输入和门槛下通过，十项既有失败集合不变。全部任务容器退出、八卡显存释放，日志无资源析构/CUDA error 告警，原始证据已归档。源码/测试匹配 c4040906，之后仅文档更新。EP 精度支持、原十项数值失败和完整 A–E 范围继续，未提交或部署。

## EP 的模型中间精度契约

Hopper BF16/block128 FP8 EP 现在支持模型声明的 `round_intermediates`。路由概率先按 BF16 舍入，再以 DeepEP 要求的 FP32 形式传递；masked/expanded 专家激活保留同样的舍入步骤。prefill 的分块计算按 top-k 槽保存已加权的 BF16 贡献，额外空间计入已有 workspace 的分块容量探测；decode 逐槽执行加权 combine。Transfer 分别搬回各槽，Compute 用 FP32 累加后得到 BF16 routed 小计；共享专家仍由模型 recipe 处理。

Planner 的传输绑定区分是否保留独立贡献，Runner 检查两侧约定一致，并在最终求和后记录就绪事件。多次 combine 使用 CUDA stream 依赖后才复用通信槽；中途异常保留已完成输出、在途权重和源资源，隔离后续提交。默认模式保持原通信方式。保留舍入的路径需要 top-k 次 combine，尚未测量吞吐和延迟，不声称已有性能收益。

H100 专项 `ep-rounding-fixed.xml` **90 passed / 90**，99.17 秒，无 skip/deselect：包含原双卡路径、两种 FP8 scale 策略的新 top-k=4 路径、跨 rank 精确值检查、空专家拥有者、双 microbatch、更新输入/路由后的 decode Graph，以及 12 项新激活舍入检查。首次运行 88 通过、2 失败来自新夹具的规格 top-k=4 与权重声明 top-k=2 不一致，失败发生在 Planner，修正夹具后执行代码未变。源码/测试快照 `7d13b0eceb3b5bd3c4b9938b60716fee799d89f94540152d7771ea635b4896df`。最终完整回归 **2709 passed、10 failed / 2719**，465.33 秒，原 2705 项无遗漏，新增 14 项全过，十项失败身份不变。原始证据已归档；本轮任务容器退出、八卡显存释放，日志无资源析构/CUDA error 告警。此结果不代表正式 Qwen3.5-MoE EP checkpoint、完整 EP 服务、跨节点或性能验收；原十项 BF16 失败及全部 A–E 剩余范围保留。

## Qwen3 RoPE 的乘积舍入

Dense MTP 的两个 BF16 greedy 失败此前已排除 accepted-prefix 缓存恢复差异：重新计算完整前缀也得到相同 logits。本轮按原输入隔离 RoPE、普通/QK norm、激活、projection 和 prefill attention。单独替换 norm、激活或 projection 时结果未变；RoPE 按输入 dtype 分别舍入两个乘积后再相加，三种 backend 的原验证均通过。改变 attention 也会改变这一近似平局，因此不把该观测扩大为所有 BF16 偏差只有同一原因。

Qwen3 的计算 recipe 现在显式要求 `rotary_emb_fwd(..., round_intermediates=True)`。融合 kernel 在乘法和加减之间执行明确的 FP32→输入 dtype→FP32 转换，并禁止这一路的编译器浮点融合；仍只执行同一个 RoPE kernel。其它 recipe 的默认参数保持原有行为。新增 12 项 FP16/BF16、64/128 head dim、部分旋转、非连续视图、可选 K 与 Graph 输入更新检查，要求结果与独立 Torch 逐值相同，并先确认测试输入能够区分原融合计算与分步舍入。

H100 `dense-rope-precision-focused.xml` **15 passed / 15**，10.67 秒，无 skip/deselect，包含原来失败的 Triton/FA3 和已通过的 FlashInfer BF16 MTP 用例。原输入、参考、RMS/最大误差/greedy 断言全部保留。前两个候选分别暴露 Triton helper 返回 dtype 不一致、以及单纯转换仍未保住舍入边界，失败日志已归档。当前源码快照 `b2a259c52fb34c4fd6ffe3940f21d6b6a6720828ffaec2bbbbf9073f03f0f143`。最终完整回归 **2723 passed、8 failed / 2731**，463.45 秒；原 2719 项全部保留，新增 12 项全过，恰好消除原 Triton/FA3 Dense MTP 两个失败，其余八个身份不变。完整退出无资源析构/CUDA error 告警，任务容器退出、八卡显存释放。源码/测试匹配上述快照，之后仅更新文档，未提交或部署。此结果不代表整个 BF16 或完整 A–E 重构验收。


## Reward 与视觉包装模型的原生装配（2026-11-04）

新增八个独立 Adapter：Qwen2Reward、InternVL 的 Llama/Qwen2/DeepSeek2、Tarsier2 的 Llama/Qwen2/Qwen2-VL，以及 Qwen3 Omni MoE Thinker。各自选择 checkpoint 权重、预处理和计算 recipe，复用既有 Dense/MLA AttentionBackend 与 Linear/MoE 绑定。包装模型启动统一通过模型的配置规范化方法读取 `llm_config`、`text_config`、`thinker_config.text_config`，不再重复处理字段别名，保留 RoPE 和专家配置。

Reward 评分头原先使用已失效的 ROWMMWeight 构造参数。现在根据 checkpoint 的两个矩阵和偏置确定实际维度，允许四个字段跨 shard、由加载线程乱序收集；完整且兼容后建立两个独立、TP 复制的原生 Linear 绑定。缺字段或维度不匹配在启动拒绝，不分配错误的评分头。只创建 embedding、最终 norm 和评分头，不创建无用的语言模型输出头。

Omni 的 packed 专家拆分改用专家权重声明的实际前缀，处理 `thinker.model.layers.N.mlp.experts`；混合 Dense 层也声明 thinker 前缀。计算后端没有增加按模型名称选择的分支。此次检查覆盖 packed 专家和包含 Dense 层的 Omni recipe。

静态清单复核：原 84 项注册及 59 retain / 25 清退候选不变；保留主模型 27/29 个注册显式接入，对应 24 类，加两类已接入草稿共 26 类。剩余主模型入口为 DeepSeek V4 普通/视觉两项；草稿仍 10/30 项接入，其余 20 项及所有原 A–E 验收范围不变。入口声明不是完整模型、全模式或性能验收。

新检查用独立 Transformers 语言模型与 Torch 评分头，对照分 shard 实际加载、两请求图片/音频原生特征、chunked prefill、前缀复用、decode、decode Graph、更新输入和特征后的 prefill Graph。旧模型工厂、NoQuantization.apply、weight-loader MoE 和运行时 State 查询在测试中禁止调用。模型使用 FP16；此次没有改变原 BF16 数值测试，也没有运行视觉/音频编码器或正式图像 HTTP。

H100 初次 `model-wrappers-focused.xml` 67 passed、3 failed / 70，19.42 秒；三项 Omni 在启动缺专家权重，暴露 packed 前缀问题。修复后 `model-wrappers-fixed-focused.xml` 72 passed、6 failed / 78，20.72 秒；Omni 全部数值对照通过，但测试的 prefill Graph 容量 11 小于含音频的 13-token 输入，未触发捕获，Graph 断言正确失败。夹具容量改为 16，输入、参考和 Graph 必须存在的断言保留。上述失败版本和原日志均保留。第三轮 `model-wrappers-graph-focused.xml` 为 51 passed、27 failed / 78，20.78 秒；测试直接调用 `_context_forward` 时 11/13 token 未填充到 12/16，触发 GraphRunner 的容量保护。夹具改走公开 `model.forward → BatchRunner`，保留独立 padding 请求槽、输出裁剪与 Graph 断言，不修改生产容量保护。最终候选 `29f024e6e8510d01a9b43bacb3296c5846caf956a94cf70f6f87d3e4d08a3a43` 的 H100 专项 **78 passed / 78**，22.85 秒，零 error/skip/deselect。含 40 个新 wrapper 参数、29 个原绑定检查和 9 个原 VL 模型检查，不与完整回归累加。Graph padding 使用独立 HOLD 请求槽，结果经 BatchRunner 裁剪；该版本的完整回归 `model-wrappers-runner-full.xml` **2762 passed、9 failed / 2771**，476.65 秒，零 error/skip/deselect；40 个新增检查全过，出现原配置检查的一项新失败：InternVL-Qwen2 的父类配置初始化依赖 max_total_token_num。已经让 InternVL-Qwen2 和 Tarsier-Qwen2 调用公共基类配置读取，保持 CPU/GPU 使用同一 normalize 结果；新 wrapper 配置检查同时加强为无运行时容量依赖、与类接口结果相等。原配置测试输入和断言未改，修复后本地 34 项配置检查通过（7.16 秒）。新快照 `6f9bb3ae3d713910fe605aaedbd3769157c91f3f577f9703cf3a16abde591b69` 的 H100 专项 **103 passed / 103**，20.66 秒，零 error/skip/deselect；比前次专项增加 25 个原配置检查，先前失败项已通过。同快照完整回归已完成，结果见下。


最终 `model-wrappers-normalized-full.xml` **2763 passed、8 failed / 2771**，471.49 秒（JUnit 471.464），零 error/skip/deselect，退出 1。`/tmp/lightllm-a716ad58-audit-wrappers.py` 按用例身份与次数确认旧 2731 项无遗漏、40 新增全过、八个失败与上一完整基线相同；CPU/GPU 配置一致性的新增回归已关闭。103 项专项不与完整结果累加。完整及最终专项日志无资源析构或 CUDA error 告警，46 个第三方/空 Graph 子段等 warning 保留。

完整证据 `/tmp/lightllm-a716ad58-model-wrappers-normalized-full-evidence.tar.gz` 已归档并解包；源码/测试匹配 6f9bb3ae3d71，之后仅两份文档变动。所有任务容器已退出，八卡显存均为 0 MiB，原 CI 保留。没有提交或部署。当前主模型装配入口剩 V4 普通/视觉，草稿仍有 20 项未迁；八项 BF16 数值失败、其余量化/EP 模式、完整资源规划、生命周期/失败收敛、正式多模态服务、任务精度与性能，以及完整 A–E 目标继续进行。


## Qwen3-MoE 无 Attention 草稿装配（2026-11-04）

新增 `Qwen3MoEMTPModelAdapter`，基于可执行 recipe 声明，Attention 与 State 均为空。共享主模型 embedding、lm_head 和 final_norm，权重仍按原 checkpoint 层号加载。每个草稿的全局执行层号由主模型及前序草稿深度确定；Dense/MoE 选择使用 checkpoint 局部坐标，再映射到全局执行 ID。旧实现直接按全局层号选择 FFN，会在混合 Dense/MoE 或间隔专家层时与加载的权重不一致。主模型与草稿共用 `qwen3_moe_specs`，Planner 绑定原生 Linear/MoE，草稿不创建 Attention Runner 或额外 KV/bounded 状态。

原 84 项注册与 inventory 逐项一致，59 retain / 25 清退候选保持。主模型仍 27/29 个入口、24 类；草稿从 10/30 增至 12/30 个入口、3 类，共 27 个 Adapter。剩余主入口是 V4 普通/视觉，草稿仍有 18 项注册。此处是声明覆盖，不是全模式验收。

新增 `test_qwen3_moe_draft.py`：4 个无 CUDA 的声明/层映射检查；6 个 H100 参数覆盖 vanilla_no_att/eagle_no_att × 全 MoE/显式 Dense 混合/按间隔选择专家。使用真实随机 FP16 safetensors，vanilla 连续装配两个双层草稿，Eagle 装配一个双层草稿；独立 FP32 Transformers/Torch recipe 对照检查相对 RMS ≤1%、最大误差 ≤峰值/32、greedy 相同。验证共享对象、局部/全局层号、原生绑定、两请求 prefill/decode、两类 Graph、变更 hidden 后回放及主 KV 不变，并禁止旧 model factory、loader compute/MoE 和运行时 manager 查询。没有运行正式 Qwen3-MoE 草稿 checkpoint、HTTP、多卡、BF16 或性能。

本地新 4 项加原 no-attention 6 项共 **10 passed**，5.28 秒。初始 H100 `qwen3-noatt-focused.xml` **22 passed、6 failed / 28**，29.87 秒；新 GPU 夹具在 finish_state_registration 之前查询存储，现有保护正确拒绝。生产代码不变，夹具调整为所有模型 prepare 后统一 finish，再查询和初始化 backend；保护及数值断言保留。修正快照 `c374e9917133eca64bfc9d91e53fb573d8ad2840287b6788c7674f2bb7ae2fc7`（2137 文件）的 `qwen3-noatt-registration-focused.xml` **28 passed / 28**，30.23 秒，零 error/skip/deselect。两轮原始 log/XML 已归档，完整回归正在相同快照运行，尚未计为通过。


无 Attention 草稿最终完整 `qwen3-noatt-registration-full.xml` **2773 passed、8 failed / 2781**，477.76 秒（JUnit 477.738），零 error/skip/deselect，退出 1。`audit-noatt.py` 已执行通过：原 2771 项 Counter 全部保留，新增 10 项全部通过，八个失败身份完全不变；28 项专项不重复累加。完整日志无资源析构、CUDA error 或 CUDA IPC producer 告警，46 个原 warning 保留。证据 `/tmp/lightllm-a716ad58-qwen3-noatt-full-evidence.tar.gz` 已归档并解包。

本轮 c374e9917133 源码/测试通过静态及逐文件校验，验证后继续开发 EAGLE3，故该完整结果不覆盖后续 EAGLE3 修改。无 Attention 草稿的全部容器/SSH 已退出，退出时八卡均 0 MiB，原 CI 保留；完整 A–E 范围和八项 BF16 失败继续。


## EAGLE3 原生装配与压缩词表（2026-11-04）

新增 `Qwen3EagleModelAdapter`，checkpoint 层号与全局执行层号分别绑定，启动创建模型特有 recipe，并由 Planner 绑定 Attention/Linear。压缩词表沿用主模型 embedding，保留草稿 lm_head、d2t 偏移与 t2d mask；普通词表使用独立 embedding。`normalize_model_config` 统一 CPU/运行时的 target_vocab_size 与 draft_vocab_size，重复规范化不覆盖原目标词表。

目标多层 hidden 仍在创建 InferState 前经过已绑定的 FC 投影，递归 hidden 直接进入相同固定宽度 Graph。注意这条投影位于 Graph 之外，测试验证的是后续同一张 decode Graph 在两种输入来源之间复用；没有把两种不同宽度直接塞入一张 Graph。

新增 2 个 CPU 配置检查与 6 个 H100 参数：普通词表/head_dim=64 和压缩词表/head_dim=128，各覆盖 Triton/FA3/FlashInfer。随机 FP16 safetensors 实际加载，独立 FP32 Torch recipe 完整计算 norm、Q/K/V、RoPE、因果 GQA、残差、FFN 和 lm_head，门槛为相对 RMS ≤1%、最大误差 ≤参考峰值/32、greedy 相同。测试包含目标 token ID 超过草稿词表范围、d2t 非恒等偏移、共享 embedding、两请求 prefix/decode、目标→递归→目标 hidden 共用一张 decode Graph，以及 prefill Graph 更新 hidden 后回放；禁止旧 factory/loader compute/运行时 manager 查询，并检查主模型 KV 未被草稿更改。

本地相关检查 **88 passed、3 skipped**，7.13 秒；三项为原 CUDA 检查，本地无 GPU。H100 `eagle-native-focused.xml` **126 passed / 126**，11.53 秒（JUnit 11.529），零 error/skip/deselect：包含新 8 项、原 EAGLE 输入 4 项、原 speculative_utils 85 项及原绑定 29 项。快照 `08aef636cd724bcdbe9f2129ed5bfe9cd217b6fad9567265d364cc79a1eabbdd`（2138 文件），完整回归仍在运行。未验证正式 EAGLE checkpoint、真实投机服务的接受率/任务、多卡、BF16、量化和性能；完整 A–E 不变。


EAGLE3 首次完整 `eagle-native-full.xml` **2863 passed、11 failed / 2874**，483.11 秒（JUnit 483.083），零 error/skip/deselect，退出 1。新增 3 个原状态提交测试失败（test_commit_selects_state_for_one_role 及 fixed/dynamic widths 两项），原因是新加入执行的旧 speculative_utils 测试直接写 LIGHTLLM_START_ARGS 并留下缺 enable_dp_prompt_cache_fetch 的 EasyDict；后续 CPU 状态页分配读到该不完整配置。保留失败日志与快照，未修改状态提交实现或其输入/断言。

为 `test_speculative_utils.py` 添加环境恢复夹具：保存/恢复原启动参数，结束时清除相关函数缓存；捕获原函数引用以兼容用例内部 monkeypatch。原 85 个测试体和断言保持。修正快照 `32409ce1555bc9763260f61d75a62e0076db4ed4e4a1156fa1568e3231681254`（2138 文件）生产代码与首轮相同，`eagle-isolation-focused.xml` **130 passed / 130**，11.38 秒，零 error/skip/deselect；在旧 speculative_utils 后紧接运行 4 项状态提交检查，三项失败全部通过。14 个第三方 warning 保留；原始证据已归档。相同快照完整 `eagle-isolation-full` 正在运行，尚未宣称完整复验通过。

并行的 GDN verify 隔离诊断在 GPU2 使用生产源码只读快照，临时 kernel 仅将 Q/K 的 FP32 归一化结果转回输入 dtype、sigmoid 门值转回 b 的 dtype，再继续 FP32 累加，以检查与 prefill 实际中间精度的一致性。原 8 项 `test_tiny_recurrent_verification_precision` **全部通过**，117.20 秒，零 error/skip/deselect，包含原 BF16 失败。没有改生产 kernel、测试数据或门槛；诊断通过不计作完整失败已修复。脚本前缀 `/tmp/lightllm-a716ad58-gdn-verify-rounding-`，kernel SHA256 `4d43128fa8d9868f140b1826a1b9161c71349e5eba62257a19885e96457e49be`，原日志/XML 归档 `*-diagnostic-evidence.tar.gz`。旧 kernel 等价性测试基线正在另行检查，后续须保留旧调用语义并验证原生装配路径的精度规则。


追加原有 `test_mtp_fused_recurrent_equiv.py` 的基线检查：`gdn-legacy-baseline.xml` **7 failed / 7**，8.57 秒，未修改生产 kernel 或原测试。该文件此前未纳入既定完整集合，因此这些失败必须另行保留，不能只报主集合八项失败。它要求旧 autograd 入口与直接 MTP kernel 输出逐位一致；现有直接 kernel 每个 token 将递推状态转换为存储 dtype，旧实现循环内部不做该转换，需继续验证这一差异。`git show HEAD` 证实直接 kernel 的状态转换已存在于起始提交，本轮未改它。最大输出差约 0.000488–0.000977，原逐位断言保留。原始证据 `/tmp/lightllm-a716ad58-gdn-legacy-baseline-evidence.tar.gz` 已归档并解包；该结果不与主集合通过数相加。


## EAGLE3 完整复验与 GDN 精度边界（2026-11-04）

修正测试隔离后的完整 `eagle-isolation-full.xml` **2866 passed、8 failed / 2874**，485.28 秒（JUnit 485.251），零 error/skip/deselect，退出 1。源码 `32409ce1555b...`；`audit-eagle.py` 已执行，通过 Counter 检查确认原 2781 项全部保留、93 项新增执行全过、八项失败身份不变。三个状态提交失败已消除，130 项专项不重复累加。完整日志无 BufferError、Exception ignored、信号量泄漏、CUDA error 或 IPC producer 告警。原日志/XML 归档 `/tmp/lightllm-a716ad58-eagle-isolation-full-evidence.tar.gz`。

另一个隔离实验只在旧 `fla/ops/fused_recurrent.py` 的 speculative 分支增加逐 token 状态 dtype 舍入，原七项 `test_mtp_fused_recurrent_equiv.py` 全过，1.31 秒，无 skip/deselect，退出 0。原逐位输出断言不变；普通 decode 分支不受影响。实验 kernel SHA256 `2b943a03cd1170b8037e06e6ba4e097a302b0ee32582a2814f6f7eda06b1c3fc`，注入脚本 `49aab02966bb97f5a019b8fe12a31d6848e64686c1703a4bd67c538652372337`；`gdn-state-rounding-diagnostic.xml/log` 已一并归档。

正式修改包含上述状态一致性修复，以及直接 MTP kernel 的显式 `round_intermediates` 参数：默认保持旧 Q/K 与 beta 精度，原生 BoundGDN verify 启用与 prefill 独立算子一致的中间舍入，FP32 累加保留。新模式使用独立 autotune key，旧 key 不变。新增独立 Torch 递推对照、FP16/BF16 输入与 FP32/BF16 SSM、不同配置、空序列/独立写槽、变更输入和接受游标后的 Graph 回放。原七项等价性及完整 autotune 测试加入后续完整集合，未改已有门槛、输入或断言。本地六项 CPU 检查与 Black/flake/diff 检查通过。源码快照 `a53e67ca799145415d062b4874aac0f3b1a854a5fb11d072e6bb74a83006baf1`（2139 文件）正在 H100 GPU2 专项验证；尚未声称正式修复通过，全部 A–E 剩余范围保持。


GDN 首轮正式候选 `a53e67ca7991...` 的 `gdn-precision-focused.xml` **102 passed、7 failed / 109**，568.00 秒（JUnit 568.002），零 error/skip/deselect。原 8 项 verify 精度和 7 项旧入口等价性均通过；5 组形状分别遍历全部 80 个 autotune 候选也通过。新增的失败有两类：3 个旧首次 decode 检查用包装函数计数，暴露默认参数读取错误；4 个新 FP32 SSM 参考检查复用了保存状态的视图，后续原地递推改写了旧候选。原失败 XML/log 已归档 `/tmp/lightllm-a716ad58-gdn-precision-initial-focused-evidence.tar.gz`。另跑原 `test_fused_recurrent_strided.py` 三项全部通过，8.34 秒。

修复让 Autotuner 在构造时保存原函数参数表，参数位置与默认值使用同一份声明；不再每批检查可能被包装的执行函数签名。新增 CPU 检查覆盖包装后的默认/显式参数、缺失必选参数与真实调用转发。新数学参考在状态落盘后复制递推值，避免改写已保存候选；所有原断言、容差与输入不变。本地 44 项通过，5.44 秒；Black/flake/diff 通过。新快照 `aafb942aa3209b6771b42d69f6cdf98394770925ff1554ba5b237dac619a0650`（2139 文件），专项新增原 autotuner 整个文件和三项 strided 检查；完整集合保留原 2874 项并预计新增 35 项。专项和完整复验正在运行，未以首轮部分通过关闭数值缺口。

### MRoPE 的独立诊断，未改生产

在 a53 只读快照上，临时替换 Qwen3.5/SenseNova recipe 的 MRoPE，实现乘积分别舍入后相加；保持原测试数据、参考和门槛。`mrope-rounding-diagnostic.xml` **16 passed、3 failed、1 skipped / 20**，94.41 秒。原 Loop BF16 普通 Triton/FA3 及 mHC2 FlashInfer 三项通过；mHC2 Triton/FA3 仍有 greedy 差异（该失败阶段为 [6,35] 对 [123,35]）。正式 Qwen3.5 最大 RMS 为 **1.9313%**，高于原 1.7853%，仍不通过。

单卡启动使正式 SenseNova 测试因要求两张可见 CUDA 卡而跳过，随后在 GPU3/4 补跑同一检查：`mrope-loop-checkpoint.xml` **1 failed**，28.02 秒，零 skip/deselect，最大 RMS **9.9618%**，仍超过 1%。这不是正式修复，也不能把隔离变体通过从主集合扣除。原始日志/XML 分别保存在 `mrope-rounding-diagnostic-evidence.tar.gz` 和 `gdn-precision-initial-focused-evidence.tar.gz`，均已解包；实验源码位于 `/tmp/lightllm-a716ad58-mrope-rounding-kernel.py`，SHA256 `7ff891aeefadcf58f8e4d3cd7c819e72338f0b7ea4e84e96f2b51f926ea8194c`。生产 MRoPE 未修改。


修正快照 aafb942aa320 的 H100 `gdn-precision-fixed-focused.xml` **155 passed / 155**，141.01 秒，零 error/skip/deselect，退出 0。相对首轮增加 43 项完整 autotuner 检查（含新包装默认值检查）与原 3 项 strided 检查；首轮七项失败全部通过，原 GDN 八项精度、七项逐位等价、所有 80 个候选及 Graph 检查保留。25 个第三方 warning 保留，原始 XML/log 归档 `gdn-precision-fixed-focused-evidence.tar.gz`。相同修正快照完整 2909 项正在 GPU3/4 运行，旧 a53 完整候选仍在 GPU0/1 收集失败证据。专项不重复累加为完整回归，尚未以该结果宣称全部 BF16 或完整 A–E 验收。


首轮 a53 完整 `gdn-precision-full.xml` 最终 **2891 passed、14 failed / 2905**，773.45 秒（JUnit 773.420），零 error/skip/deselect，退出 1。已按 Counter 核对：原 2874 项全部保留，新增 31 项；原八个 BF16 失败中仅 GDN verify 已消除，另外七个不变，新增七个失败恰为上文已修的包装默认参数/新参考视图问题。原始日志无资源析构、CUDA error 或 IPC producer 告警，归档 `gdn-precision-initial-full-evidence.tar.gz` 已解包。该首轮容器已退出；修正候选 aafb 的完整回归仍在运行，不能用 155 项专项代替其结果。


GDN 最终完整 `gdn-precision-fixed-full.xml` **2902 passed、7 failed / 2909**，490.14 秒（JUnit 490.119），零 error/skip/deselect，退出 1。已执行 `audit-gdn-fixed.py`：原 2874 项 Counter 全部保留，新增 35 项全部通过，原八项 BF16 失败仅消除 GDN verify，其余七个身份和正式 Qwen/SenseNova RMS 不变。首轮新增七项包装接口/参考代码失败全部通过。完整日志无 BufferError、Exception ignored、信号量泄漏、CUDA error 或 IPC producer 告警。快照 `aafb942aa320...` 的所有生产/测试文件与本地一致，之后仅文档更新。归档 `/tmp/lightllm-a716ad58-gdn-precision-fixed-full-evidence.tar.gz` 已解包；所有任务容器退出，八卡显存均 0 MiB，原 CI 容器保留。没有提交或部署，完整 A–E 目标及剩余七项 BF16 验收继续。

## DFlash / DSpark 的声明阶段与原生装配（2026-11-04）

新增 `Qwen3DFlashModelAdapter`、`Qwen3DSparkModelAdapter`、`Qwen35DFlashModelAdapter`，分别绑定权重映射和原有模型 recipe；Qwen3.5 DSpark 复用 Qwen3 DSpark Adapter。四个模型类显式接入，覆盖六个注册入口（含 Qwen3.5 text 别名）。checkpoint 层号仍从零开始，执行及状态层号接在主模型和先前草稿之后。Qwen3.5 DFlash 共享主模型 embedding/lm_head，Qwen3.5 草稿使用自身 rotary cache；模型特有关系在装配时保留。

`ModelAdapter.attention_phases` 默认声明 prefill/decode；这组 block 草稿只声明 decode，AttentionSpec 声明 `causal=False`。Planner 只匹配实际执行阶段，多个 proposal token 使用 verify 绑定；Runner 为只提交 KV 的 prefill 保留 writer，不创建未使用的 attention 计划或元数据。FA3 的普通及静态 FP8 实现接受非因果语义，Triton 和 FlashInfer 的现有能力限制保持。DFlash2 的 selector/window 算法未接入这组声明，继续独立保留旧路径。

新 `test_block_draft.py` 的八个模型 GPU 参数使用真实加载的 FP16 随机 safetensors 和独立 FP32 数学参考，覆盖四类模型 × 单 token / 三 token proposal。验证初次上下文提交、多行接受结果提交、块内全部 token 可见、DSpark gated/rnn Markov 与 confidence、Graph 更新 token/position、共享权重和主 KV 不变；旧工厂、量化执行入口及运行时 manager 查询被置为失败。DFlash logits 保留 RMS ≤1%、最大误差 ≤参考峰值/32 和 greedy 一致检查。另有六项 FA3 FP8 非因果检查，覆盖 FP16/BF16 输入、per-tensor/per-head KV、动态/静态 Q scale 与 Graph，采用已解码 FP8 的独立 SDPA 参考及原 FP8 误差门槛。

本地 95 项 CPU 相关检查通过；H100 首轮模型专项 103/103，补 FP8 后 109/109，均无 skip/deselect/error。原有 DSpark、Runner、Planner、绑定测试均保留。新 21 项加入原 2909 项完整集合，最终 2923 passed、7 failed / 2930，全部新增检查通过，原七个 BF16 失败身份不变；这不构成正式 DFlash/DSpark checkpoint、接受率/HTTP、全模型 BF16/量化、多卡或性能验收。

正式 DSpark 的差异发现记录（修复及最新验证见下）：H100 上已有 `/home/devsft/models/Qwen3.8-27B-DSpark`（2,718,576,122 字节 safetensors）及只改 model_type 的 `-lightllm-adapter` 配置目录。只读检查原文件头与自带 `dspark.py`，确认该发布 checkpoint 不含 embedding/lm_head，明确依赖目标模型；它有 5 层、hidden 5120、head 128、vanilla Markov rank 256、confidence head 和 YaRN。当前 Qwen3.5 DSpark Adapter 沿用旧的自有 embedding/lm_head 布局，新随机权重检查也覆盖该布局，未覆盖上述正式共享权重格式。需要补充明确的共享声明/加载与实际 checkpoint 验证，同时保留已有自有权重布局；不能把正式文件已存在或专项通过当成正式 DSpark 已可用。目标 `/home/devsft/models/Qwen3.8-27B` 为 qwen3_5，64 层、hidden 5120、Dense head 256，约 55.56 GB；尚未加载或运行这对正式模型。

### DSpark 正式共享布局的启动装配

启动阶段现在按实际 checkpoint 名称识别 vocabulary 权重布局：完整 embedding/lm_head 对继续独立加载；发布格式 `DSparkDraftModel` 在两者均省略时绑定已加载目标权重，校验 hidden/vocab/dtype；部分缺失和未声明的格式直接报错。共享布局在创建权重对象时就省去 vocabulary 矩阵分配。header 查询与真正加载使用同一组 shard 选择规则，safetensors 只读取名称，bin 使用 meta tensors；没有新增运行时资源查询。

DSpark 的配置展开与自身 rotary cache 从 Qwen3.5 子类移至公共 DSpark 模型，依据发布 backbone 自己构造 Qwen3RotaryEmbedding 的语义。原始 checkpoint 的 qwen3 model_type 因此可直接与 Qwen3.5 目标配对，无需修改远端 config；原注册保留。

本地布局/加载及旧 DSpark 共 52 项通过，GPU 项明确排除；H100 `dspark-shared-focused.xml` **124/124 passed**，14.88 秒，零 error/skip/deselect。快照 `2df416feb1c93dfd532218c8fccf52ff6ac33e686f370bfae1ffa2c0dd58a119`，原自有布局检查保留，新增共享布局包含两注册类、block1/3、不同目标/draft head 形状和 Graph 更新。证据已归档解包。新正式 checkpoint 测试正在 H100 双卡执行，尚无通过结论；其独立参考使用发布 backbone 返回的最终归一化 hidden 计算 DSpark heads，另行核对现有 confidence 输入语义。旧 2930 项完整结果不覆盖这些后续修改。

正式模型后续结果：shared layout / normalized-head 修复的 96f0 快照上，124 项布局检查再次通过；正式 Qwen3.8-27B + DSpark 的 FP16 eager、capture、目标 decode/hidden 提交后的 Graph replay 通过，18 个预测 token 与独立 FP32 draft 参考一致，最大 logits RMS 0.06347%。BF16 首次 token 一致、logits RMS 0.43677%，但一个 confidence 值误差 0.0036378 超过既定门槛，后续阶段未通过验收。只提高 confidence 尾部精度与 FP32 残差的隔离实验均未完全解决，未合入生产。当前 2945 项完整复验进行中，全部原失败及新增失败保留；正式目标模型任务精度、接受率服务、多卡与性能尚未验收。


### DFlash2 的原生窗口与专用 recipe

新增 `Qwen3DFlash2ModelAdapter`，显式声明共享目标 embedding/lm_head、decode-only 非因果计算，以及所有层的双侧窗口 `(width-1, width-1)`；零窗口继续完整可见。它选择 DFlash2 自己的权重、动态卷积和 top-16 selector recipe，checkpoint 层号与执行/状态全局层号继续分开。原生 attention 和 KV writer 消费启动绑定的视图，DFlash2 运行时不再反查状态 manager；动态卷积投影和 selector projection 也由原生 Linear 绑定执行。

FA3 普通精度的能力探测允许非因果右窗口，因果和 Triton 的右窗口限制保留；没有新增 FP8 窗口支持声明。原 DFlash2 per-head FP8 拒绝规则保持。本次不改变注册别名、selector 算法、量化格式或缓存保留策略。

本地 100 项检查通过，32 项依赖 GPU/checkpoint 的检查显式排除；H100 `dflash2-native-focused.xml` 为 **154 passed / 154**，19.76 秒，无 error/skip/deselect。新增 3 项语义能力检查和 8 项真实 FP16 随机 checkpoint 模型检查（proposal 宽度 1/3 × 完整/双侧窗口 × 固定/动态 selector），旧检查均保留。独立 FP32 参考计算块内卷积、窗口 GQA、MLP、top-16 条件选择和 confidence；检查初次 KV 提交、proposal、Graph capture、追加接受行后的 Graph replay、目标 KV 不变，禁止运行时 manager 查询和旧计算工厂。最终 hidden 的 RMS ≤1%、peak ≤1/32、预测 token 一致、confidence 2%/0.002 门槛未放宽。原 DFlash2 未继承普通 block 语义的 guard 更新为检查独立 Adapter 和专用 recipe。

当前代码快照 `f0458a3efcc7a5a3deb0cf82d50cc0feee70f86a0fa9766d46a2bb9e1b5e399c`，2141 文件；该轮完整回归最终为 2948 passed、8 failed / 2956，新 11 项全过。该阶段的 32 Adapter/草稿 20 个注册项是声明覆盖，不能等同于正式 DFlash2 checkpoint、BF16、TP、接受率服务或性能验收；这些仍未完成。原八项 BF16 失败继续保留。


### MLA verify 与 DeepSeek / GLM4 MTP

`MLAAttentionBackend` 与 `FlashInferMLABackend` 现在声明独立 verify 阶段，Runner 在启动时按相同 latent 格式分配工作区并绑定。FA3 将连续同请求行组装为因果多 query 序列，保留 HOLD 行；Triton 和 FlashInfer 按每行的实际序列长度执行，支持重复请求及动态提议宽度。Dense metadata 的窗口逻辑只用于 DenseSpec，避免 MLA 被错误要求提供 window 字段。MTP 的原生 opt-in 扩展到普通 MLA，Sparse MLA/KDA 的未完成验证路径仍未因此开放。

新增 `Deepseek3MTPModelAdapter` 与 `Glm4MoeLiteMTPModelAdapter`，共享主模型 embedding/lm_head，保留草稿自己的归一化和投影。checkpoint 层号、MoE 选择和全局状态层号显式分开；链式草稿复用 checkpoint 权重坐标，同时独占执行/状态层。原生 Linear、MoE 与 MLA 直接消费绑定资源。MTP prelayer 的目标 hidden 归一化改用独立输出，避免 Graph warmup/capture 或连续草稿原地改写上游输入。

首轮 H100 199 项专项中 181 通过、18 失败：6 个 GLM 草稿在回放后的继续解码偏差，12 个主模型在 verify 后的 suffix prefill 偏差。只读诊断确认 eager/Graph 输出和 KV 逐值一致、写入位置正确；逐层 latent 相对参考仅小幅舍入偏差。进一步核对安装的 Transformers DeepSeek V3 与 GLM4-MoE-Lite 源码，发现原 `grouped_moe_specs` 只在 `(groups, selected_groups, top_k) == (8,4,8)` 时声明组内 top-2，其他形状误用 top-1，遗漏了模型数学语义。

现在由模型 Adapter 声明默认组评分，显式 `topk_method` 决定 greedy / group_limited_greedy / noaux_tc；组内 top-2 不再由维度组合猜测。原失败数据、种子、独立参考与阈值保留。修复后的 H100 扩大专项 `mla-mtp-routing-focused.xml` **272 passed / 272**，46.44 秒，无 error/skip/deselect；Counter 核对首轮 199 项全部保留、原 18 项失败全部消失，新增 4 项路由声明检查及 69 项既有 MoE/GLM 回归也通过。

新增验证包括 12 项 MLA verify（FP16/BF16、latent256/512、三后端，动态分组/顺序/HOLD/两 lane/Graph）、12 项主模型 MTP verify（DeepSeek/GLM4 × 三后端 × 固定/动态提议，拒绝后缀覆盖与 suffix prefill）、12 项草稿完整随机 FP16 checkpoint（两模型 × 三后端 × vanilla/eagle；链式草稿、共享词表、checkpoint/执行层映射、输入 hidden 不变、prefill/decode/capture/replay/prefix prefill），以及 8 项 CPU 声明检查。原数值门槛不变。本地 83 项通过、68 项 GPU/checkpoint 检查显式排除，15 个 Python 文件 Black/flake8 和 diff 检查通过。

源码/测试快照 `8d7c688cfaeed064c0054d77dc4775a228fcc0de492d70c0478350871a29b178`（2142 文件）的完整复验最终为 **2992 passed、8 failed / 3000**，567.82 秒（JUnit 567.795），零 error/skip/deselect，退出 1。Counter 审计确认原 2956 项全部保留、新增 44 项全过，八项 BF16 失败身份不变。46 个第三方 warning 保留，无 BufferError、Exception ignored、信号量泄漏、CUDA error 或 IPC producer 告警；原始 XML/log 已归档。测试容器退出、八卡显存均 0 MiB，原 CI 容器保留。源码/测试此后不变，仅更新文档。当前结果不代表正式 DeepSeek/GLM MTP 权重、模型组合接受率、HTTP、多卡、FP8 或性能验收，完整 A–E 目标继续未完成。


### GLM5 KDA/池化 verify 与 MTP Adapter（验收进行中）

KDA 的原生绑定现在包含候选槽位与共享接受游标。verify 用明确的候选行构造 Conv/SSM 元数据，直接调用已有卷积和 recurrent kernel；后续 prefill 只读取已接受状态与对应卷积窗口，输出归一到基准槽位。同一状态的各层在执行前保存批次游标，Graph 元数据可以更新请求顺序和候选宽度。没有在本次迁移中改写 KDA 数值 kernel。

Pooled NoPE 为 verify 注册独立计划，按逐行长度读取历史，并把候选偏移交给池化 kernel。Indexer tail 的 FormatSpec 声明候选容量，StateStore 分配 pool_size + candidate_slots - 1 个环形位置；草稿的容量同时覆盖主模型展开行。普通与 group64/E8M0 KV、Triton 与 FlashMLA/SGL 继续使用原生视图、既有 writer 和独立执行 lane。

新增 Glm5NextMTPModelAdapter：一层 NoPE、关闭草稿 mHC，checkpoint 层号与全局执行/状态层号分开；共享目标 embedding/head/final norm，保留自身 fusion 与 shared-head norm，绑定 Linear/MoE。目标 hidden 的第一层 norm 使用独立输出，避免 warmup、Graph 或后继草稿改写上游输入。旧草稿存储声明遗漏的 rope_dim=0 同时补齐；原默认 64 与 GLM NoPE 不符。

新增 29 项检查，原用例数据、身份和门槛保留。3 项 CPU 检查声明、候选格式和层号映射；4 项 KDA 独立 Torch 检查覆盖 FP16/BF16 × FP32/BF16 SSM、所有候选、不同接受点、拒绝覆盖、suffix prefill、Graph 与 lane；6 项池化检查覆盖普通/FP8、两后端、长上下文、池边界与拒绝后的 tail、动态候选宽度/HOLD/Graph，以及独立 weighted pool 和字节结果。8 项草稿模型检查覆盖两后端/两格式、vanilla 两个草稿与 EAGLE 单草稿、共享权重、状态独占、上游输入不变和两类 Graph，参考为相同阶段的旧模型执行；普通 KV 另以完整 prefill 重算。它们不是正式 checkpoint、实际主草稿接受率、HTTP、多卡或性能验收。

另外 8 项完整 BF16 主模型 verify 检查仍失败：固定/动态宽度 × 两后端 × 普通/FP8 KV，首次候选执行相对整段 prefill 的最大差约 0.02832（普通）与 0.0625（FP8），超过既定 rtol=0.02、atol=0.005。后续 Graph/接受/suffix 的整模型验收尚未到达，不能以独立 kernel 通过替代。Q/K 中间舍入的只读诊断将普通用例最大差降至 0.0234375，但两项仍失败，未合入生产。

首轮 KDA 专项 95/95 通过；扩大专项先为 103 通过、16 失败。修正两请求夹具被单请求误用及旧 NoPE 声明后为 107 通过、12 失败；FP8 草稿的只读诊断确认新旧逐步执行逐值相同，双方与整段 prefill 同差 0.0390625，因为 prefill 当前 token 保留 BF16，而 decode 使用编码后的 KV。因此该新测试按相同阶段和历史比较，未放宽误差门槛；输出在调用另一模型前复制以保留生命周期。最新 a038240bdf66 快照（2143 文件）的 119 项专项为 **111 passed、8 failed**，46.30 秒；所有原 95 项与扩大 119 项身份均保留。完整 3029 项正在 GPU0/1 运行，包含原 3000 项及其八个 BF16 失败；新增主模型失败同样保留。原始失败、诊断及通过结果已归档，完整 A–E 目标继续未完成。
