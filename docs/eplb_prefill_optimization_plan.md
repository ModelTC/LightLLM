本文规划在 `eplb` 分支上优化 PD 分离的 prefill 服务。基线为 `d24f08561`；比较对象为 `wzj_eplb@eb61641cd`。规划编写日期为 2026-09-23。以下先记录首版交付范围，再保留完整路线图；未完成的路线图项目不代表已实现或已验证收益。

**首版实现（仅单元测试）**

- Triton 与 DSv4 CUDA 的物理副本选择统一使用 32 位混合哈希，保留单 token 选择及融合计数/路由语义。
- 默认 `redundant` 模式保留固定主专家，同时比较当前布局、粘性贪心、无粘性贪心与局部搜索结果。每个种子、每层至多搜索 4 步，每步最多评估 256 个替换/交换；沿用模型级收益门槛。搜索先累积改进，再检查粘性余量，避免每一步门槛阻断可获益的搜索路径。
- 新增显式 `full` 模式：完整物理布局、全专家覆盖和 rank 内唯一性校验、按负载分配副本数、节点内装箱与局部搜索。节点拥有的专家集合保持不变；支持 R=0。启动仍使用确定性默认布局，加载器通过完整布局索引权重及量化参数。
- 迁移源来自真实当前布局，完整模式目标使用绝对行号。GPU 传输沿用 CUDA IPC/NIXL，整层待写行先 staging，所有 rank 就绪后在原有安全边界发布。完整模式只分配一层全行 staging，DSv4 显存预留与实际分配共用同一策略。
- 每层维护 `physical_to_logical` 和 `placement_generation`，在权重/映射提交时更新；部分层已提交时，CPU 当前布局也反映已发布的真实状态。

启动参数示例（本次未启动服务）：

```text
--run_mode prefill --enable_ep_moe --enable_prefill_eplb
--eplb_placement_mode full --eplb_num_redundant_experts_per_rank 0
```

`--eplb_placement_mode` 默认为 `redundant`，其 R 仍必须大于 0。`full` 仅允许 PD prefill，R 必须非负且满足每卡无重复的容量约束。既有 EP、禁用 prefill CUDA graph、非 SM100 等限制继续适用；本次没有扩展 decode 支持。

单元测试包括独立逐来源负载评分、固定主专家反例的穷举最优比较、均匀/偏斜负载、来源节点约束、R=0、非法布局、版本发布、内存预留、两种路由 kernel 的周期性输入，以及两卡/三卡覆盖环与八卡原有迁移回归。GPU 迁移测试使用小型人工权重和真实 CUDA IPC，检查权重及 scale、多层缓冲复用、连续两代迁移和无任务 rank；没有启动模型服务，也没有验证跨机器 UCX 传输。

验证记录：H200 / PyTorch 2.11.0+cu130 环境中，CPU 与单 GPU 单测 242 项通过，两卡/三卡完整布局迁移 2 项通过，八卡原有 staging 复用回归 1 项通过，共 245 项。没有运行服务或迁移带宽基准。复现命令：

```bash
python -m pytest -q unit_tests/common/fused_moe/test_eplb.py \
  unit_tests/common/fused_moe/test_eplb_full_layout.py \
  unit_tests/common/fused_moe/test_eplb_hash.py \
  unit_tests/models/deepseek_v4/test_moe_topk.py \
  unit_tests/models/deepseek_v4/test_memory_profile.py \
  unit_tests/server/test_api_start_eplb.py --disable-warnings
python -m pytest -q unit_tests/common/fused_moe/test_eplb_transfer_gpu.py \
  -k 'full_layout or bounded_staging_reuse' --disable-warnings
```

本版暂未实现 PR0 的新增物理负载/耗时采集和服务基准、PR5 的时间成本模型/回本判断/布局持久化，以及跨节点重排、加权分发或逐 batch 求解。规划器的收益是同一评分函数下的估算负载改进，不能当作 TTFT 或吞吐收益。完整模式虽保持节点专家集合，但缺少本地副本的来源仍可能因副本数变化改变远端流量比例，通信成本需要后续实测。服务与 PD 全链路验证按本次要求留待后续。

**后续路线图及设计依据**

目标是在相同硬件、模型精度和 TTFT SLO 下提高 prefill 有效吞吐，并控制重平衡期间的尾延迟和显存开销。最终指标是服务性能；专家 token 均衡率和规划器估算负载是诊断指标。

保留现有实现中已经有价值的部分：融合 top-k/计数/映射，按批次和来源节点保存负载，128 对齐估算，持续在线采样，收益门槛，槽位复用，以及 CUDA IPC/NIXL/UCX 异步迁移。增加完整布局能力、质量更好的候选规划、真实成本反馈和布局持久化。

借鉴来源与边界如下：

| 来源 | 借鉴内容 | 集成时的处理 |
| --- | --- | --- |
| wzj_eplb | 更稳健的副本哈希、完整槽位布局、规划器接口、历史布局加载、迁移覆盖依赖处理 | 不采用“最热 R 个专家全部复制到每卡”的默认策略；保留 eplb 的 GPU 直接迁移 |
| DeepSeek EPLB、vLLM、SGLang | 按负载分配副本数、完整物理布局、容量约束装箱、适用时按专家组分层放置 | 作为候选生成器；必须通过本实现的批次、拓扑和迁移成本评估 |
| Libra | 将规划和迁移开销与推理重叠的设计原则 | 现阶段增强异步执行；预测下一层专家属于后续研究项 |
| METRO | memory-bound 时，活跃物理专家数量和权重读取成本不能忽略 | 小批量 prefill 也需实测；不把“支持 decode EPLB”作为本项目的收益目标 |
| TEMPO | 根据实测计算/内存/分块成本优化最慢 rank 的耗时 | 先建立简单可校准模型；该工作是预印本，不把论文模型直接作为生产事实 |
| LPLB | 分离较慢更新的布局与较快更新的 token 分发 | 背景加权分发先行；逐 batch 求解只在测得开销可回收后开启 |

上游参考：[DeepSeek EPLB](https://github.com/deepseek-ai/EPLB)、[vLLM 默认策略](https://github.com/vllm-project/vllm/blob/94f4170df37fe29d68bd2f7e5a496501b78d55ae/vllm/distributed/eplb/policy/default.py)、[SGLang DeepSeek 策略](https://github.com/sgl-project/sglang/blob/40048f6e51891e984258d34c33c5a9f2c1f542f6/python/sglang/srt/eplb/eplb_algorithms/deepseek.py)。

实施依赖为 `PR0 → PR1 → PR2 → PR3 → PR4`，其中观测与成本校准贯穿各阶段；PR5 在前述基础上收敛在线策略；PR6 是由实测决定是否实施的扩展。PR3 的完整布局执行接口必须先完成，PR4 的完整布局规划才能在服务中生效。

| 批次 | 交付 | 预期价值 | 放行条件 |
| --- | --- | --- | --- |
| PR0 | 可重放的负载数据、阶段耗时和服务基准 | 判断瓶颈和剩余优化空间 | 采样不显著扰动服务，数据可按批次/层/布局版本对齐 |
| PR1 | 修复 Triton 与 DSv4 CUDA 的副本哈希 | 消除已复现的周期性集中分发 | 对抗输入通过，输出语义一致，融合路由性能无显著退化 |
| PR2 | 多候选与局部搜索，先保持主专家固定 | 以较低改造成本缩小现有贪心算法的差距 | 同一代价函数下不劣于当前候选，并通过独立轨迹验证 |
| PR3 | 完整布局表示、加载和通用 GPU 迁移 | 为移动原始专家打通运行时 | 多 rank 迁移、覆盖环、量化参数和 overlap 正确性通过 |
| PR4 | 完整专家重排；先节点内，后有条件跨节点 | 突破固定主专家的均衡上限 | 端到端收益成立，跨节点通信和 staging 成本受控 |
| PR5 | 迁移回本判断、漂移触发、布局持久化 | 将规划收益转成持续服务收益 | 稳态减少无效迁移，热点切换能及时响应 |
| PR6 | 加权副本分发；必要时逐 batch 求解 | 在固定布局上缓解短期波动 | 分发与统计新增成本低于减少的等待时间 |

PR0 应复用 `ep_balance_monitor.py` 已有的对齐后计算负载和压力漂移统计，避免重复建设整套监控。补充每层实际接收的物理专家计数、活跃物理专家数，以及采样式 dispatch/GEMM/combine 耗时。来源侧的逻辑专家计数继续用于规划，目的端物理负载用于验证规划是否兑现。

每份样本携带 prefill round、microbatch、层、布局版本和来源节点标识，区分完整服务轮次与一次 dispatch。必要时保留少量来源 rank 统计，识别节点内偏斜。异步读取 CUDA event，不能为观测引入逐层全设备同步。对 overlap 路径，分阶段耗时用于诊断，不能简单相加替代实际 forward 墙钟时间。

首先建立四类基线：EPLB 关闭、现有 EPLB、相同副本预算的固定布局、各阶段的新实现。对比同时覆盖初始化、稳态和迁移阶段。使用同一请求轨迹，并把用于生成布局的区间与用于评价的区间分开。

PR1 同时修改 `grouped_topk.py::_eplb_replica_index` 和 DSv4 `moe_topk_eplb.cu` 中的物理副本选择。采用 wzj 分支的 32 位混合思路，保留 top-k、映射、计数的融合执行。这里修改的是物理副本选择，不修改 DSv4 的逻辑专家 hash routing、logical top-k、gate weight 或校正偏置。

需要覆盖周期性 token 下标、2/3/4/8 个副本、热点集中、单 token、空 batch 和多来源相关输入。不要机械加入 TP rank 随机盐：实际独立 dispatch 来源可以去相关，但对于复制输入后求和的执行路径，副本选择必须遵守其跨 rank 一致性约束。现有 source-rank 路由表旋转也应纳入测试。

已复现的回归案例是：专家 7 在 token 下标 `0,4,...,4092` 出现，候选副本数为 4。现有线性取模的 GPU 结果为 `[0,1024,0,0]`，wzj 混合哈希为 `[266,258,244,256]`。这是测试用例，不是吞吐收益承诺。

PR2 抽出统一的候选生成和候选评分接口。先保持现有 `[layer, rank, redundant_slot]` 运行布局，候选包括当前布局、现有贪心结果、负载驱动的其他初始化，以及有限次数的副本替换/跨 rank 交换结果。

当前算法先选择负载最低且有空位的 rank，再选择专家。改造后可对少量目标 rank 与热点专家组合联合打分，随后做有预算的局部搜索。规划器后台执行，设置确定性的迭代预算；候选再多，也不能阻塞推理线程。当前布局始终是可选候选，复用同一份按批次、来源节点和对齐后的评估器，并保留收益门槛。

这里的“不劣”只指相同输入、相同估算函数下的比较，不能直接等价为线上速度保证。离线测试包括上次发现的两类反例：均匀负载被错误重排，以及现有贪心方案明显落后于固定主专家约束下的穷举最优。

PR3 将运行状态统一为完整物理布局：

```text
physical_to_logical[layer, rank, local_slot]
logical_to_physical[layer, logical_expert, replica_slot]
replica_count[layer, logical_expert]
placement_generation[layer]
```

“初始主专家”只用于默认启动布局，不再作为执行期间不可变的位置公式。第一版保持每卡固定槽位容量、逻辑专家全覆盖、同一 rank 内不重复放置同一逻辑专家。上游装箱结果可能不满足最后一个约束，需要修复或拒绝，不能直接照搬。以后如有必要再扩展同卡多副本。

主要改动涉及 `expert_parallel_state.py`、`fused_moe_weight.py`、`eplb_placement.py`、`eplb_transfer.py`、`eplb_manager.py` 和 DSv4 内存预留。传输源从当前布局查询；目标槽使用绝对 local_slot，移除“目标槽加原始专家数”的假设。固定 shape 的映射 tensor 和 live 权重尽量原地更新。

第一版完整重排只对显式 PD prefill 模式开放。当前 decode 路径把前 E/W 行当作主专家，DeepEP decode 也按 E 而非完整物理专家数初始化；在混合/decode 模式适配完成前不能放开这个限制。prefill 服务的 warmup、空请求、最后一个 chunk 和与 MTP 相关的实际执行路径仍需验证。独立 draft 模型的专家布局不随目标模型一起隐式修改。

完整重排的关键是迁移正确性。现有 staging 只覆盖 R 个新增槽位；完整布局可能改变更多行，不能仅修改规划器张量形状。第一版采用逐层准备：

```text
旧布局继续服务
→ 将该层全部待写行放入 staging
→ 所有相关源读取/目标接收完成，统一确认 generation 就绪
→ 在安全推理边界提交权重、量化参数和映射
→ 后续 forward 使用新 generation
```

该机制允许处理 A→B→C→A 的覆盖环。提交前，任何仍被用作传输源的 live 行都不能提前覆盖。沿用独立控制组、全局有序边界与 overlap stream 事件排序。提交前失败维持旧布局；不可恢复的通信错误统一失败退出。提交后要回到旧布局，需要真实恢复权重，不能只把映射表改回去。

GPU staging 使用字节预算。可先为一个层的本地全部可能变化行预留缓冲，减少流水深度，而不是直接将原来的 8 层冗余缓冲扩大成 8 层完整缓冲。预算、quant scale、通信 workspace 和 KV cache 在启动 profile 时统一计入。CUDA IPC/NIXL 继续负责数据传输；从 wzj 借鉴依赖处理，不切回 CPU/Gloo 权重传输主路径。

PR4 在完整布局执行能力上增加负载驱动的副本数量规划和容量约束装箱。先按 `load / replica_count` 生成副本数候选，再进行装箱和局部改善。用现有按批次负载评估器统一比较完整布局、受限布局与当前布局。

第一步只允许节点内完整重排，保留节点拥有的专家集合，减少跨节点迁移与通信变化。之后再加入跨节点候选：模型确实使用分组路由且拓扑满足条件时，借鉴 DeepSeek 的专家组→节点→GPU 分层策略；其他模型按来源流量与实测链路代价评分，不能强套分组约束。

跨节点通信估算不能只看专家边际计数：同一个 token 的多个 top-k 专家可能共享一次跨节点传输。需要小比例 token-to-destination-node 轨迹或实际通信计数校准，避免重复计算通信量。当前 node-first 规则可作为低通信成本候选；远端分流只有在整体耗时收益覆盖网络开销时才启用。

完整布局模式应支持 R=0，让“仅重排、不增加专家权重副本”成为独立基线和可选方案。R 在第一版仍作为启动时固定预算，不同时引入逐层异构槽位容量和运行时显存扩缩容。

PR5 将收益门槛从对齐 token 数逐步升级为校准后的时间与回本判断。按 GPU、量化方式、专家形状和 batch 区间校准 GEMM 时间；以实际通信测量校准 dispatch/combine；小批量时加入活跃物理专家数对应的内存成本。计算、内存和通信存在重叠时，直接校准 block 时间，避免重复相加。

迁移是否值得执行，可先使用如下条件：

```text
H × Δt_forward > C_rebalance + 估算不确定性余量
```

H 是预计该布局仍适用的后续 forward 数；Δt_forward 是每次 forward 预计节省的时间；C_rebalance 是规划、传输争用与提交造成的累计服务时间损失。后台传输的墙钟时间不是其服务损失，二者要分别记录。也要计算迁移完成后剩余的有效收益窗口。

保留稳定负载下的采样退避；结合已有压力漂移指标，在热点变化时提前采集新的连续窗口。维持有限的新鲜历史，区分短期突发与持续漂移，防止旧负载长期淹没新热点。迁移完成前若布局预测明显过时，可在尚未提交时取消；一旦已提交部分层，应按已发布的真实版本继续管理。

持久化模型 revision、层号、专家数、EP 拓扑、量化参数、布局版本和完整布局。加载时校验覆盖、容量、本地唯一性及拓扑，一致性通过后由 rank 0 广播确认。启动可直接按布局加载 checkpoint，减少暖机期间的迁移。保存布局只是启动提示，不意味着跳过上线后的负载再验证。

PR6 先实现背景更新的副本分发权重：慢速规划布局，较快根据新近负载更新同一布局中的分配比例，在现有融合 kernel 内完成抽样/选择。这不等价于每个 batch 的全局最优分发，但避免为每层引入新的同步求解。

只有当前布局已较好、短期批次波动仍明显、且单次 MoE 计算足够长时，才评估 LPLB 类逐 batch 求解。统计通信、求解耗时和额外 kernel 必须包含在收益比较中。短 chunk、小 batch 或通信受限场景优先使用较便宜的分发策略。Libra 式预测下一层需要改变更多执行依赖，作为后续研究方向单独验证。

主要代码改动对应如下：

| 文件/模块 | 计划职责 |
| --- | --- |
| `lightllm/common/basemodel/triton_kernel/fused_moe/grouped_topk.py` | 稳健副本哈希，后续可选加权选择 |
| `lightllm/models/deepseek_v4/triton_kernel/csrc/moe_topk_eplb.cu` | 与 Triton 路径一致的物理分发语义 |
| `lightllm/common/basemodel/layer_weights/meta_weights/fused_moe/eplb_placement.py` | 候选接口、统一评分、受限搜索、完整布局与拓扑策略 |
| `lightllm/common/basemodel/layer_weights/meta_weights/fused_moe/expert_parallel_state.py` | 完整布局状态及版本 |
| `lightllm/common/basemodel/layer_weights/meta_weights/fused_moe/fused_moe_weight.py` | 按完整布局加载权重和量化参数 |
| `lightllm/server/router/model_infer/mode_backend/eplb_manager.py` | 新鲜采样、候选选择、回本判断、发布与持久化 |
| `lightllm/server/router/model_infer/mode_backend/eplb_transfer.py` | 任意物理行、覆盖环、GPU staging 预算和版本一致提交 |
| `lightllm/server/router/model_infer/mode_backend/ep_balance_monitor.py` | 复用实际计算负载统计，补充物理副本与耗时反馈 |
| `lightllm/common/eplb_utils.py`、DSv4 memory profile | staging/采样峰值/量化参数的内存预算 |

实验矩阵按现有硬件选择，不预设论文中的 GPU 与线上硬件相同：单节点与多节点；短/长 prompt；小/大 chunk；低/高并发；均匀、热点、热点漂移和真实混合语料；已有量化及 overlap 路径。副本预算先扫现有合法的 R，再在完整布局支持后加入 R=0。

每次试验报告两组结果：固定 batch/相同 KV 预算下的推理差异，以及相同总显存/同一 TTFT SLO 下允许各方案使用剩余 KV 空间的服务收益。这样既能识别算法收益，也不会掩盖副本与 staging 占用显存的代价。PD 全链路还要记录 KV 传输耗时，避免将其瓶颈误判为 EPLB 无效。

| 验证层次 | 必要验证 |
| --- | --- |
| 语义 | logical top-k 与权重系数保留；同精度 logits 在既定容差内；有效 token 不丢失、不重复执行 |
| 布局 | 全专家覆盖、固定容量、本地唯一性、拓扑限制与确定性 |
| 迁移 | 量化权重及 scale 一致；覆盖环；零任务 rank；不同源/目标；多层连续更新；staging 复用 |
| 并发 | microbatch overlap、空 DP rank、全局 collective 顺序、旧/新布局切换 |
| 性能 | MoE/forward 时间、prefill tokens/s、TTFT P50/P95/P99、迁移期间尾延迟、HBM 占用和 KV 容量 |
| 长时间运行 | 稳态无效迁移数量、域切换收敛时间、预测收益与实际收益偏差 |

放行以重复运行且超出测量噪声的吞吐收益、满足既定 TTFT SLO 和显存预算为准。均匀/小批量负载应能保留原布局或选择便宜策略。若真实工作负载上完整重排不能覆盖其迁移成本，可以发布 PR1/PR2/PR5 的收益，完整重排维持可选；无需为了形式上接近其他框架强行启用。

关于 decode 的补充：所核对的 vLLM 与 SGLang 均支持在 decode 中使用 EPLB 物理布局，并推进周期性重平衡状态；不等于每个 decode step 都重新搬权重。其完整布局能力更接近 wzj_eplb，持续控制和部分异步机制又与 eplb 有相似之处。具体后端/模型仍受各自支持条件约束。[vLLM 调用入口](https://github.com/vllm-project/vllm/blob/94f4170df37fe29d68bd2f7e5a496501b78d55ae/vllm/v1/worker/gpu_model_runner.py#L4737)、[SGLang 调用入口](https://github.com/sgl-project/sglang/blob/40048f6e51891e984258d34c33c5a9f2c1f542f6/python/sglang/srt/model_executor/model_runner.py#L1765)。

METRO 的结论针对 memory-bound 时按 token 均分副本的代价；它保持 EPLB 的专家布局，并替换 token 分发。论文以 PD 混部为主要收益场景，§VII-A 讨论 PD 分离时，无额外副本优于传统高副本 EPLB 的情形，以及 METRO 相对无副本的较小收益。不能据此断言 decode 永远不应重排、复制或均衡。[METRO 正文](https://arxiv.org/html/2512.09277v1)。

“观点有实验支持”“论文已被正式录用”“主流框架已经采用”需要分别核实。所查作者主页仍将 METRO 标为 In Submission；未据此宣称正式录用或业界共识。TEMPO 是讨论并扩展 token/活跃专家两种近似目标的后续预印本，也不能作为已形成共识的证明。[作者主页](https://yanpeng-yu.com/publication/)、[TEMPO](https://arxiv.org/abs/2608.13057)、[LPLB](https://github.com/deepseek-ai/LPLB)、[Libra](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9ff1ac9a659085fed0735362cafe5e53-Abstract-Conference.html)。
