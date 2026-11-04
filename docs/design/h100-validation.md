# H100 验证记录

2026-11-03 起按用户要求，通过 `ssh h100` 进行 GPU 验证。七模块重构仍在进行中。

最近完整结果：`mla-mtp-routing-full.xml` **2992 passed、8 failed / 3000**，567.82 秒。原 2956 项全部保留，新增 44 项全过，原八项 BF16 失败身份不变，零 error/skip/deselect；272 项专项不重复累加。完整七模块验收仍未完成。

## 环境与复现

- GPU：NVIDIA H100 80GB HBM3，驱动 580.167.08。完整回归使用 0、1；独立 checkpoint 检查使用 2；NCCL/NIXL 驱动检查使用 2、3。每次启动前检查占用。
- 复用镜像：`sha256:707f6da52112ca52f94a4a1702a26fcd0a0706be530f0b00163650bbcf26c51f`。
- Python 3.10.22、Torch 2.11.0+cu130、Triton 3.6.0、Transformers 5.8.0、sglang-kernel 0.4.5、FlashInfer 0.6.18、FlashMLA 1.0.0+47c35a7、FlashQLA 0.1.2、DeepGEMM 2.5.0+891d57b、NIXL 1.2.0、light-mem 0.1.0。
- 隔离目录：`/home/devsft/models/lightllm-refactor-a716ad58`。代码、测试清单和逐文件 SHA256 同步后校验；容器退出即删除。HTTP 验证只监听隔离容器内的 localhost，不发布端口或部署宿主服务。pytest 8.4.2 及其纯 Python 依赖从现有本地环境复制至独立目录，没有修改共享 Python 环境或镜像。
- 代码只读挂载；Triton、FlashInfer、CUDA、DeepGEMM 及 LightLLM autotune 各自使用独立可写缓存。首次遗漏 LightLLM 源码内的 autotune 目录，造成三个测试在执行 kernel 前失败，已修正挂载。
- 正式权重：`/home/devsft/models/Qwen3.5-0.8B`，只读挂载，不下载模型。

远端日志位于上述目录的 `logs/`，本地副本位于 `/tmp/lightllm-a716ad58-h100-evidence/logs/`。复现脚本和清单保存在远端，运行前重新检查对应 GPU 是否空闲。

## 验证中修复的问题

1. **INT4/INT8 BF16 scale 下溢。** 新工具链可保留 BF16 次正规 scale，先前依赖类型转换自动归零的写入结果会变化。现在 writer 显式将 BF16 次正规 scale 归零；参考检查覆盖零值、极小值、正常值及阈值附近输入，payload 和 scale 逐字节比较。
2. **新版 RoPE 配置。** 通用配置归一化读取平坦的 `rope_parameters`，同步 `rope_scaling`、`rope_theta` 和 `partial_rotary_factor`，保留已有明确的旧字段。DeepSeek 默认 RoPE 不要求存在 `factor`，缺省按 1 处理。CPU 状态声明与模型启动共用归一化结果。
3. **norm 权重相加精度。** `NoTpGEMMANormWeight` 和 `QKGEMMANormWeight` 用 FP32 保存 `1 + checkpoint_offset`，避免先在 BF16 中相加丢掉小偏移。输入输出 dtype 保持模型声明值；独立 Torch 算子参考覆盖 FP16/BF16。
4. **参考模型构造兼容性。** Transformers 5.8 对专家数量进行整数校验；随机 Dense MLA 用有效专家数量并由 `first_k_dense_replace` 保持全部层为 Dense。新版内部融合的专家参数导出为 checkpoint 的逐专家 gate/up/down 权重，数学参考及误差断言不变。
5. **V4 FlashMLA 输出接口。** 远端库不接受 `out=`。backend 在启动时检查签名；支持时直接传入已绑定输出，不支持时将库结果复制到该输出，保持 Graph 和调用者持有的地址稳定。旧接口会增加一次临时分配和复制，性能尚未验证。

## 已取得的证据

- MoE 集成后当前合并：`moe-final.log` / `moe-final.xml`，**1621 passed, 1 failed**，296.52 秒，1622 项无 skip 或 deselect。包含 108 项完整随机模型、29 项原生 MoE、24 项 Linear、18 项 GLM、既有 MoE kernel/激活和实际双卡搬运；唯一失败仍为 Qwen3.5 BF16 独立参考 RMS 1.785284% >1%。各专项与此集合重叠，不累加。
- 前轮 Linear 集成后合并：`linear-final.log` / `linear-final.xml`，**1571 passed, 1 failed**，295.56 秒，1572 项无 skip 或 deselect。包含 105 项完整随机模型、18 项 GLM 原生、24 项 Linear、37 项 GLM kernel、5 项原普通矩阵检查及实际双卡搬运；唯一失败仍是 Qwen3.5 BF16 独立参考误差。详见下方 Linear 章节。前轮分批数字与此集合重叠，不累加。
- GLM 迁移前合并：`final.log` / `final.xml`，**1485 passed, 1 failed**，260.34 秒；无 skip 或 deselect。唯一失败是下述 BF16 独立数值检查。包含 101 项完整随机模型、24 项 GLM5 Next 层、42 项 V4 FlashMLA、51 项 CPU 状态资源及 8 项双卡量化搬运。上述子集合与该结果重叠，不累加；当时 NCCL/NIXL 驱动专项另行执行。GLM 迁移后的合并及复验见下。
- 量化专项：`quantized-fixed.log`，**43 passed**，9.31 秒。包含编码、prefill/decode、Graph 和双执行 lane；与合并集合重叠。
- 双 H100 卡的 NCCL/NIXL：`transfer.log`，**2 passed**，27.24 秒。两个独立 CUDA 进程通过实际驱动搬运 packed KV、已接受的 recurrent 状态和尾部数据。属于传输驱动验证，不是完整 PD 服务验收。
- norm 专项：`fp32-reference.log` 中三个 norm 测试通过，包含既有 QK fused 检查及两个新增精度检查。该日志整体另有一个独立 checkpoint 精度失败。
- 正式权重对照：`precision.log`，**2 passed, 1 failed**，20.38 秒。通过项分别为 BF16 新旧 LightLLM 一致性、FP16 原生执行与独立 FP32 Transformers 对照。
- 完整随机模型 **101 项**、GLM5 Next 层 **24 项**、CPU 状态资源 **51 项**在 `full-h100.xml` 中全部通过。GLM NoPE decode 使用远端支持 `only_qv` 的 SGL 接口；此前本地旧 SGL 的 API 限制不能据此视为通用兼容性已解决。
- V4 FlashMLA：`v4-compatible.log`，**42 passed**，19.58 秒。包括真实 sparse kernel、输出地址复用和当前 query/长度变化后的 Graph 回放。初次合并的 29 个 V4 失败均为进入 kernel 前拒绝 `out=`，失败日志保留。

独立参考使用 Transformers 的 eager attention、Torch 卷积与 recurrent 运算。两个中文 prompt 分别经过 prefill、四次 decode、后缀 prefill，以及 Conv/SSM 保存、清零、恢复后的 decode。FP16 对照在全部 14 个检查位置通过逐请求 RMS 误差 1%、最大误差相对参考峰值 1/32 和 greedy token 一致性检查。

BF16 在全部 14 个位置的 greedy token 一致，但逐请求 RMS 误差最高约 **1.7853%**，超过新增检查设定的 **1%**，因此该检查仍然失败。FP32 参考有助于区分两个 BF16 实现自身的舍入误差；这不代表已经完成 BF16 精度验收。尚未测正式任务分数、完整多模态、MTP 或服务负载。

### BF16 差异定位

`trace-v2.log` 使用相同 checkpoint、相同 14 个输入位置，另外运行纯 Torch 的 BF16 Transformers。它相对 FP32 参考的最大逐请求 RMS 误差为 **2.4124%**，14 个 greedy token 也全部一致。首轮 prefill 的 24 层输出逐层比较中，LightLLM 与 BF16 Transformers 相对 FP32 的误差处于相近范围；第 24 层分别约为 1.1418% 和 1.1431%（对两请求全部输入 token 汇总，与最终 logits 的逐请求指标不同）。

这支持“低精度执行本身会累积超过 1% 的误差”这一判断，尚不能将失败定位为某个重构算子缺陷，也不能据此宣称没有精度问题。1% 检查未放宽、未 skip、未 xfail。后续需要正式任务精度和更广输入覆盖来确定可接受的数值边界。诊断脚本为远端隔离目录的 `trace.py`，使用同一 `8a92319b505e` 源码快照；诊断退出码不作为测试通过证据。

## GLM5 Next 原生绑定验证

新增 `Glm5NextModelAdapter`、KDA 与 pooled NoPE backend，并由 Planner 分别绑定历史 KV、Conv/SSM 和 indexer tail。实现与剩余范围见 [重构实施记录](refactor-progress.md)。本轮仍复用上述 H100 镜像和隔离目录。

- 合并回归：`glm-final.log` / `glm-final.xml`，**1540 passed, 3 failed**，313.00 秒，1543 项无 skip 或 deselect。包含原 1486 项、18 项新增 GLM、37 项既有 GLM kernel、2 项实际 NCCL/NIXL 驱动验证。101 项原完整随机模型及新增 4 项 GLM 完整小模型全部通过。
- 两个新增失败发生在长上下文测试初始化：底层存储额外保留一行，夹具将 16384 行数据复制到 16385 行视图，未进入 attention kernel。按实际容量修正夹具后，`glm-long-fixed.log` / `.xml` 的 **18 项全部通过**，18.46 秒。两个快照之间只有该测试文件和进度文档改变，执行代码相同。没有再次运行整个 1543 项集合，不能把此结果描述成一次“1542 passed, 1 failed”的合并运行。
- 尚未解决的失败仍为 Qwen3.5 BF16 的 1% RMS 检查，最大误差 **1.785284%**，与上一轮一致；阈值、输入和断言未改变。

新增 GLM 检查的证据范围：

1. KDA 对独立 Torch 卷积、门控和 recurrent update：FP16/BF16 输入、FP32 SSM，混合新请求与前缀请求，脏槽位复用、Conv/SSM 保存清零恢复、decode Graph 与请求重排。
2. pooled NoPE 对独立 Torch attention 和 weighted-ReLU/top-k：普通 FP16/BF16、BF16 group64/E8M0 KV；短序列捕获后跨过 top-k 阈值，HOLD 输出零。长序列覆盖 64 heads、2048 top-k、3955/3081 token 及 Graph 回放到 4093/3987 token，包含双 lane 索引区隔离。另有独立 Hadamard 与 FP8 payload/scale 字节比较。
3. 四种完整随机 GLM 小模型：Triton、SGL、FP8 KV、mHC 开关与 MoE，验证真实权重加载、八个逐层阶段绑定、logits/greedy、前缀续算及两类 Graph；绑定后禁止 manager 查询。参考为旧 LightLLM，属于迁移一致性，不是官方 GLM checkpoint 或独立整模型精度。
4. 既有 37 项 GLM kernel 全部通过，包括 KDA、池化边界、碎片 token、百万 token 的池化索引、超过 2 GiB 的 packed KV、mHC 和 NoPE Graph。它们不表示新的完整模型已经在百万 token 或多卡上验收。

验证中修正了 KDA decode 的 128 维支持条件、FP8 kernel 的 H100 shared-memory 超限、FP16 Hadamard 输入检查、SGL sparse prefill 的 BF16 限制，以及 decode Graph 从短序列增长时的 indexer 分支选择。SGL FP16 prefill 在 probe 拒绝；auto 可组合 Triton prefill 与 SGL decode。Graph 测试快照补存历史 KV，避免只恢复 bounded 状态形成不完整恢复点。

失败证据保留在 `glm-native.log`、`glm-native-v2/v3/v4.log`、`glm-native-v5-focused.log` 和 `glm-final.log`。`glm-native-v5.log` 是误用旧清单后主动停止的运行，不作验收证据。合并退出还记录了 CUDA IPC 引用释放顺序警告，后续生命周期审计仍需处理，驱动测试通过不代表完整 PD 服务验收。

相关 14 个 Python 文件通过 Black、Flake8 和编译检查，`git diff --check` 通过。最终本任务临时容器均已退出，八张 GPU 均为 0 MiB；保留既有 `lightllm-ci-agent` 容器，没有修改共享依赖或启动服务。

## Linear Compute 启动绑定验证

已有九个无 MTP ModelAdapter 的权重加载完成后，由 Planner 绑定普通 Linear/BMM、Triton FP8 per-tensor/per-channel group64/128/block128、SGL FP8 per-tensor。推理入口直接调用持有原生张量和分配函数的 `BoundLinear`，不再通过权重 loader 的 `QuantizationMethod.apply()` 计算。未迁移的 AWQ/Marlin、vLLM、DeepGEMM 及 MoE 仍保留在剩余范围，不能据此宣称 Compute 已全部迁移。

- `linear-final.xml` 确认 **1571 passed, 1 failed**，295.56 秒，无 skip/deselect；包含上一轮全部 1543 项，加上 24 项 Linear 和 5 项原普通矩阵检查。此前两个 GLM 长上下文夹具修复已在这次完整集合中通过。
- 24 项 Linear 检查覆盖格式/设备不兼容拒绝、FP16/BF16 普通矩阵/BMM/bias/非连续视图/FP32 输出/空批次、FP8 per-tensor/per-channel/block128 的独立 Torch 量化与矩阵参考、显式输出和 Graph 更新。量化结果要求 RMS 小于 1%、最大误差小于参考峰值的 1/64。
- 加载集成包含 TP=2、rank=1 的本地权重切片及融合 gate/up 分批加载；未加载完整时拒绝绑定。绑定后替换 loader 的 `WeightPack`，并禁止旧 `apply()`，已绑定执行仍使用原视图并与旧执行逐值一致。这是单卡 TP 切片检查，不是两卡完整模型 TP 验收。
- 原 101 项完整随机模型在执行前禁止 `NoQuantization.apply()`，验证普通 Linear 已经过新绑定路径；另有 4 项完整 GLM 小模型。正式 Qwen3.5 的 legacy 一致性与 FP16 独立 FP32 参考通过，BF16 仍以 **1.785284% > 1%** 失败，数值与上一轮一致。
- SGL 在未传入输出缓冲时直接返回库结果；传入输出缓冲时复制回原地址。六个相关 Python 文件通过 Black、Flake8、编译与 diff 检查；FP8 临时张量和库内存的完整峰值预算、性能验收尚未完成。

首轮 `linear-native.xml` 为 149 passed、1 failed：新 TP 夹具使用 `gate`/`up` 而非 checkpoint 的 `gate.weight`/`up.weight`，导致 block128 的 scale 名称与权重名称冲突，失败发生在加载。修正夹具并补全格式、SGL 能力检查后，`linear-fixed.xml` 的 24 项通过；之后的完整合并同时验证 SGL 可选输出路径。上述专项与完整结果重叠。初始本地仅执行两个 CPU 契约检查，GPU 验证全部通过 `ssh h100`。

`linear-final.log` 仍记录 CUDA IPC 引用释放顺序警告，未把传输驱动通过当作生命周期全部完成。所有本任务临时容器退出，八卡显存释放；未安装共享依赖、启动服务或提交代码。

## MoE Compute 启动绑定验证

- 新增 `test_moe_compute.py` 29 项：CPU 声明/格式检查；普通及 block128 FP8 × FP16/BF16 × softmax、分组 top-1/top-2、共享专家门控和两种截断激活；route 权重/专家集合、独立 Torch 专家数学、CUDA Graph 更新、空批次、路由回调；TP=2 rank1 本地加载/切片/绑定；160 专家与两条并发 Graph 的输入和暂存隔离。
- 路由权重检查 `rtol=1e-5, atol=1e-6`、专家集合 exact；专家输出检查全张量 RMS <1%、最大差 <参考峰值/32。参考显式包含已声明的 BF16/FP16 存储舍入、FP8 量化，以及 clamp 在 FP32 中计算的语义。阈值未放宽。
- 普通 TP 切片绑定输出与旧路径 exact；FP8 改用独立 Torch 参考，因为原生绑定明确选择了 activation 量化实现，不能再要求与依赖 SGL 隐式选择的旧路径逐位相同。二者均在绑定后替换 loader 的 WeightPack、禁用旧执行入口，再验证原生视图仍然可执行。此为单卡 TP 切片，尚非完整分布式 TP 模型。
- 原完整模型回归增加 3 项 Qwen3Next MoE 配置（Triton、FA3、FPI/FlashQLA attention），检查带共享门控的模型、prefix、decode/prefill Graph。模型回归逐层禁用旧 `FusedMoeWeight.experts()`；已有 DeepSeek/GLM MoE 使用原生绑定。普通完整随机模型现为 108 项（原 105 +新增 3），正式任务精度仍需另测。
- 首轮 `moe-native.xml`：116 passed、15 failed，68.44 秒，达到 maxfail 后终止。11 个模型启动失败来自全 Dense 测试配置带有不适用的 MoE 字段，已改为只声明实际 MoE 层；4 项 FP8 数值失败保留。
- `moe-trace.log` 对同一快照的诊断显示：SGL activation 编码与独立 Torch 参考 scale 一致，但 17×256 输入有 5 个 E4M3 字节不同，34×256 中间输入有 13 个不同。原生后端因此显式选择 LightLLM Triton activation 量化；未修改第三方库。诊断在打印 Graph 内部张量时触发 capture 错误，仅作为首次 eager 量化证据，不是通过的回归。
- `moe-v2.xml`：151 passed、5 failed，85.90 秒；104 项原模型文件及 18 项 GLM 全通过。剩余为参考提前舍入 clamp 常量、FP8 旧路径逐位比较、并发 Graph 测试未持有第一条 Graph 的输入张量。修正参考与输入生命周期后，`moe-fixed.xml`：**34 passed**（29 项原生 +5 项既有激活接口），7.59 秒，无跳过。执行实现与 v2 相同，测试阈值不变。
- 最终 `moe-final.xml`：**1621 passed, 1 failed**，296.52 秒，无跳过/排除。完整集合为上一轮 1572 项，加 3 项完整 Qwen MoE、29 项原生 MoE、5 项既有激活、9 项既有 grouped MoE、4 项 shared-expert 追加。唯一失败为原 Qwen3.5 BF16 独立 RMS 检查，最大误差与上一轮完全相同。退出仍出现 CUDA IPC producer 先释放引用警告；保留生命周期待审计项，NCCL/NIXL 通过不代表完整 PD 服务验收。
- 最终核对本任务容器全部退出，八张 H100 均为 0 MiB；既有 `lightllm-ci-agent` 未动。本轮 16 个 Python 文件通过 Black、Flake8 和 compile，文档 diff 检查通过；无提交、部署或共享依赖安装。
- 当前仍不覆盖 EP/DeepEP、其它专家量化、正式 MoE checkpoint、完整服务或性能。路由 callback 的 Runner 归属、真实全峰值预算、模型多卡及原 Qwen3.5 BF16 独立精度失败仍未完成。

## Runner 批次协调验证

- 通用 token 收集、padding、执行容量选择和输出裁剪已经接入 `BatchRunner`；模型保留专有状态构造、位置初始化和 recipe 回调。新增 5 项 CPU 契约检查覆盖空 prefill、双 microbatch、TP 对齐、单批次 Graph 容量、就绪事件顺序、原始输入/Graph 输出保留，以及失败时不释放可能在途的传输缓冲。
- 本地 `test_batch_runner.py` 与 `test_model_output.py` 共 **75 passed**。H100 完整集合再加入 EAGLE 输入、DSpark/DFlash 输出、无 attention 草稿的既有测试。`batch-final.xml` 为 **1671 tests：1666 passed, 5 failed**，295.91 秒，无跳过/排除；108 项完整随机模型、实际 NCCL/NIXL 驱动及原生 Compute 回归通过。
- 其中 4 个新增失败来自草稿测试通过 `__new__` 手工构造模型，却没有设置真实 `_pre_init()` 必定设置的 `mtp_previous_draft_models`。失败发生在读取状态角色时，未进入计算。补入空列表并检查 `att_state_role == "draft0"` 后，本地四项通过；H100 `batch-fixed.xml` 重跑全部 49 项新增集合，**49 passed**，6.55 秒，无跳过/排除。
- fixed 快照与完整回归相比，只改变上述测试文件及进度文档，执行代码相同。两次结果有重叠：去重后 1671 项覆盖中 1670 项已有通过结果，1 项仍失败；这不是一次 1670 passed 的完整运行，未将两次通过数相加。
- 唯一仍失败的是 Qwen3.5 BF16 独立 FP32 对照，最大 RMS 仍为 **1.785284% > 1%**，阈值未放宽，也未 skip/xfail。完整回归退出时仍出现 CUDA IPC producer 引用释放顺序警告；两卡传输驱动通过不代表完整 PD 服务或生命周期验收。
- 本轮 6 个 Python 文件通过 Black、Flake8、编译检查；完整 Runner 迁移仍缺 InferState 组装、warmup、缓存策略与路由记录。草稿辅助测试不是完整 MTP 模型/服务验收。
- 两轮容器均已退出，最终复核八张 H100 全部为 0 MiB，既有 `lightllm-ci-agent` 未动；没有共享依赖安装、提交或部署。日志与 XML 已复制到本地证据目录。

## InferState 组装与启动预热验证

- 新增 `BatchStateFactory` 和 `WarmupRunner`，默认批次直接使用绑定的状态工厂；decode/prefill Graph 不再持有模型来生成预热输入。autotune 与最大长度检查使用自己申请的请求槽及整页 KV，设备操作结束后仅释放这些资源；Graph/HOLD 预热不申请普通页，也不再调用共享分配器的 `free_all()`。
- `test_startup_runner.py` 的 14 项 CPU 检查覆盖 draft 角色元数据、原生/旧 backend 构造、两条 lane、原始字段引用、已有请求保留、分配/forward 异常清理、同步失败保留分配并停止、HOLD Graph 输入、层数与 autotune 阶段恢复。资源测试使用真实请求链表和 KV 分配器，替换的是设备同步和共享统计机制。
- 增加 4 项完整小模型启动检查：Llama Triton/FlashInfer、Qwen3Next Triton、DeepSeek2 Triton。先保留一个已分配请求，再执行 autotune、HOLD 初始化、decode/prefill Graph 捕获和最大长度检查，核对已有映射与容量，然后用真实输入回放并对照独立 Transformers 参考。GPU kernel、Graph 和同步为真实执行；单进程测试将 process-group barrier 替换为空操作，不是多卡启动验收。
- 首轮 `startup-native.xml` 为 **259 passed, 3 failed**，82.42 秒。一个真实启动失败来自 Qwen3Next autotune 假定配置必有 `full_attention_interval`，而显式 `layer_types` 可以省略该字段。现在按实际 recipe 的 attention/FFN 组合选择覆盖前缀，资源配置也以显式层类型为准。另两个 token 收集测试没有为手工假模型提供状态构造回调，已补齐并断言准备 token 时不会构造状态。
- 修正后的 `startup-fixed.xml` 为 **39 passed**，13.02 秒，无跳过/排除：4 项完整模型启动、14 项启动契约、3 项 overlap、18 项显式层类型/资源映射。此次新增检查还发现原 autotune 会生成超出小词表的随机 token，并可能选择超过实际模型深度的层数；均改为使用实际词表和 recipe 上限。
- 本地状态/批次/draft 组合 85 项通过，启动/Graph 组合 77 项通过，显式层类型/启动组合 32 项通过；这些集合有重叠，不能相加。另一次启动契约与 backend 组合为 14 passed、49 skipped，跳过项需要 CUDA，未把本地跳过视为通过。
- 最终同进程 `startup-final.xml`：**1706 passed, 1 failed**，1707 项，309.95 秒，无跳过/排除。集合为上一轮 1671 项，加 14 项启动契约、4 项完整模型启动、18 项显式层类型资源映射；专项通过数不再累加。112 项完整随机模型均通过，其中 4 项包含本轮新增启动流程。
- 唯一失败仍是 Qwen3.5 BF16 独立 FP32 对照，max RMS 为 `0.017852840945124626 > 0.01`，与此前完全相同；没有放宽阈值或 skip/xfail。退出仍有 CUDA IPC producer 引用释放顺序警告。该结果仍不构成正式任务、完整服务、多卡模型或性能验收。
- 12 个相关 Python 文件通过 Black、Flake8、编译及 diff 检查。完整回归与修正专项使用同一执行源码；之后仅更新两份设计文档。
- 三轮测试容器均已退出，最后检查八张 H100 全为 0 MiB，既有 `lightllm-ci-agent` 未动。完整日志与 XML 已保存到本地证据目录；没有共享安装、提交或部署。

## 代码快照

- 启动迁移首轮快照：`eede0cc7310653ddbc1c0c94e5a9d5a759af8713475f9c262cd51bc82bf3b961`（2109 个文件），脚本 `/home/devsft/models/lightllm-a716ad58-startup-native.sh`。修正后快照：`c3b25d9934e4497e73f48cdae4cf4876004e7f8cb36eec12552268881f9fe2b2`，脚本 `/home/devsft/models/lightllm-a716ad58-startup-fixed.sh`。
- 启动迁移完整合并复用上述修正后快照，脚本 `/home/devsft/models/lightllm-a716ad58-startup-final.sh`；日志为 `startup-final.log/xml`。

- Runner 完整合并源码摘要：`7d9ef52417aad3ee5581bceb3b42133e0c9f012a874e91cfe2f814b64dc80a6c`（2106 个文件），脚本 `/home/devsft/models/lightllm-a716ad58-batch-final.sh`。测试夹具修正摘要：`b023c68aadd385fe12452bd5a4f4f302a6b1bd892b7eda22dfd6fc2b22bcd6b4`，脚本 `/home/devsft/models/lightllm-a716ad58-batch-fixed.sh`；两者执行源码完全相同。

- MoE 最终合并源码摘要：`a506187f9d6bf195bf016c4f6ca7da6b63b3f7281ad29cd6da1a4d5a1736966f`（2104 个文件）。脚本 `/home/devsft/models/lightllm-a716ad58-moe-final.sh`。MoE 专项通过快照 `d839efe8a363908073f289150137d3899c7307fd4fc38ed0004ff1421e745142`；与 v2 的执行代码相同，仅修正测试。最后合并之后仅更新文档，执行代码未改变。

- 最初 H100 快照：`884e94c5845ad9679cf15513419085ef433e45f4baea99fbcba4bc9e4af0fc66`。
- 完整回归快照：`0287208121644a6eafa8e01e54d5ea4eb03d079010b20ef9318b5af81a648fdd`；脚本 `/home/devsft/models/lightllm-a716ad58-full-h100.sh`。
- 增加 FP16 独立对照后的快照：`51b872280017eeb2138726637b5496f20e1dafc6a28f0c7cd715ba01bf5c2310`；脚本 `/home/devsft/models/lightllm-a716ad58-precision.sh`。与前一快照的执行代码相同，仅扩展 checkpoint 测试。
- V4 输出接口修复后的快照：`8a92319b505e1995346239a2dc775b41c660d65b3ab5979acc34979852072739`；最终合并脚本 `/home/devsft/models/lightllm-a716ad58-final.sh`。
- GLM 原生迁移合并：`73f8acc3ad557cb44749b9d38f30bbc4bc1733650f2fce3f296399a58e7bec96`；脚本 `/home/devsft/models/lightllm-a716ad58-glm-final.sh`。
- GLM 长上下文夹具修正：`718e116bbd902b450b137c7e92b85e8f1ca75d14ee64cd723dd695876861e49c`；脚本 `/home/devsft/models/lightllm-a716ad58-glm-long-fixed.sh`。
- Linear 当前合并：`6e488d8a0e0e769db923a65178e2bd47f64c2050409074f97a22e0e957d9b9ec`；脚本 `/home/devsft/models/lightllm-a716ad58-linear-final.sh`。

快照摘要包含同步的源码、测试和当时的文档；本记录之后的文档更新不包含在上述摘要内。历史失败日志保留在 `baseline*`、`combined*`、`models*`、`checkpoint*`、`normalized*`、`fp32-reference*` 中，不与后续通过结果累加。


## 批次位置和模型元数据启动绑定

运行时不再调用 `init_some_extra_state(model)`。通用序列位置、普通/多组 RoPE、Qwen 动态 NTK/logn、多模态时空位置、V4 层需求及 C128 Graph 容量改为启动绑定；每批次只传 InferState。普通批次直接调用绑定函数，DSpark commit 使用同一初始化入口。该变化不等于 V4/MTP/多模态 Compute 全部迁移，相关旧 backend 和资源查询仍有剩余。

- `metadata-focused.xml`：**263 passed**，82.54 秒，无 skip/deselect。包含 34 项新增 metadata 检查、Qwen-VL 既有位置检查、112 项完整随机小模型、批次输出/overlap，以及 V4 层子集/双 lane/Graph 刷新、DSpark 因果提交与整块提议、图像缓存对齐、HOLD prefill 无 CUDA 标量读取和 target/MTP/DSpark 索引选择。
- 34 项 metadata 中 16 项 CPU 通过显式序列替身检查绑定与生命周期，16 项 CUDA 执行真实长度/位置与旋转表索引，另有 Gemma4 图像前缀可见性和 Graph 后创建容量检查。删除原模型、清空配置并替换原表属性后仍可运行，两批输出互不覆盖；不把 CPU 替身视作真实 kernel 验收。
- 首次完整 `metadata-final.xml` 在 **1 passed、15 failed** 后达到 maxfail，9.26 秒。全部失败来自资源启动夹具以 `__new__` 构造 Llama/Qwen，省略 RoPE 表；新启动绑定立即报告 `_cos_cached` 缺失。夹具补齐表并在 warmup 验证已经绑定同一张量，生产启动要求没有放宽。修正后的 `metadata-fixed.xml` 全部 **49 passed**，7.58 秒。
- focused 与 fixed 的执行代码相同，快照仅改变启动测试夹具和进度文档。首个快照 `c729af8fc0aaf48f3cb0e0b55db3f85108a9eac3700908b7f67f8bf5f5beee18`，修正快照 `3f0d0338545c86eda603652da653b4631691abb8798cd673e2588018dc0ca4f8`，均为 2111 个文件。脚本位于 `/home/devsft/models/lightllm-a716ad58-metadata-{focused,fixed,combined}.sh`，combined 复用 fixed 源码。失败日志没有覆盖。

- 最终完整 `metadata-combined.xml` 为 **1770 passed、1 failed**，1771 项，304.62 秒，零 skip/deselect。集合为上一轮 1707 项加本轮 64 项；263 项专项及 49 项启动检查均已包含，不能累加。XML 核对 108 项模型文件、18 项 GLM（含 4 项完整小模型）、34 项新增 metadata、26 项 V4、49 项共享启动与 2 项网络驱动均通过，112 项完整随机小模型覆盖保持不变。
- 唯一失败仍为 Qwen3.5 BF16 对独立 FP32 参考，max RMS `0.017852840945124626 > 0.01`，与前轮完全一致；没有放宽阈值、skip 或 xfail。退出仍有 CUDA IPC producer 引用释放顺序警告，生命周期和正式任务/服务/多卡模型/性能验收继续列为未完成。
- 本轮 22 个 Python 文件通过 Black、Flake8、编译和 diff 检查；最终 GPU 快照之后只有两份设计文档变化。四轮日志/XML 已保存到 `/tmp/lightllm-a716ad58-h100-evidence/logs/metadata-*`。所有本任务临时容器退出，八张 H100 显存均为 0 MiB，既有 `lightllm-ci-agent` 未动；没有共享依赖安装、提交或部署。


## 实例查询缓存与 CUDA IPC 引用回收

- 使用上一轮 metadata-fixed 源码分组诊断。`ipc-baseline-gqa.xml` 为 1 passed（15.07 秒），`ipc-baseline-runtime.xml` 为 6 passed（54.69 秒），`ipc-baseline-network.xml` 为 2 passed（25.68 秒）。仅运行态 IPC 组退出复现 `Producer process has been terminated before all shared CUDA tensors released`，不能把数据检查通过当作引用生命周期通过。
- 修复类级缓存持有 manager/State，以及 manager/Operator 双向强引用。查询结果改为实例缓存，复制和 IPC 序列化不携带本地缓存/Operator；导入后建立新的弱 owner 关系。移除测试对全局查询缓存的强制清理，新增关闭循环 GC 的回收检查和计划持有原生 Tensor 的真实 CUDA 搬运检查。
- `query-focused.xml` 共 481 项：479 passed、2 failed，133.07 秒，无跳过；6 项新增生命周期、6 项运行态 IPC、普通及全格式 peer IPC、实际 NCCL/NIXL 驱动均通过，进程退出没有 CUDA IPC producer 警告。两项失败均来自旧有界状态 CPU 快照夹具传入 CPU 空索引，按既有接口改为 CUDA 后 `query-fixed.xml` 的 2 项通过，3.06 秒。
- 定向快照 `25aee9fcce92d8726f50f9bed913deb01fa3baf1ec163e445532f6a5eb90e1d7`，修正快照 `915cf45253d5094c06ea08c9a00662e2d9e098588d416044570b888afaa1cbb2`，均为 2113 文件；两快照只差 `test_capacity.py` 的两处索引设备。脚本 `/home/devsft/models/lightllm-a716ad58-query-{focused,fixed,final}.sh`，final 复用 fixed archive/manifest。
- 本地 149 passed、47 skipped 只代表 CPU 可执行范围；新 GPU 检查在这之后加入，已由 H100 定向覆盖。另保留最初夹具漏填页参数、导入替换错误的失败日志。旧快照的回收断言失败与 FP8 标定夹具失败分别在 `/tmp/lightllm-a716ad58-query-before-v2.log`、`query-calibration-before.log`，修复没有降低校验门槛。
- 最终同进程 `query-final.xml`：**2147 passed、1 failed**，2148 项，325.72 秒，零 skip/deselect。完整清单去重后有 97 个文件/用例选择入口，包含此前完整集合与新纳入的容量、状态身份、CPU/DP/PD 传输检查。XML 核对 `test_model_cuda` 108 项、`test_native_glm` 18 项（含 4 个完整模型）、`test_query_lifetime` 6 项、`test_runtime_transport` 32 项和网络驱动 2 项全部通过；112 个完整随机模型覆盖保持不变。
- 唯一失败仍为 Qwen3.5 BF16 独立 FP32 对照，max RMS `0.017852840945124626 > 0.01`；没有放宽阈值、skip 或 xfail。完整退出日志已无 CUDA IPC producer 警告。未启动读者、部分导入、取消、peer 异常退出及跨 TP 失败仍不在本轮已验证范围；正式任务、服务、多卡模型和性能也尚未验收。
- 完整运行复用上述 fixed 源码；逐文件核对本地仅两份设计文档晚于快照。31 个相关 Python 文件通过 Black、Flake8、编译和 diff 检查。最终日志及 XML 已复制到本地证据目录，H100 八卡显存 0 MiB，`sudo -n docker ps` 仅见原有 `lightllm-ci-agent`；任务容器已退出。没有共享环境安装、提交或部署。

## group128 FP8 FlashMLA 原生执行

- `flashmla-native.xml`：36 passed、5 failed，41 项，10.96 秒，无跳过。三个 head 数的 prefill 与动态 Graph 已通过；五个 decode/完整模型失败均为 FlashMLA 的页内 stride 校验，未执行 decode kernel。将绑定视图的两个长度为 1 的维度设为 656 字节 stride，保留实际 token stride，没有复制整份缓存。
- `flashmla-fixed.xml`：43 passed，10.05 秒，无跳过。包含旧稀疏/Runner/Graph 检查、新增 16/64/128 heads × prefill/decode、长上下文 2048 top-k、调度/lane 检查，以及三个 FP8 完整随机模型组合。此处调度分组先直接构造，最终集合改为真实 Runner.prepare/bind/batch 调用，不以手工建立字典代替接入验证。
- 初始快照 `61d7aca4e3344e1ad4acfa388832d685c2639feda12a2a292156bc116a49065b`，stride 修正快照 `ac4bc237a6ae3be70e027be6f02e0c67bf316c2689a73ab31ef397d46d5687a5`；最终合并快照 `073a0ffa5761ef10a49505362a2899ecf8d734deaaff25af5ce0f6efaa59975d`，均为 2113 文件。脚本 `/home/devsft/models/lightllm-a716ad58-flashmla-{native,fixed,final}.sh`；final 使用上轮完整清单，新增参数自动纳入。
- 本地 sparse/Runner/Graph 集合 10 passed、23 skipped；后续 GLM/sparse 集合 6 passed、32 skipped，重叠不能累加。CUDA 检查由 H100 执行；6 个相关 Python 文件已格式化、编译，并通过 Flake8 和 diff 检查。
- 最终同进程 `flashmla-final.xml`：**2159 passed、1 failed**，2160 项，320.15 秒，零 skip/deselect。与 `query-final` 使用同一完整选择清单，文件内新增 9 项稀疏检查和 3 项完整模型。XML 核对 `test_sparse_mla` 20 项、`test_model_cuda` 111 项、`test_native_glm` 18 项、网络驱动 2 项和引用生命周期 6 项全部通过；完整随机小模型为 111 + 4 = 115 项。
- 唯一失败仍是 Qwen3.5 BF16 对独立 FP32 参考，max RMS `0.017852840945124626 > 0.01`；无阈值放宽或失败排除。完整日志没有 CUDA IPC producer 警告。最终快照后仅两份设计文档不同；三轮 FlashMLA 日志/XML 均已取回，失败证据保留。所有任务容器退出，八卡显存 0 MiB，`sudo -n docker ps` 仅见原有 `lightllm-ci-agent`。

## Qwen3.5 BF16 舍入隔离诊断

以 `query-fixed` 快照为基础，用独立 `rounding.py` monkeypatch 三个计算位置，不改变正式源码或测试断言。三次都执行原 checkpoint 测试的两个 prompt、prefill、四次 decode、后缀和 Conv/SSM 恢复后 decode，共 14 个 logits 检查位置。

| 临时计算改动 | 最大相对 RMS | 最大误差/参考峰值 | 原 1% RMS 门槛 |
| --- | ---: | ---: | --- |
| FP32 残差累加，norm 输出仍 BF16 | 0.0185214356 | 0.0110076191 | 失败 |
| 最终 logits 投影使用 FP32 输入/权重副本 | 0.0177764669 | 0.0124408491 | 失败 |
| 同时修改两者 | 0.0184496753 | 0.0106229866 | 失败 |

因此没有将这些改动合入，也没有据此判定 BF16 误差已解释或精度合格。脚本 `/home/devsft/models/lightllm-a716ad58-rounding.sh`，三个 `rounding-{residual,head,residual_head}.log` 已取回本地证据目录。脚本捕获断言以输出诊断数值，进程 exit 0 不代表精度检查通过。

## GLM5 Next group64/E8M0 FP8 FlashMLA 原生执行

- `pooled-native.xml`：37 passed，24.96 秒，零 skip/deselect。包括 GLM 原生执行、Runner 和 GraphProgram，新增两个 mHC 完整随机模型（有/无 MoE）、FP8 prefix/decode Graph、长 top-k 与双执行通道检查。
- `pooled-workspace.xml`：6 passed，6.31 秒，零 skip/deselect。使用至少 65536 长度的历史，覆盖 8/16/64 heads 的六种 query 形状、预绑定直接归约和选中 KV workspace、通用 gather、HOLD、预算不足时拒绝及通道隔离；数值与独立 Torch 参考比较。
- native 快照 `087be6d40f57a3fb7b5089ea715e72a37f88fa076c701235d1bcd126e954105f`；增加六项 workspace 检查后的快照 `d186801c0738d5693978fdaf1a32b71ed38bb1bfdea703c66f930f96cdcf9064`，均为 2113 文件。两份快照的生产代码完全相同，仅测试不同。
- 完整回归脚本 `/home/devsft/models/lightllm-a716ad58-pooled-final.sh` 复用 `pooled-workspace` 源码、archive 和 manifest；完整选择清单与 `flashmla-final` 相同，新增参数和测试自动纳入。相关固定 workspace 已计入 Planner；通用 prefill 行表、gather、动态 activation 和库内部临时张量不在完整峰值预算验收范围。
- 本地 10 passed、25 skipped（无 CUDA）；五个 Python 文件已通过 Black、Flake8、编译与 diff 检查。专项日志/XML 已取回 `/tmp/lightllm-a716ad58-h100-evidence/logs/`。
- `pooled-final.xml`：**2169 passed、1 failed**，2170 项，335.86 秒，零 skip/deselect。XML 核对 `test_model_cuda` 111 项、`test_native_glm` 28 项、网络驱动 2 项和查询生命周期 6 项全部通过；GLM 内含 6 个完整随机模型，完整模型总数为 117。专项与完整回归重叠，不累加通过数。
- 唯一失败仍为 Qwen3.5 BF16 对独立 FP32 参考 max RMS `0.017852840945124626 > 0.01`，阈值未改，未 skip/xfail；完整退出无 CUDA IPC producer 警告。完整日志/XML 已取回；GPU 快照后只改两份设计文档，生产和测试源码保持一致。全部 SSH/exec 完成，任务临时容器退出，八卡显存 0 MiB，`sudo -n docker ps` 仅有既存 `lightllm-ci-agent`。该结果不是正式任务、完整服务、多卡模型或性能验收。


## Qwen3-MoE / SenseNova Loop 模型入口

- 首轮 `loop-native.xml` 在 15 failed 后达到 maxfail，55.02 秒：Qwen3-MoE 六项因新版 `num_local_experts` 未映射而误走 Dense 权重加载；Loop FP16 九项在初始 prefill 数值失败。逐层诊断确定后者来自 `rope_parameters.partial_rotary_factor=1.0` 被序列化的旧顶层默认值 0.25 覆盖。修复两处配置处理，没有改变测试门槛。
- `loop-fixed.xml`：**111 passed、5 failed**，116 项，104.63 秒，零 skip/deselect。Qwen3-MoE 六项、Loop FP16 九项、BF16 九项中的四项、九项 mHC kernel/Graph 及 CPU/专家绑定检查通过。五项 BF16 失败保留：Triton/FA3 普通 Loop 的最大误差略超峰值的 1/64；三种 backend 的二流 mHC greedy token 不一致。
- 两份源码快照均为 2113 文件。初始 `c47d778f6e06a2a8fd3c0efc72eb5b4b3f8a0d9a2a147deca8327339f3247b72`；修复后 `6faf27ebc0ff689dd43123a33f3a8d761104f5694c88f0d375a9e2fabdfa15b8`。新增测试还覆盖六个专家配置组合、物理权重只加载一次、共享专家跨执行层绑定，以及 RoPE 优先级。脚本 `/home/devsft/models/lightllm-a716ad58-loop-{native,fixed,final}.sh`，final 沿用 fixed archive/manifest。
- `loop-diagnose.log` 的首次逐层诊断因只构造一个请求而与双请求夹具不匹配，pytest 内部错误退出；后续 `loop-diagnose2.log`、`loop-diagnose3.log` 修正诊断请求数并保留原失败，最后一份直接核对 rotary 维度和逐层偏差。无效诊断与原失败日志均保留。
- `loop-fp32-reference.xml`：6 passed、3 failed、9 deselected，51.22 秒。独立脚本在已导出 BF16 checkpoint 权重后将参考转换成 FP32，选择全部九个 BF16 组合，FP16 九项按诊断范围排除。二流 mHC 三项仍失败；原生最大两项 logits 分别同为 0.640625 或 0.9375，FP32 参考可区分。RMS 约 1.00%–1.05%、最大误差/峰值约 0.89%–1.06% 均通过既有门槛，但 greedy 检查未通过。
- `loop-fp32-logits.xml`：7 passed、2 failed、9 deselected，51.47 秒。保留上述 FP32 参考，额外临时将原生最终词表投影改为 FP32；FlashInfer/FlashQLA 二流和四流 mHC 仍在 decode 或后缀步骤有 greedy 不一致。脚本真实返回 exit 1。两个诊断均未改变生产代码或正式测试，不将换精度后的实验通过数计入验收。
- 本地 51 passed、32 skipped（无 CUDA）；11 个 Python 文件通过 Black、Flake8、编译与 diff 检查。完整回归继续包含原 Qwen3.5 BF16 正式 checkpoint 精度失败，不排除任何已知失败；随机小模型不能替代服务、多卡或正式模型任务验收。

- 最终 `loop-final.xml`：**2207 passed、6 failed**，2213 项，335.85 秒，零 skip/deselect；保持上一轮完整清单并增加 `test_loop_mhc.py`，原清单内新增用例自动纳入，共增 43 项。复用 fixed 快照，六项失败为同一组五个 Loop BF16 用例与 Qwen3.5 BF16 正式 checkpoint 独立参考检查，后者 max RMS 仍为 `0.017852840945124626 > 0.01`。没有放宽、skip 或 xfail；专项与完整结果不能累加。
- XML 核对 `test_model_cuda` 130 passed、5 failed；GLM 28 项（含六个完整随机模型）、mHC kernel/Graph 九项、查询生命周期六项和网络驱动两项全部通过。因此完整随机模型为 136 passed、5 failed（141 项），不是正式 checkpoint 任务或多卡服务验收。完整退出无 CUDA IPC producer 警告。
- 日志/XML 已归档到本地证据目录，源码与 fixed 快照一致，仅两份设计文档后来更新。全部任务临时容器退出，八卡显存均为 0 MiB，仅原有 `lightllm-ci-agent` 在运行；未共享安装、提交或部署。


## Hopper FP8 EP 计算与传输分离

- EP 镜像：`sha256:d75da36ca205cdf6c9a9de79354c4570f5e5e7e4c17d990a70f414a0d50a9c94`，Torch 2.11.0+cu130、DeepEP 2.1.0+60d4403、DeepGEMM、NCCL 2.28.9。该镜像的 SGL 不支持 `only_qv`，只用于 EP 专项；完整模型回归继续使用原 `707f6da52112...` 镜像。没有改变共享安装。
- 双卡 EP 使用 GPU 0、1；隔离容器显式映射 `/dev/infiniband` 并设置 `--ulimit memlock=-1`，保持 `--network none`。库报告 NIC GPU 映射不可用而使用 CPU fallback，测试没有证明最优网络性能。完整回归另用 GPU 2、3。
- 初始 `ep-native.xml` 39 passed、2 failed/41，25.07 秒，失败为容器缺少 RDMA 设备、ElasticBuffer 的 NCCL GIN 初始化失败。映射设备后，`ep-fixed.xml` 42 passed、2 failed/44，43.33 秒；`ep-runtime.xml` 42 passed、2 failed/44，44.33 秒，分别暴露测试夹具缺少 global rank、node rank 等环境初始化。随后按生产初始化补齐全部分布式环境字段，增加四个模型族 × 两阶段的 recipe 检查。
- `ep-ranked.xml` 50 passed、2 failed/52，153.26 秒：非幂次 scale prefill 数值失败；幂次 scale 已进入双批次 decode，但测试未及时消费 receive hook 而发生 DeepEP timeout/CUDA launch failure。prefill 改为显式 Triton 量化，测试按实际模型 recipe 的顺序消费 hook 后，`ep-quant.xml` 51 passed、1 failed/52，52.89 秒。剩余非幂次 decode RMS 为 0.015833476558327675，大于 0.015；最大误差比 0.02216312102973461 通过。
- 独立 `ep-trace` 与 `ep-trace-rounding` 诊断保留失败条件，各 1 failed，30.13/30.56 秒。接收数据与原参考存在少量 FP8 编码差异；以实际接收数据为输入的专家计算 RMS 为约 0.20%–0.75%。镜像内 `/root/DeepEP/csrc/kernels/legacy/utils.cuh:494` 和 `internode_ll.cu:223` 确认：下限幅值为 1e-4，非幂次 scale 用 `amax * (1 / 448)`，编码则乘 `448 / amax`。原参考除以已舍入的 dequant scale，FP8 中点舍入不同。
- 正式独立 Torch 参考据此区分 prefill 和 decode 输入量化：用 FP64 除法再存成 FP32 表达 CUDA float 除法，之后仍按 BF16/FP8 存储舍入与 Torch 专家矩阵/激活计算。没有调用被测量化 kernel 生成参考，没有修改 RMS <1.5% 或最大误差 <峰值/32 的门槛。
- `ep-reference.xml` **52 passed**，54.40 秒，零 skip/deselect；包括两个完整双 rank 子链路测试。最终快照 `4936943e4b7dcfb0357db01f34e366cfdb67b4b489a30adb4d820ccf6cc840ad`，2117 文件。绑定后清除权重 loader 和全局通信 buffer 入口，验证原生视图仍可执行；跨层共享 Transfer；两种 scale 模式均验证正常 prefill/decode、冷专家 rank、decode Graph 更新、两个在途 microbatch。
- 历史快照依次为 native `44b2ca3fae0e10e2228345c64a853b7d943f699378f3a6daaa2529d4088f4d6d`、fixed `41498b8e8bb7a739c907ddd46424ff43ad8793c8950fcaaa800bd350ab16cbd3`、runtime `46874970fb31f43ec6a4b8d614ca1c6d60aa179c1db9f002c67144ed93e37c1d`、ranked `7186b8c8316a883a09779d28c0099dc1fed8e085410b7e5f7e6d52130a48d46b`、quant `c98aecbfacab02e0495316480fff5c528d6fd1f3bc38e4ec27c4fe6af15048df`。两个隔离诊断分别 `951c5d2c1771a4dfb8d7a8b33b8997b7c44bf900ae5537a757ccde9ffb465b03` 和 `f35144dd2ef3720f58dfc57b777bf0b7d67a1d2a8cdcef6c1536a0eea36c2c04`，诊断插桩没有合入正式测试。
- 本地 14 passed、2 skipped（无 CUDA）；13 个修改 Python 文件通过 Black/Flake8/compile/diff。EP 专项不是完整模型 EP、正式 checkpoint 任务、服务或性能验收。此前六个 BF16 失败继续保留，整体目标未完成。

- 最终 `ep-full.xml`：**2233 passed、6 failed**，2239 项，350.69 秒（JUnit 350.672 秒），零 skip/deselect。复用 `ep-reference` 快照和原完整 2213 项清单，增加 14 个 EP CPU 契约/recipe 检查与 12 个 DeepGEMM activation 组合；两组真实双卡 EP 用例单独在 DeepEP 镜像验证，不以缺依赖跳过它们来宣称完整通过。
- XML 对照确认六个失败的 classname/name 集合与 `loop-final.xml` 完全相同；Qwen3.5 BF16 max RMS 仍为 0.017852840945124626。`test_model_cuda` 130 passed、5 failed/135；GLM 28 passed（含六个完整模型）；新增 26 项全部通过，原有网络/状态/Graph 回归无新增失败，未改变既有精度门槛。
- `ep-full.log` 没有 CUDA IPC producer 退出警告。所有 EP 初始失败、两轮独立诊断、专项与完整 log/XML 已取回本地证据目录。复现脚本 `/home/devsft/models/lightllm-a716ad58-ep-{reference,full}.sh`；完整回归镜像与 EP 镜像不同，不能合并成一次全模式验收。
- 最后核对：执行源码/测试保持验证快照，仅设计文档后改；所有本任务容器退出，八卡显存均 0 MiB，仅原有 `lightllm-ci-agent`，全部 SSH/scp 会话终结。未安装共享依赖、提交或部署。正式多卡模型、服务、剩余精度与性能仍待验收。


## 路由记录绑定与 BF16 隔离诊断

- `loop-output-precision.xml`：6 passed、3 failed/9，50.71 秒，9 个 FP16 组合按诊断范围 deselect。只把参考及原生的最终词表投影输出改为 FP32，中间保持 BF16；二流 mHC 的 Triton/FA3 decode 和四流 mHC 的 FlashInfer/FlashQLA 后缀 greedy 检查仍失败。与此前整份参考升 FP32 的实验不同，本次没有改变参考中间计算精度；也没有合入正式测试或生产。
- `gdn-rounding.xml`：16 passed、6 failed/22，123.18 秒，11 项 deselect。隔离脚本让 decode 使用 prefill 的 L2norm 舍入和 sigmoid 门值舍入，保留全部原始断言。12 个 GDN 状态/恢复/Graph 用例通过，五项 Loop BF16 与 Qwen3.5 BF16 仍失败；Qwen3.5 max RMS 从 0.017852840945124626 变为 0.01941545121371746，因而不合入。这两轮使用此前 `4936943e4b7d...` 只读快照，脚本及失败 log/XML 保留。
- 路由记录重构初始快照 `d776b37e1cbd8186e5ebfebf30deb39cfa40aeb53d490a20b5c0a887be8841fd`，2119 文件；`route-native.xml` **77 passed**，54.52 秒，零 skip/deselect。在原 EP 专项 52 项上加入路由绑定 12 项和既有 route manager 13 项，已有真实双 rank 用例也增加独立逻辑 ID/槽位断言。
- 最终快照 `c1826ed1d93eeca0b14c59cce614ce69c7e946e641acde8279cc3ffc64ae4b0b`，2119 文件；明确拒绝隐式复制 CPU token 索引，并在测试中同步 GPU 后清空 pinned 记录。`route-fixed.xml` **77 passed**，56.03 秒，零 skip/deselect。双卡 EP 的两种 scale 模式、decode Graph 更新、两个 microbatch 均核对真实 routed ID，数值门槛未变。
- 新记录测试覆盖 uint8/int16、两执行层、两批次、prefill/decode Graph 的索引更新、HOLD/负索引过滤、shared expert 前的非连续 routed 列、释放 manager 后存储引用安全、绑定变化拒绝回放；CPU 检查覆盖启动 rank/主模型隔离和旧 DeepGEMM split API 的逻辑 ID 选择。仍未验证真实 HTTP routed_experts 响应、生产负载或完整 EP checkpoint。
- 最终本地 31 passed、15 skipped（无 CUDA），15 个修改 Python 文件通过 Black/Flake8/compile/diff。脚本 `/home/devsft/models/lightllm-a716ad58-route-{fixed,final}.sh` 分别使用 DeepEP 镜像 GPU 0/1 与原完整回归镜像 GPU 2/3；源码只读、模型只读、依赖和缓存均沿用任务隔离环境。

- 最终 `route-final.xml`：**2258 passed、6 failed/2264**，347.75 秒（JUnit 347.732 秒），零 skip/deselect。原完整 2239 项全部保留，追加 route binding 12 项及既有 route manager 13 项；25 项全部通过，后端启动 49 项也通过。失败 classname/name 与 `ep-full.xml` 精确相同；Qwen3.5 BF16 max RMS 仍为 0.017852840945124626。完整随机模型仍为 136 passed、5 failed/141。
- 完整日志无 CUDA IPC producer 退出警告；`route-{native,fixed,final}.log/xml` 与两轮精度诊断证据均取回本地目录。最终代码/测试未偏离快照，只有两份设计文档后改；全部 SSH/scp 终结、任务容器已退出、八卡 0 MiB，仅既有 `lightllm-ci-agent`。未共享安装、提交、部署或启动 HTTP 服务。


## 真实 HTTP 服务与模型 dtype 通信探测

隔离容器中启动 `lightllm.server.api_server`，使用正式 Qwen3.5-0.8B 权重、FP16、Triton Dense/GDN、FP32 SSM、decode Graph、128-token chunked prefill。监听容器内 `127.0.0.1:18111`，`--network none`，没有宿主端口或共享服务变更。健康检查只是进入验收的条件。

- `service-native.json`：单卡服务通过，记录 14 个检查点。三个 prompt 各生成 12 个 greedy token；验证前缀复用、三请求并发、流式输出与普通输出一致、中断流后再次生成、错误参数返回 400 后再次生成。独立 Transformers FP32 eager attention / Torch recurrence 对照全部 36 个 token 一致。
- `service-cpu-fixed.json`：CPU cache 服务通过，记录 25 个检查点，包含相同生成链路、10 个逐出请求和恢复。目标请求 `service-cpu-fixed-19` 的服务日志证实 **GPU 缓存命中 0、CPU 命中 224、disk 命中 0**；恢复后的 12 个 token 与初次生成相同，三个 prompt 对独立 FP32 参考的 36 个 token 仍全部一致。
- 上述两项源码均为 `c1826ed1d93eeca0b14c59cce614ce69c7e946e641acde8279cc3ffc64ae4b0b`。检查点不是 pytest 测试数，不与完整回归累加，也不代表正式任务准确率、性能或其它模型服务验收。
- 初次 `service-cpu.json` 因验证脚本读取不存在的 `cpu_prompt_cache_len` 响应字段失败。现有 HTTP 协议只导出合计缓存量；修正脚本以唯一 `X-Request-Id` 匹配已有 debug 日志，核对三级命中量，没有修改服务协议。初始失败日志/JSON 保留。
- 初次真实双卡 `service-tp2.json` 在启动 warmup 失败，尚未接受请求。原环境探测只验证 BF16，不能证明模型 FP16 可运行；实际 two-shot 算子抛出 `NotImplementedError: "two_shot_all_reduce" not implemented for 'Half'`。

`SymmMemAllreduce` 现在在实际进程组建立后，使用模型 dtype 和所选 two-shot/multimem 算子执行启动探测。缺少算子或 dtype 未实现时完成已提交操作后关闭该候选，保留 FlashInfer/NCCL 分发；真实运行错误继续抛出。运行期调用已选定的算子，不在每批重新选择。

- `collective-cleanup.xml`：双 H100 **19 passed**，17.65 秒，零跳过；包括三种 dtype（FP16/BF16/FP32）的真实双 rank 归约、Graph 回放与输入更新，以及启动回归。BF16 必须实际启用 SymmMem；不支持的 dtype 验证 NCCL 回退结果。
- 最终快照 `542563a487456356580a3a7d0daf847d414c5ec3447f5033e3ab09acd9a7357b`，2120 文件。脚本与 manifest：`/home/devsft/models/lightllm-a716ad58-collective-cleanup.*`。
- 首轮 `collective-fixed` 在测试结束清理时等待，没有生成最终 XML；两 rank 的 gdb 栈均停在 `ncclCommFinalize`。测试仍持有引用 communicator 的 CUDA Graph，修正为同步后先 `graph.reset()` 再销毁通信组，并给子进程加入有界等待和回收。原日志、双 rank 线程栈和手动终止退出码 137 保留，不能记为通过。
- 本地 **2 passed、3 skipped**（无 CUDA）；两个 Python 文件通过 Black、Flake8、编译与 diff 检查。此前完整 `route-final.xml` 仍是旧快照的 2258 passed/6 failed；没有将专项结果拼成新的完整运行。

单卡服务正常退出码为 0，但 resource tracker 仍报告 4 个 semaphore 待清理；这项退出生命周期问题尚未解决。上述验收没有覆盖 OpenAI chat、多模态、HTTP routed_experts、PD/MTP/完整 EP 服务、disk 恢复或压力性能。六项 BF16 精度失败和其它架构迁移缺口继续保留。


修复后 `service-tp2-fixed.json`：**TP=2 真实 HTTP 请求链路通过**，14 个检查点、服务退出码 0，使用上述 `542563a48745…` 快照。服务日志证明两 rank 启动时关闭不支持 FP16 的 SymmMem two-shot，FlashInfer 仍启用；长 prefill、decode Graph、三请求并发、前缀复用、流式输出、中断恢复和错误请求恢复均完成，三个 prompt 共 36 个 token 与独立 FP32 参考一致。命令见远端 `lightllm-a716ad58-service-tp2-fixed.sh`，验证脚本 `service-final.py` SHA256 为 `da2c44a62e93a00e1936e624527f9680e3e8bd60c1bee3a57a6ba48df846d5a8`。

TP=2 退出另外暴露 `SharedMemory.__del__` 的 `BufferError: cannot close exported pointers exist`，以及 5 个 semaphore 待清理告警。请求链路通过不代表退出生命周期完整通过；没有忽略、修饰或删除这些日志。所有 service/collective 结果和首轮双 rank 栈已取回本地证据目录。测试容器全部退出，八张 H100 均 0 MiB，仅既有 CI 容器保留。只有两份设计文档偏离最终验证快照；未安装共享依赖、提交或部署。


## HTTP 后台任务与进程退出生命周期（2026-11-04）

本轮没有改变模型计算或放宽六项 BF16 检查。先用独立诊断记录资源创建/释放位置，再修复正常退出链路。

- `service-lifecycle-trace` 在原 `542563a48745…` 快照重现退出问题。诊断指向 HTTP 的 `lightllm_resource_lock`：`AtomicLockItem` 缓存的 memoryview 在最后一个 owner 关闭 mmap 时仍被导出。本地最小复现也触发相同 `BufferError`。现在切片仅在原子操作的 context 中持有，锁继续持有底层存储；新增检查覆盖最后持有者、同步/异步包装及异常后的再次加锁。
- 诊断中的五个未释放 semaphore 分别来自 Router 的 multiprocessing Queue（三个）和两个模型进程的 tqdm 锁。原 launcher 在 HTTP 退出后直接 SIGKILL 内部进程。现在内部进程通过 `graceful_registry` 注册 SIGUSR2 退出请求；launcher 排空 HTTP 后通知全部已注册 worker，统一等待五秒，再对剩余进程执行原强制回收。SIGTERM 对内部 worker 继续延迟处理，SIGINT/故障保留强制终止。没有修改共享环境或宿主进程。
- 第一阶段快照 `1f5ac225f80beeaf0d8e2745e6b0edf45fbbbc5475d2697b52e087291e1516ce`，2121 文件。`lifecycle-fixed.xml` **35 passed**，31.19 秒；TP=2 的 `service-lifecycle-fixed.json` 请求链路 14 检查点通过、退出码 0，原 BufferError/semaphore 告警消失，但出现未显式 `destroy_process_group()` 的警告。因此该 JSON 的请求成功不能视为完整退出验收。

进一步增加活动请求排空：TP=2 + CPU cache，流式请求收到首 token 后向 launcher 发 SIGTERM，要求完整返回 512 token，再检查退出日志。

- 初次 `service-lifecycle-drain.json`：26 个检查点执行，CPU 恢复 GPU 命中 0 / CPU 命中 224，三个 prompt 的 36 token 与独立 FP32 参考相同，活动流完整返回 512 token。但 HTTP 未在验证脚本的 30 秒期限内退出，脚本回收进程，记录 **failed、server exit -9**。
- 独立 `service-drain-trace` 将诊断窗口延长为 90 秒，保留 launcher 原 60 秒上限。HTTP 任务栈停在框架 `_cancel_all_tasks()` 等待 `HttpServerManager.recycle_resource_loop()`，后者在 `wait_for(Event.wait())` 中继续运行；最终 launcher 强杀 HTTP。记录 **failed**，没有把延长等待当修复。
- 新增 `HttpServerTasks` 明确拥有普通服务和 PD Master 的后台任务。关闭先设置 `closing`，再取消并等待任务；回收/接收循环同时观察关闭状态，PD Master 明确传播 `CancelledError`。API shutdown 在框架退出前完成这一过程。新增检查模拟 wait 完成消耗取消信号的情况，确认回收循环仍能结束，并验证 API 的清理顺序。
- 扩展本地 HTTP 回归发现旧日志夹具只调整 caplog level，无法捕获 `propagate=False` 的应用 logger；日志实际已输出。修正为将 caplog handler 直接挂到目标 logger，保持生产日志与所有内容断言。修复后本地该组与新增关闭检查 **56 passed**。本地缺少 xgrammar，未将缺依赖的检查算通过；全部 HTTP 组合在已有依赖的 H100 镜像验证。

最终 HTTP 快照 `479d80588fcd8df15bd05433ecb930cde2a0be83ffd5df3dea35e0066c09a743`，2123 文件：

- `http-shutdown-fixed.xml` **505 passed**，12.25 秒，零 skip/deselect。覆盖 HTTP、PD 控制/流式、语法约束、输出停止、后台关闭、进程管理与锁生命周期；这是 H100 容器中的服务代码回归，不是 505 项 GPU 数学检查。
- `service-http-shutdown-fixed.json` 的严格验收仍为 **failed**，26 个检查点。TP=2 + CPU cache 恢复仍为 GPU 0 / CPU 224，36 个 token 对独立 FP32 参考一致；活动流完整返回 512 token。16:18:08 UTC 收到 SIGTERM，16:18:13 HTTP 正常退出，无 HTTP 超时强杀、BufferError 或 semaphore 告警。
- **未解决：**模型工作进程仍出现 `terminate called without an active exception` 和 `destroy_process_group() was not called`。launcher 退出码 0 不能证明各工作进程正常退出。两个后台推理线程持续执行通信；需要先协调各 rank 停止提交并等待在途工作，再释放 decode/prefill Graph 和通信组，不能直接在主线程插入 `destroy_process_group()`。
- 505 项后仅从 API startup 删除一个未使用的局部 `loop` 变量；本地对应三项检查重新通过。除此项和随后更新的设计文档，源码与最终 GPU 服务快照一致。本轮 12 个 Python 文件通过 Black、Flake8、语法编译与 diff 检查。初始本地 Unix socket 检查因沙箱禁止 bind 失败，允许本地 socket 后单项重跑通过；其它对应 44 项通过。

脚本位于 `/home/devsft/models/lightllm-a716ad58-service-{lifecycle-fixed,lifecycle-drain,drain-trace,http-shutdown-fixed}.sh`；源码 tar/manifest 使用 `lifecycle-fixed`、`http-shutdown-fixed` 前缀。所有日志、JSON、XML 和诊断事件已取回本地证据目录，初始失败保留。任务容器全部退出，八张卡均 0 MiB，既有 CI 容器保留。未提交或部署。完整模型回归仍为前阶段 `route-final.xml` 的 2258 passed/6 failed，本轮没有重跑或累加为新的全量结果；七模块迁移的其它范围继续未完成。


## 推理工作进程与 Graph / 通信资源退出（2026-11-04）

本轮修复上一阶段的两个模型工作进程退出告警。沿用隔离 `707f6da52112...` 镜像、只读 Qwen3.5-0.8B checkpoint、FP16/Triton/FP32 SSM，未改变数学实现或六项 BF16 门槛。

- `ModeBackend` 通过原有请求广播/请求数 gather 传递停止标记。TP、多节点 TP 和 DP 的各 rank 在一致的批次边界停止；两个推理线程完成最后一轮 overlap 事件握手后退出，没有增加正常每批的集合通信次数。
- RPC 关闭与模型初始化/RL 操作互斥，等待后台权重加载结束。停止推理线程并同步设备之后，释放主/草稿模型 prefill GraphProgram、decode Graph、FlashInfer workspace、DeepEP/Mega MoE buffer，最后销毁进程组。同步失败仍向上抛出，不能提前释放在途资源。多节点/PD/EP/草稿完整服务的退出尚未实测，公共接口和 CPU 传播检查不替代这些验收。
- 第一个源码快照 `31b91c7147a4fb77a87ef42d4c7ca39259490e230ef3fc389afccc3531aa8b90`，2124 文件。`infer-shutdown.xml` **90 passed**，19.363 秒，零 skip/deselect：新增线程停止/资源顺序检查、Graph runner/program、真实双 rank FP16/BF16/FP32 collective、49 个后端启动检查，以及 launcher/信号回归。
- `service-infer-shutdown.json`：TP=2 + CPU cache，26 检查点通过，launcher exit 0；两个模型 rank 均记录 Graph/通信组清理完成，原 C++ 终止、NCCL 未销毁、共享内存与 semaphore 告警消失。该次尚未独立记录工作进程 OS 退出码，后续两组补上该证据。
- `service-dp2-shutdown.json`：DP=2 + prefill/decode Graph，16 检查点通过；除普通请求链路外，明确指定 DP rank 1 执行一次生成，输出相同。两个模型进程由父进程 `waitpid` 实际回收，退出码均为 0。
- `service-tp2-prefill-shutdown.json` **failed**：TP=2 + CPU cache + prefill/decode Graph 的全部生成、恢复和 512-token 活动流检查成功，两个模型进程退出码均为 0，但 Router（该次 PID 1296）在原五秒统一退出窗口末被 SIGKILL，OS exit -9。原失败日志/JSON/退出事件保留，没有删去 `Killing process` 检查或把 launcher exit 0 当通过。

正常退出窗口随后从五秒改为 30 秒，让 GPU 子进程释放资源后，Router 完成 multiprocessing join 与自己的 finalizer；强制退出兜底不变。验证脚本自身的退出等待仍为 30 秒，没有因修复延长验收等待。新脚本要求所有带服务进程名的受管 worker 都被实际回收且退出码为 0，并保留所有退出告警/强杀检查。

最终源码快照 `e321c84164163fb13d1e1409a56ac13ca04cd83983270cc51b2d915defa9888c`，2124 文件；相对上一快照只改变 launcher 等待窗口和对应测试。两组各运行 14 项 launcher 回归（0.92 / 0.94 秒），全部通过；重复项不累加。

| 真实服务复验 | 检查点 | 工作进程退出证据 | 结果 |
| --- | --- | --- | --- |
| `service-tp2-grace.json`：TP=2、CPU cache、prefill/decode Graph | 26 | 模型两 rank、Router、HTTP、detoken、metric、请求 ID、CPU cache 共 8 个进程，OS exit 全部 0 | passed；launcher exit 0、无强杀/退出告警 |
| `service-dp2-overlap-shutdown.json`：DP=2、decode 双 microbatch、prefill Graph | 16 | 模型两 rank及其它服务共 7 个进程，OS exit 全部 0 | passed；launcher exit 0、无强杀/退出告警 |

两组均完成真实生成、三并发、prefix cache、流式一致性、取消/错误请求后恢复，并对三个 prompt 的 36 个 token 做独立 Transformers FP32 eager/Torch recurrence 对照，全部相同；退出期间的 512-token 流均完整返回。TP 组目标请求 `service-tp2-grace-19` 的缓存证据为 GPU 0 / CPU 224 / disk 0。DP 组额外指定 rank 1 生成，输出相同。这是限定 checkpoint、dtype、输入和模式下的服务验收，不是正式任务准确率或负载性能。

源码 tar/manifest 前缀为 `lightllm-a716ad58-infer-shutdown` 与 `...-infer-shutdown-grace`；远端复现脚本 `lightllm-a716ad58-service-{infer-shutdown,dp2-shutdown,tp2-prefill-shutdown,tp2-grace,dp2-overlap-shutdown}.sh`。退出码记录由隔离验证目录的 `sitecustomize.py` 包装 `os.waitpid` 获得，未安装到共享环境；其 SHA256 为 `bbf3c8e588fb08996f44da616bf201213a3093e5496b12f72e2e222762fbc2f6`。最终 `service-managed-shutdown.py` SHA256 为 `6f68bc3b4a2e6ebe9451a9a66ef79caa208b9be6fdbb0a0ab24e9452ea5afde0`。

本地相关 28 项通过，最后仅窗口变化又验 14 项通过；14 个 Python 文件通过 Black、Flake8、AST 和 diff 检查。全部成功与失败 log/XML/JSON/退出事件归档至 `/tmp/lightllm-a716ad58-h100-evidence/logs/`。最终八张 H100 显存 0 MiB，任务容器全部退出，原有 `lightllm-ci-agent` 保留；未共享安装、提交或部署。只有设计文档在最终快照后更新。最新完整模型回归仍为原 `route-final.xml` 2258 passed/6 failed，不把本轮专项拼成新全量结果；其余模型/格式/规划/跨节点故障收敛及完整精度性能范围继续未完成。

## BF16 逐层误差与残差累加诊断（2026-11-04）

两次实验继续使用 `e321c8416416...` 只读源码、`707f6da52112...` 镜像和 H100 GPU 0。诊断脚本只在隔离进程中安装 hook，不改生产代码、正式参考、误差门槛或跳过规则。

- `qwen-layer-trace.xml`：Qwen3.5 BF16 正式 checkpoint 用例仍 **1 failed**，5.081 秒，最终 logits 最大 RMS 与原结果一致。记录 7 步、24 层、5 个位置，共 840 条归一化、attention、MLP 和层输出对照。首次 prefill 第 0 层输出 RMS 为 0.599%，第 23 层为 1.142%；后续一轮 decode 的第 23 层为 1.855%。所见误差在多个算子和层间积累，尚不能据此认定某个 kernel 是唯一根因。
- `qwen-residual-precision.xml`：仅在诊断中将普通层残差累加改为 FP32，归一化输出、计算、存储和最终 logits 保持 BF16。选中 Qwen checkpoint 及三个普通 Loop 后端组合，**3 passed、1 failed**，50.493 秒，17 项未选中。Loop 的 Triton/Triton、FA3/Triton、FlashInfer/FlashQLA 三组通过；未包含 mHC。Qwen 最大 logits RMS 为 **0.01852143555879593 > 0.01**，高于原 0.017852840945124626。提高残差精度不能作为当前问题的通用修复，因此没有合入，也不从正式六项失败中扣除任何一项。

脚本位于任务目录 `qwen-layer-trace.py`、`qwen-residual-precision.py`，运行入口为 `/home/devsft/models/lightllm-a716ad58-qwen-{layer-trace,residual-precision}.sh`。残差诊断脚本 SHA256 为 `772b1c64a5074ba6c5a59783ec310c707c3e50c9e1cfaa296049e9f00ccf07be`。逐层 JSON、参考实现源码、log/XML 均已归档至上述本地证据目录。实验容器已退出，八张卡显存均为 0 MiB；没有共享安装、提交或部署。后续仍需区分 GDN 状态计算与低精度算子自身误差，再按实际模型语义修复并做完整回归。

### 相同输入与初始状态的 GDN 对照

继续在 `e321c8416416...` 上运行隔离脚本 `qwen-common-input.py`，SHA256 `7140d6c942bce7cf143842f4978591c155ac758b5908aa9dde454022850df86d`。FP32 数学对照关闭 TF32；原生与参考使用同一批投影输入及调用前的 Conv/SSM 状态，分别比较卷积保留 FP32、卷积按原 dtype 舍入两种条件。归一化和 MLP 也以相同原生输入对照 FP32 参考，避免把前层误差计为当前算子自身误差。

- `qwen-common-input-observe.xml`：**1 failed**，5.434 秒，正式 checkpoint 最大 logits RMS 仍为 0.017852840945124626；只增加观测，共 1008 条记录。卷积结果舍入保持一致时，GDN prefill 状态平均 RMS 为 0.2196%、最大 0.3926%；decode 状态平均为 0.0000350%、最大 0.001637%。归一化平均约 0.166%，MLP 平均约 0.357%。这定位了误差的分布，但没有证明单一根因。
- `qwen-common-input-replace.xml`：**1 failed**，5.373 秒。诊断中将 GDN 输出及最终状态替换为 FP32 卷积/递推结果，输出仍存为 BF16；全模型最大 logits RMS 变为 0.020045721903443336，恢复状态后的最后一步还出现 greedy token 不一致。说明单独提高该子链路精度不足以修复完整模型，未合入实验或改动正式门槛。

两次 log/XML/JSON 已取回证据目录。普通 Loop、mHC 和正式 checkpoint 的六项失败继续保留；不把局部数学对照当成模型精度验收。

## 多模态输入资源绑定（2026-11-04）

源码快照 `0083b7e52e90e036705c933bcc28bb1494a6417a2123bc0b7f28bffd6746021e`，2126 文件，沿用隔离 `707f6da52112...` 镜像。生产变更与上述数值诊断分开：这次没有更换 GDN 或其它数学算子。

- 启动协调将 embedding cache 原生张量传给主/草稿模型；`MultimodalInputs.bind()` 校验维度/dtype 并持有视图，纯文本路径只创建一次空视图。`BatchStateFactory` 在计算前准备图片/音频 token 范围和缓存偏移，各批次/并发 lane 的索引独立。越界引用在提交计算前拒绝。
- Qwen-VL 共用 pre-layer、Qwen3-VL/3.5、GLM 主/草稿等继承路径直接消费已绑定视图。Qwen3-VL 重复实现合并为已有公共实现的导出别名；Gemma3 保留自身缩放 recipe 并复用相同 embedding 计算。这些 pre-layer 已不再查询 `g_infer_context.cpu_embed_cache_client`。
- Graph 回放要求 embedding 存储绑定不变，避免通用状态复制把另一份缓存的数据写入旧资源。Deepstack 仍由动态步骤读取本批索引，支持相同 Graph 下图片数、token 顺序与 chunk 变化。
- `multimodal-focused.xml`：**106 passed**，10.647 秒，零 skip/deselect。覆盖新增 12 项绑定/真实 CUDA 检查、CPU pinned/CUDA cache、FP16/BF16、图像与音频混合、分块后 Graph、Gemma3 缩放、GLM 主/草稿图像 embedding 融合，以及批次、Graph、路由记录、启动和既有多模态 kernel 回归。CPU 本地两组分别 34 passed/8 skipped、66 passed/10 skipped，重叠项不累加；跳过均需 CUDA。
- `service-multimodal-binding.json`：**26 检查点 passed**。Qwen3.5-0.8B FP16、TP=2、CPU cache、prefill/decode Graph，36 个 token 与独立 FP32 参考相同；目标请求 `service-multimodal-binding-19` GPU 命中 0、CPU 命中 224、disk 0。取消/错误请求后恢复通过，退出期间 512-token 流完整返回。8 个受管进程实际 OS exit 全部 0、launcher exit 0，无强杀或退出告警。这是文本服务回归，不是带图片的完整 HTTP 链路或多模态模型任务准确率验收。

复现入口 `/home/devsft/models/lightllm-a716ad58-multimodal-{focused,full}.sh` 和 `lightllm-a716ad58-service-multimodal-binding.sh`；源码 tar/manifest 前缀 `lightllm-a716ad58-multimodal`。服务使用 GPU 0/1，与完整回归的 GPU 2/3 分离，不作性能结论。11 个 Python 文件通过 Black、Flake8、AST 和 diff 检查；没有共享安装、提交或部署。

第一次完整 `multimodal-full.xml` 为 **2276 passed、12 failed/2288**，343.23 秒。除原六个数值失败外，新增失败来自两个 GLM mHC autotune 用例及四个无 attention 草稿用例：它们手工创建模型/批次，绕过启动绑定，分别缺少 embedding cache 字段与 pre-layer 对象。修正测试装配：GLM 调用真实 `bind_inputs()`，草稿的资源隔离夹具明确提供不准备额外输入的 pre-layer；保留全部原有数学、mHC 形状与资源断言。测试文件的既有幂运算空格同时按 Black 规范化；生产代码未改。

最终快照 `00c4485d5c8adea2e0f44a8770b42ecdaf2d9a317a3b8f0ccd3c003cc7ba141e`，2126 文件；相对服务验证的 `0083...`，差异仅为两份测试及两份设计文档，全部 `lightllm/` 文件哈希相同。`multimodal-fixtures.xml` **20 passed**，8.67 秒；随后同一快照的 `multimodal-final.xml` **2282 passed、6 failed/2288**，345.85 秒（JUnit 345.837 秒），零 skip/deselect。逐项核对确认原 `route-final.xml` 的 2264 项无遗漏，新增 24 项；失败 classname/name 集合完全相同，Qwen BF16 最大 RMS 仍为 0.017852840945124626。没有放宽门槛、增加 xfail 或排除失败。

最终源码 tar/manifest 前缀 `lightllm-a716ad58-multimodal-final`，入口为 `lightllm-a716ad58-multimodal-{fixtures,final}.sh`。新增修正的两份 Python 文件也通过 Black/Flake8/AST，合计 13 份；无 attention 草稿本地 6 项通过。全部成功/失败 log/XML、诊断 JSON、服务 JSON/进程退出事件已归档至本地证据目录。只读导出 root 所有的退出事件，没有改远端权限。全部任务容器已退出，八张 H100 均 0 MiB，既有 CI 容器保留；无共享安装、提交或部署，完整重构继续进行。

## Qwen VL 原生 Adapter（2026-11-04）

最终源码快照 `e159ccaa52731434bed5a7fa2d22f49139bb4be4b270784f0b0a2a8d27290ea8`，2127 文件，沿用 `707f6da52112...` 镜像。Qwen2-VL/2.5-VL、Qwen3-VL、Qwen3-VL-MoE 显式选择各自 ModelAdapter，复用已有 Attention/Linear/MoE 后端。原权重映射、MRoPE/deepstack recipe 保留；启动改用公共配置归一化，支持序列化 rope 参数与专家数字段别名，不引入文本模型的滑窗默认值。

- `vl-adapters.xml`：58 passed、3 failed，9.66 秒。三个失败均为 MoE 参考配置启用了共享 embedding/head，测试却使用独立输出权重。两个 Dense 模型 × 三后端通过。
- 修正 MoE 参考后，`vl-adapters-corrected.xml`：80 passed、6 failed，8.96 秒。MoE 三组通过；六组 Dense 参考因配置没有 `tie_word_embeddings` 属性而在构造时失败，后按同一未声明默认值处理。
- `vl-adapters-final.xml`：**86 passed**，9.21 秒，零 skip/deselect。9 个完整语言模型用例是 Qwen2-VL、Qwen3-VL、Qwen3-VL-MoE × Triton/FA3/FlashInfer。使用实际 FP16 权重和 CPU pinned 图像缓存，独立 Transformers eager 参考明确实现图像网格三轴位置与 deepstack；检查图片中途分块、前缀续算、decode、两类 Graph 和更新图片内容。保留 FP16 数值门槛并要求 greedy token 相同。
- 运行时禁止旧模型 factory、manager 状态查询、全局 embedding cache client、`NoQuantization.apply` 和旧专家计算入口；packed VL 专家权重按 `[E,H,2I]` / `[E,I,H]` 导出并经过实际 loader。其它专项包含配置/CPU 布局、原生绑定、tokenizer、MRoPE 和多模态 embedding 回归。本地相关 56 项通过，5 个 Python 文件 Black/Flake8/AST 和 diff 检查通过。

本次只覆盖随机小型语言模型消费图像特征，不覆盖 vision encoder、图片 HTTP、真实多模态 checkpoint 准确率、多卡、MTP 或 VL BF16。原六项 BF16 失败和完整 A–E 重构缺口仍保留。出现的 empty CUDA Graph 提示来自 deepstack 动态段之间的空捕获段，包含变更图片内容的 replay 数值检查已通过；不据此宣称 Graph 性能通过。

专项使用 GPU 0/1，完成后完整回归使用重新核对空闲的 GPU 2/3。脚本为 `lightllm-a716ad58-vl-adapters-{final,full}.sh`，源码 tar/manifest 前缀 `lightllm-a716ad58-vl-adapters-final`。原失败和最终专项 log/XML 已只读导出至本地证据目录，无共享安装、提交或部署。

最终同一快照的 `vl-adapters-full.xml`：**2300 passed、6 failed/2306**，344.12 秒（JUnit 344.106 秒），零 skip/deselect。逐项比较确认 `multimodal-final.xml` 的全部 2288 项都在本轮，增加 9 个 VL 完整模型、3 个绑定和 6 个配置用例；新增 18 项全部通过。六项失败 classname/name 集合完全相同，Qwen BF16 最大 RMS 仍为 `0.017852840945124626`。所有原失败保留，没有改数值门槛、skip 或 xfail。

完整 log/XML 已导出至本地证据目录。5 个本轮 Python 文件通过 Black/Flake8/AST/diff；生产及测试哈希与快照一致，其后仅两份设计文档更新。所有工具会话终结，任务容器退出、八张 H100 均 0 MiB，只有原有 CI 容器保留；未共享安装、提交或部署。

## Dense MTP verify（2026-11-04）

新增 `verify` offer 并绑定普通 Dense、INT4/INT8、静态 FP8 三类路径。最大候选宽度写入 `ExecutionLimits`；Triton 的共享前缀 kernel 使用预留 scratch，FA3/FlashInfer 按连续候选行建序列，保持 HOLD 行独立和元数据固定形状。Graph 仍使用相同绑定视图；FlashInfer 在 replay 前执行规划。主模型的全部 attention 语义为 Dense 才开启；recurrent/MLA/sparse 和草稿 Adapter 的迁移仍待完成。

初源码 `f645f072c8f51aa3a9ccdd382babe5e75ef9c2b5f896edb7ded896135611015a`，2127 文件；`dense-verify.xml` **35 passed、2 failed/37**，13.18 秒，135 项非专项 deselected，无 skipped。新增完整小模型验证：Llama、Qwen3、Qwen3 sliding、Qwen3-MoE × Triton/FA3/FlashInfer × 固定/动态布局；两请求分别接受不同前缀、覆盖拒绝候选的物理 KV、更新候选分组/HOLD 行并重放 Graph，之后 suffix prefill。量化测试包括 Triton INT4/8、FA3 FP8 per-head/per-tensor、FlashInfer per-tensor；额外三个 Qwen3 BF16 用例。

最终生产/测试快照 `784c6f86233c481616822a1b62185c27c6c6ac1100691a9420e7a36aa0295cab`，2127 文件。Triton verify 使用绑定的序列容量选择 stage 配置；新增 12 项 paged verify × FP16/BF16 × MHA/GQA × 三后端，覆盖 768-token 容量、并发 lane 与增长/重排后的 Graph。`dense-verify-capacity.xml` **71 passed、2 failed/73**，14.03 秒，141 项 deselected，无 skipped；包含原 24 项长序列 decode/prefill 回归。原 37 项模型中失败身份不变，新的 kernel 检查全通过。

失败分别是 `test_tiny_model_bf16_verification[triton]` 和 `[fa3]`：前面的 prefill、verify 和 Graph 数值检查通过，suffix prefill 的 greedy 参考为 token 107，原生为 106。BF16 的 RMS <1.5%、最大误差 <参考峰值/64 已通过，greedy exact 失败；没有放宽该标准。

隔离脚本 `verify-bf16-diagnose.py` 仅在上述快照上增加观测，生产与正式测试未改。`verify-bf16-diagnose.xml` **1 passed、2 failed**，6.20 秒，169 deselected。三个 backend 的 resumed/fresh prefill logits 最大差和 RMS 都为 0，表明该输入重算完整前缀也得到相同结果。Triton/FA3 原生候选 106、107 均为 0.69140625；参考 107 为 0.69921875、106 为 0.69140625。FlashInfer 原生 107 为 0.6953125，因此该用例通过。此结果缩小数值定位范围，不构成完整 MTP 或 BF16 验收。

本地相关 52 项通过；9 个本轮 Python 文件通过 Black、Flake8、AST 检查。原始专项和诊断 log/XML/JSON 均只读导出，路径前缀 `logs/dense-verify*`、`logs/verify-bf16-diagnose*`。生产快照和运行脚本前缀分别为 `lightllm-a716ad58-dense-verify-capacity`、`lightllm-a716ad58-dense-verify-{capacity,full}.sh`。

最终完整 `dense-verify-full.xml`：**2495 passed、8 failed/2503**，350.19 秒（JUnit 350.171 秒），零 skip/deselect。对比 `vl-adapters-full.xml`，原 2306 项全部保留；扩展 197 项为 37 个模型 verify、12 个页式 kernel verify 和 148 个既有 MTP manager/planner/proposer、状态提交、固定/动态布局 kernel 检查。扩展集合仅上述两个 BF16 用例失败。原六失败 classname/name 集合全部保留，Qwen3.5 BF16 最大 RMS 仍为 `0.017852840945124626`，没有修改数值门槛、skip 或 xfail。

生产和测试与 `784c6f86233c...` 快照逐文件哈希一致，之后仅两份设计文档修改。完整 log/XML 已只读取回本地证据目录。任务容器已退出，八张卡显存均为 0 MiB，仅既有 `lightllm-ci-agent` 保留；无共享安装、提交或部署。仍未完成原生草稿、recurrent/MLA/sparse verify、完整 MTP 服务、全模式任务精度和性能验收。

## GDN MTP verify 与候选状态恢复（2026-11-04）

新增原生 GDN verify，绑定加宽 Conv、按请求展开的 SSM 槽位和接受游标。固定/动态布局共用显式候选索引，HOLD/零长度尾部兼容 Graph；prefill 从接受槽读取后归一基准状态。GDN 页保存、DP 运行态复制和 PD 页写出选择接受窗口，恢复可使用不同候选数量。KDA/MLA/sparse verify 和草稿 Adapter 仍未迁移。

首运行源码 `0f99e1c3b43e6c73f0b2c4aeb891c7f9e5b8c29a436821386be0c25459a603a5`，2127 文件；`gdn-verify.xml` 在 15 项失败后停止，57.66 秒、193 deselected。各例 eager verify 已通过，失败在 Graph 重放：新测试捕获后只恢复 recurrent 状态，未恢复捕获改写的 Dense KV。修正为完整状态快照，并扩展 Loop 及 prefill Graph。此前未运行的 `310f81874f69...` 打包漏导入 torch，在静态检查时修正，没有作为 GPU 验证证据。

最终生产/测试源码 `179bf3d28dbd839b0057aa9ddbbbab2718b3d7159bf8429659156c1934ab068f`，2127 文件；与首运行相比，生产文件哈希全部相同，仅模型测试改变。`gdn-verify-graph.xml` **64 passed、1 failed/65**，138.20 秒（JUnit 138.203 秒），193 deselected，无 skipped。集合为六种模型（Qwen3Next、显式交错层、MoE、Qwen3.5、Loop、mHC2）×三后端组合×固定/动态候选 36 项、8 项计算/SSM 精度组合、9 项接受游标 0/1/2 经页缓存/DP/PD 搬运、原 12 项 GDN 独立递推/恢复检查。

唯一失败 `test_tiny_recurrent_verification_precision[triton-bfloat16-dtype1]`：BF16 计算和 BF16 SSM 下，第二批候选的 Graph 输出相对独立 Transformers 参考最大误差 `0.01953125`，参考峰值 `1.1015625`，超过峰值/64。原 RMS/最大误差/greedy 断言全部保留，不改输入、阈值、skip 或 xfail。其它 64 项通过不代表完整 MTP 服务、任务精度、多卡和性能验收。

本地相关 56 passed、12 skipped（无 CUDA）；9 个 Python 文件通过 Black、Flake8、AST 检查。快照 tar/manifest 和专项脚本前缀 `lightllm-a716ad58-gdn-verify-graph`；完整回归脚本 `lightllm-a716ad58-gdn-verify-full.sh` 使用同一只读源码，保留全部上轮集合。专项原始 log/XML 均归档至本地证据目录；完整结果与隔离诊断另记。

首次完整 `gdn-verify-full.xml` 使用 `179bf3d28dbd...`，**2538 passed、9 failed/2547**，365.80 秒（JUnit 365.773 秒），零 skip/deselect。逐项确认 `dense-verify-full.xml` 原 2503 项全部保留，增加 44 项模型检查；原八失败身份不变，仅新增上述 BF16 SSM 用例。9 项新增传输检查当时仅在专项中，最终集合另纳入整个 `test_bounded_state.py`，不将专项通过数重复累加。

独立诊断初次导入 `execution.decode_graph` 暴露循环导入：Graph 模块仅为类型注解导入 InferState，却触发 basemodel 包初始化并回到尚未定义的 CudaGraph。decode/prefill Graph 移除未使用的 ModelOutput 导入，InferState 改为 TYPE_CHECKING 注解依赖。新增两个全新 Python 进程检查两种导入顺序；本地 `test_graph_runner.py` 四项通过。原导入错误日志保留，没有将该失败当作数值结果。

最终源码 `f34226606cebbd5379a625627987cdd2229bfbdc0d1e7367bbf208fc0638e8e0`，2127 文件，相对 `179bf3d28dbd...` 仅两份 Graph 源码、对应测试及两份文档改变；12 个本轮 Python 文件通过格式、Flake8、AST 检查。最终回归脚本 `lightllm-a716ad58-gdn-verify-final-full.sh` 保留全部原集合并增加状态传输和新进程导入检查。

隔离诊断 `gdn-verify-replay-diagnose.py`（SHA256 `b6295a6ef697456e222c264444b13d8ef9d39b43740a8eba28a4fc5bda8a30c0`）在相同候选状态上分别执行 Graph 和 eager，随后恢复现场并继续原正式断言。`gdn-verify-replay-diagnose.xml` **3 passed、1 failed/4**，67.67 秒，212 deselected，无 skipped；四组 BF16 计算 × Triton/FlashQLA prefill × FP32/BF16 SSM 的 Graph/eager 最大差与 RMS 都为 0。失败组相对参考 RMS `0.013616247102618217`，greedy `[19,27,28,24]` 一致，但最大误差 `0.01953125` 超限。该输入的差异不由 Graph 重放引入，仍未通过 BF16 数值验收。诊断原始 JSON/log/XML 已归档，正式生产和测试未加入这些观测钩子。

最终 `gdn-verify-final-full.xml`：**2566 passed、9 failed/2575**，378.38 秒（JUnit 378.360 秒），零 skip/deselect。逐项对比原 `dense-verify-full.xml` 的 2503 项、首轮 `gdn-verify-full.xml` 的 2547 项均无遗漏。对前轮扩展 72 项，其中 44 个模型测试、26 个 bounded-state 测试（17 个既有、9 个新增）、2 个新进程导入检查；仅新增上述 BF16 SSM 失败，原八项失败全部保留。Qwen3.5 BF16 最大 RMS 仍为 `0.017852840945124626`。没有修改数值门槛或跳过失败。

源码/测试与 `f34226606ceb...` 逐文件哈希一致，之后仅两份设计文档更新，diff 检查通过。所有专项、诊断和两次完整日志/XML/JSON 已只读归档到本地证据目录；最终归档为 `/tmp/lightllm-a716ad58-gdn-verify-final-full-evidence.tar.gz`。所有测试和导出会话终结，任务容器退出，八卡 0 MiB，仅既有 CI 保留，无共享安装、提交或部署。

后续草稿验证资源已只读核实：现有 `/home/devsft/models/Qwen3.5-0.8B` 的 text config 声明 `mtp_num_hidden_layers=1`、`mtp_use_dedicated_embeddings=false`，safetensors header 包含 15 个 MTP 权重键。该发现仅确认权重可用，尚未验证原生草稿 Adapter 或完整 MTP 服务。

## Qwen3.5 原生草稿 Adapter 与 MTP 服务（2026-11-04）

初次源码 `a1e4bca122bc1d90b30ea22805b5669556b7a2a826fbab490d127875a1c4ec1d`，2128 个文件。新增 `Qwen35MTPModelAdapter`，分开执行层号与 checkpoint 层号，显式注入共享 embedding、输出头和主模型 norm。独立 FP32 参考用 checkpoint 的 MTP 权重构造 fusion 与单层 Transformers 模型；三种 attention 后端均禁止旧工厂、运行时 manager 查询和量化 loader 执行入口。`draft-adapter.xml` 52 passed，76.16 秒，无 skip/deselect。初次完整 `draft-adapter-full.xml` 2572 passed、9 failed / 2581，401.87 秒；逐项核对上一轮 2575 项全部保留，仅新增三个 CPU 声明检查和三个 GPU checkpoint 检查，原九个失败身份不变。

首次真实 TP2 服务 `service-mtp-native` 使用 FP16 权重、FP32 SSM、EAGLE with attention、`mtp_step=2`、CPU cache 与 prefill/decode Graph。主/草稿均报告 bound，三个顺序请求接受草稿 token 为 6/8/6，验证 token 为 21/15/21，前缀命中 224 token。但三请求并发失败：草稿第一步会补写 target 的完整验证行，原预算仅每请求一行，触发 `batch exceeds the planned attention capacity`。客户端等待 180 秒后超时，异常退出出现共享内存 `BufferError`；宽限期后验收脚本强制清理，launcher exit -9，三个受管进程缺少正常退出记录。初次失败的 JSON、server log、客户端 log 与进程退出记录均保留；它也证明单请求通过不足以验收完整 MTP 服务。

修复在 Planner 装配前按自身 decode 与 target verify 的最大宽度预留 batch/workspace；草稿执行阶段仍为 decode。扩展实际 checkpoint 检查，补入两请求共六行的草稿 KV catch-up，并逐行与相同 FP32 参考比较。新源码 `eacee72374e80cefd5f62d1d8206a5d1107c3762aa1ffba69a390ddf8126fd89`，2128 个文件，较初次快照只有 Runner 容量和该测试改变。`draft-capacity.xml` 再次 52 passed，23.34 秒，无 skip/deselect；本地相关 74 passed，7.18 秒，六个 Python 文件 Black/flake/AST 与 diff 检查通过。

修复后 `service-mtp-capacity.json` **30 检查点通过**，包括主/草稿原生绑定、实际候选验证、三并发、前缀复用、流式输出、取消后恢复、400 后恢复、CPU 缓存卸载恢复。CPU 恢复日志明确 GPU=0、CPU=224、disk=0；三个 prompt 的 36 个 greedy token 与独立 FP32 同 checkpoint 完全一致。首 token 后发送 SIGTERM 的流式请求完整返回 512 token；launcher exit 0、8 个受管工作进程的实际 OS exit 均为 0，无共享内存/NCCL/强杀等清理告警。该正常路径结果不关闭首次异常退出暴露的 C 类故障收敛缺口。

所有运行隔离于任务容器，源码和模型只读，未发布宿主端口、未部署或修改共享环境。脚本、manifest 和 tar 使用 `/tmp/lightllm-a716ad58-{draft-adapter,draft-capacity,service-mtp-*}` 前缀；远端位于 `/home/devsft/models/`，源码分别在任务目录下 `source-a1e4bca122bc-draft-adapter` 与 `source-eacee72374e8-draft-capacity`。当前归档为 `/tmp/lightllm-a716ad58-draft-interim-evidence.tar.gz`，日志副本在既有本地 evidence/logs 目录。

同一源码的 `service-mtp-dynamic.json` 增加 `--mtp_dynamic_verify` 后也通过 30 个检查点；36 个参考 token、224-token CPU 恢复、512-token 退出流均通过，launcher 与 8 个受管工作进程实际退出码全为 0，无清理告警。实际日志显示压缩后每步平均验证 1.0–1.36 行，两个并发请求分别接受 3/2 个草稿 token，证实动态路径实际运行；本负载多数步骤退化为单行验证，不能据此宣称性能收益或动态策略已优化。固定和动态服务均限于该正式 checkpoint 的文本流程，不覆盖完整任务精度、图片、多节点或全部模型/模式。

最终同一修复源码的 `draft-capacity-full.xml`：**2572 passed、9 failed / 2581**，396.56 秒（JUnit 396.539 秒），无 skip/deselect。对照前轮 `gdn-verify-final-full.xml`，原 2575 项全部保留；与本轮首次 full 的 2581 项集合完全相同，九个失败身份均未改变。最后仍是五个 Loop BF16、Qwen3.5 BF16 checkpoint、两个 Dense MTP BF16 greedy 和一个 GDN BF16 SSM 峰值误差失败；阈值、输入、断言未放宽。新的六行草稿补写检查包含在三个 checkpoint 测试中，未把同一测试的阶段数计成独立用例。

最终归档 `/tmp/lightllm-a716ad58-draft-capacity-full-evidence.tar.gz` 和 `/tmp/lightllm-a716ad58-service-mtp-dynamic-evidence.tar.gz`；初次失败等在上述 interim archive，均已导出本地。最终源码/测试逐文件与 eacee72374e8 快照一致，之后仅修改两份设计文档；diff 检查通过。所有测试、服务、上传和导出命令已终结；最终仅保留既有 `lightllm-ci-agent`，八卡均为 0 MiB。没有提交、部署或共享依赖安装；完整重构目标继续进行。

## 推理线程致命异常传播与退出（2026-11-04）

首版 `54ab26e8923f9dc68947572b0653c0ea34762cfb13c088e26292a9c32d744f41`（2129 文件）在模型 RPC 进程入口安装既有 `install_fatal_thread_excepthook`；新增独立子进程测试，确认线程异常退出 1、正常 RPC 关闭仍调用资源释放。`worker-failure-unit.xml` **42 passed**，49.96 秒，零 skip/deselect，补验本地 Unix socket 权限导致的一项环境失败（本地其余 41 项通过）。

注入通过容器专用 `sitecustomize.py` 在已完成实际生成后、选定 TP rank 的草稿 `create_metadata` 抛 Python 异常，生产代码没有测试开关；使用 TP2、Qwen3.5-0.8B、FP16/FP32 SSM、MTP、CPU cache、prefill/decode Graph。注入器 SHA256 `25c3035e7f7ea1f1d32f71fb339812aff46d63ec615454b222ee1d2645f8d4c9`，初始 harness `2668ca1f3e68ea04ff7e7c032b6608f1b9e892068fd16b1e282192fe7e37ee9d`。

`inference-failure.json` 首次 rank 0 故障使三个请求约 2.07 秒收到连接失败、launcher 4.323 秒后退出 1；工作进程、GPU、服务 POSIX/SysV 均已释放，但仍有五个 `sem.mp-*` 信号量，严格验收失败，未继续 rank 1 或重启。保留全部初始日志和 `/tmp/lightllm-a716ad58-inference-failure-initial-evidence.tar.gz`，不把进程失败可见性当资源回收完成。

第二版 `a8018131df9e66b7871057f0dc00162867bacdfcec34cedf722ac30715400de7`（2129 文件）增加 HTTP 管理进程的 SIGTERM/10 秒有限退出期，超时才强制清理，且 HTTP 退出竞争不会阻止其它进程清理。`worker-failure-v2-unit.xml` **44 passed**，50.53 秒，零 skip/deselect；相关本地 27 passed，6.16 秒；四个 Python 文件 Black/flake/AST 与 diff 检查通过。

`inference-failure-v2.json` rank 0 实际注入九行草稿批次；三个请求约 5.145 秒返回 HTTP 500，launcher 12.032 秒后退出 1。实际 OS 记录为故障模型 1、HTTP worker 0、其余六个 worker -9；工作进程、显存、POSIX（包含信号量）与 SysV 已无残留。脚本当时在这些资源清空后立即检查 IPC socket，遇到独立清理进程尚在处理的四个 socket，判定失败。v3 将 IPC 也纳入原先同一个 15 秒清理等待条件，没有增加期限或删除零残留断言；生产源码保持 a801 不变。v2 日志还有 `BufferError` 与资源跟踪器信号量告警，不能称故障退出无告警。第二次失败证据在 `/tmp/lightllm-a716ad58-inference-failure-v2-evidence.tar.gz`。

v3 harness SHA256 `298f6588b305d3bef63660f0e2c2e5e5b19c3d5b8d5f8028334c497a2daf021c`。脚本 `/tmp/lightllm-a716ad58-inference-failure-v3.sh`，远端 `/home/devsft/models/` 同名；生产源码目录 `source-a8018131df9e-worker-failure-v2`。同一隔离容器依次测试两个 rank 的故障，最后重新运行完整固定 TP2 MTP 文本服务验收。数值回归本轮未重跑，之前 2581 项及九项 BF16 失败的证据仍独立保留。

v3 在 a801 源码上完成 rank 0 的严格验证：三个请求约 5.105 秒返回 HTTP 500，launcher 11.580 秒后退出 1，故障模型退出 1、HTTP worker 0、其他六个 worker -9，全部工作进程、GPU、POSIX（含随机信号量）、SysV 与 IPC socket 无残留。BufferError、信号量跟踪和故障 peer 的 NCCL Broken pipe 告警仍保留。随后 rank 1 的同端口启动被 `validate_ports` 拒绝，失败原因是端口预检查与 Hypercorn 实际监听的复用选项不一致；它把已关闭 TCP 连接的 TIME_WAIT 视为活动服务。未更换端口、未等待 TIME_WAIT 消失，失败日志归档 `lightllm-a716ad58-inference-failure-v3-evidence.tar.gz`。

第四版源码 `9730ed022321ef71fc322f64ae86d627b63a1f26f4b546897ee81df1bc09602e`，2130 文件。`validate_ports` 使用与实际 HTTP listener 一致的 SO_REUSEADDR；新增真实 TCP 检查先确认普通 bind 因 TIME_WAIT 返回 EADDRINUSE，再验证允许重启，另保留活动 listener 必须拒绝和重复端口拒绝。相关本地 28 项通过，5.83 秒，真实 socket 检查在 H100 执行。v4 脚本仍使用同一 v3 故障 harness、相同端口和全部截止时间，先跑生命周期/TCP 检查，再测试两个 rank 故障与正常重启。故障注入仍是 Python 元数据异常，不是 GPU illegal access 或完整多节点故障验收。

最终 `worker-failure-v4-unit.xml` **47 passed**，50.24 秒（JUnit 50.238 秒），零 skip/deselect。`inference-failure-v4.json` 两个 rank 的严格故障检查全部通过：

| 注入 rank | 三个请求 | 故障至 launcher 退出 | 实际工作进程退出 | 最终残留 |
| --- | --- | --- | --- | --- |
| 0 | HTTP 500，5.220–5.222 秒 | 11.480 秒，exit 1 | 故障模型 1、HTTP 0、其余六个 -9 | 进程/GPU/POSIX/SysV/IPC 均为零 |
| 1 | HTTP 500，5.235–5.237 秒 | 11.429 秒，exit 1 | 两个模型 1、HTTP 0、其余五个 -9 | 进程/GPU/POSIX/SysV/IPC 均为零 |

两例均先完成四 token 的真实请求，再在选定 rank 的九行草稿批次抛异常；退出码来自实际 waitpid 记录。POSIX 检查包含随机 multiprocessing 信号量。故障日志仍有 BufferError、信号量跟踪器及故障 peer 的通信告警，最终零残留不能替代析构告警修复；这些强制退出也不能称正常关闭。

两例之后在同一容器、相同端口 18111 立即运行 `service-failure-restart`，**30 检查点全过**。源码仍为 9730ed 快照，服务 harness SHA256 `9ca4c354dea89ec6959b7a76760737a84b036f517fbaf4aa34765ccf7f06f115`。检查覆盖主/草稿原生绑定和实际候选验证、三并发、前缀、stream、取消后恢复、400 后恢复，以及 CPU 卸载恢复（GPU=0、CPU=224、disk=0）。三个 prompt 共 36 个 greedy token 与同 checkpoint 独立 FP32 参考完全一致。首 token 后 SIGTERM 的请求完整返回 512 token；launcher exit 0、8 个受管工作进程实际 OS exit 全为 0，`shutdown_errors=[]`。

完整 v4 脚本会话退出 0。所有 XML、JSON、server/client log、故障触发记录与 OS 退出记录已只读导出 `/tmp/lightllm-a716ad58-inference-failure-v4-evidence.tar.gz`，并解包至 `/tmp/lightllm-a716ad58-h100-evidence/logs`。任务容器已退出，八张 H100 显存均为 0 MiB，只保留既有 `lightllm-ci-agent`；无共享安装、提交或部署。本轮未重跑数值全量，最近完整结果仍为 eacee72374e8 快照的 2572 passed/9 failed；完整 A–E 范围及故障析构告警继续未完成。

## 故障视图与信号量定位（2026-11-04）

在 9730ed 快照上仅加入容器诊断：包装 SharedMemory.close 记录 BufferError 的 PID/名称并重新抛出；包装 SemLock 创建与 finalizer 记录名称和调用栈。`shm-diagnose.json` 单 rank 0 故障检查通过，三请求 HTTP 500 约 5.39 秒、launcher exit 1/11.783 秒、最终资源零残留，告警照常保留。诊断脚本未进入源码，证据归档 `/tmp/lightllm-a716ad58-shm-diagnose-evidence.tar.gz`。

实际出错映射为 HTTP worker PID 2856 的 `29b691b5e42449e7_0_req_shm_total`。记录到 28 个信号量创建、23 个清理；未清理的五个分别来自 Router PID 1297 的 info_queue（三个），以及模型 PID 1820/1822 在 hf_load_utils 中首次创建 tqdm 时各一个递归进程锁。正常模型崩溃后不会运行这些进程的 finalizer，不能依赖它们完成本地资源清理。

修复快照 `bb047fd2792b30b3e52da8f4fd5f99a8dad62077162d51466fb14468b04943fa`（2131 文件）：HTTP 关闭按后台任务、请求视图、本地共享映射的顺序执行，保留共享租约和名称；仅 PD 模式创建 info_queue；权重加载采用 ThreadPoolExecutor 和私有 tqdm 子类的线程锁。新增真实共享映射及多 shard 正常/异常加载检查。九个 Python 文件 Black/flake 与全部快照 AST 检查通过；本地完整 HTTP 收集因缺 xgrammar 失败，H100 集合保留该文件。

脚本 `lightllm-a716ad58-inference-failure-v5.sh` 使用只读 bb047 源码，运行 HTTP、共享内存、PD 启动、生命周期检查后，执行原双 rank 故障与同端口完整服务恢复。v5 harness 在原先全部截止时间、实际退出码和零残留断言之外，新增 BufferError/信号量/未销毁 PG 告警必须为空的断言；不删除或放松原检查。原注入器保持不变。

`worker-failure-v5-unit.xml` **340 passed**，64.16 秒，零 skip/deselect。随后 rank 0 故障在新增告警断言失败：HTTP 显式关闭 req_shm_total 仍有导出视图，造成 lifespan shutdown failed，另剩两个信号量告警；未执行 rank 1 或恢复服务。原始日志/XML/JSON/退出记录归档 `/tmp/lightllm-a716ad58-inference-failure-v5-evidence.tar.gz`，不将单元通过替代真实故障验收。

第二轮 `shm-diagnose-v2` 保持 bb047 生产源码，使用 GC 引用诊断，确认请求视图被两个已取消后台协程 `handle_loop`、`recycle_resource_loop` 的 traceback frame 局部变量 `req` 持有。两个剩余信号量来自 `WarmupRunner.autotune` 的 tqdm 锁。此轮 HTTP worker 实际退出 1、launcher 1/11.981 秒、请求约 5.57 秒失败；原资源回收条件通过，不能称 HTTP 正常退出。归档 `/tmp/lightllm-a716ad58-shm-diagnose-v2-evidence.tar.gz`；诊断注入器 SHA256 `6da28d0fa4e7ae10e855ef56271bc25777fbe665b0f4009d184d8c6a5f00d666`，harness `82a827ad4e4d79d91ef0c09ce949c84fcab1534133d07c4c9f8f9632ff315c30`。

新增本地复现让已取消 Task 继续存活、让其协程局部持有真实请求视图；bb047 实现出现相同 BufferError（1 failed、1 teardown error）。HttpServerTasks 现在在 gather 确认结束后取出 Task 原始 CancelledError，并清除其 traceback 中已结束 frame 的局部引用；不吞掉新的 SHM 关闭错误、不提前释放共享槽位。加载和 warmup 共用仅持线程锁的 ThreadProgress。修复后相关本地 **36 passed**，8.62 秒；十二个 Python 文件 Black/flake、快照 AST 与 diff 检查通过。

新快照 `a44b88bc9dbeac63bd22e48f0ac3b3f28a8ff701f2e721dc3e49c2a1395ebfbb`（2132 文件）。v6 故障 harness 与 v5 完全相同，SHA256 `3914c0e3ae55439693222b46eb2c14affcbfd3345f72b0c0d8ada71b32daf3fd`；专项新增原 warmup 回归集合。完整回归脚本保留此前全部 2581 项并增加两项真实 shard 加载检查，尚未用任何结果替代九项旧 BF16 失败。

v6 在 H100 单元阶段停止：**353 passed、1 failed、1 teardown error**，64.33 秒，失败仍为保留取消 Task 的真实共享视图复现，未运行 GPU 故障或恢复服务。归档 `/tmp/lightllm-a716ad58-inference-failure-v6-unit-evidence.tar.gz`。主机 Python 3.10.12 最小 CPU 复现确认：task.result() 抛出的取消异常自身仅有调用者 frame，原始 reader frame 位于 `__context__` 异常链；本地 Python 3.12 的直接 traceback 检查不足。修复遍历原取消异常的 context 链、清理已结束 frame 局部引用，保留异常链及栈位置。

第七版快照 `02d662a9cc2b6229cc8ae2ae585af4dfdc82f6858185e02d0be13d96abb1c7a7`（2132 文件）；与 a44b 的生产/测试差异仅为上述异常链处理，测试门槛不变。v7 继续使用同一严格故障 harness，完整数值脚本只更新源码快照，不改变集合或失败阈值。

`worker-failure-v7-unit.xml` 354 passed，65.29 秒；真实 rank 0 故障中 HTTP 正常退出 0，BufferError 已消失，但仍因两个 semaphore 告警未通过严格断言，未运行 rank 1 或恢复。证据归档 `/tmp/lightllm-a716ad58-inference-failure-v7-evidence.tar.gz`。最后统一项目自有 tqdm 入口：CPU KV 注册、autotuner、V4 MTP/DSpark 权重加载也使用 ThreadProgress；不修改第三方库全局锁。

第八版服务快照 `450fad16241aa00cfaea5d98f67dd5fd392826cac8dcddd77c3b8f66c7736c2e`（2132 文件），`worker-failure-v8-unit.xml` **354 passed**，64.76 秒（JUnit 64.761 秒），零 skip/deselect。保留首轮只读信号量跟踪注入器，SHA256 `60226569e95c5973922f6874262a784590e82e59e8307e841e915bd10c800dc5`；故障 harness `7bbcebb08bb47c1d168c5f168939dc66c2f68a72a723c266258de76bbdf9b22d`，相对 v5 仅改变记录的注入器路径，故障条件、时限、零残留和无析构告警断言不变。

| 注入 rank | 三个失败请求 | launcher 退出 | 信号量创建/正常清理 | 资源与析构检查 |
| --- | --- | --- | --- | --- |
| 0 | HTTP 500，5.082–5.084 秒 | exit 1，13.282 秒 | 15 / 15 | 本服务新增进程/GPU/POSIX/SysV/IPC 零残留，warnings=[] |
| 1 | HTTP 500，5.345–5.347 秒 | exit 1，11.081 秒 | 15 / 15 | 本服务新增进程/GPU/POSIX/SysV/IPC 零残留，warnings=[] |

两例 HTTP worker 实际 OS exit 都为 0；rank 0 故障时注入模型 1、其余六个 worker -9，rank 1 故障时两个模型 1、其余五个 worker -9。SemLock 创建/cleanup 事件按名称逐项核对无遗漏，不再依赖 resource_tracker 兜底删除。故障 peer 的 NCCL 失联信息仍存在，这不是正常集体通信退出，也不表示 GPU illegal access/PD/EP/多节点故障已验收。

`service-failure-restart-v8.json` **30 检查点全过**：同容器同端口 18111，主/草稿原生绑定、真实候选验证、三并发、prefix/stream、取消和 400 后恢复、CPU 恢复 224 token（GPU=0、disk=0），三个 prompt 的 36 token 与独立 FP32 参考一致，首 token 后 SIGTERM 的流完整返回 512 token。launcher 及 8 个受管 worker 实际 OS exit 全为 0，shutdown_errors=[]。v8 脚本终结 exit 0，所有原始日志、XML、JSON、注入记录与退出码已归档 `/tmp/lightllm-a716ad58-inference-failure-v8-evidence.tar.gz`。

最终完整回归快照 `b06f455707a55341fb6117babd2ff3e2669940055999544bbad0f8bfe79f9744`（2132 文件），相对服务快照仅移除 autotuner 的一行既有多余空白，所有 Python AST 完全相同。完整脚本 `lightllm-a716ad58-worker-failure-final-full.sh` 保留原 2581 项，并增加两项 shard 加载和 42 项既有 autotuner 检查，预期 2625 项；42 项 autotuner 本地先行通过（1.76 秒）。数值验收仍保留原九项 BF16 失败及全部原阈值，完整运行结果另记。

最终 `worker-failure-final-full.xml`：**2616 passed、9 failed / 2625**，396.96 秒（JUnit 396.933 秒），零 error/skip/deselect。对比 `draft-capacity-full.xml`，原 2581 项逐一保留，新增 44 项全过；九个失败的 classname/name 集合完全相同。Qwen3.5 BF16 最大 RMS 仍为 `0.017852840945124626`，未修改输入、断言、阈值或跳过失败。完整日志无 BufferError、信号量泄漏或 CUDA IPC producer 退出告警。

完整回归已终结 exit 1（上述九项数值失败）；原始 XML/log 已归档 `/tmp/lightllm-a716ad58-worker-failure-final-full-evidence.tar.gz` 并解包本地 logs。十六个本轮 Python 文件通过 Black/flake，所有快照 Python AST 与 diff 检查通过；数值快照之后仅更新两份设计文档。最终全部任务容器退出、八张 H100 显存为 0 MiB，仅既有 `lightllm-ci-agent` 保留，无共享安装、提交或部署。故障析构修复在上述 TP2 MTP 文本服务范围内已形成实测闭环；完整 A–E 迁移、九项 BF16、其它故障模式和完整性能验收仍未完成。

## Qwen3.5 的 BF16 参考基线与 norm 权重诊断（2026-11-04）

沿用 `b06f455707a55341fb6117babd2ff3e2669940055999544bbad0f8bfe79f9744` 只读源码、同一 checkpoint、镜像和 H100 GPU 0。原正式测试仍使用独立 FP32 参考、原输入、七步共 14 个 logits 位置及原断言。隔离脚本额外加载 Transformers BF16 eager/Torch recurrence，在同一输入上观测；没有把它替换为正式参考，也没有修改生产代码或验收门槛。

| 计算路径，对照原 FP32 参考 | 最大相对 RMS | 14 个位置平均相对 RMS | greedy 一致 |
| --- | ---: | ---: | ---: |
| 当前原生 BF16 | 0.017852840945124626 | 0.012813032205615724 | 14/14 |
| Transformers BF16 | 0.024123532697558403 | 0.014672862166272742 | 14/14 |
| 仅原生 GDN norm 权重保留 FP32 的隔离实验 | 0.01974412053823471 | 0.012186984564842922 | 14/14 |

首次 `qwen-bf16-baseline.xml` **1 failed**，8.61 秒（JUnit 8.607），零 skip/deselect。当前原生的 14 项 RMS 与既有失败一致；Transformers BF16 的 14 项都超过 1%，原生 13 项超过。参数逐项比较还发现 BF16 加载会舍入 18 个 `linear_attn.norm.weight`，最大权重差 0.00390625，其余参数升回 FP32 后相同。因此该对照包括正常 BF16 权重转换与计算舍入，不能把全部差异归为某个执行 kernel；也不能仅凭原生误差较小宣称通过正式门槛或完整任务精度。

第二次只在隔离进程中将原生这 18 个 norm 权重改为原 FP32 参考值，确认原加载值恰好等于这些值的 BF16 转换。`qwen-bf16-norm-weight.xml` 仍 **1 failed**，8.69 秒（JUnit 8.692），零 skip/deselect。平均误差下降，但最大误差增大至 1.9744%，故未合入该实验，也未减少九项正式失败。此结果排除了“只提高这些 norm 权重精度即可解决该项”的修复方向，未确定完整误差根因。

诊断脚本 `qwen-bf16-baseline.py` SHA256 `cb0de76083bd34bf6e9933b4d133c81be4d5d9f9b5c7ffa8e51f16d4022172c3`；`qwen-bf16-norm-weight.py` SHA256 `dbcd4c74cab699a97022b1135c96937d7ba49862e131fdd5ff8e1b3daa162c6f`。远端脚本在任务根目录，运行入口为 `/home/devsft/models/lightllm-a716ad58-qwen-bf16-{baseline,norm-weight}.sh`。两份原始 log/XML/JSON 和脚本均只读导出 `/tmp/lightllm-a716ad58-qwen-bf16-{baseline,norm-weight}-evidence.tar.gz`，并解包既有本地 evidence 目录。

两次会话都终结 exit 1；全部任务容器退出、八卡显存 0 MiB，既有 CI 保留。此次只增加诊断与文档，未修改生产或正式测试，因此没有重复完整回归；最近完整集合仍为 2616 passed、9 failed / 2625。后续需继续区分原有 BF16 数值差异与重构引入的偏差，保留全部 A–E 范围，不降低验收标准。

## GPT-OSS 原生 Adapter、sink 与专家格式（2026-11-04）

首版快照 `829225dbc04146f28f0cda9f8cebde31eb35df282db075990b7f5e7b7974ac4c`（2133 文件），同一 H100 镜像、GPU 0、只读源码、隔离缓存。新增 GPT-OSS Adapter；普通及 static FP8 FA3 的 sink 静态绑定；MoE 交错布局、bias 与 top-k softmax；纯权重 loader 支持普通张量和 MXFP4/E8M0 checkpoint 的跨 shard 组装及 TP 切片。YaRN 读取非截断修正区间和显式参数。

`gpt-oss-native.xml` 在 maxfail=12 停止，**32 passed、12 failed / 44**，10.97 秒。GPT-OSS 首批 15 项中 13 项通过、两个 BF16 数值失败；其后已有 MoE 集合出现 CUDA Graph capture 失效，并污染后续随机张量分配。相同源码独立运行全部 `test_moe_compute.py`，`gpt-oss-moe-isolate.xml` **31 passed**，7.54 秒，没有复现 capture 错误；不能将首次后续十个失败直接计为十个不同生产根因。

修订版 `e2826f4c2388179acdcad67fa8995ed722c6a85502c2e5f1e532691278b1a7f1`（2133 文件）让完整小模型测试在所有退出路径同步并关闭 Graph，原来只有 verify 分支关闭；没有放宽数值断言。另补八个独立 sink 数学与 Graph 更新检查，并限制 MXFP4 解码临时索引的分块大小。`gpt-oss-v2.xml` **203 passed、2 failed / 205**，11.65 秒（JUnit 11.645），零 skip/deselect；同进程原 MoE、Dense、FP8 和 Planner 集合均完成，没有再次出现 capture 污染。

新增 23 项包括：六个 TP1/2/4 × 普通/MXFP4 的并发跨 shard 加载检查、一个声明检查、八个完整随机模型、八个普通/FP8 KV × full/window × prefill/decode sink 检查。完整随机模型六个通过；两个失败均为默认 RoPE 的 BF16：普通权重在后缀 prefill 的 greedy 为 `[63,98]`，参考 `[63,63]`，RMS 和峰值门槛已过；MXFP4 权重在首次 decode 的最大误差 `0.015625` 超过参考峰值 `0.859375 / 64`。YaRN 四组全过，FP16 两种 RoPE/两种格式全过。没有改变输入、阈值或排除这些新失败。

本地首次相关集合 51 passed、34 skipped；新增 sink 后 7 passed、16 skipped，跳过均需 CUDA。十九个 Python 文件通过 Black/flake，快照全部 Python AST 与 diff 检查通过。初次失败归档 `/tmp/lightllm-a716ad58-gpt-oss-native-evidence.tar.gz`；修订专项与独立 MoE 归档 `/tmp/lightllm-a716ad58-gpt-oss-v2-evidence.tar.gz`，均已解包既有本地 logs。

完整 v2 回归使用同一 e282 快照，`gpt-oss-v2-full.xml` **2637 passed、11 failed / 2648**，395.65 秒（JUnit 395.623），零 error/skip/deselect。审查时另发现 YaRN 的可选 `None` 值需按默认参数处理；最终 `e3a318eb523f04c29264c801b5c7e6dfe21735dc0420eb103dc21dfad8a73cbc` 保留原测试，并新增两个独立 Transformers RoPE 参数对照，分别覆盖空值和显式 attention factor。

最终 `gpt-oss-final.xml` **205 passed、2 failed / 207**，11.19 秒（JUnit 11.188）；`gpt-oss-final-full.xml` **2639 passed、11 failed / 2650**，404.02 秒（JUnit 403.991），均无 error/skip/deselect。按 classname/name 及出现次数核对，原 2625 项全部保留，新增 25 项；v2 的 2648 项也全部保留，仅增加两个通过的 YaRN 对照。完整回归的失败集合与 v2 完全一致：原九项仍在，新增 GPT-OSS 两项；Qwen3.5 BF16 最大 RMS 仍为 `0.017852840945124626`。没有更改原输入、数值阈值、断言或隐藏失败。

审查 `Quantcfg` 的 MXFP4 元数据路径未发现新加载阻断：默认线性仍选 `none`，专家由上述专用 loader 解码；这只是代码检查，不能替代正式 GPT-OSS checkpoint 启动验证。最终源码和测试与 e3a 快照逐文件一致，之后仅更新两份设计文档；最后两个修改文件的 flake 与 diff 检查通过。完整日志无 BufferError、信号量泄漏、CUDA IPC producer 或 CUDA error；日志中的第三方弃用、空 Graph、TVM 注册和缺少 Neo FA3 等 40 个 warning 保留，未把日志称为完全无告警。

原始 v2 完整回归和最终专项归档 `/tmp/lightllm-a716ad58-gpt-oss-interim-full-evidence.tar.gz`；最终完整回归归档 `/tmp/lightllm-a716ad58-gpt-oss-final-full-evidence.tar.gz`，均已解包本地 evidence/logs。完整 v2 和最终回归各以 exit 1 结束（数值失败）；所有任务容器已退出，八张 H100 显存 0 MiB，仅既有 `lightllm-ci-agent` 保留。无提交、部署或共享安装。正式 GPT-OSS checkpoint、多卡模型、服务、任务精度、性能及完整 A–E 仍未验收。

## GPT-OSS 中间精度的定位与修复（2026-11-04）

沿用 e3a 快照、原两项失败的输入、BF16 Transformers 参考和全部断言。隔离脚本 `/tmp/lightllm-a716ad58-gpt-numerics.py` 在启动绑定之后记录 native norm/FFN 的输入与输出；Graph 捕获中的记录张量随回放更新，参考计算在捕获之外执行。对同一份 FFN 输入分别运行 Transformers、向量化 BF16 分步计算和固定路由权重的 FP32 专家计算。该 FP32 对照仅用于区分舍入位置，没有替换正式参考或门槛。

`gpt-numerics-observe.xml` **2 failed**，1.85 秒，复现原两处失败。普通权重观察八个 layer/step，MXFP4 解码权重观察四个；所有观测的 attention norm 和 FFN norm 与参考逐值相等。原生专家与 Transformers 的最大相对 RMS 分别为 `0.0068462463`、`0.0071713151`，最大绝对误差为 `0.001953125`、`0.03125`；相同输入的向量化 BF16 专家与 Transformers 逐值一致。

原生融合专家在固定路由的 FP32 对照下最大 RMS 分别为 `0.0041086604`、`0.0041895760`，Transformers BF16 则为 `0.0058529181`、`0.0058539105`。因此不能把这次修复描述为降低 FP32 数学误差：所需的是保留模型的中间 dtype 舍入边界。只在隔离脚本中替换 native 专家为 BF16 分步计算后，`gpt-numerics-bf16.xml` **2 passed**，1.76 秒，完整 prefill/decode、前缀与两类 Graph 均执行到末尾。原始两组 log/XML/JSON 及诊断脚本归档 `/tmp/lightllm-a716ad58-gpt-numerics-evidence.tar.gz`。

生产修复由 `GptOssModelAdapter` 声明 `MoeSpec.round_intermediates=True`，在启动绑定后传入融合专家 kernel。GEMM 在 bias 前及路由乘法前保留输入 dtype 的舍入，clamped SwiGLU 的 alpha 乘法、sigmoid、gate 与 up+1 同样保留必要边界；仍使用原 grouped GEMM、激活和归约 kernel，没有引入逐 token 复制专家矩阵的诊断实现。其它模型默认不启用。DeepGEMM EP 暂不支持此精度要求，probe 明确拒绝，避免忽略声明；GPT-OSS 原有 TP-only 边界没有扩大。

快照 `c1f497cb3c74fb3d5bce4d757187946d538a87249088bbab37e4c3e7f6ce1d57`（2133 文件）的 `gpt-rounding.xml` **211 passed**，16.98 秒：原专项 207 项全部保留并通过，新增 top-k=4 的 FP16/BF16 × 普通/MXFP4 四个完整模型用例。未改原输入、参考、阈值或断言。归档 `/tmp/lightllm-a716ad58-gpt-rounding-evidence.tar.gz`。

候选 `b53d602da79e19ed3091590d3b855cad52fcd3d0fd3b923933f60bf638174ff7`（2133 文件）额外包括 EP 能力拒绝检查及契约说明。本地 **9 passed、22 skipped**，5.57 秒，跳过项需 CUDA；九个 Python 文件 Black/flake、快照 AST 和 diff 检查通过。`gpt-rounding-final-full.xml` **2645 passed、9 failed / 2654**，400.50 秒（JUnit 400.474），零 error/skip/deselect；原 2650 项全部保留并增加四项，GPT-OSS 两项通过，剩余失败精确等于原九项。原始 XML/log 归档 `/tmp/lightllm-a716ad58-gpt-rounding-full-evidence.tar.gz`。

同一 kernel 的审查发现，`grouped_matmul` 把所有非 BF16 的输出都选为 FP16，包括已被 `MoeBackend` 接受的 FP32。隔离 `gpt-float32.py` 在 b53 源码上用 FP32 的 128 维输入和精确的二进制权重复现：up projection 为 65536，期望最终输出 `2**23`，实际 128 个元素全为 `inf`，脚本 exit 1。这是隐式 dtype 转换导致的范围错误，与前述 BF16 兼容舍入不同。

修复将 compute type 显式映射为 FP16/BF16/FP32；原 FP16/BF16 分支不变。新增两个用例分别覆盖是否启用中间舍入，严格要求 eager 结果 `2**23`，Graph 输入从 1 更新为 2 后结果 `2**25`，均为 `rtol=0, atol=0`。候选 `771261154555810094c89a5ced75091da1b2762f7a3ff42bfeb52380df86a588`（2133 文件）的 `gpt-precision-final.xml` **213 passed**，12.34 秒（JUnit 12.339），零 error/skip/deselect；新增两个文件检查通过 Black/flake，所有快照 Python AST 与 diff 检查通过。该检查证明所测 FP32 范围和 Graph 更新，未用它声称完整 FP32 数学精度或性能验收。

修复前 FP32 复现脚本和日志、最终专项 XML/log 归档 `/tmp/lightllm-a716ad58-gpt-precision-focused-evidence.tar.gz`。诊断脚本 SHA256：`gpt-numerics.py` 为 `e6e48d697bb95faf5325043e20c980589609be1fa6c4274bb1c95436c94871b2`；`gpt-float32.py` 为 `a75c84b9e3d4f05eeb07b0f353f35a911d84a20d8579f5cba526bab1d6905e82`。

最终 `gpt-precision-final-full.xml` **2647 passed、9 failed / 2656**，402.72 秒（JUnit 402.693），零 error/skip/deselect。按 classname/name 的 Counter 逐一核对，原 2650 项、上一候选 2654 项全部保留，仅增加四项 top-k=4 与两项 FP32；剩余 failure 集合与 b53、原 b06 九项完全相同。Qwen3.5 BF16 最大 RMS 仍为 `0.017852840945124626`。日志无 BufferError、信号量泄漏、CUDA IPC producer、CUDA error 或 Exception ignored；原第三方及空 Graph 等 40 个 warning 保留。

最终完整回归 exit 1（原九项数值失败），原始 XML/log 导出 `/tmp/lightllm-a716ad58-gpt-precision-full-evidence.tar.gz` 并解包本地 evidence/logs。生产代码与测试逐文件匹配 771 快照，随后仅两份文档更新，diff 检查通过。本轮十个 Python 文件均通过 Black/flake；全部隔离任务容器已退出，八卡显存 0 MiB，无计算进程，仅既有 CI 保留。未提交、部署、修改共享环境或创建子代理；完整重构仍未完成。

后续正式模型资源的只读复核：`/home/devsft/models/SenseNova-Loop-9B-Preview-27abea77` 的 index 所需九个权重分片均存在，35.52 GiB；配置为 32 个物理层、loop 4–28 重复两次、mHC 两路，含本地 `modeling_sensenova_loop.py`。`Qwen3.5-27B` 所需 11 个分片均存在，51.75 GiB；`GLM-5.3-Flash` 所需 62 个分片均存在，305.79 GiB。这里只验证文件清单和配置，尚未加载这些模型或完成其精度/服务验收。Kimi 目录无 shard index，未判定完整。

## SenseNova Loop 正式权重对照（2026-11-04）

新增 `unit_tests/common/execution/test_sensenova_checkpoint.py`，使用上述本地 9B checkpoint，GPU 0 执行原生路径，GPU 1 执行 checkpoint 自带的 Transformers 实现，参考权重保持 FP32、eager attention 和 Torch GDN/Conv。参考每次重新计算完整前缀，两条中文输入均由参考 greedy token 驱动后续输入。检查 32 个物理层展开为 56 次执行时的权重共享和状态独立性，禁止运行时 `get_state` 查询；prefill、两步 decode、decode Graph、后缀 prefill、prefill Graph、Conv/SSM 页保存后清零并恢复的 decode 共七个阶段，与参考比较 logits。保留相对 RMS ≤ 1%、归一化最大误差 ≤ 1/32 和 greedy 完全一致的门槛。

首轮快照 `268c8707292a8c203fa94228b7dcfeca7c4e1f06f75ebcbee52158c043b0fe18`（2134 文件）的 `loop-checkpoint.xml` **1 failed**，31.28 秒。九个原生权重分片和 856 个参考权重加载成功，原生 prefill 已执行；参考 checkpoint 代码向当前 Transformers 的 GDN 传入不接受的 `use_cache` 参数，尚未产生数值对照。验证器仅在该签名缺少此参数时去掉 decoder 标志，不改参考数学计算、checkpoint 文件、权重和数值门槛。Tokenizer 显式允许读取本地模型代码，运行容器仍无网络。

第二轮快照 `39d0b67d2b63` 的 `loop-checkpoint-v2.xml` **1 failed**，33.02 秒：前五个阶段已得到数值，随后直接调用 prefill 内部函数而未补齐 Graph 输入，触发 `prefill inputs must be padded before Graph execution`。修正为通过 `native.forward()` 进入 Runner，在独立 HOLD 请求及 KV 范围完成 padding，保持原输入与断言。

第三轮快照 `be738a7f0de88aecd212a430ad50ae4e0f5dd117e8e3f720b8cab0b12ee6edb9`（2134 文件）的 `loop-checkpoint-v3.xml`，FP16 **1 passed**，33.91 秒（JUnit 33.909），零 error/skip/deselect。七个阶段的 14 个 greedy token 全部一致，最大相对 RMS `0.0030066343024373055`，归一化最大误差 `0.0018050353974103928`。原生预留 workspace 为 3,170,304 字节；这不是完整峰值显存。原始失败和通过日志/XML 分别归档 `/tmp/lightllm-a716ad58-loop-checkpoint-{initial,v2,v3}-evidence.tar.gz`，并解包到本地 evidence/logs。

同一快照、同一输入和门槛的 `loop-checkpoint-v3-bf16.xml`，BF16 **1 failed**，36.45 秒（JUnit 36.449），零 error/skip/deselect。七个阶段的 14 个 greedy token 也全部一致，但最大相对 RMS `0.10378676652908325`（decode Graph 阶段），归一化最大误差 `0.0599900558590889`，分别超过 1% 和 1/32；测试在 RMS 断言失败，完整指标保存在 XML。后缀 eager 与 prefill Graph 的 RMS 分别为 `[0.0325711891, 0.0286075063]` 和 `[0.0325714089, 0.0286077484]`，不能仅因最大误差出现在 decode Graph 阶段就判定根因在 Graph，尚需相同状态的隔离对照。归档 `/tmp/lightllm-a716ad58-loop-checkpoint-v3-bf16-evidence.tar.gz`。

本轮未更改生产代码；该正式模型检查不替代最近完整回归的 2656 项和原九项失败，新增正式 BF16 失败单独记录，也不代表 HTTP、MTP、多卡、vision encoder、CPU/disk 缓存、任务精度或性能通过。状态页在 GPU 上完成保存与恢复；prefill Graph 覆盖捕获时的首次回放。日志保留 Transformers tokenizer regex 提示、缺少调优配置及 14 个第三方弃用 warning；双方使用相同 token 输入，未据此声称 tokenizer 或自然语言任务质量已经验收。验证器通过 Black/flake，快照 Python AST 与 diff 检查通过；全部任务容器退出，八卡显存释放，既有 CI 容器保留。

### BF16 同状态和参考精度对照

沿用 be738a 快照的隔离诊断 `/tmp/lightllm-a716ad58-loop-numerics.py`（SHA256 `6457854c6b61176801da14e727ca56bb1f9e66279d69b02bac86565e46be4872`），保留七阶段原输入、FP32 参考、门槛及最终断言，在第三张 H100 加载独立的 Transformers BF16 模型。`loop-numerics.xml` **1 failed**，45.33 秒，复现原生最大 RMS `0.10378676652908325`，没有改生产代码。

同一 decode 输入、同一完整 KV/Conv/SSM 快照分别执行 eager 和 Graph，logits 最大绝对差和相对 RMS 均为零。该输入从完整前缀重新 prefill 与缓存 decode 的 RMS 差为 `[0.0210121665, 0.0191813800]`，greedy 相同，说明两种计算路径有数值差别，但没有证据把原最大偏差归因于 Graph。

独立 Transformers BF16 相对 FP32 的最大 RMS 为 `0.09665697813034058`，归一化最大误差 `0.0599900558590889`，也未达到原门槛；14 个 greedy token 全部相同。原生与 Transformers BF16 之间最大相对 RMS 为 `0.02885107323527336`。此结果区分参考本身的低精度偏差与原生执行差别，不能证明两者全部等价，也不用于放宽验收或移除失败。完整脚本、XML 和日志归档 `/tmp/lightllm-a716ad58-loop-numerics-evidence.tar.gz`。

### 补齐 mHC 的显式计算语义

参考实现声明的 `pre_scale` 和 `fixed_identity` 未进入原生 recipe，且 loader 总是按完整 mix 路由分配权重。修复后，启动归一化检查有限正数 scale 和布尔 identity；恒等映射仅加载 pre/post 的 `2*S` 行，普通映射仍加载 `2*S + S*S` 行。kernel 根据绑定的选项生成单位矩阵或执行 Sinkhorn，不访问恒等格式中不存在的 mix 字段；恒等格式使用 Torch 投影。stream 合并先按激活 dtype 舍入，再应用 pre_scale 并舍入，随后做 RMSNorm。默认 `1.0/False` 保持既有计算语义。

候选 `396a53eacd6d133658d0eb69c5fe10f2dafe0e7f9f3f53cd3901ff46a069d1d6`（2134 文件），七个修改 Python 文件通过 Black/flake，快照 AST 与 diff 检查通过。本地配置集合 **18 passed、2 skipped**，7.65 秒，两个跳过项需要 CUDA。H100 `loop-options.xml` **56 passed、5 failed / 61**，106.90 秒（JUnit 106.904），零 error/skip/deselect。新增 20 项全部通过：八个 FP16/BF16、2/4 stream 的非默认路由与 Graph 更新用例，六个完整 FP16 小模型 prefill/decode/前缀/Graph 用例，以及六个无效配置检查。小激活 kernel 用例使 scale 在 RMSNorm epsilon 下可观测，避免仅验证被归一化抵消的缩放。

专项的五项失败身份均为原九项中的 Loop BF16 用例，没有修改它们的输入、参考或门槛。此修复补齐模型语义，不声称解决正式 BF16 偏差。专项证据归档 `/tmp/lightllm-a716ad58-loop-options-focused-evidence.tar.gz`。

同一源码的完整 `loop-options-full.xml` **2668 passed、10 failed / 2678**，451.50 秒（JUnit 451.477），零 error/skip/deselect。按 classname/name 的 Counter 核对，原 2656 项全部保留，新增上述 20 项及两个正式 SenseNova 参数；失败集合精确为原九项加正式 Loop BF16。正式 Loop FP16 的最大 RMS 仍为 `0.0030066343024373055`，BF16 仍为 `0.10378676652908325`，各 14 个 greedy token 一致；Qwen3.5 BF16 最大 RMS 仍为 `0.017852840945124626`。没有放宽或替换数值门槛。

完整 XML/log 归档 `/tmp/lightllm-a716ad58-loop-options-full-evidence.tar.gz` 并解包本地 evidence/logs；日志没有 BufferError、Exception ignored、信号量泄漏、CUDA error 或 CUDA IPC producer 警告，保留原 40 个第三方、空 Graph 等 warning。全部任务容器已退出，八卡显存 0 MiB，仅原有 CI 容器运行。生产代码与测试匹配 396a53 快照，之后只更新两份文档，diff 检查通过。未提交、部署、修改共享环境或创建子代理，完整 A–E 目标仍未完成。

## Qwen3.5-MoE 主模型与草稿（2026-11-04）

主模型候选 `9d149b36fe30b29bf261285be3aa7040a0e55ca088b285a4c4b6ef355c0bb074`（2134 文件）接入显式 `Qwen35MoEModelAdapter`。小模型的 packed 专家权重直接交由专用 loader，独立 Transformers 参考使用 Torch GDN/Conv；运行时禁止旧模型工厂、manager 查询和 loader 计算入口。`qwen35-moe.xml` **78 passed、3 failed / 81**，93.18 秒，零 error/skip/deselect。集合包括 27 项绑定、12 项新增完整主模型/verify、原 recurrent 小模型和原生 MoE 检查。

新增 FP16 三种 backend 组合及固定/动态 verify 六项全部通过。BF16 三项均在 decode Graph 检查位置失败：Triton、FA3 的 RMS 为 `0.10618439316749573`，FlashInfer/FlashQLA 为 `0.10584104806184769`，超过既有 `0.015` 门槛。不能仅由失败位置判定 Graph 根因；原输入、参考和阈值保持。原始 XML/log 已归档 `/tmp/lightllm-a716ad58-qwen35-moe-evidence.tar.gz`。

草稿候选 `dbd426326a38aebb2e95f0b735f83195cb8cf0c3cafb33e6fab9c38c6f093988`（2134 文件）增加 `Qwen35MoeMTPModelAdapter`，复用已有绑定和共享权重契约。`qwen35-moe-draft.xml` **35 passed / 35**，25.28 秒，零 error/skip/deselect：29 项绑定、三个已有 Qwen3.5 正式 checkpoint 草稿回归，以及三个新 MoE 草稿后端。新草稿使用随机 packed checkpoint，对照独立 FP32 fusion、norm 和 Transformers 单层 recipe，检查前缀/普通 decode/两类 Graph/六行补写及主模型 KV 不变；不是正式 MoE 权重或端到端服务验收。

本地绑定 29 项通过（7.09 秒），五个 Python 文件 Black/flake 与全快照 AST/diff 通过。源码和测试逐文件匹配 dbd426，之后仅更新文档。快照、清单、脚本位于两端 `/tmp/lightllm-a716ad58-qwen35-moe-draft*`（远端 `/home/devsft/models/`）；证据归档 `/tmp/lightllm-a716ad58-qwen35-moe-draft-evidence.tar.gz`，包含首轮诊断因参数次序错误而失败的日志。专项与之前完整回归不能累加；全部既有失败和 A–E 剩余验收保留。

同状态诊断 `qwen35-moe-diagnostic-v2.xml` **1 failed**，8.03 秒，保留原 RMS 断言；脚本 SHA256 `313a4998ddebddb7c0c6ac3b50f1da1c71e70ad2e2a5d8c0488cb47cc28890e8`。使用同一 KV/Conv/SSM 快照、相同输入分别执行 Graph 与 eager，logits 最大差为 0。逐层记录当前 token 的 MoE 输入和 router logits：前三层 top-k 一致；最后一层第二个请求原生选 `[3,1]`，参考选 `[3,0]`。该层原生 logits 前两项为 `[0.015625,0.02099609375]`，参考为 `[0.0198974609375,0.016845703125]`，MoE 输入最大绝对差 `0.0234375`。说明当前失败伴随专家路由分歧，不是该次 Graph 重放差异；上游误差来源及计算精度契约尚需继续定位，不能把诊断当作修复或验收通过。日志/XML 归档 `/tmp/lightllm-a716ad58-qwen35-moe-diagnostic-v2-evidence.tar.gz`。

### MoE 舍入隔离与修复

隔离脚本 `qwen35-moe-rounding.py`（SHA256 `e95bc27bb770fb6a5ef9a22d9d0667a04fbfbc9aea1c24c070a34f8bad5c33b6`）在 dbd426 快照上保留原测试输入和断言，分别测试 baseline、down projection 舍入、路由权重舍入、routed/shared 小计舍入、三者同时，以及静态 Torch 专家实现。`qwen35-moe-rounding.xml` **2 passed、4 failed / 6**，9.38 秒：同时保留三个边界及 Torch 实现通过；baseline/down/sum 的 RMS 分别约 0.10618/0.10610/0.10629，route 单独修复通过误差检查但 greedy 102 与参考 45 不一致。baseline 同输入 MoE RMS 约 0.00413–0.00519。归档 `/tmp/lightllm-a716ad58-qwen35-moe-rounding-evidence.tar.gz`，四个失败保留。

生产修复由模型声明 `round_intermediates=True`，原生 router 将权重按声明 dtype 保存；共享 gate 的 sigmoid 在相同 dtype 的输出中舍入。grouped GEMM 保留 projection 的舍入；`moe_sum_reduce` 分别累积 routed 和 shared 小计，按输入 dtype 舍入后相加。新求和方式单独使用含 routed 边界的 autotune key，原默认路径及其 key 保持。未添加模型名称的 kernel 分支或运行时资源查询。现有 DeepGEMM EP 仍拒绝该精度要求，完整 EP 模式迁移尚未完成。

候选 `10c5f42fd9b5ac2d17b4515d95d63b296d1f57ccccadabb7c7271cfc37765416`（2134 文件），`qwen35-moe-precision.xml` **126 passed / 126**，112.91 秒，零 error/skip/deselect。按原 case identity 核实三个 BF16 失败全部通过。集合包括 29 个绑定、12 个 Qwen3.5-MoE 主模型、六个草稿、37 个 MoE Compute、29 个 GPT-OSS、九个 grouped MoE 及四个 shared-gate 算子检查。新增四项通过构造 routed 的大数与小数、shared 的相反大数，精确验证舍入小计的行为及 Graph 输入更新；普通/保留舍入两种声明分别检验，没有依赖宽泛误差门槛。

最终候选 `c40409060065865271c35358ebcc9894371186f614a2cbc1f0f99234a560902f`（2134 文件），仅比 10c5 增加四项 FP16/BF16、普通/block128 FP8 的共享专家精度用例及条件参考分支；原测试身份与默认行为保留。`qwen35-moe-precision-extra.xml` **4 passed、37 deselected**，8.23 秒，定向选择新参数；不是完整 Compute 集合。路由精确对照、独立 Torch 专家数学、路由记录、空批次、Graph 更新均通过。两组原始证据分别归档 `/tmp/lightllm-a716ad58-qwen35-moe-precision-evidence.tar.gz`、`/tmp/lightllm-a716ad58-qwen35-moe-precision-extra-evidence.tar.gz`。

本地绑定 **29 passed**，6.94 秒；七个 Python 文件 Black/flake、完整快照 AST 与 diff 检查通过。完整回归命令的用例选择与上一轮 loop-options-full 保持一致，各文件中的新增用例自动加入。最终脚本/清单为 `/tmp/lightllm-a716ad58-qwen35-moe-precision-final{.tar.gz,-manifest.json,-full.sh}`，远端同名位于 `/home/devsft/models/`，只读源码目录 `source-c40409060065-qwen35-moe-precision-final`。

最终 `qwen35-moe-precision-final-full.xml` **2695 passed、10 failed / 2705**，469.46 秒（JUnit 469.436），零 error/skip/deselect，退出 1。用 `(classname,name)` 的 Counter 核实上一轮 2678 项全保留，新增 27 项均通过：四项模型/草稿绑定、12 个主模型参数、三个 MoE 草稿、四个确定值精度/Graph、四个普通/FP8 精度参数。三个原 MoE BF16 failure identity 均通过；剩余 failure 集合与 loop-options-full 十项完全相同，没有改旧输入/阈值或排除用例。

完整日志无 BufferError、Exception ignored、leaked semaphore、CUDA error、CUDA IPC producer 告警；原第三方及空 Graph 等 40 个 warning 保留。XML/log 归档 `/tmp/lightllm-a716ad58-qwen35-moe-precision-final-full-evidence.tar.gz` 并解包本地 evidence/logs；核对脚本 `/tmp/lightllm-a716ad58-audit-moe-precision.py` 通过。全部验证容器已退出，八卡显存 0 MiB，仅既有 CI 容器运行。生产代码/测试匹配 c4040906，后续只有文档更新；未提交、部署、修改共享环境或创建子代理。完整重构继续，尤其不能将 Qwen3.5-MoE 的非 EP 精度通过推广到目前被 probe 拒绝的 DeepGEMM EP 组合。

### EP 舍入支持的双卡验证

DeepEP 的常规 combine 将加权与求和融合，无法直接表示每个专家加权结果各自舍入为 BF16 的语义。本轮增加按 top-k 槽传输贡献的绑定方式，最后由 Compute 求和；activation 和 routing 同时遵守模型声明。正常模式不切换协议。每次 combine 的通信由当前流等待后再复用内部槽，最终输出事件覆盖 Compute 求和；部分提交失败保留中间输出和在途权重。每层现在可能增加到 top-k 次 combine，性能尚未测量。

初始快照 `17472770d31c1932d4d222d739cd2554b2e39a7eaa29e0683555e9e675ce2378`（2134 文件），`ep-rounding.xml` **88 passed、2 failed / 90**，99.16 秒。两个失败均为新双卡夹具将 MoeSpec 改成 top-k=4，却仍把模拟权重声明为 top-k=2，Planner 在执行前正确拒绝不一致。保留原日志和 XML，未放宽启动校验。

修正快照 `7d13b0eceb3b5bd3c4b9938b60716fee799d89f94540152d7771ea635b4896df`（2134 文件）只比初始版本改变上述测试夹具；`ep-rounding-fixed.xml` **90 passed / 90**，99.17 秒，零 error/skip/deselect，退出 0。使用既有 DeepEP 镜像 d75da36ca205，GPU 0、1 和 RDMA 设备，source/model 只读，网络禁用。精确值检查将四个专家分布到两个 rank，以 BF16 权重构造必须先舍入再求和的结果，prefill 和 decode 均逐值相同；独立随机数学对照、两种 scale 策略、空专家拥有者、双 microbatch、更新输入与路由后的 Graph、12 项新增激活参数和两项中途失败保留检查均通过。原两项双卡用例及默认激活断言保留。

本地 CPU 契约 **16 passed、4 deselected**，5.46 秒，排除四个双卡用例并由 H100 补验。九个 Python 文件 Black/flake、完整快照 AST 与 diff 检查通过。首次和修正证据分别在 `/tmp/lightllm-a716ad58-ep-rounding{,-fixed}-evidence.tar.gz`；源码快照、清单和脚本同前缀，远端位于 `/home/devsft/models/`，目录 `source-7d13b0eceb3b-ep-rounding-fixed`。完整回归保留原全部选择，并加入两项部分提交失败回归与 12 项激活检查，结果如下。专项不累加为全量通过数，不扩展为正式 EP 模型、完整服务、多节点和性能验收。

最终 `ep-rounding-full.xml` **2709 passed、10 failed / 2719**，465.33 秒（JUnit 465.310），零 error/skip/deselect，退出 1。审计脚本 `/tmp/lightllm-a716ad58-audit-ep-rounding.py` 用 case identity Counter 核实原 2705 项全保留、14 项新增全过、十个失败身份完全相同。正式 Qwen3.5 与 SenseNova BF16 最大 RMS 仍分别为 0.017852840945124626 和 0.10378676652908325，原门槛、输入、参考保留。完整与双卡专项日志均无 BufferError、Exception ignored、leaked semaphore、CUDA error、CUDA IPC producer 告警；原第三方和空 Graph 等 40 个 warning 保留。

完整日志/XML 归档 `/tmp/lightllm-a716ad58-ep-rounding-full-evidence.tar.gz`，三个证据包均已解包本地 evidence/logs。源码/测试匹配 7d13b0eceb3b，之后仅更新两份设计文档；未提交或部署。全部本轮任务容器退出，八卡显存均为 0 MiB，仅既有 CI 容器运行。EP 精度支持在上述算法/格式/双卡范围通过，但正式 EP 模型、完整服务、多节点、性能和完整 A–E 重构范围仍未完成。

### Dense MTP BF16 的 RoPE 隔离与修复

在 EP 最终 7d13b0eceb3b 只读快照上，`dense-bf16-isolate.py` 逐项替换原验证中的计算步骤。首次 24 组合 **10 passed、14 failed**，9.11 秒：12 个原数值断言失败，两个 FlashInfer attention/all 诊断因其 metadata 外层包装未解包而失败。修正诊断后只重跑这两个参数，`dense-bf16-isolate-v2.xml` **2 passed、22 deselected**，6.05 秒；生产与正式测试未改。单独 RoPE 替换使三个 backend 的原验证通过。Triton/FA3 suffix prefill 的 greedy 从 106 变为参考 107，RMS 从 0.008229815401136875 降至 0.00702272541821003；native token 106/107 的 logit 从 0.69140625/0.69140625 变为 0.69140625/0.6953125。普通/QK norm、激活、projection 单项替换均不改变该输出。隔离结果不等于全部计算路径逐位一致；attention 替换也能改变近似平局，组合替换并不保证单调接近参考。

首候选 `552e1ebdd2e983685e85e7d528e89b45a0d9fb73dde0e1d447a1205e8bc6cd2c`（2135 文件）新增融合 RoPE 的精度参数、Qwen3 recipe 声明及 12 项零误差算子检查；`dense-rope-focused.xml` **15 failed**，10.09 秒，因 helper 两个分支的 FP32/输入 dtype 返回类型不一致而编译失败。修正返回类型后的 `6021142cafc802214e3a721a279e2cbef4a65f5460dd4e7392aaa65679aef752`，`dense-rope-fixed-focused.xml` **1 passed、14 failed**，10.17 秒：12 个新精确值检查及两个原模型失败仍未通过，说明转换没有保住所需边界。两个失败候选的原始日志均保留，未调整断言。

最终候选 `b2a259c52fb34c4fd6ffe3940f21d6b6a6720828ffaec2bbbbf9073f03f0f143`（2135 文件）显式先扩展为 FP32 计算乘积，转回输入 dtype 再扩展相加，并设置 `enable_fp_fusion=False`；默认路径仍使用原融合选项。`dense-rope-precision-focused.xml` **15 passed / 15**，10.67 秒，零 error/skip/deselect，退出 0。12 项算子检查包括 FP16/BF16、64/128 dim、部分旋转、非连续 Q/K、没有 K，以及 Graph 回放前修改 Q/K/cos/sin，所有断言 rtol=atol=0；三个完整 BF16 MTP 参数沿用原输入及全部门槛。没有改为 Torch 执行或按 token 特判。

三个修改 Python 文件 Black/flake、全部快照 AST/hash 和 diff 检查通过。本轮脚本/清单/源码位于 `/tmp/lightllm-a716ad58-dense-rope-precision*`，远端 `/home/devsft/models/` 同名，源码目录 `source-b2a259c52fb3-dense-rope-precision`。两轮诊断和三个候选的 log/XML（诊断另含 JSON）均归档在相同任务前缀的 `*-evidence.tar.gz`。完整回归保留原 2719 项并加入 12 项算子检查，结果如下；尚不代表其它模型的 BF16 差异、正式任务精度、性能或完整 A–E 范围完成。

最终 `dense-rope-precision-full.xml` **2723 passed、8 failed / 2731**，463.45 秒（JUnit 463.426），零 error/skip/deselect，退出 1。`/tmp/lightllm-a716ad58-audit-dense-rope.py` 核实旧 2719 项 Counter 无遗漏、12 项新增全过、恰好移除原 `test_tiny_model_bf16_verification[triton/fa3]` 两个 failure identity，没有新增失败。其余八项仍为 GDN verify 一项、Loop 小模型五项、正式 Qwen3.5 与 SenseNova BF16 各一项，原输入和门槛保持。

完整及专项日志均无 BufferError、Exception ignored、leaked semaphore、CUDA error、CUDA IPC producer 告警，原第三方/空 Graph 等 40 个 warning 保留。完整证据 `/tmp/lightllm-a716ad58-dense-rope-precision-full-evidence.tar.gz` 已本地归档并解包 evidence/logs。源码/测试逐文件匹配 b2a259c52fb3，之后只有两份文档更新；三个 Python 文件格式/flake、全部快照 AST/hash、diff 检查通过。全部本轮容器已退出，八卡显存均为 0 MiB，原 CI 保留；没有提交、部署或改共享环境。正式任务精度、性能和完整 A–E 剩余迁移仍未完成。


## Reward / InternVL / Tarsier / Omni 装配验证（2026-11-04）

新增八个 Adapter，复用 Dense/MLA/Linear/MoE 的绑定实现。Reward 的旧权重构造接口已失效，改为从四个 checkpoint 字段确定评分头维度、检查形状、收集分 shard 加载后建立两个 TP 复制的 Linear；不再创建无用 lm_head。Omni 的专家打包拆分使用声明的实际 `thinker.model` 前缀，混合 Dense 层也采用该前缀。包装配置经统一 normalize 流程保留 RoPE、专家字段与嵌套层级。

新 `test_model_wrappers.py` 包含九个配置参数（Omni 另含 Dense 层变体）、四个无 CUDA 的非法评分头检查，以及九个模型配方 × 三个 backend 的 27 项 H100 完整语言链路检查。参考为独立 Transformers 与 Torch MLP，随机 FP16 小权重以实际 safetensors 文件加载；Reward 四个字段分文件，LOADWORKER=2。输入包含两请求、图片、Omni 音频与 deepstack 特征、chunked prefill、前缀恢复、decode Graph，以及公开 BatchRunner 的 prefill padding/裁剪和更换特征后的 Graph 回放。禁止旧 factory、NoQuantization.apply、weight-loader MoE 与运行时 State 查询。未运行视觉/音频编码器、正式包装模型 checkpoint、HTTP、多卡或性能。

本地配置/非法评分头/旧绑定集合 **42 passed**，7.10 秒；本地无 CUDA，不计作 GPU 数值验收。所有原 BF16 输入、参考及门槛保持。

隔离主镜像 `707f6da52112`、GPU0/1、source/model 只读、网络关闭；之前确认八卡 0 MiB 且仅原 CI 容器运行。快照命令由 `/tmp/lightllm-a716ad58-build-wrappers.py <label>` 生成，沿用 b2a259c52fb3 的全部路径，并新增 wrapper 检查，逐文件 SHA 校验，2136 文件。初轮结果：

| 标签 / 快照 | 结果 | 原因与处理 |
| --- | --- | --- |
| model-wrappers / cd957b676e58 | 67 passed、3 failed / 70，19.42 秒 | Omni packed 专家前缀未拆分，加载阶段失败；修复实际前缀 |
| model-wrappers-fixed / bfd98ab74cc4 | 72 passed、6 failed / 78，20.72 秒 | 新增 4 非法评分头和 4 Omni Dense 参数；数值通过，Omni 13-token 超过夹具 Graph 上限 11，Graph 存在断言失败 |
| model-wrappers-graph / 276ad5fdef59 | 51 passed、27 failed / 78，20.78 秒 | 上限改 16 后，测试直接内部调用未做 padding；GraphRunner 拒绝 11/13 对 12/16 容量；修正夹具走公开 BatchRunner |

三轮均实际退出 1，无 skip/deselect，失败原始日志/XML 保留。没有降低数值或 Graph 断言，生产容量保护不变。前两轮已归档 `/tmp/lightllm-a716ad58-model-wrappers-initial-evidence.tar.gz`。最终候选 `29f024e6e8510d01a9b43bacb3296c5846caf956a94cf70f6f87d3e4d08a3a43`：`model-wrappers-runner-focused.xml` **78 passed / 78**，22.85 秒（JUnit 22.855），零 error/skip/deselect，退出 0。第三轮与成功专项归档 `/tmp/lightllm-a716ad58-model-wrappers-focused-evidence.tar.gz`，全部已解包本地 evidence/logs。四轮均无 BufferError、Exception ignored、leaked semaphore、CUDA error、CUDA IPC producer 告警；成功专项有 26 条原第三方/空 Graph 子段 warning，主 Graph 存在与实际输出检查通过。最终八个 Python 文件 Black/flake 与快照 AST/hash 通过，之后只有文档变动；相同快照的完整回归正在运行。


`29f024e6e851` 的完整 `model-wrappers-runner-full.xml` **2762 passed、9 failed / 2771**，476.65 秒，零 error/skip/deselect，退出 1（session1127已结束）。原 8 BF16 失败之外，新增失败是原 `test_multimodal_cpu_layout_matches_runtime[internvl_chat-qwen2-llm_config]`：包装模型开始继承 Qwen2._init_config，该方法给 sliding_window 写入运行容量；纯配置初始化没有 max_total_token_num，且会破坏 runtime/class normalize 相等的契约。修复两个 Qwen2 包装模型调用 TpPartBaseModel._init_config；其它 normalized 配置和 recipe 保持。新增 wrapper 配置测试加强为不预置容量并逐值比较类接口结果；原失败测试未更改。修复后本地配置集合 34 passed，7.16 秒。

该失败完整回归已归档 `/tmp/lightllm-a716ad58-model-wrappers-runner-full-evidence.tar.gz` 并解包。最终候选 `6f9bb3ae3d713910fe605aaedbd3769157c91f3f577f9703cf3a16abde591b69`（2136 文件）的 `model-wrappers-normalized-focused.xml` **103 passed / 103**，20.66 秒，零 error/skip/deselect，退出 0。增加原 25 个配置检查，先前失败项通过；26 条第三方/空 Graph 子段 warning 保留。归档 `/tmp/lightllm-a716ad58-model-wrappers-normalized-focused-evidence.tar.gz` 已解包。源码/测试 hash 与 AST 及八个改动 Python 的 Black/flake 通过；之后仅文档更新。同快照完整回归已完成，结果如下。


最终 `model-wrappers-normalized-full.xml` **2763 passed、8 failed / 2771**，471.49 秒（JUnit 471.464），零 error/skip/deselect，退出 1（session50730已终结）。审计 `/tmp/lightllm-a716ad58-audit-wrappers.py` 通过：相对 b2a259c52fb3 完整基线，2731 项 Counter 全保留，新增 40 项全过；failure identity 集合严格相同，新增配置回归已修复。正式 Qwen3.5 和 SenseNova BF16 最大 RMS 仍为 0.017852840945124626、0.10378676652908325；原输入/参考/门槛不变。

成功完整/最终专项均无 BufferError、Exception ignored、leaked semaphore、CUDA error、CUDA IPC producer 告警；完整 46 条第三方/空 Graph 子段等 warning 保留。证据 `/tmp/lightllm-a716ad58-model-wrappers-normalized-full-evidence.tar.gz` 已归档并解包 evidence/logs。最终源目录 `source-6f9bb3ae3d71-model-wrappers-normalized`，tar/manifest/focused/full 脚本均以 `/tmp/lightllm-a716ad58-model-wrappers-normalized` 为前缀，远端 `/home/devsft/models/` 同名。源码和测试逐文件匹配，后续仅文档更新。八卡最终均 0 MiB，仅原 lightllm-ci-agent 容器；本轮测试与所有 SSH/scp 句柄终结，无提交/部署/共享环境修改。

此结果只证明本轮八个 Adapter 的既述 FP16 语言链路及集合回归；不代表视觉/音频编码器、正式包装模型服务、全部分布式模式、八个 BF16 数值缺口、性能或完整七模块重构验收已完成。


## Qwen3-MoE FFN-only 草稿验证（2026-11-04）

实现和边界见 [重构实施记录](refactor-progress.md)。源码基于 6f9bb3ae3d71 完整清单，新增一个检查文件，共 2137 文件；builder `/tmp/lightllm-a716ad58-build-noatt.py <label>`，完整集合保留旧 2771 项并加入 10 项新检查。使用同一主镜像 707f6da52112、GPU0/1、source/model 只读及关闭网络；启动前八卡均 0 MiB，仅原 CI 容器。

- 初始 `d60d776fc685025501f35ff8e35611c9e91041d772f057b5ed42baac5c64f3de`：`qwen3-noatt-focused.xml` **22 passed、6 failed / 28**，29.87 秒。新夹具提前查询尚未注册完成的 State，未进入数值；保留失败快照与日志。
- 修正 `c374e9917133eca64bfc9d91e53fb573d8ad2840287b6788c7674f2bb7ae2fc7`：`qwen3-noatt-registration-focused.xml` **28 passed / 28**，30.23 秒（JUnit 30.226），零 error/skip/deselect，退出 0。只修夹具初始化顺序，生产代码相同。包含新 10 项、原 no-attention 6 项、原 Qwen3-MoE 主模型 6 项与 Qwen3.5 主/草稿 6 项。原数值输入、门槛和失败测试未改。

两轮均无 BufferError、Exception ignored、leaked semaphore、CUDA error、CUDA IPC producer 告警，14 个第三方弃用 warning 保留。证据 `/tmp/lightllm-a716ad58-qwen3-noatt-focused-evidence.tar.gz` 已归档并解包。新测试覆盖 FP16 随机权重的独立数值与 Graph、无额外 KV 及共享权重；不构成正式模型、BF16、多卡、HTTP 或性能验收。

完整 `qwen3-noatt-registration-full` 正在同一快照运行；前缀 `/tmp/lightllm-a716ad58-qwen3-noatt-registration`，远端 `/home/devsft/models/` 同名。不能将专项相加为完整回归，也不能在运行结束前称为通过。全部 A–E 剩余目标及原八个 BF16 失败继续保留。


无 Attention 草稿最终完整 `qwen3-noatt-registration-full.xml` **2773 passed、8 failed / 2781**，477.76 秒（JUnit 477.738），零 error/skip/deselect，退出 1。`audit-noatt.py` 已执行通过：原 2771 项 Counter 全部保留，新增 10 项全部通过，八个失败身份完全不变；28 项专项不重复累加。完整日志无资源析构、CUDA error 或 CUDA IPC producer 告警，46 个原 warning 保留。证据 `/tmp/lightllm-a716ad58-qwen3-noatt-full-evidence.tar.gz` 已归档并解包。

本轮 c374e9917133 源码/测试通过静态及逐文件校验，验证后继续开发 EAGLE3，故该完整结果不覆盖后续 EAGLE3 修改。无 Attention 草稿的全部容器/SSH 已退出，退出时八卡均 0 MiB，原 CI 保留；完整 A–E 范围和八项 BF16 失败继续。


## EAGLE3 装配与 Graph 专项（2026-11-04）

实现及数值范围见 [重构实施记录](refactor-progress.md)。原 no-attention 完整容器退出且八卡均 0 MiB 后，继续复用隔离主镜像 707f6da52112、GPU0/1、只读源码/模型和关闭网络。快照 `08aef636cd724bcdbe9f2129ed5bfe9cd217b6fad9567265d364cc79a1eabbdd`（2138 文件），builder `/tmp/lightllm-a716ad58-build-eagle.py <label>` 基于 c374e9917133 清单，新增 `test_eagle_draft.py`；原 `test_speculative_utils.py` 已在源码清单，现加入完整执行集合。

`eagle-native-focused.xml` **126 passed / 126**，11.53 秒（JUnit 11.529），零 error/skip/deselect，退出 0。包含新 8 项、原 EAGLE 输入 4 项、原 speculative_utils 85 项与原绑定 29 项。相对上一完整集合共新增 93 项：8 个新检查及 85 个原有测试；不能全称为新编写用例。14 个第三方弃用 warning 保留。专项日志/XML 已归档 `/tmp/lightllm-a716ad58-eagle-native-focused-evidence.tar.gz` 并解包。两个 Python 文件 Black/flake 与全快照 AST/hash 检查通过，之后仅文档更新。

相同快照的 `eagle-native-full` 正在运行，预期保留原 2781 项并增加上述 93 项。脚本/manifest/tar 前缀 `/tmp/lightllm-a716ad58-eagle-native`，远端 `/home/devsft/models/` 同名；源目录 `source-08aef636cd72-eagle-native`。不以专项代替完整回归，尚未宣称正式草稿权重、HTTP 接受率/任务、多卡/量化/BF16 或性能验收。


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

## DFlash / DSpark 与 FP8 非因果块执行（2026-11-04）

上一轮所有任务退出、八张 H100 显存均 0 MiB 且原 CI 容器保留后，启动本轮隔离验证。沿用主镜像 `707f6da52112`，源码/模型只读、网络关闭、任务缓存独立；专项 GPU2，完整 GPU0/1。builder `/tmp/lightllm-a716ad58-build-block-draft.py <label>` 基于 aafb 完整清单和执行集合，仅增加 `test_block_draft.py`。

首轮快照 `802d0ab4c18a01a40aaa9e0d20fdd59d9a231ae6a627c2181865a8fcdd24e3a8`（2140 文件），`block-draft-native-focused.xml` **103 passed / 103**，12.02 秒，零 error/skip/deselect，退出 0。含新 7 项 CPU + 8 项 GPU 模型检查及原有 88 项 DSpark/Runner/Planner/绑定检查。证据 `/tmp/lightllm-a716ad58-block-draft-native-focused-evidence.tar.gz` 已归档并解包。

增加六项 FP8 非因果独立数学与 Graph 检查，生产代码相同。新快照 `00a5e18beb515b86f20ca2276a8442a9fcfd34c875b9ed577f83ae04b55706a9`（2140 文件），`block-draft-fp8-focused.xml` **109 passed / 109**，12.32 秒，零 error/skip/deselect，退出 0。两轮均保留 14 个第三方弃用 warning，Black/flake/diff 检查通过。范围与误差门槛见实施记录，专项不重复累加为完整回归。

相同快照 `block-draft-fp8-full.xml` 最终 **2923 passed、7 failed / 2930**，489.78 秒（JUnit 489.752），零 error/skip/deselect，退出 1。`audit-block-draft.py` 已执行通过：原 2909 项 Counter 全部保留，新增 21 项全过，原七项 BF16 失败身份及正式 Qwen/SenseNova RMS 不变，未改输入、参考、阈值或跳过用例。109 项专项不重复累加。46 个第三方 warning 保留，无 BufferError、Exception ignored、信号量泄漏、CUDA error 或 IPC producer 告警。三轮原始 archive 均已解包；本轮完整证据为 `/tmp/lightllm-a716ad58-block-draft-fp8-full-evidence.tar.gz`。全部任务容器/SSH 已退出，八卡均 0 MiB，原 CI 容器保留。源码/测试逐文件匹配 00a5 快照，之后仅文档更新。正式 DSpark 共享目标权重格式的只读检查暴露另一个待补装配边界，见实施记录；尚未宣称正式草稿模型/任务、HTTP、多卡或性能验收完成。

## 正式 DSpark 共享权重与预测头（2026-11-04）

`dspark-shared-focused.xml` **124 passed / 124**，14.88 秒（JUnit 14.881），快照 `2df416feb1c93dfd532218c8fccf52ff6ac33e686f370bfae1ffa2c0dd58a119`。包括共享/自有 checkpoint、部分缺失拒绝、目标规格核对、metadata-only 名称读取及新共享布局 Graph；无 error/skip/deselect。该阶段尚未按正式发布模型核对 confidence 的归一化边界。

随后加入正式 `Qwen3.8-27B` 与原始 `Qwen3.8-27B-DSpark`（不改 config）的双卡测试：目标原生模型产生自然语言前缀的辅助 hidden，草稿共享目标 embedding/lm_head，独立 FP32 draft/GQA/FFN/YaRN/Markov/confidence 参考在另一张卡执行。源/model 只读、网络关闭，GPU3/4，主镜像保持。新文件 `test_dspark_checkpoint.py` 的 FP16/BF16 两项要求 token 相同、logits RMS ≤1%、peak ≤1/32、confidence rtol 2%/atol 0.002；没有放宽原检查。

首轮快照 `7938827b44e4307ba10a1f2fec4cfff371a4ee40e092d7111f58321fe99a2a71`（2141 文件）`dspark-checkpoint-focused.xml` **2 failed / 2**，47.14 秒：FP16 首次 token 已一致，最大 logits RMS 0.0617%，但 confidence 六值全错（最大差 21.9822），原因是现有 post 使用 norm 前 hidden，而发布 backbone 返回 final-normalized hidden；随后 BF16 被首个失败 traceback 保留的 27B 权重影响而 OOM，不能计作 BF16 数值失败。

Adapter 现在为明确的发布格式绑定 normalized-head 语义，自有旧格式保持；新共享布局的独立参考按发布模型修正。正式测试在外层保存完整错误文本并退出内部模型作用域，恢复 monkeypatch 后 GC/清缓存，避免失败 traceback 占有下一用例的权重。原断言、数据和门槛不变。

修正快照 `96f0a5f93a5d104eff8a8c318e0435ee7539ca77f1ed2096c77275de465c7a24`：`dspark-normalized-heads-layout.xml` **124 passed**，15.51 秒；正式 `dspark-normalized-heads-focused.xml` **1 passed、1 failed / 2**，71.37 秒。FP16 的 eager/capture/提交目标新 hidden 后 replay 全部通过，最大 RMS **0.06347%**、peak **0.09541%**，三阶段每次两请求各三 token 均同参考。BF16 首次 logits 最大 RMS **0.43677%**、peak **0.57522%**、token 相同，但 confidence 一值差 **0.0036378354**（相对 6.96%），仍未通过 2%/0.002 门槛；Graph 后续阶段尚未到达。OOM 已消除。原始两轮 archive 均已解包。

正在只读 96f0 快照上隔离 confidence 的 FP32 线性和归一化计算，生产尚未修改精度规则。首次诊断因包装函数形参名不匹配退出，原日志另存；已修签名，以新 label 复跑且保留原测试断言。完整回归尚未重跑，七个原 BF16 失败及新增正式 DSpark BF16 confidence 缺口均继续保留。

修正调用签名后的 `dspark-confidence-precision.xml` 原 BF16 检查仍失败，29.82 秒：同一失败值从原 -0.0559082 变为仅 FP32 linear 的 -0.0559960，重算 FP32 final RMSNorm + linear 为 -0.05566156，参考 -0.05227037。误差只降到 0.00339119，仍超门槛，未改生产。

再加 FP32 master residual（norm 后转模型 dtype 执行原生 attention/FFN，最终 confidence 保留 FP32 norm/linear）的独立变体 `dspark-residual-precision.xml` 仍失败，31.10 秒：初次 eager/capture confidence 已过，但提交新 target hidden 后的 replay 有一值为 0.60571325，参考 0.58629489，误差 0.01941836，超过 2%/0.002。原测试输入、数据与断言不变；该变体也未进入生产。两轮原始 archive 已保存。

当前生产仍为 96f0 快照的共享装配与 normalized-head 语义修复，正式 FP16 已过、BF16 confidence 未过。`dspark-normalized-heads-full` 开始 GPU0/1 完整回归，预计原 2930 + 新布局/loader 13 + 正式两项 = 2945。原七项 BF16 和新增正式 DSpark BF16 均不删除或放宽，完整结果待收。


`dspark-normalized-heads-full` 已结束：同一 96f0 快照，**2937 passed、8 failed / 2945**，560.59 秒（JUnit 560.560），零 error/skip/deselect，退出 1。原始 XML/log 已归档至 `/tmp/lightllm-a716ad58-dspark-normalized-heads-full-evidence.tar.gz` 并解包；实际 Counter 审计确认旧 2930 项全部保留、新增 15 项，其中 14 项通过；原七项 BF16 失败身份保持，唯一新增失败是正式 DSpark BF16 confidence。Qwen3.5/SenseNova 正式 RMS 仍为 0.017852840945124626 / 0.10378676652908325。46 个第三方 warning 保留；BufferError、Exception ignored、leaked semaphore、CUDA error、IPC producer 告警均为零。该全量不覆盖随后 DFlash2 修改。

## DFlash2 专用 Adapter 与非因果窗口（2026-11-04）

快照 `f0458a3efcc7a5a3deb0cf82d50cc0feee70f86a0fa9766d46a2bb9e1b5e399c`，2141 文件。H100 GPU2、主镜像 707、source/model 只读、network none，`dflash2-native-focused.xml` **154 passed / 154**，19.76 秒，零 error/skip/deselect，退出 0。原始 XML/log 已归档 `/tmp/lightllm-a716ad58-dflash2-native-focused-evidence.tar.gz` 并解包。

含原 124 项 block/DSpark/loader/Runner/Planner/绑定检查、新 3 项 FA3 窗口能力检查和 8 项 DFlash2 FP16 完整随机模型检查，以及 19 项原 DFlash2 proposer/collector/selector 检查。8 个新模型参数覆盖 proposal 宽度 1/3、窗口 full/3、固定/动态 confidence；真实 safetensors 加载，目标 embedding/lm_head 共享、两请求、全局层 ID、native Linear/StateView、独立 FP32 卷积/窗口 attention/MLP/selector 对照、初始提交、Graph capture 和追加 target hidden 后 replay 均通过。RMS/peak/token/confidence 原门槛保持；未验正式 DFlash2 权重、BF16/TP/HTTP/接受率或性能。

同一快照 `dflash2-native-full` 已启动 GPU0/1，保留原完整 2945 项及八个 BF16 失败断言，新增 11 项，预计总计 2956；完整结果尚待收取。专项不重复累加。源代码此后无变化，只有本验证记录更新。


`dflash2-native-full` 最终 **2948 passed、8 failed / 2956**，555.92 秒（JUnit 555.898），零 error/skip/deselect，退出 1。`/tmp/lightllm-a716ad58-dflash2-native-full-evidence.tar.gz` 已归档解包。实际 Counter 审计确认原 2945 项全部保留、新 11 项全过，八项 BF16 失败身份未变。46 个第三方 warning；BufferError/Exception ignored/leaked semaphore/CUDA error/IPC producer 告警为零。该结果不覆盖随后 MLA/MTP 修复。

## MLA verify、MTP 与显式 MoE 组评分（2026-11-04）

首轮快照 `2fb2fb655d97df49e12f980eb28ee454c92faf6aff8cf79537d5550b8950abee`，2142 文件，GPU2 主镜像 707、源码/模型只读、网络关闭。`mla-mtp-native-focused.xml` **181 passed、18 failed / 199**，30.74 秒，无 error/skip/deselect。失败为 GLM4-MoE-Lite 草稿的 6 个参数，以及 DeepSeek/GLM4 主模型 verify 后 suffix prefill 的 12 个参数。全部原始日志/XML 已归档 `/tmp/lightllm-a716ad58-mla-mtp-native-focused-evidence.tar.gz` 并解包。

同快照只读 `mla-mtp-diagnostic` 的两个失败样本中，Graph/eager logits 与每层 KV 逐值一致；GLM draft 写入位置为 4/19，主模型动态 verify 为 4/19/20/21，未越过这些位置。另行逐层 latent 诊断显示 GLM FP16 对 FP32 参考峰值约 0.0014–0.0027，主模型对同精度参考约 0–0.0039，未复现最终 logits 的大偏差。原日志和诊断 XML 归档 `/tmp/lightllm-a716ad58-mla-mtp-diagnostics-evidence.tar.gz` 并解包；这些临时观测脚本没有修改生产或数值门槛。

原组评分从 `(8,4,8)` 形状猜 top-2 的声明不符合 DeepSeek V3 / GLM4-MoE-Lite 路由；已改为 Adapter 默认语义与显式 topk_method。新快照 `8d7c688cfaeed064c0054d77dc4775a228fcc0de492d70c0478350871a29b178`，2142 文件；`mla-mtp-routing-focused.xml` **272 passed / 272**，46.44 秒（JUnit 46.434），零 error/skip/deselect，退出 0。原 199 项 Counter 全部保留，18 失败全解决；增加 4 项路由声明和 69 项已有 MoE/native GLM 检查。14 个第三方 warning，上述析构/泄漏/CUDA 告警为零。`/tmp/lightllm-a716ad58-mla-mtp-routing-focused-evidence.tar.gz` 已归档解包。

同一快照 `mla-mtp-routing-full` 在 GPU0/1 完成：**2992 passed、8 failed / 3000**，567.82 秒（JUnit 567.795），零 error/skip/deselect，退出 1。实际运行 `/tmp/lightllm-a716ad58-audit-mla-mtp-full.py` 按 classname/name Counter 核对，原完整 2956 项全部保留、新增 44 项（12 kernel verify + 12 主模型 verify + 12 草稿模型 + 8 CPU 声明）全部通过；八项 BF16 失败身份不变。专项不与完整集合相加。

原始 XML/log 已归档至 `/tmp/lightllm-a716ad58-mla-mtp-routing-full-evidence.tar.gz` 并解包。46 个第三方 warning 保留，BufferError、Exception ignored、leaked semaphore、CUDA error、IPC producer 告警为零。任务容器退出，八卡显存均 0 MiB，既有 `lightllm-ci-agent` 保留。源码/测试此后不变，仅更新这两份记录；没有提交、部署或共享依赖安装。GLM5 的原生 KDA/pooled NoPE 仍仅支持 prefill/decode，verify 的候选状态、接受游标和尾部池化恢复尚待迁移；本轮只检查了其现有实现，未新增该路径的通过声明。


## GLM5 verify 与 MTP 草稿（2026-11-04，验收进行中）

新增 test_glm_verify.py，保留全部原始测试，构建器为 /tmp/lightllm-a716ad58-build-glm-verify.py。沿用主镜像 707，源码/模型只读、network none、任务缓存隔离。

- eb2597c94e8b6f7b0898f2d057550a05d5a69ceb8ce6d2034b7409e9478531f6（2143 文件）：glm-verify-initial-focused.xml **95 passed**，33.25 秒，GPU2，退出 0。含 5 项新 KDA 和原 GLM/kernel/Planner/Runner 检查。
- c23535cff0aa7207be2972a9183654a8ca07717924e836478014d9df5df45048：glm-mtp-native-focused.xml **103 passed、16 failed / 119**，48.31 秒，GPU2，退出 1。8 项主模型参考误用固定两请求夹具；4 项普通草稿暴露旧 NoPE 状态声明的 rope_dim=64；4 项 FP8 草稿在 decode 与整段 prefill 比较时最大差 0.0390625。
- 5c359bceeee1a440f1bebf66aede47084abd8fdd1f66c330544bb6854376cffc：glm-mtp-contract-fixed-focused.xml **107 passed、12 failed / 119**，47.81 秒，GPU3，退出 1。修夹具与 rope_dim 后，普通草稿四项全部通过；主模型八项进入实际 verify 后数值失败，FP8 草稿四项参考阶段差异仍在。
- c235 只读 glm-fp8-reference.log：新旧 prefill 和 decode 分别逐值相同；双方 decode 对整段 prefill 同差 0.0390625，greedy [203,75] 一致。prefill 当前 token 的 BF16 KV 与 decode 的编码 KV 解释了参考语义不同。生产不改，测试按相同阶段/历史比较 FP8，普通 KV 仍完整 prefill 重算；阈值不变。
- a038240bdf66134ad919db3c7be1c7b03028f890bc22a718fe421b98ed9fa5a3：glm-mtp-phase-reference-focused.xml **111 passed、8 failed / 119**，46.30 秒，GPU2，退出 1。Counter 审计保留原 95 和 119 项全部身份，草稿 8 项与独立 kernel 均通过；剩余是主模型八个 verify 对整段 prefill 的 BF16 数值差异。全部三轮专项均无 error/skip/deselect，14 个第三方 warning 保留，原始 XML/log 均已归档解包。

同 a038 的只读 kda-qk-rounding.xml 两项原普通 KV 失败仍未通过，3.35 秒；Q/K 归一化按输入 dtype 舍入将最大差从 0.0283203 降为 0.0234375，未解决。另一只读 kda-fp32-prefill.xml 将 prefill 中间计算提升为 FP32，两项仍失败，7.62 秒，最大差 0.02734375。两个变体均未合入，原始日志/XML 已归档。主模型 Graph/接受/suffix 检查在原数值断言之后，尚未到达；不据此声称端到端 GLM5 MTP 通过。

同 a038 的 glm-mtp-phase-reference-full 已在 GPU0/1 启动，预计原 3000 + 新 29 = 3029；原八项 BF16 与新增八项主模型 verify 失败检查均保留。结果待收。最新本地 53 passed、26 个 GPU 检查显式排除，7.17 秒；12 个 Python 文件 Black/flake 与 diff 检查通过。注册 AST 审计确认 84 live 与 inventory 一致，主模型 27/29、草稿 28/30、Adapter 类 35；剩余入口为 V4，声明数不代表全部能力或精度验收。


`glm-mtp-phase-reference-full` 已结束：同 a038 快照，**3012 passed、17 failed / 3029**，592.14 秒（JUnit 592.102），零 error/skip/deselect，退出 1。原始 XML/log 已归档 `/tmp/lightllm-a716ad58-glm-mtp-phase-reference-full-evidence.tar.gz` 并解包。实际 Counter 核对旧 3000 项全部保留，新增 29 项；旧八项 BF16 失败仍在，新增八项 GLM5 主模型 verify 数值失败。此外，旧 `test_target_and_mtp_prefill_embed_images_before_hidden_fusion` 失败：测试内 main_norm_weight_ 的 lambda 不接受保护上游 hidden 所新增的 alloc_func 参数，尚未修复复验。46 个第三方 warning；BufferError、Exception ignored、leaked semaphore、CUDA error、IPC producer 告警均为零。任务容器已退出，八卡显存均 0 MiB。

用户清除自动目标后，本次只收取已启动测试并记录状态，没有继续实现或新启动测试。完整七模块目标仍未完成；数值问题、V4/其它量化、规划/生命周期与旧分支清理、正式任务/服务/多卡模式及性能验收仍待完成。
