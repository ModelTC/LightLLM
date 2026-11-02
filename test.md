# V4.1 移植到 support_ds4 的验证记录

## 2026-10-30：基线与迁移原则

- 目标分支 `support_ds4_v41`，worktree `/mtc/wanzihao/lightllm-ds4-v41`；基于远端已核对的 `support_ds4:6dc33df85cb92c2dfefb2206a92dff6b28375ba9`。
- 功能参考为旧 V4.1 `23d53ba5e`，额外迁入已经验证的 tokenizer 别名 medium→low、xhigh→high。原两个工作区未修改。
- 保留新基线的 DP-shared checkpoint、异步 CPU cache、紧凑 PD 协议、WRITE 生命周期、FlashMLA out/split workspace、融合 SwiGLU 和请求 presence all-reduce。未迁入旧 8 步调度、token-budget/underfill 或分组 FP8 量化实验。
- DeepEP V2 布局转换参考当前 support_ds4 的未提交适配，但显式使用旧 V4.1 已验证的 compute-stream ownership（dispatch/combine 的 async_with_compute_stream=False、allocate_on_comm_stream=False），避免曾复现的多层 graph allocator 膨胀。没有整文件覆盖新 MoE/EPLB 框架。
- 四台预检 .17.62/.17.55/.23.225 空闲，.23.201 已有新的 GPU 任务，不关闭未知进程。本轮先在前两台单卡测试，不启动 LLM Bench 压测。

## 测试环境

远端 `/opt/conda/bin/python` 3.10.20，Torch2.11.0+cu130；实际 FlashMLA 来自 `vllm/v1/attention/ops/flashmla.py`，支持 out，且 `_flashmla_C.sparse_decode_fwd_with_workspace` 已注册。测试使用 GPU7、现有本地 kernel cache；nofile1048576。未安装或替换推理依赖。远端没有 pytest，复用本机已有容器的纯 Python pytest 模块至独立 `test_deps/`，仅测试 PYTHONPATH 引用，未写 site-packages。首次本机 Python3.8 compileall 无法解析基线既有 match 语法，后续使用远端 Python3.10。

原始日志：`/mnt/vqa/wanzihao/dsv41_port_support_ds4_20261030/tests/`。初始失败轮次保留，不作为通过记录。

- initial_v41_vendor：128 passed / 3 failed；发现 Marlin fixture 的设备来源及移植中 DSpark 共享 stream 初始化不完整，随后修复。
- v41_second：146 passed / 5 failed / 7 skipped；8组真实小模型 FP4/FP8/视觉/DSpark 分块推理均已通过，失败来自旧视觉测试引用新基线已不存在的 block_start_idx/block_end_idx。按 start_idx/token_num 当前契约更新测试。
- v41_third：229 passed / 1 failed / 1 skipped；真实 checkpoint 图像处理参考已启用。剩余 V2 初始化测试引用旧 workspace-size helper，改为新基线的既有函数。
- base_regression/base_second：测试夹具缺少 dp/nnodes、prefill workspace、C128 scratch，导致旧基线接口测试失败。只补齐实际框架要求的测试输入，不在生产代码增加 fallback。

尚未完成整模型准确率、性能或四机 PD 验收；不将这些内核/小模型结果写成整机加速收益。

## 迁移回归通过

- `v41_all_final.log`：231 passed，30.84s，无跳过；包含 checkpoint 原生图像处理及 prompt 展开参考、FP4/FP8 小模型、CED、DSpark、parser、V2 ownership 与 autotuner 回归。
- `base_final.log`：644 passed，37.59s；覆盖 V4 推理、mHC、FlashMLA、shared expert、CUDA Graph pool、多卡缓存搬运、CPU cache 事件与 PD checkpoint。
- 改动文件 flake8、`git diff --check` 通过。测试使用已有依赖，无推理包安装或替换。
- 四机 PD 实测按用户要求推迟，等待通知；本轮仅启动 .17.62/.17.55 的双机 DPEP16 独立服务，不占用 .23 两台。

## 双机 EP16 MoE 实测

冻结源码 `7de03bd92`，.17.62/.17.55 共16张H100；走本分支实际 `dispatch → masked grouped GEMM → weighted combine`，H=5120、experts=384、topk=6、intermediate=2304、capacity=384。独立 Torch 参考覆盖 eager、CUDA Graph、三次改变输入/路由/路由权重后的 replay。全部16个 rank、4类路由通过，相对L2最大2.012%（阈值5%，用于FP8数值校验，不代表模型准确率）。

| 路由/每rank输入行数 | 完整MoE graph耗时中位数 | 最大相对L2 |
| --- | ---: | ---: |
| 随机 / 6 | 548.95 μs | 1.658% |
| 随机 / 48 | 617.26 μs | 1.576% |
| 集中到6个专家 / 384 | 4158.37 μs | 1.493% |
| 奇数rank1行、偶数rank48行 | 591.43 μs | 2.011% |

这些是单层局部耗时，没有旧版本同轮对照，不宣称加速。每档5轮、每轮50次graph replay取rank最大耗时再取中位数；原始结果 `standalone/moe_attempt02/`，两节点正常退出、16个ALL_PASSED。首次探针PATH遗漏/usr/sbin导致DeepEP找不到ibstat、带宽探测除零，原始失败日志 `standalone/moe_node*.log` 保留；仅修正探针环境，正式launcher原本已包含/usr/sbin，未添加生产fallback或改依赖。

双机完整模型使用 `tools/launch_deepseek_v41_dpep16.sh`，源代码通过git archive冻结并逐文件SHA256校验，部署身份和命令保存在 `standalone/manifest.json`、`standalone/service/*identity.json`。未设置max_total_token_num，交由新基线profile；chunk4096沿用旧V4.1已验证的显存边界，未换成此前并发16K场景OOM的8192。

## 完整启动发现并修复 ViT FA3 ABI 不匹配

第一轮在视觉峰值预热失败：公共 ViT 后端直接调用 `sgl_kernel.fwd` 时传37个参数，当前安装的算子schema只有35个；额外的sparse_mask_fine/only_qv不受支持。VisionPeakVramHolder将RuntimeError统一包装为OOM提示，但实际栈为参数数量错误，没有据此降低mem_fraction或跳过预热。两机旧进程树全部退出、GPU和SysV清空，日志与源码保留在 `standalone/attempt01_service/`、`standalone/attempt01_src/`。

在既有公共 FA3 后端及旧ViT入口复用已经导入的 `flash_attn_varlen_func(..., out=o, causal=False)`，由安装包负责底层ABI，删除手写参数列表，无新兼容层。修复前原有 `test_varlen[_vit_att_fwd-*]` 两项实测失败，修复后ViT全套与V4.1视觉共22项通过（8.32s，`tests/vit_abi_after.log`）。独立提交 `98e5ae729`；未安装、更换包或硬编码V4.1 backend。

第二轮冻结源码 `98e5ae729`（launcher与参数不变），两台并发启动。Vision预热通过。新基线共享内存带服务UUID，Engram冷启动需要为两台合计四张94.42GiB表（每台两张、约188.84GiB/台）重新预触页/注册；采样栈 `standalone/engram_startup_stack.txt` 定位在预触页的NumPy读取，第一张表约164s后进入下一张。此启动成本独立记录，不当作TTFT/TPOT或吞吐数据。尚未修改Engram初始化策略。

## MTP加载后的峰值复检迁移

补查启动时序发现基线会把DISABLE_CHECK_MAX_LEN_INFER永久设为1，只在draft加载前检查target。迁入旧V4.1的必要修复：draft构造期间临时禁用检查、随后恢复调用者环境；全部draft及backend缓存就绪后再检查target。保留新基线disable_eplb_model_init及EPLB生命周期，不整段覆盖。回归原实现4 failed/2 passed，修复后该模块19 passed/2 skipped（本轮CUDA_VISIBLE_DEVICES为空，两项GPU用例跳过），环境unset/1/空串及DSpark/链式draft均覆盖。提交d06ec314f；还需最终启动观察每rank的第二次target检查。

## 完整模型首轮验收（98e5ae729）

- 两台health200，文本42、8路算术并发、真实红图均通过；每台31长期进程/16 GPU进程，所有nofile1048576，最大FD374/347，fatal为空、cgroup oom/oom_kill0。
- GSM8K 8-shot，训练集前8题作示例，完整测试集1319题，temperature0/top_k1/top_p1，1024输出上限、parallel256，沿用旧runner及数据hash。**1196/1319=90.6748%**，invalid0、truncated2、125751输出token；接收90758/178700=50.7879%，3.53940 committed tokens/step。旧shared-overlap记录1198/1319=90.8264%，不能称分数完全不变。
- 首轮耗时193.784s，最早完成请求耗时118.388s；中途栈记录rank0位于prefill DeepEP dispatch，部分GPU等待、部分忙，且多种实际prefill布局首次出现。不能把193.8s或等待栈直接归因为稳态kernel速度，也尚未隔离冷路径的具体贡献。保留`standalone/gsm8k_precheck/`及`gsm_prefill_stack.txt`，最终版本完成预热后另测。

## 最终版本启动与缓存验收（d06ec314f）

仅增加启动阶段复检，推理实现与98e5ae729相同。保留同环境生成的autotune配置与本地编译cache，重新核对全部生产文件SHA256；两台并发启动。每台target出现16次`check max_len 8192 infer ok`（8 rank各在draft加载前后检查一次），target/draft最大graph捕获完成，health及实际42生成通过。每台31长期进程/16GPU进程，nofile均1048576，最大FD374/347，fatal与OOM为空。原始profile5,055,391/rank；4096/8192等batch参数未改变。

Prompt cache实测：fresh seed1158461164，先在两节点16路填入同一4096-token前缀，再用该前缀追加64-token后缀做16路请求。**16/16请求均命中3584 token**，输入4160、输出8计数正确、HTTP200。此为刻意复用前缀的功能测试，不是测速；结果`standalone/validation/prompt_cache.json`。

第二轮关闭时GPU释放有延迟，初次stop断言失败记录保留；复查两台GPU进程及SysV均为空后才启动最终版本。部署证据和首轮GSM日志分别保存在attempt02_service与gsm8k_precheck。

## 最终三轮性能对照

相同两台H100、FP8权重、DPEP16、DSpark5、mem_fraction0.75、batch8192/chunk4096、graph64/rank，沿用旧OpenAI SSE客户端。各档先用32输出token预热；正式各三轮、每轮fresh seed，temperature0/top_k1/top_p1/ignore_eos；正式期间没有profiler。15组正式结果全部HTTP200、输入输出token计数正确、无失败请求。下表为**每轮请求平均TTFT/TPOT的三轮中位数**及每轮总输出吞吐的三轮中位数，预热不计入。

旧值来自此前同机shared-overlap版本的冻结记录，不是本轮交替AB测试；该版本包含旧分组量化路径，新分支按迁移范围保留support_ds4当前路径。比较包含基线架构与多项迁移差异，不能单项归因。原始旧JSON/source manifest已复制到historical_reference，新数据在standalone/lightllm/bench/final_measure*/，汇总standalone/benchmark_summary.json，保留全部seed、逐请求SSE、metrics和单轮值。

| 负载（输入/输出token） | TTFT旧 ms | TTFT新 ms | TPOT旧 ms | TPOT新 ms | token/s旧 | token/s新 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C16持续128请求，512/1024 | 207.318 | 209.280 | 9.736 | 9.209 | 1533.486 | 1622.737 |
| C128共256请求，512/1024 | 505.341 | 677.631 | 15.120 | 15.456 | 5990.059 | 5357.941 |
| C128，16384/256 | 7587.619 | 8439.025 | 22.735 | 23.344 | 2079.745 | 1986.873 |
| C16，65536/256 | 6048.244 | 6501.213 | 8.219 | 7.661 | 443.012 | 406.779 |
| C1，260000/128 | 13768.717 | 14426.840 | 5.278 | 6.258 | 8.857 | 8.443 |

C16吞吐+5.82%、TPOT−5.42%，TTFT+0.95%；C128短输入吞吐−10.55%、TPOT+2.22%、TTFT+34.09%；16K吞吐−4.47%、TTFT+11.22%；64K吞吐−8.18%、TTFT+7.49%；260K吞吐−4.68%。**存在明确的版本级回退观测，不能称全面加速或无性能退化**。C128三轮吞吐5899.952/5357.941/5099.744，不只保留最好一轮。下一步需独立profile定位C128与prefill差距，本轮未隔离其原因。

MTP平均token/step来自每轮完成请求的Prometheus直方图sum/count差值，为请求加权指标，不等于按verify步加权接收率：

| 负载 | 旧三轮中位 | 新三轮中位 |
| --- | ---: | ---: |
| C16持续128请求，512/1024 | 4.10023 | 4.27386 |
| C128共256请求，512/1024 | 4.19759 | 4.08083 |
| C128，16384/256 | 4.02644 | 4.04161 |
| C16，65536/256 | 3.85860 | 4.02846 |
| C1，260000/128 | 4.74074 | 4.12903 |

未做Graph pool或autotuner的独立端到端消融：前者的输出生命周期GPU回归通过，后者缺配置的重复文件查询回归通过；不把上述整机变化分别当作这两项的收益。

## 最终 GSM8K 与验收结论

三轮矩阵结束后运行同一8-shot全量runner：**1212/1319=91.8878%**，invalid=0、truncated=1、124568输出token，38.564s。MTP接受89929/176880=50.8418%，3.54209 committed tokens/step。配置、数据hash、逐题预测和计数保存在`standalone/gsm8k_final/`。首轮90.6748%与历史90.8264%一起保留；单轮91.8878%不证明迁移提高了准确率，也不与官方不同评测配置的成绩直接等同。38.56s仍高于历史28.03s；预热、请求批形状与浮点归约未隔离，不称纯代码加速。

全部测试后两台各31长期进程/16GPU进程，FD最大374/347、上限1048576，fatal为空、cgroup oom/oom_kill=0，详见`service/audit_final_after_all_node*.json`。保留正在运行的双机DPEP16服务：.17.62/.17.55的18941端口，生产源码d06ec314f。**四机PD未启动，按用户要求等待通知**；只有模板、相关缓存/PD回归，未把它们表述成四机实测通过。

## 2026-10-30：独立 EIP 的 LLM Bench 20% 回放

用户要求避开正在使用的 `10.210.6.10:10418`。确认 H100-msw 的旧8020代理无活跃连接且健康请求失败后，备份旧配置，将 `10.210.6.10:11046 → H100-msw:8020` 转发到本轮 `.17.62/.17.55:18941`，HTTP leastconn，后端健康检查。原10066代理PID3077179和配置SHA256 `9c5c88ca4e155ac0582574f44bbd50a700b1904f195bc2641575bb4ade559f42` 均未改变；8019未修改。当前拓扑仍为双机DPEP16独立服务，不是四机PD。

公网health、模型列表与真实42推理均HTTP200；平台inference preflight成功、929ms（只作功能预检）。两台启动前审计分别31长期进程/16GPU进程，FD最大374/347，上限1048576，cgroup oom/oom_kill=0，fatal为空；未重启模型服务或更改包/启动参数。

复制源任务 `73cbd186-218e-4531-b480-543b6cb2dd83`，保留原数据集 `80d718de-af61-468b-b06c-97adb0fe0c01`、session_ratio=20、factor=1、50ms调度、prompt cache与clear_cache_before_run=false；目标模型为DeepSeek-V4.1-Flash-FP8，新seed `PVbmo`。首个6位seed被平台422拒绝、未创建任务，改为平台要求的5位seed后仅成功创建一个任务，不使用force，不修改源任务或10418上的在跑任务。

任务名 `v41-ds4-DP16-20p-PVbmo-1030-163658`；ID `0a8db498-b4f9-4d15-88ac-2a7fb8c48d3b`。报告：http://10.9.66.47:5174/spaces/default/env/default/reports?region=intranet-node&taskId=0a8db498-b4f9-4d15-88ac-2a7fb8c48d3b 。HTTP201初始为pending，待确认运行及递增进度；不把创建成功表述成压测完成。

本轮配置、预检、创建响应与进度采样归档在H100-msw的 `/mnt/vqa/wanzihao/dsv41_port_support_ds4_20261030/llm_bench_11046/0a8db498-b4f9-4d15-88ac-2a7fb8c48d3b/`；同级保留代理配置、旧配置备份与身份记录。

已确认running且进度1→2→31/70646递增，最新完成31条、成功31条；公网health200。初期metric-series已有TTFT/TPOT样本，但样本很少且处于启动回放阶段，不作为最终延迟或吞吐结论。运行初期两节点FD最大930/910、上限1048576，未发现新增fatal或cgroup OOM。后续正常压测按每小时一次监测，不在期间运行profiler或额外推理。

## 2026-11-01：压测结束与当前入口失联检查

用户要求复查服务。当前EIP11046返回503；H100-msw的8020代理仍监听，但到.17.62/.17.55的8090和18941均无法连接，SSH报No route to host，无法进入节点复查当前进程、显存或cgroup。两份持久化服务日志最后均为2026-10-31 18:39:38 UTC（北京时间11-01 02:39:38）的health200，随后停止更新；不能仅据此断言容器被销毁或OOM。扫描已有完整日志，无Traceback、CUDA OOM、CUDA error、NCCL/NVSHMEM fatal或missing-WRITE-task；HTTP记录分别400=34/36、500=0。日志里仍有共享内存重名Warning，需要与逐请求失败记录区分。

任务0a8db498已completed：70646/70646，69029成功、1617失败，成功率97.7111%，dropped=0、unsent=0；UTC 10-30 22:00:03结束（北京时间10-31 06:00），实际历时47890.576秒=13.303小时，明显超过计划7199秒。报告聚合TTFT平均5164.4ms/p50 2679.1ms/p99 24655.2ms，TPOT平均464.476ms/p50 359.561ms/p99 2875.249ms，仅记录平台原值，尚未按失败类型/长尾请求核验，不作为正常稳态加速结论。此次节点失联发生在最后一次已知health200之后，晚于压测结束，不能用当前失联解释全部1617条历史失败；失败明细尚未完成归因。检查未重启服务、改动代理或占用其他机器。

## 2026-11-01：同步最新 support_ds4

从origin重新fetch，最新 `support_ds4=38fce3351cc258539f2705eca4f4c77bddc1a153`，相对移植基线6dc33df85新增1个提交。已通过merge提交 `a042cf54b` 合入 `support_ds4_v41`，未修改原support_ds4工作区的未提交内容；本轮未push、未重启服务或改动部署参数。

- 自动调优：删除移植版本对空cached_configs的重复处理，生产autotuner完整采用上游 `_missing_config_keys` 实现；保留移植时已有的level0/1/2参数化回归，不添加第二份同用途测试。DSpark顺序执行的target/draft图内存池共享不受影响。
- 多模态prefill：合入图片块内不再为hybrid checkpoint额外切分的修复，两条计算输入长度/输入IDs的路径保持一致；保留V4.1的Engram lookback生命周期。
- 共享锁：attach不再清零可能被持有的锁字；共享cache构造的barrier及所有权继续沿用上游。V4.1 FP8、CED共享KV、Engram、vision、DSpark及draft加载后target峰值复检均保留。

本机H200 GPU7，使用已有测试镜像fdf6df4b8331（Python3.10/Torch2.11cu130），独立容器、私有shm，不安装或替换任何依赖。上游autotuner/atomic-lock/multimodal三模块104 passed；backend及请求预处理36 passed/2 skipped（CPU容器未开放GPU，跳过两项GPU用例）；V4.1模型、checkpoint视觉参考、parser、DeepEP V2契约、DSpark、CUDA Graph生命周期、MTP manager和ViT FA3 GPU回归283 passed。共423 passed、2 skipped；merge提交black/flake8 hooks及git diff --check通过。

GPU首轮273 passed/2 failed，失败均为测试将源码挂成只读，而autotuner需在源码下创建NVIDIA_H200缓存目录。只为测试增加独立可写缓存挂载后完整重跑283项全部通过；没有为此修改生产代码、测试断言或包。首轮与复测原始日志均保留在 `/mnt/vqa/wanzihao/v41_update_20261101/`：upstream_cpu.log、backend_cpu.log、v41_gpu.log、block32_repro.log、v41_gpu_writable_cache.log。

迁移状态：文本/FP8、CED/Engram、vision和DSpark已有完整权重旧版本及本轮GPU小模型验证；此前DPEP16 GSM8K为1212/1319=91.8878%、MTP接收50.8418%。这些是同步前整机成绩，本次合并未重跑完整权重、GSM8K或压力测试，不能据此宣称本轮加速。历史20%回放97.7111%成功率和13.303小时长尾尚未完成失败归因；四机PD依用户要求仍待安排，prefill CUDA Graph、微批overlap及cross-DP cache fetch仍不支持。原两台DPEP16节点已失联，本轮没有把代码合并描述成服务恢复或四机验收通过。


### 2026-11-02：优化 1 — 补齐短 suffix 的 Decoder SWA Bounded Replay

短 extend 不足128行时额外重算此前最多128行 encoder（图片块内起点退到块首），decoder 始终取得整个当前 prompt 的最后 min(N,128) 行。ModelInput 分开实际计算起点和原始 KV 提交边界；调度计入重算行，但请求进度、页分配和共享前缀不回退。历史重算行禁止写 main/global KV 和 indexer K，encoder/decoder SWA 分别按 replay 起点截断，Engram lookback 对齐计算起点。保留21层全量＋19层裁剪、FP8及现有持久化布局。

H200隔离单卡、只读源码和独立可写 autotune 缓存：V4.1 GPU回归146 passed（含文本/视觉、DSpark、GPU checkpoint、CPU cache 恢复、decode graph）；输入/分页CPU回归53 passed。256-token缓存命中＋新增1 token的 decoder FFN 从1行变为128行，新增断言确认共享 global KV 前256个物理槽逐字节不变。这是语义修复，可能增加短 suffix TTFT，不当作加速。尚未跑完整权重 GSM8K 或多机 PD。日志：/mnt/vqa/wanzihao/v41_four_optimizations_20261102/replay_{cpu,gpu}.log。未修改在线实例。


### 2026-11-02：优化 2 — Hopper FP8 候选集内 Reindex

候选源输出最多2048个排序去重block ID；四个后续Reindex直接读取最多16384个候选位置的现有FP8 paged K/FP32 scale，以Tensor Core评分并复用公共Top-K/物理槽转换。不再生成全长候选bool mask或consumer全长logits；源层扫描保持原样。短上下文候选已覆盖全部位置时保留DeepGEMM并省掉无意义的mask，不强制切换较慢的新kernel。

H200数值、causal/HOLD、非连续页、CUDA Graph专项20 passed，完整V4.1 GPU回归146 passed。新旧候选评分按已有atol0.03/rtol0.01核对。真实调用使用现有deep_gemm与FP8 cache格式，无依赖升级。

同卡合成分页cache、32heads、FP8、CUDA Graph三轮median，以下为单个consumer评分＋mask/Top-K/slot转换，不含Q投影、候选源及整模型：

| Context | Query rows | 原路径 ms | 候选路径 ms | 倍率 |
| --- | --- | --- | --- | --- |
| 64K | 64 | 0.11737 | 0.07714 | 1.52x |
| 256K | 64 | 0.42725 | 0.08107 | 5.27x |
| 1M | 64 | 1.62156 | 0.08581 | 18.90x |
| 1M | 384 | 8.56060 | 0.38703 | 22.12x |

1M/64行consumer logits从256MiB降到4MiB（64倍）；物理KV布局仍1800B/token。8K/64行单独复测原0.02550ms、实际选用的DeepGEMM免mask路径0.02251ms；直接强用新kernel为0.04176ms，因而未在该范围启用。不能将局部倍率当TPOT/TPM提升。原始日志/参数：/mnt/vqa/wanzihao/v41_four_optimizations_20261102/indexer_{gpu,model_gpu}.log、indexer_bench.jsonl、indexer_short_bench.jsonl；可复跑test/benchmark/kernel/benchmark_dsv41_candidate_indexer.py。未改线上服务或重跑完整权重评测。


### 2026-11-02：优化 4 — 复用公共 mHC 融合 V4.1 shifted seam

公共hc_post_pre_norm增加可选previous_pre，复用现有post＋projection kernel及shifted Sinkhorn/mix/norm finish；V4.1的attention→FFN seam不再强制分开post与projection。保留原BF16舍入、FP32累加和系数移位，不引入模型专用后端、Mega-mHC依赖或新的权重格式；层间FFN post暂不延迟，以免改变Engram/hidden collector边界。

H200/hidden5120，CUDA Graph三轮median：1行11.072→8.325us（1.33x）；6行11.649→11.198us（1.04x）。第一版强用融合的16行11.934→13.318us，约慢11.6%；最终shifted仅在8行以内启用融合，16行复测12.089→12.114us，32/64/128/8192行沿用原算子组合且差异约噪声范围。V4原有16行阈值未变，不把小batch的局部收益外推整机速度。

共享mHC回归57 passed（新增24项含hidden128/5120、1/6/16/17/64/128行、禁用/启用DeepGEMM、更新输入后的Graph replay），加V4.1模型共203 passed，含vision/DSpark/cache/graph。日志与首版退化记录均保留：/mnt/vqa/wanzihao/v41_four_optimizations_20261102/mhc_gpu.log、mhc_model_gpu.log、mhc_bench*.jsonl；复跑test/benchmark/kernel/benchmark_dsv41_mhc.py。未改在线服务。


### 2026-11-02：优化 5 — Decode Engram 单层提前预取

复用原FP8 host/UVA lookup，将本地shard读取提前到前一层的attention/FFN执行期间，在独立CUDA stream上执行；消费者用event等待并record_stream保护embedding寿命。TP all-reduce仍在消费者原计算stream上执行，不把collective放到预取stream。每次forward至多持有一份待消费embedding，不新增持久化hidden缓存、线程或后台manager。首层Engram在_token_forward入口预取，明确置于decode graph捕获内部，避免graph replay沿用旧hash查出的embedding；DSpark draft不走target的首层预取。

首版仅decode启用；prefill沿用原路径，避免大chunk额外占用embedding缓冲。实际配置24个ngram head、每head256维时，待消费BF16 embedding为每query行12288B，384行约4.5MiB；不增加host权重副本。

H200合成组件微测：264MiB FP8/UE8M0 host表，hash每次graph replay轮转，24heads/256dim，前序计算为5120x5120 BF16投影，三轮median。1行串行23.013→预取20.869us（1.10x），6行24.932→22.156us（1.13x），64行36.368→34.880us（1.04x），384行122.929→122.676us（基本持平）。这些是可重叠工作负载的组件实验，不是实际完整前层或整模型TPOT；没有声称多机吞吐收益。原始记录engram_bench.jsonl/.stderr，复跑test/benchmark/kernel/benchmark_dsv41_engram_prefetch.py。

最终GPU回归212 passed（V4.1 155项＋共享mHC57项），覆盖首层/中间层Engram、文本/视觉、DSpark、GPU/CPU checkpoint恢复与更新输入的decode Graph。额外预取专项验证实际host lookup、更新hash后Graph结果，以及collective位于消费者stream；多rankcollective在该专项用替身检查顺序，尚未做真实TP多机性能验收。CPU输入/分页及PD生命周期回归155 passed。Black/Flake8、git diff --check通过。日志根目录：/mnt/vqa/wanzihao/v41_four_optimizations_20261102/，final_gpu.log、final_cpu.log、engram_model_gpu.log、engram_prefetch_gpu.log。

本轮未部署、未重启线上服务、未升级依赖、未跑完整权重GSM8K。Indexer/mHC/Engram数据均为独立单卡组件实验，Graph使用微测自身的capture形状；真实服务较大的固定graph_max_len_in_batch会影响indexer的静态capacity与路径选择，8K的免mask DeepGEMM微测不能当作默认1M graph容量下整机短上下文成绩。完整checkpoint质量和端到端TTFT/TPOT/TPM仍待独立服务验收。已有launcher及旧test.md未提交内容均保留，本轮commit仅包含四项实现及各自新增记录。


### 2026-11-02：CED 改为20层全量＋20层尾段

按用户要求与SGLang当前write_tail_source_kv路径对齐：decoder起点为最后一个KV source（真实配置layer20），不是其下一层。该层先对全量encoder输出完成mHC混合/归一化与共享global KV/indexer K投影和写入，随后裁剪到最后128行；该层及后19层的Q/SWA、indexer Q/候选选择、attention和FFN均只执行尾段。普通decode与完整prompt logits路径保持全量契约。KV写入移到V4.1自己的context/token forward，索引阶段不再重复写；DSpark collector按新边界对齐encoder/decoder hidden。

H200完整V4.1 GPU回归155 passed；补充首个decoder层计数断言后边界/模型回归17 passed。256行输入时，首个decoder层全量KV writer仅调用一次且输入256行，QKV与FFN均128行；既有断言确认共享global KV与未裁剪参考逐字节相同。覆盖Engram首层/中间层、FP8/历史FP4测试fixture、vision、DSpark、GPU/CPU cache恢复及decode graph。最初新断言误读测试外层config而失败，改为读取运行model.config；未为断言改变生产语义。

8192-token单请求chunk的attention/FFN token-layer预算由21*8192+19*128=174464降到20*8192+20*128=166400，少4.622%；全量global KV投影和必要归一化仍存在，该数字不是所有算术量或实测TTFT提升。21＋19的旧预算只描述之前版本，现已被本条替代。Black/Flake8通过，日志：/mnt/vqa/wanzihao/v41_ced_fetch_20261102/ced_gpu.log、ced_boundary_gpu.log；未改在线服务。


### 2026-11-02：V4.1 同节点跨 DP prompt cache fetch

移除V4.1对现有--enable_dp_prompt_cache_fetch的禁用，直接沿用DPChunkedPrefillBackend及DPKVSharedMoudle的V4公共路径：只创建owner DP的InferReq，其他DP保留radix/checkpoint引用后提供命中长度；通过CUDA IPC读取源global KV，复制实际四个source pool（C2/C2/C2/C1），继承同TP shard共享CPU池的checkpoint ID并恢复请求状态。没有新增V4.1传输后端或协议；范围与V4一致，为同一节点的DP之间，不是跨机器拉取。

请求容量选择器同步把V4.1归入无需为外国DP请求分配槽位的模型，避免打开fetch后退回每rank全局running_max_req_size、放大SWA和MTP常驻显存。保留diverse_mode原有全局容量。开关仍由已有CLI控制；未修改launcher及运行实例。

双H200真实CUDA IPC、Gloo计划交换、NCCL完成屏障、公共KV传输kernel和真实ReqManager/MemoryManager验证4项通过：V4回归、V4.1大页终点512、V4.1小页终点768/已有本地前缀256/DSpark5运行态布局，以及跨默认128MiB host注册边界的checkpoint拷贝。源物理页刻意非连续；接收端逐字节核对所有source池。checkpoint池填满后仍能继承引用，源释放引用并覆盖原GPU池后接收端仍能再次恢复SWA，最后所有checkpoint引用归零。V4额外核对compressor/indexer state，V4.1保留现有C2/C1对齐检查点契约。

同一双卡测试还模拟源进程Engram history完全缺失，真实CPU输入和preprocess生成129行encoder replay、128行decoder尾窗对应输入，保留原始提交边界；从实际replay起点恢复lookback后，两层Engram hash与整段参考逐位相同，cur_kv_len不被重算倒退。这里只验证DSpark5 cache布局，未启动完整DSpark服务或声称测过整机接收率。

公共源ID/接收引用及V4/V4.1容量契约CPU回归11 passed；允许fetch的V4.1参数与tiny model回归17 passed，含文本/vision、Engram首层/中间层、DSpark及decode Graph。将既有checkpoint GPU回归从已删除的SWA-page API更新为当前请求ring/CacheConfig接口；未为兼容旧fixture改生产接口。Black/Flake8及git diff --check通过。日志：/mnt/vqa/wanzihao/v41_ced_fetch_20261102/fetch_gpu.log、fetch_cpu.log、fetch_model_gpu.log。尚未做完整权重四机服务、GSM8K或LLM Bench性能验收，未把功能通过写成TTFT/TPM收益。


### 2026-11-02：V4.1 复用 V4 的跨层延迟 mHC post

将 support_ds4 已有的跨层衔接补到 V4.1：非末层 FFN 输出先携带 (x, residual, residual_mix, post_mix, shifted_pre)，下一层 attention 入口调用现有公共 hc_post_pre_norm(previous_pre=...)，合并上一层 post 与本层 shifted pre / RMSNorm。未新增算子、后端、依赖或 CLI 开关；生产代码仅两个文件增加22行。共享算子的原有选择不变：shifted 路径不超过8行才融合 post 与投影，较大 batch 仍使用原 post + DeepGEMM。

Engram 前一层立即完成 post，保证注入发生在完整 residual 上、下一层 mHC 投影之前。DSpark 由现有 hidden collector 的目标层声明在初始化时关闭这些层的延迟，直接收集已完成的 streams，不在 collector 与下一层重复做 post。末层按运行时 is_last_layer 完成 post，兼容 DSpark draft 构造时重新确定末层的顺序。CED 首个 decoder 层仍先对全量 encoder rows 完成混合、归一化和共享 KV 写入，再裁剪尾窗；未改变20+20契约。

H200、hidden=5120、BF16 residual、FP32 mixing、Sinkhorn20次，走真实 V4.1 _hc_ffn_out → _hc_attn_in 调用，比较立即完成与延迟完成两种路径。CUDA Graph 微测，三轮 median；LIGHTLLM_TRITON_AUTOTUNE_LEVEL 未设置（默认0），使用现有/默认 kernel 配置，没有强制新一轮 autotune。时间仅为单个跨层 mHC 衔接，不含 attention / MoE，不代表完整模型 TPOT 或 TPM。

| Query 行数 | 原立即 post（μs） | 跨层衔接（μs） | 耗时变化 |
| --- | ---: | ---: | ---: |
| 1 | 11.063 | 8.318 | -24.81% |
| 6 | 11.556 | 11.191 | -3.15% |
| 16 | 11.936 | 11.961 | +0.21% |
| 32 | 12.461 | 12.477 | +0.13% |
| 64 | 13.151 | 13.172 | +0.16% |
| 128 | 15.223 | 15.169 | -0.36% |
| 8192 | 387.393 | 388.169 | +0.20% |

16行及以上复用相同的原有 GPU 算子，变化处于本次微测噪声范围；收益集中在小 batch decode。本次只是接通现有跨层融合，未实现论文完整 Mega-mHC 或系数计算的跨流重叠。

GPU 唯一用例合计169 passed、8 skipped：专项24项（8项真实 kernel 的跨层 shifted 系数/输入不变性/更新输入后的 CUDA Graph，以及16组 tiny model），其余回归145项。跳过项依赖本容器未挂载的官方视觉参考 image_processor / tokenizer；已有 tiny vision 路径通过。tiny model 的256行 prefill 对照原立即 post 路径，logits 及 DSpark hidden 逐位相同；补充原生 DeepGEMM 开启的单 token decode 对照 logits 与 hidden，16组复验通过（已计入169项，不重复累加）。保留 BF16 舍入，native decode 采用既有数值容差。覆盖 Engram 首层/中间层、FP8/历史 FP4 测试 fixture、DSpark draft、20+20 CED、checkpoint恢复及 decode Graph。

Black/Flake8、git diff --check 通过。记录：/mnt/vqa/wanzihao/v41_cross_layer_mhc_20261102/ 下 model_gpu.log、regression_gpu.log、native_model_gpu.log、boundary_bench.jsonl/.stderr、environment.json；复跑微测使用 test/benchmark/kernel/benchmark_dsv41_mhc.py。未启动或修改在线服务，未测试完整权重 GSM8K/整机性能。既有部署文件和 test.md 未提交内容保留，本次 commit 只包含该优化与本条新增记录。


### 2026-11-02：Engram hash 融合与 Decode 提前预取

复用现有 EngramHasher、engram_stream、prefetch_engram 和单份待消费 embedding。首个 Engram lookup 改为在 target _token_forward 入口启动（仍在 decode Graph 内），不等到该层的前一层 attention 入口之后；每次消费 Engram 后，在该层原有 token_forward 预取点提前启动下一层 lookup。真实配置[1,14]因此在 decode 入口准备layer1，在layer1准备layer14，替代原先layer0/layer13准备的安排。没有增加后台manager、持久化embedding副本或新CLI；TP all-reduce及WKV投影仍在原消费stream，prefill lookup策略不变。提前调度本身的整机收益尚未单独量化。

EngramHasher 原逐项PyTorch的映射、历史收集、乘法、XOR、取模和拼接，改为现有engram.py内两个Triton kernel：先统一写当前batch的token history，再读取前驱并产生全部层的hash。分成两个kernel是必要的依赖屏障：chunk/MTP可包含同一请求的多行，跨CTA直接读正在写入的history会有竞争。保留原token压缩表、int64乘法/XOR/取模、逐层multiplier/prime/offset、视觉token截断与padding语义，hash及更新后的history与原Torch实现逐位相同。hash输出shape/layout保持不变，未改FP8表或量化算法。

H200/Torch2.11.0+cu130/Triton3.6.0，合成2层、每层24个hash（4-gram、每阶8 heads）、32768 token map。旧Torch实现保留在现有benchmark内作对照；三轮median。Graph列为hash准备的GPU执行时间；eager列为100次调用提交并同步收尾后的单次平均墙钟时间，包含Python/CUDA提交开销，符合该准备步骤目前在服务decode Graph外执行的性质。这不是完整Engram、整模型TPOT或TPM测试。

| Query 行数 | 原Graph（μs） | 融合Graph（μs） | 原eager（μs） | 融合eager（μs） |
| --- | ---: | ---: | ---: | ---: |
| 1 | 77.716 | 3.329 | 423.732 | 35.954 |
| 6 | 77.137 | 3.314 | 429.322 | 36.354 |
| 64 | 80.076 | 3.713 | 424.493 | 36.248 |
| 384 | 105.204 | 3.915 | 426.991 | 36.241 |
| 8192 | 122.406 | 5.445 | 435.057 | 35.456 |

6行样例torch.profiler记录GPU kernel从60个减到2个（_store_engram_tokens、_engram_hash）；这里位置/请求/slot元数据为int32，token ID为int64，其他dtype下旧路径cast数量可能不同。trace与kernel计数存于hash_previous_trace.json、hash_fused_trace.json、hash_profile.json。复跑现有test/benchmark/kernel/benchmark_dsv41_engram_prefetch.py；bench_final.jsonl/.stderr保存最终结果。现有serial/lookup预取微测仅验证原读取路径，不将其本来就有的重叠收益当作本轮新增收益。

撤回一项实验：直接读取分层hash切片stride、取消contiguous打包，在264MiB host表、轮转hash避免重复小工作集命中的合成重叠测试中，6/64/384行分别比先打包慢约8.9%/44.4%/22.9%；改二维grid仍未消除退化。因此最终完全保留原gather_local/_gather_table和contiguous读取路径，未把“少一次copy”直接当成更快。候选源码engram_strided_candidate.py、benchmark_strided_candidate.py及bench.jsonl、bench_2d.jsonl保留在日志目录，可核对被撤方案；未据此推断实际服务整机退化。

最终GPU回归197 passed、8 skipped（validated_gpu.log）。新增20组ragged/跨CTA/空batch/视觉padding/真实8 heads的hash逐位对照与更新输入后的Graph测试；tiny model从16组扩到24组，新增双Engram，覆盖提前串联预取、CED边界、文本/vision、DSpark target/draft、原生DeepGEMM、GPU/CPU checkpoint恢复和Graph。8项跳过仍是本容器未挂载的官方视觉参考文件。已有本地shard及consumer stream collective契约测试通过；没有改变collective位置。Black/Flake8及git diff --check通过。

日志目录：/mnt/vqa/wanzihao/v41_engram_prefetch_20261102/。未改运行服务、未安装依赖、未测试完整权重GSM8K/服务吞吐；不能将hash组件21–27倍的GPU加速解释为完整Engram或模型同倍加速。当前实际变化是每次forward更低的hash准备开销，以及decode lookup更早开始。既有部署文件与test.md的旧未提交内容保留。


### 2026-11-02：2P2D 启动参数启用 DP fetch 与动态 DSpark verify

P launcher 增加 --enable_dp_prompt_cache_fetch（同节点DP之间），D launcher 增加 --mtp_dynamic_verify；D 保留CUDA Graph。两个参数按现有调用路径分别放在P/D，不给D添加不会走到的DP前缀拉取路径。每个参数仍单独一行。

P/D 的 --mtp_step 暂保持5。P prefill仅用target hidden做draft KV-only commit，不会因为step5额外生成5个草稿；但当前DeepseekV4AttStatePageHelper按物理顺序整块传输SWA/compressor ring。get_runtime_state_shapes中C2 ring长度为4+mtp_step，P1会有5槽、D5为9槽，直接混用会造成恢复布局和物理索引不一致。要允许P1/D5，应先支持逻辑位置重排的状态传输或解耦统一传输容量，不能仅修改launcher数字。

bash -n及git diff --check通过；本条仅为命令配置更新，未重启/部署服务，未声称动态调度性能或P1/D5多机已验收。既有节点配置、DeepEP隔离路径及旧文档的未提交修改保留，本次commit只包含两个新增参数及对应说明。


### 2026-11-02：修复 V4.1 DP fetch 请求槽位重复分配

router.wait_to_model_ready 原先单独判断 model_type != deepseek_v4，漏掉 deepseek_v41；双机DPEP16、running_max_req_size=256时，开启DP fetch会给每个P rank传256个GPU请求槽位，而调度器使用32个。改为复用现有 get_running_max_req_size_per_dp，不再在router维护第二套模型例外。没有增加配置、fallback或管理器。

新增测试实际调用router初始化路径并截取8个worker的init_model参数，覆盖V4/V4.1/其他模型及diverse模式。修复前仅V4.1正常模式失败（256 != 32），其余5例通过；修复后整个请求容量文件24例通过。同步修正同文件既有测试fixture：显式初始化队列所需的per_dp容量，并按现有函数的语义检查普通full-attention容量。此时尚未部署修复，不把KV容量恢复量作为实测结论。

记录：/mnt/vqa/wanzihao/v41_pd_restart_20261102_122400/bugfix_20261102_133822/ 下 router_capacity_before.log、router_capacity_after.log。
