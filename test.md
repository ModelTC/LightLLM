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
