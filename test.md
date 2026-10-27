# DeepSeek V4.1 Flash：DPEP8 测速与 indexer 优化（2026-10-27）

## 测试配置

- H200-07：LightLLM；H200-08：vLLM 0.32.0。各 8×H200，独立单节点 DP8/EP8，端口 18942。
- 模型均为 `/sensecore/DeepSeek-V4.1-Flash-1017-FP8`，FP8 E4M3 权重、128×128 UE8M0 scale，其他参数保持 BF16，KV 使用 FP8。
- max context 262144，batch 8192，prefill chunk/threshold 4096，显存比例 0.75，prompt cache 开启；DSpark 固定 5 token，greedy，adaptive verification 关闭。
- LightLLM `--tp 8 --dp 8 --enable_ep_moe`，等价每卡 attention、8 卡 EP；vLLM TP1/DP8/EP8。LightLLM 使用 legacy DeepEP，vLLM 使用 DeepEP v2；本轮只优化 indexer。
- LightLLM 镜像为 `registry.cn-tj-01.sensecore.cn/ccr-tj/lightllm:v1.3.9-dsv4-c5bc99`，Torch 2.11.0+cu130；vLLM 官方 0.32.0，Torch 2.13.0+cu130。vLLM digest：`sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`。
- vLLM 使用原生 `--moe-backend deep_gemm`，并只读覆盖主模型、DSpark 两个 loader 的 UE8M0 scale 转换：FP32 目标按真实数值转换，uint8 目标保留位表示。因此这里不是完全未修改的官方镜像结果；推理 kernel 和调度器未修改。
- 同一官方 vLLM 0.32 bench serve 客户端，本机 localhost 发压，固定长度随机 token；每次预热和正式运行均用独立新 seed，temperature=0、top_p=1、top_k=1、ignore_eos。通常输出 512 token，131072 档输出 256，260000 档输出 128。
- TTFT/TPOT 为请求平均值，单位 ms；吞吐为包含 prefill 的端到端输出 token/s，按 usage 统计 token。C1 仅 4 个正式样本；这些是受控随机输入性能测试，不是新的线上 LLM Bench 20% 回放或 GSM8K 准确率结果。

## 优化前 LightLLM 与 vLLM 完整基线

每个引擎 11 档，共 2864 个正式请求，全部成功。表内每组数值顺序为 **LightLLM 原始 / vLLM**。

| 输入 token | 并发 | TTFT ms LightLLM / vLLM | TPOT ms LightLLM / vLLM | 输出 token/s LightLLM / vLLM | vLLM 吞吐倍数 |
|---:|---:|---:|---:|---:|---:|
| 16384 | 1 | 1085.20 / 565.24 | 3.81 / 4.92 | 168.90 / 166.34 | 0.98× |
| 16384 | 8 | 2721.84 / 1064.35 | 11.41 / 7.45 | 439.94 / 740.46 | 1.68× |
| 16384 | 64 | 10674.99 / 4142.55 | 38.80 / 33.57 | 1013.39 / 1435.57 | 1.42× |
| 16384 | 256 | 11791.97 / 10063.11 | 102.98 / 78.06 | 1945.18 / 2502.69 | 1.29× |
| 16384 | 512 | 20811.27 / 17533.84 | 165.02 / 76.43 | 2407.12 / 4331.71 | 1.80× |
| 65536 | 1 | 4159.00 / 2097.97 | 4.26 / 2.87 | 80.82 / 143.68 | 1.78× |
| 65536 | 8 | 5358.76 / 4635.21 | 16.20 / 5.97 | 270.49 / 516.43 | 1.91× |
| 65536 | 64 | 18295.63 / 11798.93 | 79.53 / 35.70 | 545.94 / 1049.14 | 1.92× |
| 65536 | 512 | 97385.00 / 128013.25 | 567.73 / 78.86 | 671.67 / 1308.26 | 1.95× |
| 131072 | 1 | 9528.98 / 4195.43 | 15.52 / 11.56 | 18.98 / 35.83 | 1.89× |
| 260000 | 1 | 24351.70 / 8742.12 | 14.79 / 10.82 | 4.88 / 12.65 | 2.59× |

两套服务实际 KV 预算不同：LightLLM 自动 profile 为 11,840,396 token/DP，vLLM 约 387 万 token/DP。相同显存比例不意味着相同 KV 预算。64K/512 并发的 vLLM TTFT 比 LightLLM 更高，但 TPOT 更低，该档 vLLM preemption 计数前后均为 0。

## Indexer 改动与架构

源码工作树 `/mtc/wanzihao/lightllm-dsv41`，分支 `support_dsv41_model`。本轮完整差异在运行目录的 `indexer_opt/indexer.patch`，未覆盖其他已有改动。

1. 移除 V4.1 原来逐 query 的 Triton `_index_logits` 和 torch 通用 top-k/sort/pad 链路。decode 复用 V4 已有的 DeepGEMM paged FP8 MQA scorer，实际有效长度由 GPU tensor 控制，避免按 256K graph 上限执行无效打分。
2. 在原 hybrid KV buffer 内把 V4.1 indexer 数据排为 64-key slab，使用只读 strided view 和页表交给 DeepGEMM；不新建 K cache 池，不做逐 query KV gather。V4 C4 原本就是 64-key slab，其布局不变。
3. Prefill 在既有 512MiB scratch 预算内，每个请求、每个 index source gather 一次 FP8 K，调用 dense FP8 MQA；单 query 或 gather 超出预算时使用 paged scorer。
4. 复用 native top-k，Triton 完成候选块最大值、最近块保留、候选 mask、排序去重、无效项压缩和物理页映射；CED 裁剪后重建 query 对应的 indexer 元数据。
5. max context 仍为 262144，KV 自动 profile 仍为 11,840,396 token/DP；没有通过缩小缓存或 graph 上限获取收益，没有新增生产文件、后端或依赖包。

## 最终 V3 端到端结果

V3 为当前最终版本：在 V2 基础上把新 kernel 的 stride/宽度改成运行时参数，消除随长度反复编译的开销。三档正式请求共 **1156 个，失败 0**。每组三个数依次为 **优化前 LightLLM / 最终 V3 LightLLM / vLLM**。

| 输入 token | 并发 | TTFT ms 前 / V3 / vLLM | TPOT ms 前 / V3 / vLLM | 输出 token/s 前 / V3 / vLLM | V3 吞吐提升 |
|---:|---:|---:|---:|---:|---:|
| 16384 | 512 | 20811.27 / 20177.59 / 17533.84 | 165.02 / 154.20 / 76.43 | 2407.12 / 2538.79 / 4331.71 | +5.47% |
| 65536 | 64 | 18295.63 / 15646.57 / 11798.93 | 79.53 / 59.79 / 35.70 | 545.94 / 688.66 / 1049.14 | +26.14% |
| 260000 | 1 | 24351.70 / 14447.01 / 8742.12 | 14.79 / 13.44 / 10.82 | 4.88 / 7.92 / 12.65 | +62.38% |

这里的预热是先用另一批独立随机输入运行 kernel，使首次编译等开销单独记录；正式测试使用新 seed，不复用预热 prompt 来获得 cache 命中。模型启动加载、单独预热不计入正式结果；正式运行中产生的新形状开销仍包含在请求时延内。V2/V3 各为一次新 seed 实测，并非相同请求的逐项配对；本表使用最终版本数字，没有选取两次中较高的结果。260K 单并发只有 4 个正式样本。

16K/512 并发最终 MTP 接收率：**67.577%**，每次验证实际输出约 **4.357 token**；优化前 LightLLM 为 66.806%，vLLM 为 66.684%。完整各档统计在 `indexer_opt/mtp_cases.json`。

最终验收：23 项 V4.1 GPU 数值/图回归通过，V4 共用路径沿用同代码下已通过的 3 项 native 分支回归；文本和图片 smoke 均通过，HTTP health=200。31 个长期进程 nofile 均为 1048576，最高 FD 使用 715，cgroup OOM/oom_kill=0，空闲请求计数为 0。服务保持运行。

## 中间版 V2 实测（修正冷编译之前）

以下保留 V2 实测，三档正式请求共 1156 个、失败 0。其稳态结果有提升，但首次长请求存在大量重编译开销；最终结果使用上面的 V3 表。

表内每组三个数的顺序为 **优化前 LightLLM / V2 LightLLM / vLLM**。

| 输入 token | 并发 | TTFT ms 前 / V2 / vLLM | TPOT ms 前 / V2 / vLLM | 输出 token/s 前 / V2 / vLLM | V2 吞吐提升 |
|---:|---:|---:|---:|---:|---:|
| 16384 | 512 | 20811.27 / 20168.96 / 17533.84 | 165.02 / 148.40 / 76.43 | 2407.12 / 2611.35 / 4331.71 | +8.48% |
| 65536 | 64 | 18295.63 / 15500.30 / 11798.93 | 79.53 / 69.49 / 35.70 | 545.94 / 624.41 / 1049.14 | +14.37% |
| 260000 | 1 | 24351.70 / 15043.77 / 8742.12 | 14.79 / 13.99 / 10.82 | 4.88 / 7.61 / 12.65 | +55.94% |

预热单独记录，不计入正式结果。V2 预热最多 64 并发、每请求输出 64 token；原基线运行过小并发到大并发的完整矩阵，paged-only 初版仅 8 请求预热，因此保留预热口径差异，不能把局部微基准倍数当作整机倍数。260K 单并发只有 4 个样本。

16K/512 并发的 MTP 接收率：优化前 LightLLM **66.806%**，V2 LightLLM **66.832%**，vLLM **66.684%**。LightLLM 接收率按 accepted / (verified rows − verify steps) 统计；V2 每次验证实际输出约 4.322 token。吞吐变化不是靠明显提高接收率得到的。

## 局部微基准与验证

Decode 微基准来自 paged 初版，prefill 微基准来自 V2 dense 版；最终 V3 的完整服务结果以上文表格为准。

| 场景（8 个 index source） | 原始 ms | 优化 ms | 加速 |
|---|---:|---:|---:|
| Decode：16K，384 query rows，CUDA graph replay | 35.93 | 4.71 | 7.62× |
| Prefill：16K，8192 query rows，CUDA graph replay | 118.14 | 13.69 | 8.63× |

这些是 indexer 局部微基准；prefill 的所有 source 都使用 8192 query rows，未模拟 CED 裁剪，不代表整个模型提速倍数。paged decode 初版的真实模型 profile 为约 24 步/1792ms，原始约 18 步/1836ms；profile 含额外开销，仅用于判断路径和量级。

- V4.1 GPU 测试 23 项通过：C1/C2、decode、dense/paged prefill、候选产生与消费、HOLD、短 prefix、打乱物理页、CUDA graph replay、CED、两请求 ragged K。
- V4 原有三项 scorer/graph 测试通过其既有 native top-k 分支。镜像的 FlashInfer wrapper 不接受原测试传入的 page_size，未改动的原始基线也复现；未修改生产 selector 或安装包绕开。
- V2 完成后文本算术返回 `42`，红色图片返回 `红色`；所有长期进程 nofile=1048576，FD 最大 715，cgroup OOM/oom_kill=0，空闲请求计数为 0。

## 冷编译问题与 V3 修正结果

V2 首个 260K 预热请求 TTFT 为 162926ms，原始基线相应预热为 34897ms；稳态 TTFT 为 15044ms。这笔首次长请求的回退单独披露，不用预热后的数字掩盖。

已定位新 kernel 把 stride/宽度声明为 constexpr，长 prefill 随长度产生大量编译变体：top-k transform 165 个、candidate scores 105 个、mask/mark 各 70 个。V3 将这些值改为运行时参数，BLOCK/TOPK 等算法常量继续编译期确定。GPU 数值回归 23 项通过；独立空 Triton cache 下测试 16K、32K、64K、128K、256K 宽度，四类 kernel 各只编译 1 个变体，首次约 3.0 秒，后续新宽度约 0.28–0.57ms。

完整服务 V3 已完成同样三档复测。以下为每档单独预热的实测，输入数和输出长度在 V2/V3 间一致，预热也使用不同 seed。

| 输入 token | 预热并发/请求数 | V2 预热耗时 s | V3 预热耗时 s | V2 / V3 预热 TTFT ms |
|---:|---:|---:|---:|---:|
| 16384 | 64 | 78.62 | 69.51 | 47510.78 / 39769.04 |
| 65536 | 64 | 64.77 | 34.18 | 52365.02 / 21782.73 |
| 260000 | 1 | 163.78 | 26.08 | 162926.05 / 25227.39 |

真实服务覆盖三档之后，新 kernel 的编译变体数如下；V3 保留 TOPK/算法常量等导致的少量合法变体。

| Kernel | V2 变体数 | V3 变体数 |
|---|---:|---:|
| `_indexer_topk_transform` | 165 | 5 |
| `_candidate_scores` | 105 | 3 |
| `_mark_candidates` | 70 | 8 |
| `_mask_candidates` | 70 | 2 |

## 可复现脚本与原始记录

- 运行目录：`/mnt/vqa/wanzihao/dsv41_h200_1p1d_20261026/engine_compare_225309`。
- 完整启动脚本：`/mtc/wanzihao/lightllm-ds4/compare-v41-dpep8/{config.txt,launch_lightllm.sh,launch_vllm.sh,run_container.sh}`；实际部署为运行目录 `deploy/`，两台哈希校验一致，挂载本地 `/root/.cache` 目录。
- 原始基线：`comparison.json` / `comparison.csv`、`bench/{lightllm,vllm}/*/{result.json,command.json,metrics_before.prom,metrics_after.prom}`；每档保存 seed、客户端完整命令、逐请求时延和输入/输出长度。
- V2：`indexer_opt/attempt2_dense_constexpr/{comparison.json,mtp_cases.json,bench/,service/,post_validation.json,service_final.log}`。远端同目录另有对应源码快照 `src/`。
- V3：`indexer_opt/{indexer.patch,deploy_sha256.json,tests_runtime_strides.log,tests_runtime_c2_first.log,cold_runtime_strides.json}`；最终汇总见 `indexer_opt/comparison.json` / `indexer_opt/comparison.csv`，逐请求数据见 `indexer_opt/bench/`，MTP 见 `indexer_opt/mtp_cases.json`，最终资源/生成验收见 `indexer_opt/service/`、`indexer_opt/post_validation.json`，编译变体数见 `indexer_opt/compiled_variants_v3.json`。
- 初版 paged-only 的 TTFT 曾回退，已保存在 H200-07 的 `indexer_opt/attempt1_paged_only/`，不算作最终收益。
- vLLM loader 兼容补丁：`vllm_patch/ue8m0_expert_loader.patch`；原始 trace、服务日志和资源审计在运行目录对应 `profile/`、`service/` 下。

## 第二轮 profile 与逐项优化（2026-10-27，进行中）

本轮起点为上文 V3。H200-07 跑 LightLLM，H200-08 保留 vLLM 0.32 对照服务；模型、DP8/EP8、256K context、4096 prefill chunk、DSpark 5、KV FP8 和显存比例保持一致。每项改动分别记录局部数值检查与完整服务结果，局部加速不作为端到端结论；回退项也保留。

新 profile 使用独立 seed 的英文随机词输入，分 prefill、全部请求首 token 后的 decode，以及混合排队三个阶段；每次约采样 2 秒，随后让全部请求自然完成，不中断客户端。其输入分布与前面的随机 token 性能矩阵不同，不能拿 profile 请求的 MTP 接收率直接对比正式基线。

| V3 profile，rank 0 | 采样窗口 ms | GPU 活跃比例 | MoE combine ms | MoE dispatch ms | 其他确认开销 |
|---|---:|---:|---:|---:|---|
| Prefill，64 请求、约 64K 输入 | 2105.80 | 84.62% | 527.81 | 119.24 | 路由专家两次 GEMM 合计 261.82 ms；gather table 79.28 ms |
| Decode，512 请求、约 16K 输入 | 2059.68 | 94.94% | 641.66 | 134.64 | 页表构建 39.87 ms / 56 次，约 712 us/次 |

这些是 rank 0 kernel 累加时长，通信内的跨 rank 等待也在其中，不能简单当成可全部消除的纯传输时间。接下来先用八卡独立 relay 核对镜像内既有 DeepEP v2 路径与 legacy 的数值、CUDA graph 和通信耗时，再决定生产改动。

原始记录目录：`engine_compare_225309/optimization_round2_20261027_0555/`；三阶段 trace 文件列表及采样控制信息保存在各 `profile_*/control.json`，数值/性能实验保留命令、完整日志和 JSON。

### E1：Hopper decode 复用既有 ElasticBuffer（验证中）

当前 common MoE 的 prefill 已用 v2，decode 则走 legacy low-latency。实验在同一个 common backend 内复用 ElasticBuffer 的 expanded dispatch、GPU expert prefix sums 和现有 DeepGEMM contiguous grouped GEMM；扩展已有 SwiGLU + FP8 量化 kernel，使其根据 GPU expert ends 跳过 padding。未新建后端、模型文件或依赖。SM100 FP8 和已有 legacy overlap 接口保持原路径。生产服务部署前先做八卡完整 MoE 数值及 graph 对照。

八卡 identity-expert relay 已通过 legacy、v2 expanded/non-expanded 的数值和 CUDA graph 检查，FP8 量化相对 L2 约 2.58%。relay 的 dequantize 布局与工作量差别很大，**其耗时不作为通信加速或整机收益**。首次 expanded relay 的数值失败来自实验误将 combine 的 `topk_weights` 参数当成加权操作；按接口实际行为显式加权后通过。首次 legacy 额外申请 7.6 GiB/卡 buffer 时因运行中服务占用而初始化失败，已在完整停止服务、确认 GPU 空闲后重测通过；失败运行不计入性能结果。

### E2：页表构建按 128 页向量化（局部通过，待服务单独验证）

只修改既有 `gather_c4_indexer_k_dsv4.py` 中的页表 kernel 和 launch grid；数据布局、页数、模型 context 和函数接口不变。12 组 C1/C2/C4 × batch 1/48/384/8192 分别对照原 kernel 和独立页映射参考，覆盖 129/16K/64K/256K 历史、HOLD、打乱物理页、以及修改 GPU 长度后直接 graph replay，全部通过。

| 页表微基准（256K 上限） | 原始 ms | 向量化 ms |
|---|---:|---:|
| C1，384 行 | 0.99972 | 0.01560 |
| C2，384 行 | 0.49224 | 0.00771 |
| C4，384 行 | 0.23901 | 0.00494 |
| C1，8192 行 | 21.01535 | 0.26557 |

每组取 5 次测量中位数，每次 20 次 graph replay。单行 C4 为 3.597 → 3.632 us，处于约 0.035 us 的微小差异；未把大 batch 的局部倍数当作整机倍数。E2 暂未叠入 E1 服务快照，以便分别统计端到端效果。

混合阶段补充 profile：rank 0 采样窗口 1811.72 ms、GPU 活跃 82.72%，prefill MoE combine 410.46 ms、dispatch 269.01 ms。所有三阶段采样客户端均自然完成，未通过中止请求清场。

E1a 的完整 MoE 实验回退，**未部署**：同一八卡进程、相同权重与路由，384 行时 legacy 约 0.654 ms，v2 contiguous + 通用加权约 1.283 ms；48 行时约 0.527 → 1.198 ms。与独立 FP32 routing/matmul 参考的相对 L2 为 legacy 约 1.5–1.8%、v2 约 0.6%，CUDA graph、空 expert 和全路由落到一张卡的检查通过。首次实验退出时的 buffer 清理错误单独记录；改为按 API 显式销毁两个 buffer 后 profile 复测正常退出。

E1a kernel profile 定位到最大容量输出上的 PyTorch 通用 `mul_`：5 次合计 3.296 ms，约 659 us/次。数值虽通过，但处理了大量无效容量；不能仅因换成 v2 就保留这版。源码、日志和 profile 保存在 `rejected_contiguous/` 及 `moe_contiguous_*`。

E1b 改为复用已有 masked grouped GEMM，用同一 common layout kernel 文件在 v2 expanded 与 masked slabs 间只搬运有效行，并融合输出加权；移除 E1a 的 contiguous helper、SwiGLU 扩展及相应单用途测试。新增布局 graph 回归 2 项、原有 SwiGLU 回归 32 项通过；接着分别检查 target 的 384 experts/top6 与 draft 的 128 experts/top3 八卡完整 MoE。

E1b 八卡完整实验均正常退出。下表为每张卡取 5 次 × 30 次 graph replay 的中位数，再取 8 卡最大值，单位 ms；随机路由以相同输入、相同权重在同一进程比较。

| Experts/top-k，行数、路由 | EP SM / DG SM | legacy ms | v2 masked ms |
|---|---|---:|---:|
| 128/3，48、随机 | 14/118 | 0.20142 | 0.21543 |
| 128/3，320、随机 | 14/118 | 0.27361 | 0.28226 |
| 128/3，48、全路由落单卡 | 14/118 | 0.20936 | 0.20773 |
| 128/3，48、随机 | 64/132 | 0.20401 | 0.22109 |
| 128/3，320、随机 | 64/132 | 0.27159 | 0.27781 |
| 128/3，48、全路由落单卡 | 64/132 | 0.19879 | 0.19663 |
| 384/6，48、随机 | 14/118 | 0.52699 | 0.53912 |
| 384/6，384、随机 | 14/118 | 0.65365 | 0.65122 |
| 384/6，48、全路由落单卡 | 14/118 | 0.39660 | 0.35023 |
| 384/6，48、随机 | 64/118 | 0.52727 | 0.54181 |
| 384/6，384、随机 | 64/118 | 0.65380 | 0.64172 |
| 384/6，48、全路由落单卡 | 64/118 | 0.39453 | 0.34055 |
| 384/6，48、随机 | 64/132 | 0.52158 | 0.53495 |
| 384/6，384、随机 | 64/132 | 0.64729 | 0.63372 |
| 384/6，48、全路由落单卡 | 64/132 | 0.36726 | 0.30990 |

随机路由下收益不足：target 约持平，draft 多数回退；偏斜路由有所改善。因此撤出 E1b 生产改动，保留原 legacy decode，没有为了更换通信库保留额外转换。实验源码保存在 `masked_v2_attempt/`，数值检查、命令与完整记录在 `moe_decode_masked_e*.json` / `moe_*.log`。后续如真实模型 profile 再显示偏斜路由通信瓶颈，再用该快照验证，不能把局部偏斜输入的收益当作实际模型收益。

### E3：按实际 overlap 模式分配通信与 GEMM SM（局部通过，待单独部署）

当前服务未开启 prefill/decode microbatch overlap，但 ElasticBuffer 默认 `prefer_overlap_with_compute=True`，并从所有 DeepGEMM kernel 中预留通信 SM。改为由现有两个 overlap 启动参数决定该偏好；同步模式下通信自动选择更多 SM，GEMM 使用全部设备 SM，开启 overlap 时沿用原有预留方式。改动位于现有通信组初始化，不增加 CLI/env 参数、模型常量或后端。

先在既有 prefill MoE 路径内拆开测量，使用同一输入、权重、buffer 容量和 7.6 GiB workspace；每组 5 次 × 10 次 eager forward，中位数再取 8 卡最大值。第一列增加通信 SM，第二步再恢复计算 SM；完整输出相对原始结果的 L2 均为 0。首次实验因测试脚本缺少 LightLLM rank 环境而中止，已修正测试环境后重跑，不计入结果。

| 单卡 prefill 行数 | 原始 EP14/DG118 ms | EP64/DG118 ms | EP64/DG132 ms | 最终降低 |
|---:|---:|---:|---:|---:|
| 128 | 0.87915 | 0.87684 | 0.88257 | -0.39% |
| 4096 | 3.15462 | 3.08442 | 2.98877 | 5.26% |
| 8192 | 5.99696 | 5.80704 | 5.62375 | 6.22% |

小 batch 128 行为约 0.4% 回退，大 batch 4096/8192 行分别降低约 5.3%/6.2%；保留全部结果，不把局部微基准当作整机结论。生产部署将放在 E2 页表整机复测之后，以便分别记录。

### 本轮 vLLM 独立复测

保留原 vLLM 服务与启动参数，使用新的独立随机 seed、相同官方客户端和预热规则。1156 个正式请求，失败 0；前后资源检查通过，health=200，24 个进程 nofile 均为 1048576，最大 FD 306，cgroup OOM/oom_kill=0。

| 输入 token | 并发 | TTFT ms | TPOT ms | 输出 token/s |
|---:|---:|---:|---:|---:|
| 16384 | 512 | 19803.34 | 80.82 | 4053.74 |
| 65536 | 64 | 11391.46 | 34.89 | 1093.88 |
| 260000 | 1 | 8654.04 | 10.35 | 12.84 |

本轮 16K/512 吞吐较最初 vLLM 基线低约 6.4%，64K/64 和 260K/C1 略有上浮，说明单次新 seed 实测有波动；后续同时保留旧基线与这次新复测，不挑选较低的对照作为最终结论。原始逐请求结果在 `optimization_round2_20261027_0555/round_page/bench/vllm/`。

### 2026-10-27 停止测试、释放机器

按用户要求停止优化与后续压测，关闭 H200-07 的 `dsv41-lightllm-compare` 和 H200-08 的 `dsv41-vllm-compare`；测试控制进程同时停止。两台均确认本轮服务进程退出、GPU compute-apps 为空。H200-07 清理了 173 个已核实归属且无进程使用的遗留共享内存资源；保留模型、编译缓存、源码和全部测试记录。H200-08 既有的其他容器未改动。

E2 整机对比尚未完成，E3 尚未部署，不据此宣称已超越 vLLM。关闭前后证据保存在 `optimization_round2_20261027_0555/shutdown_handoff_*/{before,after}.json`（对应节点）。


## H100 两机 DP16/EP16 优化准备（2026-10-27）

目标节点为经 H100-msw 跳转的 `10.119.23.201:8090`、`10.119.23.225:8090`，各 8×H100 80GB。预检查发现 .201 空闲，.225 的 8 张卡被另一套 284B vision P/D 服务占用（prefill 8019，master 60011，每卡约 71GB），已询问是否可以释放；确认前未停止这套服务，尚未启动两机模型或进行端到端测速。

沿用已有 `tools/launch_deepseek_v41_dpep16.sh`，更新为逐行参数的 DP16/EP16、两机、256K 上限、batch 8192、chunk 4096、DSpark 5、mem_fraction 0.75。KV 继续自动 profile；没有手动指定 max_total_token_num 或 graph_max_len_in_batch。两机现有环境均为 Torch 2.11.0+cu130、DeepEP 2.1.0+local，已核对实际导入路径和 ElasticBuffer 签名，无包安装或版本变更。

本轮记录目录：`/mnt/vqa/wanzihao/dsv41_h100_dpep16_20261027_1527/`。冻结当前工作树和 kernel 配置共 1809 个文件，baseline/optimized 仅 `lightllm/distributed/communication_op.py` 不同，用于单独验证按 overlap 配置分配 SM 的效果；页表向量化两边均保留。基线与优化版使用同一启动脚本和参数，未把此前 H200 的成绩充作 H100 基线。

H100 .201 上新增 indexer/页表/CED 路径 GPU 回归 **23 项通过**（44.63s），.225 也已核对两版全部 1809 个文件哈希一致。首次测试因未引用此前已有的 pytest runner 而未启动，日志单独保留；补上既有 runner 的 PYTHONPATH 后重跑通过，未安装包。此结果仅为局部正确性验证，尚无新的 DP16/EP16 吞吐或 GSM8K 分数。


用户已授权停止 .225 旧服务。已停止其 284B vision master 60011 与 prefill 8019 的完整 50 进程树；确认 GPU 全部释放、旧进程全部退出，移除其未挂接的 1TiB CPU-cache 和 4GiB SysV 段及已核实归属的 POSIX 共享资源。CUDA 退出清理较慢，初次 240s 等待超时后继续检查，最终完成释放；未在旧资源存活时启动新模型。证据位于本轮 `old_service_shutdown/`。

随后并发启动两机 V4.1 DP16/EP16 基线，主模型和 DSpark 均使用 `/mnt/vqa/models/DeepSeek-V4.1-Flash-1017-FP8`，服务端口均为 18941，跨机 NCCL rendezvous 为 .201:28941。完整命令以冻结的 `baseline/src/tools/launch_deepseek_v41_dpep16.sh` 为准，逐行列出每个选项。两台的 HTTP/router 各自接收本机 8 个 DP rank 的请求，后续性能测试会同时向两台发压，避免只有 8 个 attention ranks 有请求。


### H100 首次启动失败与配置修正

首轮 `mem_fraction=0.75` 在主模型 CUDA graph warmup 的 legacy MoE `masked_group_gemm` 输出分配处 OOM：申请 1.88GiB、空闲 1.80GiB，graph 私有池当时已有 5.04GiB。两机均出现同样错误；未开始测速，不计作性能结果。失败源码、日志和清理证据完整保存在 `attempt1_mem075/`。

完整停止失败进程、确认两机 GPU 空闲后，将两版启动脚本统一为 `mem_fraction=0.70`。同时修正请求容量的节点语义：`running_max_req_size` 按本机 8 个 DP rank 分配，因此每节点设 512，对应每 rank 64、两机合计 1024；与 graph_max_batch_size=64 和 DSpark 6 行/请求的容量保持一致。最初的 1024 是每节点容量，并非全局容量。保留 256K 上限、8192 batch、4096 chunk 和动态 prompt cache；未通过减小上下文或关闭 graph 避开 OOM。

### H100 显存定位与临时缓冲复用

用户要求先查清显存占用，因此主动停止第二轮 mem_fraction=0.70 的权重加载，保留 `baseline/service/` 日志；该轮未发生 OOM，也未完成初始化，不记作性能结果。两机 GPU 清空后进行独立 16 卡 MoE 显存测试。

当前 legacy DeepEP 的 RDMA size hint 为 8,178,896,000 bytes（7.617GiB/卡，512 tokens/rank、16 ranks、384 experts、hidden=5120）。主模型 masked MoE 的 GEMM1 BF16 输出为 1.6875GiB、GEMM2 输出为 1.875GiB；两者生命周期不重叠，但分开分配可能在 CUDA graph 私有池中保留两块不同大小的空间。Engram 的两张约 94.42GiB 表位于 pinned CPU memory，经 UVA 读取，不属于 GPU 权重显存。

首先尝试直接写 DeepEP 的 `get_next_low_latency_combine_buffer(handle)` 并开启 native zero-copy combine，16 卡复现发现原生输出带 stride，当前 DeepGEMM masked GEMM 要求 `stride(0) == size(-2)*size(-1)`，因此触发 layout assertion。已撤回这版生产改动，失败日志保存在 `rejected_native_zero_copy/`，不能声称已通过 zero-copy 验证。

替代实现只修改已有 `masked_group_gemm`：分配一个足以容纳两次输出的连续 scratch，各自取连续 view；SwiGLU quantization 消费 GEMM1 后，GEMM2 再覆盖该空间。不增加后端、配置或搬运 kernel。数值、CUDA graph、显存与延迟对照正在验证，尚无完整模型结果。


16 卡第一组已通过并正常退出：384 experts/top6 主模型、128 experts/top3 draft，分别覆盖 512 容量随机/集中/空路由及 384 容量随机路由，共 128 个 rank-case。Eager、CUDA graph、old/new graph 交替重放结果均逐位一致。每个时间取 5×20 replay 中位数，再取 16 卡最大值。

| 模型与容量，随机路由 | 原 graph 增量 GiB | 复用后 GiB | 原 MoE ms | 复用后 ms |
|---|---:|---:|---:|---:|
| target 384 experts，cap512 | 4.96875 | 3.29297 | 0.84501 | 0.84458 |
| draft 128 experts，cap512 | 1.67383 | 1.10938 | 0.38676 | 0.38688 |
| target 384 experts，cap384 | 3.73047 | 2.47461 | 0.83150 | 0.83241 |
| draft 128 experts，cap384 | 1.25391 | 0.83008 | 0.38918 | 0.38823 |

固定 cap512，仅改 scratch：两类独立 graph 的保留显存合计减少 2.24023GiB/卡，随机路由速度基本持平。瞬时活跃峰值基本未变（target 约 3.291GiB、draft 约 1.097GiB）；收益来自避免 graph 池为不同大小的中间结果保留两块 storage，不能将保留显存与活跃峰值混为一谈。

通信初始化实测同样不能只看 Torch allocator：NCCL ready 时设备使用 1.489GiB；Elastic 初始化后 3.775GiB（API buffer_num_bytes=1,377,828,864）；legacy 初始化后 13.821GiB，Torch reserved 始终仅 2MiB。也就是说此复现里 native runtime、通信缓冲和其分配开销约增加 12.33GiB/卡，不能用 `torch.cuda.memory_allocated()` 代表整个设备。以上为独立复现实测，完整服务还有独立 vision 进程，需另测。

原始 16 卡逐项结果：`moe_memory_results.json`；rank0 graph allocator snapshots：`moe_{384,128}_{before,shared_scratch}.pickle`。生产改动仅为 common MoE 5 行新增、2 行替换；原生 zero-copy 的参数与 optional output 接口均已移除。


第二组使用实际 cap384 大小重新创建 native buffer，并增加 cap384 集中/空路由，以及 GEMM1 输出宽度大于 GEMM2 的通用形状（hidden5120、intermediate4096、16 experts）；16 卡共 144 个 rank-case 全部逐位一致，两机正常退出。`moe_capacity_results.json` 保存全部结果。相同 NCCL/Elastic 条件下，legacy 初始化后的设备使用量 13.82147 → 11.82147GiB，16 张卡均精确减少 2GiB；并非仅根据 size hint 估算。

启动脚本将 decode dispatch capacity 从 512 调整为 384，依据本配置每 rank 64 请求 × 6 target verification tokens（draft 为 320 行）。没有改变请求数、上下文上限或 MTP 步数。对本轮两个独立 MoE graph 池，原 cap512/原分配合计 6.64258GiB，cap384/复用后合计 3.30469GiB，减少 3.33789GiB；加通信的 2GiB，独立复现合计约省 5.34GiB/卡。完整模型不能简单相加套用此值，实际节省以全模型阶段记录为准。

完整模型的验证源码冻结在 `memory_optimized/src/`，仅叠加 scratch 复用、cap384 和运行目录内的临时分阶段显存探针；未叠加待测的 E3 SM 分配变更。探针不进入生产工作树；记录 target/draft 构造、权重分配、req 状态、KV、native 通信、权重加载、attention workspace、graph 等阶段。此阶段保持 mem_fraction=0.70、256K、64 requests/rank、vision 和 DSpark。


重启前补充清理：进程 `setproctitle` 后不再保留可识别的 PYTHONPATH，原停机辅助脚本漏掉了部分 CPU cache/metrics/detokenizer 孤儿。按两轮 service UUID、原始日志和进程创建时间核实后，分别停止每节点 5 个本轮残留进程；等 nattch=0 后移除其 4GiB SysV 段，其余 SHM_DEST 段随最后的进程退出自动回收。未删除旧的未知 UUID POSIX 文件或其他历史资源。清理证据为 `memory_preflight_*.json`、`own_orphans_cleanup_*.json`。

随后两节点在同一时间启动 `memory_optimized`：node0 PID309724、node1 PID465254，两边均验证 1814 个源码/配置文件 SHA256。源码、模型、参数、环境和所有日志继续保存在本轮目录；生产工作树没有加入临时诊断接口或公共内存管理器。


### 完整 V4.1 DP16/EP16 初始化显存分解（memory_optimized）

主模型和 DSpark 均完成 256K CUDA graph 捕获与最长请求检查，16 个 rank 都成功初始化。每节点 30 个存活服务进程，nofile 全部为 1048576，预检最大 FD 分别 435/411；两台 health 均为 200，cgroup oom/oom_kill 均为 0。`memory/summary.json` 保存 16 卡各阶段范围，`memory/{target,draft}.pickle` 保存 rank0 allocator snapshots。

| 项目 | 每卡 GiB | 说明 |
|---|---:|---|
| 主模型 GPU 权重 | 42.064 | 转换后的 FP8 checkpoint；DP16 的 attention/dense/词表为各 rank 独立保留，只有 experts 按 EP16 分片 |
| DSpark 新增权重 | 约 1.556 | draft 自身权重统计约 4.022GiB，但其中词表和 lm_head 与主模型共享，不能重复相加 |
| 请求状态 | 1.274 | SWA、compressor tails、请求索引等；SWA 仍为各层独立，CED 复用的是 global KV |
| KV history | 4.513 | 自动 profile，无固定 max_total_token_num；4 组实际 KV source，避免按 40 层重复分配 |
| attention workspace | 0.586 | 含 8192 tokens 的 FlashMLA Q staging |
| RoPE | 0.500 | 当前沿用配置的 1M 位置表 |
| 两个 CUDA graph 私有池 | 3.893 | target 2.943GiB、draft 0.949GiB；以实际 allocator segment pool ID 去重，不是各层相乘 |

主模型自定义初始化阶段，Torch reserved 外的设备占用从 4.42468 增至 15.25671GiB，约增加 10.832GiB，包含该阶段实际新增的 DeepEP/NCCL/native 开销。这里不能把整个 15GiB 都称作通信 buffer，原有的 vision 进程、CUDA context 与 host-UVA 页表等也计在其中。Engram 的两张约 94.42GiB 表本体放在 CPU pinned memory。

初始化末尾 `cuda.mem_get_info` 所见设备总使用 70.444–70.960GiB，剩余 8.360–8.876GiB；其中 Torch allocated 约 50.889GiB，reserved 54.859–55.375GiB。NVML 与 CUDA 的统计口径存在差异，后续真实负载用 NVML 单独采样记录。`mem_fraction=0.70` 是 KV profile 当时的预算；DeepEP/graph 等后来分配，因此不是整个进程的 70% 硬上限。

完整服务前后不能作为纯代码显存差值比较：首次 OOM 用 mem_fraction=0.75、每节点 request cap1024；本次为 0.70/cap512。代码收益以此前同配置的 16 卡 MoE 对照和 native buffer 复现为准，不把降低 KV 预算的差值算作优化收益。

### 完整模型正确性

两台文本均回答 `42`（约 0.70s），红色 PNG 均回答 `红色`（约 1.67s）。GSM8K 8-shot 全量 1319 题：**1203/1319=91.205%**，invalid=0、length 截断=1；DSpark accepted/proposed=91976/188765=**48.725%**，平均 3.436 committed tokens/step。评测请求总耗时 39.32s（固定 train 八例、prompt cache 开启，不能当作冷缓存速度基准）。详情保存在 `memory_optimized/gsm8k/`，原始运行日志 `memory_optimized_gsm8k.log`。

此前同题集成绩为 1205/1319，本次少 2 题；不是本轮同配置的 kernel 前后配对评测，不能据此断言 scratch 复用导致降分。scratch 的局部结果在所有已测试输入中逐位一致。完整模型当前使用的冻结源码未叠加 E3 SM 变更，避免将此前局部 SM 收益误作已部署结果。


### 满并发、260K 与运行后显存

使用两机各自 18941 端口均匀发压；每组用独立随机 seed、精确 token IDs prompt、greedy/ignore_eos。容量组为 1024 请求、输出 128 token，长上下文组为 2 请求、输出 64 token。两组均使用新随机 prompt，未预热并重用相同 prompt，不将 prompt cache 命中当成提速。

| 输入 token | 并发 | TTFT ms | TPOT ms | 输出 token/s |
|---:|---:|---:|---:|---:|
| 16384 | 1024 | 47272.64 | 369.03 | 1355.16 |
| 260000 | 2 | 16962.89 | 9.38 | 7.27 |

表内延迟为逐请求均值，吞吐为总输出/总耗时，包含 prefill。容量组耗时 96.72s、1024/1024 成功；长上下文组耗时 17.60s、2/2 成功，实际 token 数均正确。两组的输出长度、硬件和拓扑均与早期 H200 vLLM 表不同，不能直接比较吞吐，更不能称为已经超越 vLLM。本轮改善的是显存分配；满并发 TTFT/TPOT 仍高，吞吐瓶颈尚未解决。

NVML 每秒采样，两个节点各 315 个采样点：16 张卡的逐卡采样峰值为 **77.27–79.13GiB**，最高 81029MiB（设备总量 81559MiB）。初始化、真实推理后的统计必须区分：初始化 NVML 约 70.95–71.47GiB；GSM8K 和满并发之后保留的量更高。初始化 snapshot 已准确拆分 Torch/native，但当前运行探针没有逐步采集负载期间的 Torch allocated/reserved，所以不能把运行后所有增量直接等同于活跃 Tensor、KV 或内存泄漏，也不把初始化 70.44GiB 当作整个运行过程的峰值。

末次检查两节点 health=200，均为 30 个服务进程且 nofile=1048576，最大 FD 分别为 435/411；cgroup oom/oom_kill=0。日志扫描未发现新的 traceback、CUDA OOM、NCCL WARN 或缺失 WRITE task。采样器已停止，**两台 V4.1 18941 服务保留运行**，H200 两台未改动。

原始结果、请求、随机 seed、metrics 快照和采样在 `memory_optimized/bench/`、`gpu_memory_node{0,1}.jsonl`；归档为 `memory_validation_evidence.tar.gz`。本机回收副本在本轮记录目录的 `memory_validation_evidence/`。核心生产变更仍只有原 MoE 函数复用 scratch，以及原 launcher 的 dispatch capacity 调整；原生 zero-copy 尝试已完全撤回，没有新增生产后端或诊断 API。

### 提高 KV 容量：共享 DSpark graph 池与按服务长度分配 RoPE

用户要求优先提高自动 profile 的 max_total_token_num，并保留不 OOM 的余量。本轮目录为 `/mnt/vqa/wanzihao/dsv41_h100_dpep16_20261027_1527/kv_capacity/`。完整停止上轮两个节点各 30 个服务进程，确认 GPU 进程与 SysV 段均清空，再并发启动；没有修改环境包或镜像。

本轮只增加两项生产显存优化，沿用已有 CudaGraph 和模型初始化：

- 顺序执行的 DSpark 与 target 共用 graph mempool；有 decode microbatch overlap 时不共享。DP backend 在 proposer 前完成 target logits 的采样，DSpark 在 proposal graph replay 前以 eager 路径提交 target hidden/KV，因此无需跨 graph 保留未消费的 target 输出。没有为共享池添加逐步 clone 或新 backend。
- V4/V4.1 RoPE 表按 `max_seq_length` 分配。本次 router 已给 256K 服务长度加上 31 个 MTP/边界位置，表容量为 262175；YaRN 的 factor 仍根据 checkpoint 的 1M/65536 计算，不随表容量改变。

H100 GPU 回归 13 项通过：覆盖共享池不同 graph batch 的交替 replay 和保留输出，以及 V4/V4.1 在 256K、1M 服务长度与 MTP 边界的 RoPE/YaRN 一致性；日志为 `kv_capacity/gpu_tests.log`。生产工作树未加入运行时显存探针。

`trial076/src/` 冻结本轮源码，两个节点均核对 1810 个文件哈希后启动。继续保留 256K、每 rank 64 请求、batch8192/chunk4096、vision、DSpark5 和动态 prompt cache；launcher 将 mem_fraction 从 0.70 调为 0.76。没有设置 max_total_token_num 或 graph_max_len_in_batch，也未叠加待测的 E3 SM 分配变更。

本轮自动 profile 为 **5,530,624 tokens/rank**（1800 bytes/token），上轮为 **2,691,840**，增加 **105.46%**；16 个 DP rank 合计 **88,489,984**。此处是 KV 容量，不是已完成运行验证的结论；通信初始化、graph 捕获和真实负载结果将在下面记录。

`trial076` 已完成主模型和 DSpark graph 捕获，两节点 health=200，各 30 个长期服务进程，nofile 均为 1048576，cgroup oom/oom_kill=0。文本均回答 42（0.706/0.712s），红图均回答红色（1.461/1.455s）。初始化 CUDA 设备使用约 74.80GiB/卡。rank0 allocator snapshot 中，原两个 graph 私有池合计 3.89258GiB，本轮共享后 2.94531GiB；节省约 0.94727GiB。RoPE 从 0.5GiB 降至约 0.125015GiB，两项合计约 1.322GiB/卡。

但 **mem_fraction=0.76 未通过真实负载验证，不可作为可用配置**。GSM8K 首批 prefill 在 FlashMLA `sparse_decode_fwd` 申请约 0.95–1GiB 临时空间时，两节点多个 rank 发生 CUDA OOM；当时部分卡空闲仅约 0.3–0.9GiB，Torch allocated 已约 60.8GiB。未产出有效 GSM8K 分数或吞吐结果，后续满并发/长上下文测试未执行。失败日志和初始化 snapshot 保留在 `trial076/`；两个节点各 30 个进程均已停止，GPU 已清空。

现有 `_check_max_len_infer` 在 target 构造结束时执行，而 DSpark 权重/graph 随后加载；`init_mtp_draft_model` 设置 `DISABLE_CHECK_MAX_LEN_INFER=1`，draft 阶段跳过检查。检查输入为单请求、8192 个全 1 token，并非 256K、多请求或完整 MTP 流程。主模型检查时 Torch 峰值约 57.9GiB，不能覆盖此次真实负载。当前工作树尚未修改这项检查，不应声称已有最终联合检查。

OOM 后继续检查现有 attention backend，发现已安装 FlashMLA 不支持 `out=`，而原代码仍先分配完整 output，再复制 FlashMLA 自行分配的返回值。本轮最新 diff 改为：未显式指定输出且 native API 不支持 `out` 时，直接返回 native 输出；显式传入 `out` 的 graph-split 路径保持写回契约。8192×64×512 BF16 的重复 buffer 为 512MiB，此为按 shape 计算的预期收益，**该最新改动尚未完成 GPU 回归或全模型验证**。

launcher 当前候选值为 mem_fraction=0.74，尚未启动和验证。`trial074/` 仅有未完成的源码准备，不能算作一次启动或测试。用户要求查看 diff 后暂停部署/压测，服务保持关闭。当前 HEAD 为 `7bdc227d8`，未提交的已跟踪 diff 保存于 `kv_capacity/current_worktree.patch`；未叠加或部署 communication_op.py 中待测的 SM 调度改动。

提交前补充验证：在 H100 .201 的空闲 GPU 上覆盖加载最新 attention backend，原测试中的 GPU 数值、显式/隐式输出、strided output、decode graph replay 和 prefill graph-split 回归共 **38 项通过，1 项取消选择，7.97s**。取消选择的是未修改的独立导入子进程测试，因为本次以运行目录中的单模块覆盖方式测试，子进程不会继承该覆盖。没有启动模型服务或修改包；原始日志为 `kv_capacity/commit_validation/attention_gpu.log`。这补齐最新 attention 改动的局部 GPU 验证，不代表 mem_fraction=0.74 或完整模型已验证。

### 2026-10-27：同步 PR #1593 的缓存与推理状态结构

此前六笔显存相关提交已推送至 `support_dsv41_model`（截至 `21bf7747d`）。本轮先合入 PR #1593 的 `20e1f22e9`，合并提交为 `0ea31b320`；随后接入其新增的 `369f35461`（将页配置移入 `config_utils`）。保留上游提交历史，公共路径尽量使用 PR 的实现。

主要对齐点：

- ReqManager 只持有跨 forward 的 SWA/compressor ring 和请求映射；prefill SWA workspace 由 attention backend 管理，并在自动 KV profile 前分配。workspace 按 batch/request 上限计算，launcher 删除已失效的 `DSV4_SWA_FULL_TOKENS_RATIO`。
- 当前 forward 的索引、页表和 DeepGEMM scheduler 归 infer state；V4.1 的 C1/C2 使用同一 metadata 入口，保留共享 KV source 映射与分层 candidate selection。CED 复用本轮 SWA workspace，仅重建缩短 query 后的 attention scheduler。
- PD 状态搬运使用公共 memory manager 的 `DeepseekV4AttStatePageHelper`，移除旧 model-local PD 实现；量化格式适配使用 `layer_weights/quant_utils.py`。page size 和 CPU cache 页配置统一走 `config_utils`，覆盖 V4/V4.1。
- 同步 attention state 弱引用生命周期、公共多模态 chunk/cache 边界、PD transfer progress、prefill queue 和 parser 等上游改动。保留 V4.1 的 causal LLM vision、shifted mHC、Engram continuation、原始 block32 FP8 / FP4 checkpoint 与转换后的 block128 FP8 checkpoint 支持。

接入时修复 indexer K 初始化入口、局部导入遮蔽、旧 CUDA graph 标记和 V4.1 indexer RoPE 后的 BF16 舍入。当前 H100 已安装的 FlashInfer TopK 不接受 `page_size/out` 参数，因此仅在该 API 支持时使用对应优化，否则复用已有 vLLM TopK 路径。未安装或更换依赖。此前共享 graph pool、按服务长度分配 RoPE、MoE scratch 复用和 FlashMLA native output 优化均保留。

验证在两台 H100 的空闲 GPU 上执行，源码与日志保存在：
`/mnt/vqa/wanzihao/dsv41_h100_dpep16_20261027_1527/pr1593_merge/`。

| 测试范围 | 结果 | 日志 |
|---|---|---|
| attention 生命周期/FlashMLA、分页与 CPU cache、graph layout、完整 V4.1 单测 | 446 passed，2 skipped | `r7/core_gpu.log` |
| 公共 V4、vision、FlashInfer graph state/plan、多模态边界、PD/CPU state | 262 passed，124 skipped | `r5/upstream_gpu.log` |
| PD progress/queue/cache、parser、detokenizer、HTTP lifecycle 与启动参数 | 433 passed，另 8 subtests passed | `r5/shared_server.log` |
| 最终公共 kernel 与 V4 模型回归（属于上述公共范围的复测，不叠加计数） | 130 passed | `r7/v4_kernels.log` |
| 最新页配置、V4.1 默认值/非法对齐及相关启动参数 | 137 passed | `r8/page_config.log` |
| 双 GPU PD 状态与 KV 拷贝完成事件 | 2 passed | `r8/peer_gpu.log` |

核心套件包含 7 组真实小模型构造与前向：FP4/block32、FP8/block128、单/多 O group、vision、DSpark 及二者组合；检查 chunked prefill、decode、prompt-cache/CPU-cache continuation 等数值一致性。124 项跳过是远端缺少单测硬编码路径下的 DeepSeek reference processor/vision 文件；核心的 2 项跳过需同时可见两张 GPU，随后已在双 GPU 上单独补测通过。上述结果不是模型精度评测分数。

Black 检查通过；Flake8 使用相同仓库规则在远端 Python 3.10 上通过。本机 hook 的 Python 3.8 无法解析上游 `match` / `anext`，提交时只跳过该本地 Flake8 hook，未跳过等价的 Python 3.10 检查。

本轮为结构同步与回归，未启动完整模型服务、未重新运行 GSM8K/吞吐/显存压测。两台 18941 服务仍保持关闭；`mem_fraction=0.74` 仍是未通过完整模型负载验证的候选值，不将这些单测结果当作新的容量或性能结论。

最终核对远端测试目录与提交前工作树的 **1838 个 Python/JSON 文件 SHA256 全部一致**，清单为 `final_manifest.json`；上述通过的日志已归档为同目录 `verified_logs.tar.gz`，本机与远端各保留一份。
