# PD Decode CPU Cache：代码审查与服务测试

2026-11-23，基线 `0c0b0683a`，审查并测试当前工作区的未提交实现。

**结论：223 个不同场景定义，在 8 组真实服务配置下共 602/602 项有效结果通过；另有 784 项相关回归测试通过。** 测试期间生产代码的 SHA-256 保持不变。本轮修复测试启动器的残留进程清理问题，修正测试预期和 HTML 描述，没有修改推理生产逻辑。

602 指“配置 × 场景”的去重有效结果，不是 602 种独立功能。两种 Hybrid PD 模式分别执行 222 个场景；另增加 P CPU 恢复场景，因此共有 223 个场景定义。原始执行共 620 项，其中首次有 6 项测试预期错误；18 项定向重跑均通过。原始失败未删除，也未计作原始通过。

原始证据：[汇总](artifacts/pd_decode_cpu_cache_review_20261123/summary.json)、[审查记录](artifacts/pd_decode_cpu_cache_review_20261123/review_findings.json)、[回归日志](artifacts/pd_decode_cpu_cache_review_20261123/regression.log)、[生产代码指纹](artifacts/pd_decode_cpu_cache_review_20261123/runtime_source_sha256.json)、[清理审计](artifacts/pd_decode_cpu_cache_review_20261123/cleanup.json)。

## 审查发现与处理

| 项目 | 发现 | 处理与验证 |
| --- | --- | --- |
| P2：测试启动器清理遗漏 | 启动失败后父进程退出，`p.wait()` 成功，但忽略 SIGTERM 的 detoken 子进程仍存活。本次发现 2 个。 | 修复 `launch.py`：等待父进程后也清理其独占进程组。只清除了本轮确认归属的进程；再次制造端口冲突，验证失败回滚后无所属存活进程。生产侧信号处理未改动。 |
| 已知 P2：命中统计 | 跳过 H2D 仍可能计数，CPU/disk 与 GPU 区间重叠时也可能重复统计。 | 遵守之前的要求，保留对应代码 TODO，不在本轮合入统计修复。生成正确性、传输范围、GPU 命中长度、CPU 引用分别验证。 |
| 文档落后于代码 | HTML 仍要求额外的任务提交标志，并将空 CPU 页描述为能力回退。 | 改为一次性 load 完成回调和公共就绪检查；空布局沿用非空页校验报错，无自动关闭缓存。 |
| 测试预期：PF 单 token | `max_new_tokens=1` 的 Prefill-first 在 P 完成，不会选择 D。以它预热 D cache 会超时。 | 用 8-token 请求预热；再验证目标单 token 请求在 P 结束，D 新增 CPU 命中为 0，输出等于基线。 |
| 测试预期：输出预算超限 | 输入合法时，既有 API 会自动缩短输出预算，返回 200，而非 400。 | N=3063，申请 4096 个输出 token，验证修正为 9，且 token ID 与基线的 9-token 请求严格一致。 |
| 测试配置：Hybrid 动态缓存 | 首次 P CPU 测试关闭了 Hybrid 必需的动态缓存，启动被拒绝。 | 保留生产校验，修正启动参数，通过 20 个不同前缀请求制造 GPU 淘汰，再验证 P CPU 恢复。 |
| 测试环境：端口碰撞 | 额外配置的 HTTP 端口与另一测试组的 NCCL 端口重叠。 | 使用不重叠端口重启，保留失败日志。该启动失败同时暴露了上面的启动器清理问题。 |

启动器清理修复证据：[修复前残留](artifacts/pd_decode_cpu_cache_review_20261123/cleanup_before_orphan_fix.json)、[精确清理记录](artifacts/pd_decode_cpu_cache_review_20261123/orphan_cleanup_actions.json)、[受控端口冲突结果](artifacts/pd_decode_cpu_cache_review_20261123/cleanup_fault_result.json)、[故障启动日志](artifacts/pd_decode_cpu_cache_review_20261123/cleanup_fault_injection_ready_children.log)。

本轮未发现新的阻塞级 PD CPU cache 功能问题。重点审查了回调是否一次性执行、未命中是否续接、取消/终态是否仍建立接收任务、H2D 后容量复查、TP 完成同步、失败尾部有效长度、物理尾页回收、Hybrid 页裁剪与 CPU/GPU 布局一致性。单测覆盖这些控制路径；下面的服务结果验证实际推理路径。

## 服务环境和对照

- 机器：NVIDIA H200；本轮使用 GPU 0–6，既有 GPU 7 服务未修改。P/D 通过本机网卡 IP 和 NCCL 跨 GPU 传输。
- Hybrid：`/mtc/models/Qwen3.5-0.8B`；完整注意力：`/mtc/models/Qwen3-4B`；BF16，Triton attention。
- 模型 GPU 页长 4，D CPU 页长 256，PD 传输页长 512，上下文上限 3072，KV token 池每 rank 8192。
- D 使用 legacy 缓存放置。P CPU 配置使用 legacy；P 保留 512-token 联合 KV/运行态 CPU 页，D 的 256-token 页只含 full-att KV。
- 每组有独立 Normal 基线服务，CPU cache 关闭；基线请求禁用 GPU 前缀复用。非流式及新增流式对照严格比较贪心 token ID，logprob 差异单独记录。原有 8-chunk 流式用例检查长度，新增 12 个流式组合检查 token ID。
- 个别额外配置启动阶段共用 GPU 的空余显存；本轮是正确性验证，不用时延作性能结论。实际 GPU 分配、端口、进程命令与最终归一化参数在每组 `processes.json` / `runtime_configs.json` 中。

## 配置结果

| 配置 | 有效通过 | 记录到的最大 logprob 绝对差 | 结果 |
| --- | ---: | ---: | --- |
| Hybrid 普通 PD | 222/222 | 0.09826946 | [JSON](artifacts/pd_decode_cpu_cache_review_20261123/hybrid_normal/accepted_results.json) |
| Hybrid Prefill-first | 222/222 | 0.09826946 | [JSON](artifacts/pd_decode_cpu_cache_review_20261123/hybrid_pf/accepted_results.json) |
| Hybrid 普通 PD，P/D CPU cache | 27/27 | 0.07065082 | [JSON](artifacts/pd_decode_cpu_cache_review_20261123/hybrid_pcpu_normal/results.json) |
| Hybrid Prefill-first，P/D CPU cache | 27/27 | 0.07065082 | [JSON](artifacts/pd_decode_cpu_cache_review_20261123/hybrid_pcpu_pf/results.json) |
| 完整注意力普通 PD | 26/26 | 0.06323636 | [JSON](artifacts/pd_decode_cpu_cache_review_20261123/dense_normal/results.json) |
| 完整注意力 Prefill-first | 26/26 | 0.06369507 | [JSON](artifacts/pd_decode_cpu_cache_review_20261123/dense_pf/results.json) |
| Hybrid D DP=2 + CUDA Graph | 26/26 | 0.06722194 | [JSON](artifacts/pd_decode_cpu_cache_review_20261123/hybrid_dp2_graph/results.json) |
| Hybrid P/D/基线 TP=2，D CUDA Graph | 26/26 | 0.09471476 | [JSON](artifacts/pd_decode_cpu_cache_review_20261123/hybrid_tp2_graph/results.json) |

logprob 并非逐位相等；记录到的最大绝对差约为 0.09827。所有执行严格 token 对照的场景均通过，没有通过放宽 token 一致性条件来消除失败。

## 场景分布与判定

| 每种 Hybrid PD 模式的场景组 | 数量 | 核查内容 |
| --- | ---: | --- |
| 基础主要/边缘场景 | 26 | N=1、阈值及页边界、部分命中、不同前缀、输出 1/64、并发、取消恢复、输入/参数非法、流式、近上下文上限、引用审计 |
| 预热后的边界 | 43 | 2–3063 token，覆盖 GPU 页、128 阈值、256 CPU 页及多页边界 |
| 冷前缀页边界 | 44 | 1–11 个 CPU 页，边界偏移 -1/0/+1/+2；无误命中 |
| 部分前缀 | 44 | N=1025/1538/2049/2818，在 11 个页内/页边界位置替换 token；比较共同前缀与 CPU 命中 |
| 预热后的输出长度 | 40 | N=257/258/513/514/1025 × 输出 1/2/3/4/7/16/32/63 |
| 严格流式对照 | 12 | N=257/258/513/514 × 输出 1/2/16；与基线逐 token 比较 |
| 并发/排队 | 8 | 并发 1/2/4/8 × 相同/不同前缀，含超过单批 KV 预算后的排队与恢复 |
| 参数校验 | 4 | 负输出预算、负温度、top_p>1、非法 top_k；显式启用采样，避免被贪心参数归一化掩盖 |
| 输出预算自动修复 | 1 | 3063 + 4096 输出预算修复为 3063 + 9，严格对照 |
| 合计 | 222 | 普通 PD 与 Prefill-first 分别执行 |

P CPU 两组各额外执行 1 个 GPU 淘汰后的 CPU 恢复场景，总场景定义为 223 个。多 DP、TP、完整注意力两种模式各执行 26 个基础场景。

## 关键边界证据

| 模型/模式 | N=257 CPU 新增 | N=258 | N=513 | N=514 | N=513 首个 KV 接收任务 |
| --- | ---: | ---: | ---: | ---: | --- |
| Hybrid 普通 PD | 0 | 256 | 256 | 512 | [256, 513) |
| Hybrid Prefill-first | 0 | 256 | 256 | 512 | [256, 513) |
| 完整注意力普通 PD | 256 | 256 | 512 | 512 | [512, 512) |
| 完整注意力 Prefill-first | 256 | 256 | 512 | 512 | [512, 513) |

Hybrid 两种模式在 N−1 恰为 CPU 页边界时退掉末页，保留 P 的运行态恢复路径；完整注意力不退页。完整注意力普通 PD 的 [512,512) 是零长度握手，Prefill-first 的 [512,513) 是真实的尾部 KV。

P CPU 恢复实测：N=1025，先经过 D 完整预热，再发送 20 个不同前缀请求淘汰 P 的 GPU 前缀。普通/PF 两种模式均测到 P GPU 命中 0、CPU 复用 1024；D GPU 命中 0、CPU 复用 768。生成与基线一致。结束时 P/D CPU 引用全部为 0。P 仍保留合法 GPU radix 缓存，审计时空闲 KV 为 512；D 各 rank 为 8192。缓存驻留未当作资源泄漏。

冲突参数启动使用不存在的模型目录，确认 D + CPU cache + DP fetch + dp=2 在模型访问前拒绝：[命令及结果](artifacts/pd_decode_cpu_cache_review_20261123/dp_fetch_conflict.json)。

## 原始失败和重跑

| 模式 | 首次 | 定向重跑 | 最终有效结果 |
| --- | --- | --- | --- |
| Hybrid 普通 PD | 221 通过，1 个输出超限预期错误 | 9/9 通过 | 222/222 |
| Hybrid Prefill-first | 217 通过，4 个单 token 预热预期错误、1 个输出超限预期错误 | 9/9 通过 | 222/222 |

重跑包含 5 个单 token 输出组合、1 个输出预算修复场景、健康检查和引用审计。最终合并按场景 ID 去重：正确的首次结果保留，受影响场景使用修正预期后的证据，错误的“超限必报错”用例由实际裁剪契约用例替换。

[普通首次](artifacts/pd_decode_cpu_cache_review_20261123/hybrid_normal/results.json) / [重跑](artifacts/pd_decode_cpu_cache_review_20261123/hybrid_normal/corrected_results.json)；[PF 首次](artifacts/pd_decode_cpu_cache_review_20261123/hybrid_pf/results.json) / [重跑](artifacts/pd_decode_cpu_cache_review_20261123/hybrid_pf/corrected_results.json)。首版脚本亦保留为 [initial_runner.py](artifacts/pd_decode_cpu_cache_review_20261123/initial_runner.py)。

## 清理与未覆盖范围

所有本轮服务已停止。检查 67 个所属 HTTP/NCCL 端口均关闭，35 个服务命名空间无共享内存文件或所属存活进程，10 个 SysV key 均无残留段。GPU 0–6 恢复初始占用（4/78/4/4/4/4/4 MiB），GPU 7 保持原有 10217 MiB。受控端口冲突也验证了启动器修复后的清理。

未将以下内容宣称为本轮服务验收通过：跨物理主机网络、NIXL、MTP 真服务、disk 真落盘/跨服务目录共享、量化 KV 真服务、adaptive 策略切换、微批 overlap，以及单 rank/CUDA 故障注入。部分布局、draft、量化、多模态、取消、容量不足和失败回收有单测覆盖；它们不计入 602 项真实服务结果。H2D 后容量不足的受控门控与 HTTP busy 通道由回归测试验证，本轮没有制造真实 GPU OOM。

## 复跑

先启动服务，再从另一终端执行测试；更换为新的输出目录保存每次证据。启动器退出会清理其独占进程组。

```bash
python test/service/pd_decode_cpu_cache/launch.py hybrid_normal \
  --model-dir /mtc/models/Qwen3.5-0.8B \
  --output-dir /tmp/pd-cache-review-new \
  --prefill-gpus 0 --decode-gpus 1 --baseline-gpus 2

python test/service/pd_decode_cpu_cache/run_cases.py hybrid_normal \
  --extended --output-dir /tmp/pd-cache-review-new
```

将配置名改为 `hybrid_pf` 可跑另一组 222 项。`hybrid_pcpu_normal` / `hybrid_pcpu_pf` 各跑 27 项；`dense_normal` / `dense_pf` 需要 Qwen3-4B，各跑 26 项；`hybrid_dp2_graph` / `hybrid_tp2_graph` 各跑 26 项，GPU 分配见对应 manifest。并行启动时需确保各组 HTTP 与 NCCL 端口集合互不重叠。

定向重跑可用 `--case-filter`、`--result-label corrected_` 和 `--tag-prefix corrected-`，保留首次结果并隔离提示词缓存。每组目录含命令、端口、模型、归一化参数、用例 JSON、P/D/Master/基线日志。

## 222 个场景逐项索引

以下两列均指按上述规则合并后的有效结果；P CPU 新增场景在前文单列说明。`N` 为输入 token 数，`O` 为输出 token 数，`mutation` 为替换 token 的位置。

| 序号 | 场景 ID | 普通 PD | Prefill-first |
| ---: | --- | --- | --- |
| 1 | `01_health_and_node_registration` | PASS | PASS |
| 2 | `02_one_token_prompt` | PASS | PASS |
| 3 | `03_below_cpu_threshold_127` | PASS | PASS |
| 4 | `04_page_minus_one_255` | PASS | PASS |
| 5 | `05_exact_page_256` | PASS | PASS |
| 6 | `06_full_match_boundary_257` | PASS | PASS |
| 7 | `07_full_match_plus_one_258` | PASS | PASS |
| 8 | `08_two_pages_minus_one_511` | PASS | PASS |
| 9 | `09_exact_two_pages_512` | PASS | PASS |
| 10 | `10_two_page_full_match_513` | PASS | PASS |
| 11 | `11_two_page_plus_one_514` | PASS | PASS |
| 12 | `12_multi_pd_pages_1025` | PASS | PASS |
| 13 | `13_shared_prefix_divergent_tail` | PASS | PASS |
| 14 | `14_distinct_prefix_no_false_hit` | PASS | PASS |
| 15 | `15_single_generated_token` | PASS | PASS |
| 16 | `16_long_decode_64_tokens` | PASS | PASS |
| 17 | `17_concurrent_identical_prompts` | PASS | PASS |
| 18 | `18_concurrent_distinct_prompts` | PASS | PASS |
| 19 | `19_cancel_active_stream` | PASS | PASS |
| 20 | `20_recovery_after_cancellation` | PASS | PASS |
| 21 | `21_invalid_zero_output_budget` | PASS | PASS |
| 22 | `22_input_exceeds_context_limit` | PASS | PASS |
| 23 | `23_stream_finishes_without_duplicate_tokens` | PASS | PASS |
| 24 | `24_max_context_near_limit` | PASS | PASS |
| 25 | `E_warm_boundary_N2` | PASS | PASS |
| 26 | `E_warm_boundary_N3` | PASS | PASS |
| 27 | `E_warm_boundary_N4` | PASS | PASS |
| 28 | `E_warm_boundary_N5` | PASS | PASS |
| 29 | `E_warm_boundary_N7` | PASS | PASS |
| 30 | `E_warm_boundary_N8` | PASS | PASS |
| 31 | `E_warm_boundary_N15` | PASS | PASS |
| 32 | `E_warm_boundary_N16` | PASS | PASS |
| 33 | `E_warm_boundary_N31` | PASS | PASS |
| 34 | `E_warm_boundary_N32` | PASS | PASS |
| 35 | `E_warm_boundary_N63` | PASS | PASS |
| 36 | `E_warm_boundary_N64` | PASS | PASS |
| 37 | `E_warm_boundary_N65` | PASS | PASS |
| 38 | `E_warm_boundary_N126` | PASS | PASS |
| 39 | `E_warm_boundary_N128` | PASS | PASS |
| 40 | `E_warm_boundary_N129` | PASS | PASS |
| 41 | `E_warm_boundary_N130` | PASS | PASS |
| 42 | `E_warm_boundary_N254` | PASS | PASS |
| 43 | `E_warm_boundary_N259` | PASS | PASS |
| 44 | `E_warm_boundary_N383` | PASS | PASS |
| 45 | `E_warm_boundary_N384` | PASS | PASS |
| 46 | `E_warm_boundary_N385` | PASS | PASS |
| 47 | `E_warm_boundary_N386` | PASS | PASS |
| 48 | `E_warm_boundary_N510` | PASS | PASS |
| 49 | `E_warm_boundary_N515` | PASS | PASS |
| 50 | `E_warm_boundary_N767` | PASS | PASS |
| 51 | `E_warm_boundary_N768` | PASS | PASS |
| 52 | `E_warm_boundary_N769` | PASS | PASS |
| 53 | `E_warm_boundary_N770` | PASS | PASS |
| 54 | `E_warm_boundary_N1023` | PASS | PASS |
| 55 | `E_warm_boundary_N1024` | PASS | PASS |
| 56 | `E_warm_boundary_N1026` | PASS | PASS |
| 57 | `E_warm_boundary_N1279` | PASS | PASS |
| 58 | `E_warm_boundary_N1280` | PASS | PASS |
| 59 | `E_warm_boundary_N1281` | PASS | PASS |
| 60 | `E_warm_boundary_N1282` | PASS | PASS |
| 61 | `E_warm_boundary_N1535` | PASS | PASS |
| 62 | `E_warm_boundary_N1536` | PASS | PASS |
| 63 | `E_warm_boundary_N1537` | PASS | PASS |
| 64 | `E_warm_boundary_N1538` | PASS | PASS |
| 65 | `E_warm_boundary_N2049` | PASS | PASS |
| 66 | `E_warm_boundary_N2561` | PASS | PASS |
| 67 | `E_warm_boundary_N3063` | PASS | PASS |
| 68 | `E_cold_page1_offset-1_N255` | PASS | PASS |
| 69 | `E_cold_page1_offset+0_N256` | PASS | PASS |
| 70 | `E_cold_page1_offset+1_N257` | PASS | PASS |
| 71 | `E_cold_page1_offset+2_N258` | PASS | PASS |
| 72 | `E_cold_page2_offset-1_N511` | PASS | PASS |
| 73 | `E_cold_page2_offset+0_N512` | PASS | PASS |
| 74 | `E_cold_page2_offset+1_N513` | PASS | PASS |
| 75 | `E_cold_page2_offset+2_N514` | PASS | PASS |
| 76 | `E_cold_page3_offset-1_N767` | PASS | PASS |
| 77 | `E_cold_page3_offset+0_N768` | PASS | PASS |
| 78 | `E_cold_page3_offset+1_N769` | PASS | PASS |
| 79 | `E_cold_page3_offset+2_N770` | PASS | PASS |
| 80 | `E_cold_page4_offset-1_N1023` | PASS | PASS |
| 81 | `E_cold_page4_offset+0_N1024` | PASS | PASS |
| 82 | `E_cold_page4_offset+1_N1025` | PASS | PASS |
| 83 | `E_cold_page4_offset+2_N1026` | PASS | PASS |
| 84 | `E_cold_page5_offset-1_N1279` | PASS | PASS |
| 85 | `E_cold_page5_offset+0_N1280` | PASS | PASS |
| 86 | `E_cold_page5_offset+1_N1281` | PASS | PASS |
| 87 | `E_cold_page5_offset+2_N1282` | PASS | PASS |
| 88 | `E_cold_page6_offset-1_N1535` | PASS | PASS |
| 89 | `E_cold_page6_offset+0_N1536` | PASS | PASS |
| 90 | `E_cold_page6_offset+1_N1537` | PASS | PASS |
| 91 | `E_cold_page6_offset+2_N1538` | PASS | PASS |
| 92 | `E_cold_page7_offset-1_N1791` | PASS | PASS |
| 93 | `E_cold_page7_offset+0_N1792` | PASS | PASS |
| 94 | `E_cold_page7_offset+1_N1793` | PASS | PASS |
| 95 | `E_cold_page7_offset+2_N1794` | PASS | PASS |
| 96 | `E_cold_page8_offset-1_N2047` | PASS | PASS |
| 97 | `E_cold_page8_offset+0_N2048` | PASS | PASS |
| 98 | `E_cold_page8_offset+1_N2049` | PASS | PASS |
| 99 | `E_cold_page8_offset+2_N2050` | PASS | PASS |
| 100 | `E_cold_page9_offset-1_N2303` | PASS | PASS |
| 101 | `E_cold_page9_offset+0_N2304` | PASS | PASS |
| 102 | `E_cold_page9_offset+1_N2305` | PASS | PASS |
| 103 | `E_cold_page9_offset+2_N2306` | PASS | PASS |
| 104 | `E_cold_page10_offset-1_N2559` | PASS | PASS |
| 105 | `E_cold_page10_offset+0_N2560` | PASS | PASS |
| 106 | `E_cold_page10_offset+1_N2561` | PASS | PASS |
| 107 | `E_cold_page10_offset+2_N2562` | PASS | PASS |
| 108 | `E_cold_page11_offset-1_N2815` | PASS | PASS |
| 109 | `E_cold_page11_offset+0_N2816` | PASS | PASS |
| 110 | `E_cold_page11_offset+1_N2817` | PASS | PASS |
| 111 | `E_cold_page11_offset+2_N2818` | PASS | PASS |
| 112 | `E_partial_N1025_mutation128` | PASS | PASS |
| 113 | `E_partial_N1025_mutation255` | PASS | PASS |
| 114 | `E_partial_N1025_mutation256` | PASS | PASS |
| 115 | `E_partial_N1025_mutation257` | PASS | PASS |
| 116 | `E_partial_N1025_mutation383` | PASS | PASS |
| 117 | `E_partial_N1025_mutation511` | PASS | PASS |
| 118 | `E_partial_N1025_mutation512` | PASS | PASS |
| 119 | `E_partial_N1025_mutation513` | PASS | PASS |
| 120 | `E_partial_N1025_mutation767` | PASS | PASS |
| 121 | `E_partial_N1025_mutation768` | PASS | PASS |
| 122 | `E_partial_N1025_mutation769` | PASS | PASS |
| 123 | `E_partial_N1538_mutation128` | PASS | PASS |
| 124 | `E_partial_N1538_mutation255` | PASS | PASS |
| 125 | `E_partial_N1538_mutation256` | PASS | PASS |
| 126 | `E_partial_N1538_mutation257` | PASS | PASS |
| 127 | `E_partial_N1538_mutation383` | PASS | PASS |
| 128 | `E_partial_N1538_mutation511` | PASS | PASS |
| 129 | `E_partial_N1538_mutation512` | PASS | PASS |
| 130 | `E_partial_N1538_mutation513` | PASS | PASS |
| 131 | `E_partial_N1538_mutation767` | PASS | PASS |
| 132 | `E_partial_N1538_mutation768` | PASS | PASS |
| 133 | `E_partial_N1538_mutation769` | PASS | PASS |
| 134 | `E_partial_N2049_mutation128` | PASS | PASS |
| 135 | `E_partial_N2049_mutation255` | PASS | PASS |
| 136 | `E_partial_N2049_mutation256` | PASS | PASS |
| 137 | `E_partial_N2049_mutation257` | PASS | PASS |
| 138 | `E_partial_N2049_mutation383` | PASS | PASS |
| 139 | `E_partial_N2049_mutation511` | PASS | PASS |
| 140 | `E_partial_N2049_mutation512` | PASS | PASS |
| 141 | `E_partial_N2049_mutation513` | PASS | PASS |
| 142 | `E_partial_N2049_mutation767` | PASS | PASS |
| 143 | `E_partial_N2049_mutation768` | PASS | PASS |
| 144 | `E_partial_N2049_mutation769` | PASS | PASS |
| 145 | `E_partial_N2818_mutation128` | PASS | PASS |
| 146 | `E_partial_N2818_mutation255` | PASS | PASS |
| 147 | `E_partial_N2818_mutation256` | PASS | PASS |
| 148 | `E_partial_N2818_mutation257` | PASS | PASS |
| 149 | `E_partial_N2818_mutation383` | PASS | PASS |
| 150 | `E_partial_N2818_mutation511` | PASS | PASS |
| 151 | `E_partial_N2818_mutation512` | PASS | PASS |
| 152 | `E_partial_N2818_mutation513` | PASS | PASS |
| 153 | `E_partial_N2818_mutation767` | PASS | PASS |
| 154 | `E_partial_N2818_mutation768` | PASS | PASS |
| 155 | `E_partial_N2818_mutation769` | PASS | PASS |
| 156 | `E_warm_output_N257_O1` | PASS | PASS |
| 157 | `E_warm_output_N257_O2` | PASS | PASS |
| 158 | `E_warm_output_N257_O3` | PASS | PASS |
| 159 | `E_warm_output_N257_O4` | PASS | PASS |
| 160 | `E_warm_output_N257_O7` | PASS | PASS |
| 161 | `E_warm_output_N257_O16` | PASS | PASS |
| 162 | `E_warm_output_N257_O32` | PASS | PASS |
| 163 | `E_warm_output_N257_O63` | PASS | PASS |
| 164 | `E_warm_output_N258_O1` | PASS | PASS |
| 165 | `E_warm_output_N258_O2` | PASS | PASS |
| 166 | `E_warm_output_N258_O3` | PASS | PASS |
| 167 | `E_warm_output_N258_O4` | PASS | PASS |
| 168 | `E_warm_output_N258_O7` | PASS | PASS |
| 169 | `E_warm_output_N258_O16` | PASS | PASS |
| 170 | `E_warm_output_N258_O32` | PASS | PASS |
| 171 | `E_warm_output_N258_O63` | PASS | PASS |
| 172 | `E_warm_output_N513_O1` | PASS | PASS |
| 173 | `E_warm_output_N513_O2` | PASS | PASS |
| 174 | `E_warm_output_N513_O3` | PASS | PASS |
| 175 | `E_warm_output_N513_O4` | PASS | PASS |
| 176 | `E_warm_output_N513_O7` | PASS | PASS |
| 177 | `E_warm_output_N513_O16` | PASS | PASS |
| 178 | `E_warm_output_N513_O32` | PASS | PASS |
| 179 | `E_warm_output_N513_O63` | PASS | PASS |
| 180 | `E_warm_output_N514_O1` | PASS | PASS |
| 181 | `E_warm_output_N514_O2` | PASS | PASS |
| 182 | `E_warm_output_N514_O3` | PASS | PASS |
| 183 | `E_warm_output_N514_O4` | PASS | PASS |
| 184 | `E_warm_output_N514_O7` | PASS | PASS |
| 185 | `E_warm_output_N514_O16` | PASS | PASS |
| 186 | `E_warm_output_N514_O32` | PASS | PASS |
| 187 | `E_warm_output_N514_O63` | PASS | PASS |
| 188 | `E_warm_output_N1025_O1` | PASS | PASS |
| 189 | `E_warm_output_N1025_O2` | PASS | PASS |
| 190 | `E_warm_output_N1025_O3` | PASS | PASS |
| 191 | `E_warm_output_N1025_O4` | PASS | PASS |
| 192 | `E_warm_output_N1025_O7` | PASS | PASS |
| 193 | `E_warm_output_N1025_O16` | PASS | PASS |
| 194 | `E_warm_output_N1025_O32` | PASS | PASS |
| 195 | `E_warm_output_N1025_O63` | PASS | PASS |
| 196 | `E_stream_N257_O1` | PASS | PASS |
| 197 | `E_stream_N257_O2` | PASS | PASS |
| 198 | `E_stream_N257_O16` | PASS | PASS |
| 199 | `E_stream_N258_O1` | PASS | PASS |
| 200 | `E_stream_N258_O2` | PASS | PASS |
| 201 | `E_stream_N258_O16` | PASS | PASS |
| 202 | `E_stream_N513_O1` | PASS | PASS |
| 203 | `E_stream_N513_O2` | PASS | PASS |
| 204 | `E_stream_N513_O16` | PASS | PASS |
| 205 | `E_stream_N514_O1` | PASS | PASS |
| 206 | `E_stream_N514_O2` | PASS | PASS |
| 207 | `E_stream_N514_O16` | PASS | PASS |
| 208 | `E_queue_concurrency1_sameFalse` | PASS | PASS |
| 209 | `E_queue_concurrency1_sameTrue` | PASS | PASS |
| 210 | `E_queue_concurrency2_sameFalse` | PASS | PASS |
| 211 | `E_queue_concurrency2_sameTrue` | PASS | PASS |
| 212 | `E_queue_concurrency4_sameFalse` | PASS | PASS |
| 213 | `E_queue_concurrency4_sameTrue` | PASS | PASS |
| 214 | `E_queue_concurrency8_sameFalse` | PASS | PASS |
| 215 | `E_queue_concurrency8_sameTrue` | PASS | PASS |
| 216 | `E_invalid_negative_output` | PASS | PASS |
| 217 | `E_invalid_negative_temperature` | PASS | PASS |
| 218 | `E_invalid_top_p_above_one` | PASS | PASS |
| 219 | `E_invalid_negative_top_k` | PASS | PASS |
| 220 | `25_health_after_cache_pressure` | PASS | PASS |
| 221 | `26_cpu_page_references_released` | PASS | PASS |
| 222 | `E_output_budget_clipped_to_context` | PASS | PASS |
