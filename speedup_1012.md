# 2026-10-12：1P1D vision 性能优化

按用户指定顺序实施：③ PD master/HTTP 转发 → ② 不开 microbatch 的 MoE 通信重叠 → ① C128 compressor。每项独立部署、验证、记录，再提交。

## 环境与比较口径

- 起点：LightLLM `91fb7a95bf1b295c4219e99fe947f61c1e725039`；参考本轮归档的 vLLM 0.31.0 实际运行源码。
- P：`10.119.25.62`，API 8019；D：`10.119.25.57`，API 8121；master 在 P 节点 60011。经 H100-msw HAProxy 10066 对外提供 `http://10.210.6.10:10418`。
- 两台各 8×H100；TP8/DP8/EP；DSpark MTP 3；decode microbatch 关闭。主模型为自定义 sensenova vision 256K 检查点，draft 为 `/mnt/vqa/models/DeepSeek-V4-Flash-Vision-Exp-BF16`。
- 沿用已验证的 `1p1d-vision-256K` 启动脚本和 config，模型、镜像、量化、cache、batch 和 graph 参数保持一致。CPU cache 1024 GiB、disk cache 5120 GiB 位于 P 节点。
- 所有运行日志、结果和采集文件：`/mnt/vqa/wanzihao/bench-logs/speedup-1012-20261012/`；原始双方 profile 位于 `/mnt/vqa/wanzihao/bench-logs/pd-tpot-comparison-20261010/`。
- 服务对比是受控文本/贪心 decode 实验：回放归档的 16K token IDs，128/512 并发；每项测量前显式生成 8 tokens 预热同一前缀，正式每请求固定输出 4096 tokens。有意使用相同输入隔离 decode/转发变化，不作为独立随机 seed 的冷 cache 压测，也不代表原始视觉/采样工作负载。
- TPOT：逐请求 `(完成时间−首输出时间)/(usage.completion_tokens−1)` 的均值，单位 ms；吞吐：整轮实际输出 tokens/整轮耗时，包含 prefill 供给和尾部，不是窗口 Peak TPS。测速时 profiler 关闭。
- runtime overlay 保留原比较使用的 master 原始 token-ID 兼容和 D profiler 标记；两者不进入正式优化提交。每阶段的 overlay 单独保存，不覆盖旧归档。

## ③ PD master 与 completions 转发

改动：`/v1/completions` 合并同一 ready batch、同一 choice 的文本后序列化；首 chunk、批次结束和 finish 立即发送，最多暂存 32 tokens/64K 文本字符。usage 仍逐 token 计数。`n=1` 直接消费并关闭源生成器，省去多 choice 合并专用 task/Queue。异常前已经缓冲的输出先发送。chat 的 reasoning/tool 解析协议保持原有路径。

回归：`test_completion_stream_batching.py`、已有 chat stream 和 PD master 测试共 41 项通过。测试依赖从本机缓存复制到日志目录，只在 pytest 进程的 PYTHONPATH 使用，服务依赖未更换。

部署：只重启 master。第一次启动遇到旧连接关闭后的 60011 bind 占用；等待实际 bind 成功再启动。首次 smoke 在 D 的 KV 状态连接重连前发出并超时，排除；连接恢复后的外部文本/视觉请求分别返回 `2`、`Red`，HTTP 200。正常对比随后开始。

| 输入/并发 | 成功（前→后） | TPOT 前→后 (ms) | TPOT 降幅 | output TPS 前→后 | TPS 增幅 | 每请求文本 SSE 前→后（均值） |
|---|---|---:|---:|---:|---:|---:|
| 16K / 128 | 128/128 → 128/128 | 8.2967 → 8.1817 | 1.39% | 13,632.7 → 13,900.8 | 1.97% | 4096 → 2881.2 |
| 16K / 512 | 512/512 → 512/512 | 15.4485 → 13.0199 | 15.72% | 26,730.2 → 30,066.7 | 12.48% | 4096 → 482.1 |

TTFT 均值分别为 3028.2→2663.4 ms、11314.1→10532.4 ms，包含 P 供给和排队。前后均为各一轮；128 档变化很小，不能据此断言稳定提速。512 档文本 SSE 数量明显下降，高并发客户端 TPOT 与整轮吞吐改善。合并的是已经到达的输出，未增加等待凑批的定时器。

测试后 master 健康并注册 1P/1D；P/master 44 个、D 27 个进程的 soft nofile 均为 1048576，采样最大 FD 370/643，cgroup OOM 为 0。CPU 采样辅助脚本因转义错误未运行，本项不提供 CPU 降幅结论。原始摘要：`stage3-results.json`；逐请求 usage、SSE、TTFT/TPOT：`controlled/{baseline-16k-c{128,512}-stage0,after-16k-c{128,512}-stage3}/`。

本项提交：`da96b61a`。

## ② 普通 decode 的 DeepEP/shared expert 重叠

复用选中的 DeepGEMM 后端现有 low-latency dispatch、masked GEMM、combine 接口。普通 DS4 decode 顺序变为 `dispatch → dispatch recv hook → expert GEMM → combine send → shared expert → combine recv hook → routed+shared`。没有开启 microbatch，也没有新增 CUDA stream。对应 vLLM 的 `MK_INTERNAL_OVERLAPPED` 路径，其 shared expert 同样在当前 stream 执行；这项主要减少接收阶段的等待，不以不同 GPU kernel 并行执行为前提。

支持能力由现有后端声明，MegaMoE 与会覆盖输入的 Triton 后端沿用原顺序；prefill 沿用原路径。expert ID 采集仍由 FusedMoeWeight 负责；保留 quant selector、EPLB primary weights、clamp 和调用方 tensor allocator。

回归：新增 4 项验证 hook/读取顺序、后端选择、逻辑路由 ID 和 inplace 输入保护；与流式回归合计 10 项通过。实际 DeepEP API 为 `/opt/conda/lib/python3.10/site-packages/deep_ep/buffers/legacy.py`，dispatch/combine 均支持 `return_recv_hook`。停止完整 D 进程树后 GPU 进程列表为空，再按原 launcher 启动；8 rank CUDA Graph warmup 完成。外部文本/视觉 smoke 为 `2`、`Red`，HTTP 200；D 27 个进程 nofile 均为 1048576，启动后最大 FD 417，OOM 为 0。

以下“前”是已包含③的版本，“后”是③+②。正式测速均在本轮 GPU profile 之前，profiler 关闭，前后各一轮。

| 输入/并发 | 成功（前→后） | TPOT 前→后 (ms) | TPOT 降幅 | output TPS 前→后 | TPS 增幅 |
|---|---|---:|---:|---:|---:|
| 16K / 128 | 128/128 → 128/128 | 8.1817 → 7.6108 | 6.98% | 13,900.8 → 15,342.5 | 10.37% |
| 16K / 512 | 512/512 → 512/512 | 13.0199 → 12.4207 | 4.60% | 30,066.7 → 32,115.6 | 6.81% |

128 并发的额外、同样关闭 profiler 的 post-profile 回放为 9.0468→8.1931 ms。它不替代主表，表明不同运行窗口有波动，不能把单轮差异当作统计置信区间。

GPU profile 等待全部 8 个 backend 真正开始，截取内部稳定周期。前后实际请求均为 128（每 rank 16），平均序列长 17003.6→17202.8，相差 1.17%；以下时间为单 rank 区间并集的 8-rank 平均，不能把不同类别相加当作壁钟时间。

| GPU 指标 | 前 (ms/周期) | 后 (ms/周期) | 降幅 |
|---|---:|---:|---:|
| target verification | 30.0472 | 28.8086 | 4.12% |
| draft/DSpark | 2.1129 | 2.0724 | 1.92% |
| expert 通信及 collectives | 7.5498 | 6.6296 | 12.19% |
| GPU busy | 32.4570 | 31.1786 | 3.94% |

跨类别 GPU kernel 重叠为 0.0665→0.0236 ms，没有显示更多 GPU kernel 并行；当前数据支持通信等待减少和 target 时间改善。profile 本身会增加开销，其客户端延迟不进入速度主表。

原始数据：`stage2-normal-results.json`、`before2-gpu-summary.json`、`after2-gpu-summary.json`；`controlled/after-16k-c{128,512}-stage2/`；`controlled/profile-16k-c128-{before2,after2}/` 保存全部 8 rank trace 的 inventory/分析。启动与 smoke：`stage2-decode-launch.json`、`stage2-smoke.json`。

本项提交：`c261c747`。

## ① C128 compressor：64 维分块与两阶段执行

复用同一个 Triton compressor：C128 第一阶段以 64 维 tile、8 warps 计算 window softmax/weighted sum，写 FP32 scratch；第二阶段以完整 head、4 warps 完成原有 RMSNorm/RoPE/FP8 pack。C4/indexer 仍是原单阶段。scratch 由 DeepseekV4Workspace 按 microbatch slot 持有，infer_state 已有的 workspace 引用直接借用；没有新 backend、状态格式转换或每请求 scratch 分配。仍先完成压缩，再保存 partial state，保留环形 cache 的历史/当前输入与 MTP 语义。

独立脚本：`test/benchmark/kernel/benchmark_dsv4_compressor.py`。根据实际 serving TTIR 使用 FP32 KV/score/state、BF16 norm、int32 positions/slots，模拟每 rank 16/24/32/64 请求 × 4 个验证 token，覆盖全部/单个/无 C128 边界、padding、C4 latent/indexer、129/512/8192 行 prefill。对照 `c261c747` 中的 compressor；每场景 3 轮 CUDA Graph + CUDA event 计时，包含 slot 计算、所有压缩阶段及 partial-state 保存，没有 Torch profiler。

完整 21 场景通过。与 PyTorch FP32 softmax/weighted-sum oracle 的最大绝对误差 `2.98e-7`；partial-state 保存完全相同；C4/indexer 输出逐字节相同。C128 的 64 请求场景有 1 个、8192 行 prefill 有 4 个输出字节处于 FP8/BF16 舍入边界，scale 字节不变，解包值差异在 1 ULP 内；其余场景字节相同。这项保持 FP32 中间精度，但不能声称全部输出 bit-exact。

| 场景 | 行数 | 完整 compressor 前 (µs) | 后 (µs) | 倍率 |
|---|---:|---:|---:|---:|
| 16 请求，全部跨 C128 边界 | 64 | 295.77 | 18.99 | 15.57× |
| 16 请求，仅一个边界 | 64 | 228.16 | 16.90 | 13.50× |
| 64 请求，全部跨 C128 边界 | 256 | 495.43 | 31.44 | 15.76× |
| 64 请求，仅一个边界 | 256 | 229.11 | 19.05 | 12.03× |
| 64 请求，没有边界 | 256 | 11.61 | 14.96 | 0.78× |
| prefill | 512 | 312.06 | 23.04 | 13.55× |
| prefill | 8192 | 720.11 | 113.87 | 6.32× |
| C4 latent | 64 | 13.54 | 13.50 | 1.00× |
| C4 indexer | 64 | 13.04 | 13.02 | 1.00× |

无边界场景多一次 kernel launch，增加约 1.6–3.4 µs，不能将有边界的倍率外推为整模型速度。没有使用 GPU→CPU `.item()` 同步来判断运行时边界。

实际 cubin 资源：原 C128 两类 `REG=32, STACK=18008/18192 bytes`；新第一阶段 `REG=88/86, STACK=0`，第二阶段 `REG=29, STACK=0`。旧 kernel 的寄存器 spill 已在对应 int32 特化中消除。

原始数据：`kernel-c128-i32-clean/results.json`、`compiler-resources.json`、cubin/resource txt；模拟输入构造和源文件 SHA 随数据记录。初次以 int64 positions/slots 测试的旧特化只有约 10 KB 栈，保留为先行诊断，不替代上表。随后 int32 旧 kernel 与常驻 D 共用 GPU 时测试进程 OOM，该失败尝试排除；停止 D 并确认 GPU 清空后完成上表测试。

本项部署同时重启 P/D。旧 P 停止后的 GPU/pinned memory 释放有延迟，等待全部旧 PID/GPU 资源消失；旧服务清理器自动移除已识别的 SysV shmid 1（4 GiB，creator 3300）和 2（约 1 TiB，creator 6978），没有手动删除未知或附着 segment。身份、进程、日志 key/shmid 和清理后证据保存于 `stage1-p-*`。

正式服务测速均在本轮 GPU profile 之前完成；“前”为③+②，“后”为③+②+①。两档均每请求输出 4096 tokens，HTTP/usage 合约全部通过。

| 输入/并发 | 成功（前→后） | TPOT 前→后 (ms) | TPOT 降幅 | output TPS 前→后 | TPS 增幅 |
|---|---|---:|---:|---:|---:|
| 16K / 128 | 128/128 → 128/128 | 7.6108 → 7.0181 | 7.79% | 15,342.5 → 16,295.3 | 6.21% |
| 16K / 512 | 512/512 → 512/512 | 12.4207 → 11.2567 | 9.37% | 32,115.6 → 34,971.9 | 8.89% |

TTFT 均值为 2548.1→2536.6 ms、10592.8→10114.2 ms。P CPU cache 经重启重新建立，先完成各档相同前缀预热，再采集正式测量。原始数据：`stage1-normal-results.json`、`controlled/after-16k-c{128,512}-stage1/`。前后各一轮，不把变化解释为统计置信区间。

8 rank profile 的实际请求均为 128（每 rank 16），平均序列长 17202.8→17200.1，相差 0.016%。以下是内部稳定周期的 8-rank 均值，profile 客户端延迟不进入上述正式速度表。

| GPU 指标 | 前 (ms/周期) | 后 (ms/周期) | 降幅 |
|---|---:|---:|---:|
| target verification | 28.8086 | 25.9914 | 9.78% |
| draft/DSpark | 2.0724 | 2.0737 | -0.06% |
| GPU busy 区间并集 | 31.1786 | 28.3628 | 9.03% |
| CPU execute 起点间隔 | 35.3032 | 32.3505 | 8.36% |

kernel 分类中的非 GEMM/attention 部分为 6.8851→5.3146 ms，通信为 6.6296→5.3763 ms；通信数字包含跨 rank 等待，不把这次变化解释为又改了通信后端。profile 完成后额外 unprofiled 128 回放为 8.1931→7.4613 ms，同样只作另一运行窗口的参考。原始数据：`after2-gpu-summary.json`、`after1-gpu-summary.json`、`controlled/profile-16k-c128-after1/`，包含全部 rank 的 trace inventory、实际 batch/序列长度、分析与 profiler 开始/停止证据。

最终服务仍为健康的 1P1D，所有 profiler 已停止。通过外部 `10.210.6.10:10418` 的文本/真实 image_url 请求分别返回 `2`、`Red`，HTTP 200；P/master 44 个、D 27 个进程 nofile 均为 1048576，采样最大 FD 370/433，cgroup OOM 为 0。P/D 日志无 CUDA/NCCL/HTTP 500/WRITE-task 错误；master 的两条历史 traceback 是 D 重启时 KV 状态 WebSocket 断开，均早于本项正式测速。证据：`final-smoke.json`、`final-{p,d}-validation.json`。服务使用 `runtime-stage1` P/D overlay 和 `runtime-stage3` master overlay，保留原 token-ID/profiler 兼容补丁，正式仓库只提交性能改动和独立测试脚本。

## 三项累计服务结果

| 输入/并发 | 起点 TPOT (ms) | ③后 | ③+②后 | ③+②+①后 | 累计 TPOT 降幅 | 起点→最终 output TPS | 累计 TPS 增幅 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16K / 128 | 8.2967 | 8.1817 | 7.6108 | 7.0181 | 15.41% | 13,632.7 → 16,295.3 | 19.53% |
| 16K / 512 | 15.4485 | 13.0199 | 12.4207 | 11.2567 | 27.13% | 26,730.2 → 34,971.9 | 30.83% |

每阶段正式测速共 640/640 请求成功，profiler 关闭，decode microbatch 始终关闭。上述吞吐是整轮均值，与原始 LLM Bench 的不同输出长度、采样、视觉输入和 Peak TPS 口径不能直接等同；本次没有重新运行平台任务。

## 当前 LightLLM 与 vLLM 对比

LightLLM 使用三项优化后的 `1527fd44`；vLLM 使用此前 0.31.0 的 `hot-16k-c{128,512}-1011` 归档，本次未切换服务重测 vLLM。两边是同一套自定义检查点、同一组原始 token IDs（已核对解压后的 prompts 文件 SHA256 相同），16K 输入、每请求固定 4096 输出、temperature=0、ignore_eos=true，profiler 均关闭；使用同两台 8×H100 的 1P1D 服务。以下为预热前缀后的受控文本回放，前后各一轮。

| 输入/并发 | vLLM TPOT (ms) | 当前 LightLLM TPOT (ms) | LightLLM TPOT 相对差异 | vLLM output TPS | 当前 LightLLM output TPS | LightLLM TPS 相对差异 |
|---|---:|---:|---:|---:|---:|---:|
| 16K / 128 | 7.2066 | 7.0181 | -2.62% | 12,264.4 | 16,295.3 | +32.87% |
| 16K / 512 | 9.8407 | 11.2567 | +14.39% | 26,675.9 | 34,971.9 | +31.10% |

相对差异为 `(LightLLM / vLLM − 1) × 100%`；TPOT 越低越好，TPS 越高越好。两边均 128/128、512/512 成功。TTFT 均值（vLLM→当前 LightLLM）为 4215.5→2536.6 ms、17737.0→10114.2 ms。

128 并发 TPOT 已接近 vLLM，单轮略低不足以证明稳定领先；512 并发仍有约 14.4% 的 TPOT 差距。LightLLM 整轮 output TPS 较高，同时 TTFT 较低；吞吐包含 P 供给、准入/排队和尾部耗时，不能据此断言纯 decode GPU 更快。两边 cache 预热历史和粒度不同：usage 报告的平均 cached tokens 为 vLLM 两档均 16383，LightLLM 分别为 16128、15981；没有把整个运行窗口或缓存状态视为完全一致。优化后尚未复测 64K，因此本表只列本次有实际结果的 16K 两档。

可复核原始数据：`/mnt/vqa/wanzihao/bench-logs/pd-tpot-comparison-20261010/vllm/controlled/hot-16k-c{128,512}-1011/{case,summary,requests}.json` 与本轮 `controlled/after-16k-c{128,512}-stage1/`；派生对比保存于 `/mnt/vqa/wanzihao/bench-logs/speedup-1012-20261012/current-vs-vllm-results.json`。

## 512 并发补采：实际 batch 与剩余差距

应用户要求在现有优化后服务补采，没有重启、修改启动参数或开启 microbatch。沿用 vLLM 归档的 512 条 16K token IDs，每请求输出 4096；先用 8-token 回放预热，再等待至少 504 请求收到首个输出，确认全部 8 rank 的 backend profiler 开始，发出停止命令并保存全部 8 rank trace。采集请求 512/512 成功。

此前提到的平均约 410/274，是按时间加权的客户端“收到首个输出、尚未完成”请求数，不是 GPU batch，不能直接作为 TPOT 差距的归因。现在的 trace 证据如下：

| 实际执行指标 | vLLM 归档窗口 | 当前 LightLLM 稳定窗口 |
|---|---:|---:|
| 总请求 batch | 504 | 512 |
| 每 rank 请求 batch | 60–64 | 全部 64 |
| 每 rank 平均序列长（8-rank 均值） | 17541.6 | 17293.2 |
| 分析的内部周期数 / rank | 53 | 38 |

batch 只差 1.59%，LightLLM 序列还短 1.42%；这些是接近的执行窗口，并非精确同 batch/同输出序列的 A/B。它们不支持此前用 410/274 解释“大幅 batch 差异”。

LightLLM 的 all-thread trace 还覆盖了 profiler 初始化期间的推理，因此额外按 `baseTimeNanoseconds` 对齐 epoch 时间，仅保留全 rank backend 开始确认后 150 ms 至 stop ACK 前 150 ms，再去除首尾各 5 个周期。确认窗口为 `[1791784872.492364, 1791784874.925029]`；8 rank 均保留 38 个内部周期。完整跨度的原分析单独保存在 `trace-analysis-full-span.json` 与 `*-full-span.json`，不替代下表。vLLM 使用原已归档的内部稳定周期分析。

以下为 8-rank 均值，单位 ms/周期；GPU 项使用区间并集。不同分类有重叠，不能直接相加为壁钟时间；CPU execute 起点间隔也不是纯 CPU 计算时间。

| 指标 | vLLM | 当前 LightLLM | LightLLM − vLLM |
|---|---:|---:|---:|
| target verification GPU | 36.5560 | 40.4074 | +3.8515 |
| draft/DSpark GPU | 2.7851 | 3.2518 | +0.4667 |
| GPU busy 区间并集 | 40.2090 | 44.3043 | +4.0953 |
| CPU execute 起点间隔 | 43.4372 | 49.4596 | +6.0225 |
| 辅助 kernel：量化/索引/mHC/采样等 | 6.2204 | 9.1198 | +2.8994 |
| expert 通信与 collectives | 9.3058 | 10.1811 | +0.8753 |
| dense/indexer 等其他 GEMM | 8.5908 | 9.1380 | +0.5472 |
| DeepGEMM grouped masked expert GEMM | 13.0378 | 12.3863 | -0.6515 |
| attention | 3.6059 | 3.4066 | -0.1993 |

target GPU 高 10.54%，周期起点间隔高 13.86%；GPU 空闲占比为 7.43%→10.42%。剩余差距确实包含执行与同步/调度等待，不能都归给转发，也不能仅根据 GPU 空闲推断具体 CPU 函数耗时。最大的已测 GPU 分类差额在辅助 kernel，expert GEMM 和 attention 本身并未更慢。

进一步从原始 trace 提取完整 kernel 家族，仍使用相同内部窗口。下表是 kernel duration 的和 / 周期，不是壁钟区间并集；独立 kernel 分类不包括融合到其他 kernel 中的同类工作。

| 独立 kernel 家族 | vLLM 调用数/周期 | LightLLM 调用数/周期 | vLLM (ms/周期) | LightLLM (ms/周期) |
|---|---:|---:|---:|---:|
| PyTorch `direct_copy_kernel_cuda` | 9.00 | 105.00 | 0.0211 | 1.6312 |
| per-token FP8 quant | 253.12 | 253.03 | 0.8720 | 1.3809 |
| PyTorch reduction | 6.00 | 7.00 | 0.1421 | 0.4277 |
| mHC pre | 91.04 | 92.00 | 0.5673 | 0.5523 |
| mHC post | 94.04 | 94.00 | 0.5995 | 0.5939 |

独立 GPU copy 多出约 1.61 ms/周期，独立量化多约 0.51 ms；mHC 本身相近。下一步应追踪这些 copy 的生产者/消费者和目标 buffer，确认哪些可以直接写入或融合，并比较量化 kernel；不能仅凭 kernel 名称就直接删除拷贝。通信、draft 和同步等待仍有差额，本次尚未对这些剩余项实施新优化。

采集期间客户端 TPOT 16.1448 ms 含 profiler/export 开销，排除出正常测速。采集后的额外 unprofiled 回放也为 512/512 成功，TPOT 11.7391 ms、整轮 output TPS 33452.9，作为另一窗口参考；保留前述主表 11.2567 ms，不择优替换，也不把单轮差异当统计置信区间。

全部 profiler 已停止，D running=0；1P1D 健康，外部文本/真实 image_url 请求仍为 HTTP 200、`2`/`Red`。P/master 44 个、D 27 个长驻进程 nofile 均为 1048576，采样最大 FD 370/433；无新增 HTTP 500、CUDA/NCCL/WRITE-task 错误，cgroup OOM 为 0。

原始日志与结果均在 `/mnt/vqa/wanzihao/bench-logs/speedup-1012-20261012/`：`profile512-supplement1.log`、`controlled/profile-16k-c512-after1-1012-supplement1/`（原始 trace inventory、capture events、稳定与完整跨度分析）、`paired-512-supplement1.json`、`profile512-kernel-families.json`、`controlled/hot-16k-c512-after1-1012-supplement1/`、`profile512-final-{p,d}-validation.json` 与 `profile512-final-smoke.json`。

## 剩余热点排查：copy 的来源、量化粒度与等待

本次按用户要求排查，未修改服务代码、启动参数、镜像或依赖，未重启。独立脚本在空闲服务旁的 D 节点 GPU0 执行，诊断进程已结束；它们的局部倍率不是新的服务 TPOT/TPS 结果。证据目录为上述日志目录中的 `remaining-gap-investigation/`。

### attention 输出投影的两次布局搬运

确定的调用链是 `DeepseekV4TransformerLayerInfer._get_o → ROWBMMWeight/BMMWeightTpl.bmm → wo_b.mm`。H100 上当前走 BF16 grouped BMM 分支：输入 `[T, G, 4096]` 的 `transpose(0,1).contiguous()` 搬运到 group-major；BMM 输出 `[G,T,1024]` 再经 `transpose(0,1).reshape(T,-1)` 搬回 token-major。第二个 reshape 在 `G>1,T>1` 时不能只改 view。

512 档 rank0 的 38 个稳定周期中，grid `[16384,1,1]` 与 `[4096,1,1]` 的 BF16 copy 各为 43 次/周期，对应 target 的 `T=256,G=8` 输入/输出布局；draft 的 `T=192` 对应 `[12288,1,1]` 与 `[3072,1,1]`，各 3 次/周期。合计 92 次布局 copy，约 1.20 ms/周期。另有 4 次完整词表 logits 转 FP32：target 的 `_lm_head_and_gather` 明确写入 FP32 `ans_logics`，3 次 DSpark Markov 的 `local_base_logits.float()`；其 grid 与 `256×129280`、`64×129280` 元素匹配，约 0.40 ms/周期。剩余 9 次小 copy 约 0.03 ms，尚未逐一归因。上述详细 grid 归因是 rank0，不能冒充全部 rank 的逐项平均。

归档 vLLM BF16 `deep_gemm_fp8_o_proj` 直接使用 strided input 和 `out=z.transpose(0,1)`，最后 `z.flatten(1)` 为 view。当前 LightLLM 的 BMMWeight 已有 `out` 参数，因此不需要新 backend：可以让它直接写入 token-major 目标 buffer 的转置 view，并保留现有 weight 布局与 allocator。

独立复现使用 BF16、`K=4096,R=1024`，测试 G=1/8、T=16/64/128/192/256 共 10 个场景，每项 3 轮 CUDA Graph/event 计时。所有场景输出逐元素相同；G=1 的性能基本不变。G=8 的完整布局转换+BMM 如下：

| query 行数 T | 现有路径 (µs) | 输入/输出均用 stride (µs) |
|---|---:|---:|
| 64 | 36.97 | 26.32 |
| 128 | 45.44 | 28.41 |
| 192 | 54.75 | 30.29 |
| 256 | 63.72 | 33.99 |

单次 Torch GPU trace 确认原路径为 BMM+2 个 copy，strided 路径仅 BMM、0 个 copy；两个原 copy 的 grid 与服务 trace 一致。源代码、数值、计时和 trace：`benchmark_o_proj_layout.py`、`o-proj-layout-benchmark.json`、`o-proj-{original,both_strided}.trace.json`、`copy-grid-rank0.json`；vLLM 实际源码来自原 runtime tar，保存为 `vllm_o_proj.py`。

### FP8 activation quant 的 CTA 粒度

实际调用 `DeepGEMMFP8w8a8B128QuantizationMethod.apply → per_token_group_quant_fp8 → lightllm_per_token_group_quant_fp8`。普通 UE8M0 路径 `COPY_TOPK=False`，当前固定每 CTA 处理 1 个 128 元素组、1 warp；`256×4096` 会启动 8192 个小 CTA。当前同一个 Triton kernel 已支持 `GROUPS_PER_CTA` 和尾组 mask，16 组配置目前只用于大 Mega/COPY_TOPK 输入；普通路径没有用到分组能力。

独立实验复用该 kernel，以每 CTA 8 组、4 warps 代替 1 组、1 warp；FP8 字节、FP32 scale、UE8M0 rounding 和输出布局保持相同。

| 输入形状 | 当前 (µs) | 8 组/CTA (µs) |
|---|---:|---:|
| 64×4096 | 2.39 | 1.93 |
| 192×4096 | 4.85 | 2.20 |
| 256×1536 | 2.99 | 1.91 |
| 256×4096 | 6.07 | 2.40 |
| 256×8192 | 10.99 | 3.15 |

5 个计时场景的 FP8/scale 全部 bit-exact；另加 1×1536、17×3840、65×4096、127×8192、129×4096 的小 batch/尾组检查，字节、scale 与输出 sentinel 均通过。这里只验证普通 `COPY_TOPK=False` 分组路径，不扩展结论到 Mega 或其他量化 dtype。实施时仍需按实际形状选择粒度，并验证更小输入是否值得分组。

核实了运行环境的 SGL 接口后，最初“UE8 分支判断过时”的猜测没有成立：已安装的 `sgl_kernel.gemm.sgl_per_token_group_quant_8bit` 的 `scale_ue8m0=True` 写 packed exponent（4 字节/int32），与当前 H100 DeepGEMM 的 FP32 scale 格式不同。它还固定接受 eps=1e-10；零值/1e-7 输入的解包 scale 分别为 `2.27e-13/2.33e-10`，当前 Triton 都为 `2.38e-7`。普通值解包后相同，小值量化字节不同，因此不能直接移除 guard 切换 SGL，也不引入格式转换来掩盖差异。原错误格式尝试和 eps=1e-4 被拒绝的诊断单独保留，排除出同合约性能结论。

证据：`benchmark_quant_backends.py`、`quant-backend-benchmark.json`、`quant-edge-parity.json`、`sgl-packed-ue8-contract.json`；实际量化 import 为 `runtime-stage1/lightllm/common/basemodel/triton_kernel/quantization/fp8act_quant_kernel.py`，SGL callable 位于 `/opt/conda/lib/python3.10/site-packages/sgl_kernel/gemm.py`。没有安装或更换包。

### 不能把单 rank 空闲全归为 CPU 同步

继续把同一 D 节点 8 rank 的 kernel/memcpy/memset 时间转换到共同 epoch，裁剪到各 rank 内部窗口的交集，再计算“所有 GPU 都空闲”的区间。vLLM 全 rank 同时空闲约 1.566 ms/周期，LightLLM 约 1.625 ms/周期，仅差约 0.059 ms；占共同窗口的 3.60%/3.29%。这不支持用“CPU 把整个 D 卡住”解释主要差距。此前较高的单 rank 平均空闲还包含不同 GPU/stream 的工作错位、收发和负载等待，不能直接当成 CPU 时间。此项是短 profile 窗口证据，未定量证明具体路由负载不均。

证据：`analyze_global_gpu_idle.py`、`global-gpu-idle-analysis.json`。优先改动应是输出投影布局与普通量化 CTA 粒度；完整 logits FP32 搬运需结合采样/logprob/Markov 的精度合约另行设计，不能直接删 cast。上述排查尚未覆盖全部 512 TPOT 差额，也未实施服务优化。

## 输出投影布局：服务验证

按用户要求实施并部署第一项。`_get_o` 复用 `BMMWeightTpl.bmm(..., out=...)`：输入只转置为 view；输出通过现有 tensor allocator 分配为 `[T,G,wo_a.dim2]`，BMM 直接写入其转置 view；交给 wo_b 的 `[T,G*R]` 为连续 view。维度取自 weight owner；保留选中的 BF16/FP8 分支、权重布局和 allocator，不新建 backend。

新增 10 项验证 G=1/2/8、T=1/16/256 的结果、输入/输出 storage 共享关系、wo_b 连续布局与 FP8 线性分支；连同 MoE/转发共 20 项通过，black/flake8 通过。部署前重跑当前服务的 128/512 正式基线，随后停止旧 D 的完整 PGID 63684，确认进程组和 GPU context 清空、checkpoint 全部 shard 可读，再使用原 launcher/config 启动 PGID 82025。P/master 保持原 overlay 与进程；D 使用 `runtime-o-proj-validate1`。主/draft 模型、镜像、cache、batch、graph 和 Microbatch 参数不变，8 GPU SM clock 均 1980 MHz。启动和资源证据保存于 `o-proj-*-launch/stop/resource-validation` 对应文件。

前后均 profiler 关闭，预热相同前缀后固定输出 4096 tokens，每档各一轮，640/640 全部成功：

| 输入/并发 | TPOT 前→后 (ms) | TPOT 降幅 | output TPS 前→后 | TPS 增幅 |
|---|---:|---:|---:|---:|
| 16K / 128 | 7.4930 → 6.9691 | 6.99% | 14,755.8 → 15,982.5 | 8.31% |
| 16K / 512 | 11.6108 → 11.1036 | 4.37% | 31,778.9 → 34,560.2 | 8.75% |

TTFT 均值为 2471.7→2875.2 ms、10422.5→10455.5 ms。采集后的额外 unprofiled 512 回放为 11.1649 ms，512/512 成功。前后时段存在波动，不把单轮差异当统计置信区间，也不拿更早的最好一轮替换本次基线。

512 profile 与前述补采的稳定窗口比较，实际 batch 均为 512/每 rank 64，平均序列长 17293.2→17331.8，仅差 0.22%。8-rank 均值：target GPU 40.4074→38.5766 ms/周期；GPU busy 44.3043→42.4078；周期起点间隔 49.4596→46.7683。独立 copy 从 **105→13 次/周期、1.6312→0.4311 ms/周期**，全部 rank 都减少 92 次，确认目标布局 copy 在实际服务中消失。辅助 kernel 区间并集 9.1198→7.9316 ms；通信区间也变化，包含跨 rank 等待，不解释为修改了通信 backend。

profile 512/512 成功、8 trace 保存、16 个线程 profiler 开始/停止事件匹配；客户端 profile 延迟不进入主表。部署后的内部与外部文本/真实 image_url 请求均 HTTP 200、`2`/`Red`；1P1D 健康，D 27 个长驻进程 nofile 均 1048576，启动后采样最大 FD 417，无新 CUDA/NCCL/OOM/HTTP 500/WRITE-task 错误。

数据：`o-proj-normal-results.json`、`o-proj-gpu-summary.json`、`controlled/{baseline,after}-16k-c{128,512}-o-proj-validate1/`、`controlled/profile-16k-c512-o-proj-validate1/`、`o-proj-unit-tests.log`、`o-proj-final-validation.json`、`o-proj-final-smoke.json`，均在本轮日志目录。本项单独提交后，再验证量化粒度。

## 普通 UE8M0 量化 CTA 粒度：服务验证

在输出投影提交 `51fc373f` 上单独实施第二项。复用原量化 kernel：仅在 `COPY_TOPK=False`、UE8M0 scale、group_size=128、总组数 M≥2048 时选 8 groups/CTA 和 4 warps；小输入、非 UE8M0 与 Mega 分支保持原选择。没有切换 SGL backend、scale 格式、eps、dtype 或量化 rounding。

新增 GPU 回归覆盖 1/63/64/129/256/8192 行的 4096 维输入，以及 171×1536、173×3840 的非整 CTA 尾组；包括零值和 1e-7 输入。逐字节比对原 1-group kernel 的 FP8 与 FP32 scale，并验证输出 sentinel 未被覆盖。首次连同现有回归共 58 项通过；补充尾组后量化 8 项全部通过，合计覆盖 60 个不同用例；black/flake8 通过。

完整停止旧 D 的 PGID 82025、确认 GPU context 清空后，原 launcher/config 启动新 D PGID 98542，runtime 为 `runtime-quant-validate1`。P/master 保持运行。权重、MTP、cache、batch、graph、Microbatch、GPU clock 与上一项相同，未更换镜像或依赖。

主表使用上一项的正式 after 作为量化 before，保留量化部署后的第一轮正式 after；相同归档 token IDs，16K 输入、预热前缀、每请求 4096 输出，profiler 关闭，640/640 全部成功：

| 输入/并发 | TPOT 前→后 (ms) | TPOT 降幅 | output TPS 前→后 | TPS 增幅 |
|---|---:|---:|---:|---:|
| 16K / 128 | 6.9691 → 6.7898 | 2.57% | 15,982.5 → 16,791.3 | 5.06% |
| 16K / 512 | 11.1036 → 11.4263 | -2.91% | 34,560.2 → 31,965.8 | -7.51% |

TTFT 均值为 2875.2→2509.4 ms、10455.5→9946.2 ms。512 第一轮较慢，不能列为端到端加速。采集后的额外回放与再次独立回放如下，均 512/512 成功、profiler 关闭；不择优替换主表：

| 512 回放窗口 | TPOT (ms) | output TPS | TTFT (ms) |
|---|---:|---:|---:|
| 量化前，O-proj 后额外回放 | 11.1649 | 34,412.9 | 10864.6 |
| 量化后，采集后额外回放 | 11.2787 | 33,995.6 | 11497.6 |
| 量化后，再次回放 | 11.2167 | 34,399.9 | 11144.7 |

512 改后两次额外窗口与改前相近，但尚无稳定端到端收益证据；正式第一轮的退化也保留。MTP 指标的逐请求平均 tokens/step 为主表 3.9862→3.9821，额外窗口约 3.9874–3.9895，不能用明显不同的 MTP 接收解释这项变化；这不是按总验证次数加权的精确接收率。

512 profile 两边实际 batch 均为 512、每 rank 64；平均序列长 17331.8→17505.1，差 1.00%，8 rank 各取 36→35 个内部稳定周期。按全 backend 启动确认后 150 ms、stop ACK 前 150 ms 裁剪，并去掉首尾 5 周期。下表为 8-rank 均值，GPU 项为区间并集；独立量化项是 kernel duration 和，不能直接等同壁钟节省：

| 指标 (ms/周期) | 量化前 | 量化后 |
|---|---:|---:|
| 独立 FP8 quant kernel duration 和 | 1.3855 | 0.5993 |
| target verification GPU | 38.5766 | 37.9091 |
| draft/DSpark GPU | 3.1861 | 3.1432 |
| GPU busy 区间并集 | 42.4078 | 41.6817 |
| 辅助 kernel 区间并集 | 7.9316 | 7.1409 |
| CPU execute 起点间隔 | 46.7683 | 46.7401 |
| 全部 rank 同时空闲时间 / 等效周期 | 1.8344 | 2.3678 |

量化调用数仍约 253 次/周期，实际量化 kernel 时长减少 **56.75% / 0.7862 ms**，GPU busy 少约 0.7261 ms，但执行周期几乎不变；共同窗口的全 rank 同时空闲占比 3.92%→5.07%。这是该短窗口中 GPU 节省未转化为周期收益的直接证据，尚不能定位到某个 CPU 函数或调度策略，也不能证明新增空闲由量化改动引起。保留此项为数值一致、实际 kernel 更快的优化；**不计为 512 并发端到端加速**。

profile 512/512 成功，保存 8 rank trace，16 个线程 profiler start/stop 匹配。采集时出现 8 条 Kineto `External init callback must run in same thread as registerClient` 诊断，单独保存在资源检查记录，不把它们当正常测速；profile 客户端延迟仍排除。最终外部文本/真实 image_url 请求均 HTTP 200、`2`/`Red`，注册 1P/1D；P/master 44 个、D 27 个长驻进程 nofile 均 1048576，最终采样最大 FD 370/433，cgroup OOM 为 0。量化 D 启动后无运行错误；master 在停止旧 D 时的 WebSocketDisconnect 保留在历史日志，不混入新服务测速窗口。

新增证据均在本轮日志目录：`quant-normal-results.json`、`quant-gpu-summary.json`、`quant-comparison-global-idle.json`、`quant-validation.log`、`quant-repeat512.log`、`quant-unit-tests.log`、`quant-edge-unit-tests.log`、`quant-decode-launch.json`、`quant-final-{p,d}-validation.json`、`quant-final-smoke.json`，以及 `controlled/{after,profile,hot,repeat}-16k-c*-quant-validate1/` 对应目录。

## 本次验证后与 vLLM 的对比

LightLLM 为三项原优化 + O-projection + 量化 CTA 改动后的当前服务；vLLM 仍使用前述 0.31.0 的归档 hot 回放，没有切换机器重测。以下保留本次量化正式第一轮，指标定义、模型、16K token IDs、4096 输出、warm-prefix 与前述对比一致；不同引擎 cache 粒度和供给时序仍有差异。128/512 两档均全部成功。

| 输入/并发 | vLLM TPOT (ms) | 当前 LightLLM TPOT (ms) | LightLLM TPOT 相对差异 | vLLM output TPS | 当前 LightLLM output TPS | LightLLM TPS 相对差异 |
|---|---:|---:|---:|---:|---:|---:|
| 16K / 128 | 7.2066 | 6.7898 | -5.78% | 12,264.4 | 16,791.3 | +36.91% |
| 16K / 512 | 9.8407 | 11.4263 | +16.11% | 26,675.9 | 31,965.8 | +19.83% |

当前 512 的其他无 profiler 窗口为 11.2787/11.2167 ms，相对归档 vLLM 仍约高 14%；不能认为差距已消除。O-projection 的 92 次布局 copy 已验证消失，量化 kernel 也已加速；剩余周期差距需继续追踪等待、通信和其他执行路径。整轮 output TPS 包含 P 供给与尾部，仍不代表纯 decode GPU 比 vLLM 更快。
