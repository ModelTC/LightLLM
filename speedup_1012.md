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

## 2026-10-13：从现有 profile 定位剩余 TPOT 差额

继续分析已经保存的 `profile-16k-c512-o-proj-validate1` 与 vLLM `profile-16k-c512-1011`，没有新增 profiler 或为排查修改服务参数。本节使用减少拷贝后的量化原始路径，对应下述回退后的代码。实际 batch 为 LightLLM 512、vLLM 504，平均序列长 17331.8/17541.6；因此是接近负载的 profile 对照，不是完全同形状的严格 A/B。正常无 profiler 主表 TPOT 为 11.1036/9.8407 ms，差约 12.83%。

8-rank 均值的 GPU busy 区间并集为 LightLLM 42.4078、vLLM 40.2090 ms/MTP 周期，差 2.1988 ms；CPU execute 起点间隔为 46.7683/43.4372 ms，差 3.3311 ms。两项来自带 profiler 的短窗口，不能直接作为正常 TPOT 的精确分解，也不能把起点间隔当纯 CPU 耗时。

辅助 kernel 区间并集仍多 1.7112 ms/周期，dense/indexer GEMM 多 0.5296 ms，通信/collectives 多 0.2168 ms；expert GEMM 少 0.6449 ms、attention 少 0.2094 ms。剩余最大差额在辅助计算与部分 dense GEMM，不支持把 attention 或 expert GEMM 本身视为主要慢点。下表进一步定位具体 kernel 家族；数字是独立 kernel duration 和 / 周期，与前述区间并集不能相加：

| kernel 家族 (ms/MTP 周期，8-rank 均值) | vLLM | LightLLM | LightLLM − vLLM |
|---|---:|---:|---:|
| per-token FP8 quant | 0.8720 | 1.3855 | +0.5135 |
| 剩余 PyTorch direct copy | 0.0211 | 0.4311 | +0.4100 |
| PyTorch reduction | 0.1421 | 0.4275 | +0.2854 |
| wo_b FP8 GEMM：N=4096,K=8192 | 0.9007 | 1.2501 | +0.3494 |
| shared w13 FP8 GEMM：N=4096,K=4096 | 0.5349 | 0.7155 | +0.1805 |
| shared w2 FP8 GEMM：N=4096,K=2048 | 0.3711 | 0.4634 | +0.0923 |

三个 dense 家族合计差约 0.6222 ms，target+draft 每家族均约 46 次/周期，调用次数基本相同。其他 GEMM 有反向差额，因此不能把 0.6222 直接替代整个 dense 分类的 0.5296。实际 kernel 模板全部显示 LightLLM 118 SM、vLLM 132 SM，且 BLOCK_N 等参数不同。当前运行的 `communication_op.py:_set_num_sms_for_deep_gemm` 全局执行 `set_num_sms(device_sms - deepep_sms)`；这确立了资源/分块配置差异，但还没有单独修改 SM 数做 A/B，不能断言这三个 GEMM 的全部差额由预留 SM 引起，也不能据此调整 expert GEMM。

copy 的具体来源延续前述 grid 归因：target 的 `llama/layer_infer/post_layer_infer.py:_lm_head_and_gather` 第 121–122 行分配 FP32 全词表并复制转置后的 lm_head 输出；draft 复用 `qwen3_dspark/layer_infer/post_layer_infer.py:_sample_markov` 第 114–116 行，3 个 Markov 步分别 `.float()`、相加并 `torch.max`。3 次 full-vocab max 是 reduction 热点；剩余 logits copy 约 4 次大搬运/周期。它们不是已经移除的 92 次 O-projection 布局 copy。两边配置均 BF16，vLLM 没有显式设置 FP32 head；改变这些计算需保留相加、rounding、tie-breaking 与 logprob 合约，不能直接删除 `.float()`。

另一个明确工作量差异是 `generic_post_process.py:78–84`：LightLLM 在判断 `is_all_greedy` 前仍除 temperature 并执行全词表 softmax，用于返回选中 token 的 logprob；当前 trace 为 1 次/周期、0.1785 ms。vLLM 的 greedy 且未请求 logprobs 路径允许直接取 argmax；其现有 trace 没有对应的 PyTorch 全词表 SoftMaxForward。这个差异适用于本次 greedy 受控回放，不能推广到所有随机采样请求。

进一步按 CUPTI correlation 聚合每次 CUDA Graph 的实际 GPU 起止时间，匹配完整 target/draft graph，相邻图之间的平均间隔为：

| 图间间隔 (ms，8-rank 均值) | vLLM | LightLLM |
|---|---:|---:|
| target 结束 → draft 开始 | 0.6682 | 0.7902 |
| draft 结束 → 下一轮 target 开始 | 3.0004 | 3.5328 |

额外图间间隔主要出现在 draft 后进入下一轮 target 的边界，约多 0.5324 ms。间隔包含采样、更新、其他非 graph kernel，以及其他 rank 的执行错位，不能全部归为纯 CPU 空闲或某一个调度函数。现有 profile 已直接定位慢的 kernel 和阶段边界；仍未完成对全部正常 TPOT 差额的精确因果分解。

新增证据位于 `remaining-gap-investigation/after-o-proj-kernel-families.json`、`dense-gemm-matched-shapes.json`、`graph-transition-gaps.json`；vLLM 对照源码直接读取原 snapshot 的 `vllm-runtime-source.tar.gz`，选中源码保存为同目录 `vllm-vllm__*.py`，不使用当前网上版本代替实际运行代码。

## 2026-10-13：按用户要求回退量化 CTA 改动

用户决定回退 `5ab53973` 的普通 UE8M0 8-groups/CTA 选择。仅删除该选择的 2 行与本项新增测试文件；量化源码逐字节恢复到 `51fc373f`，此前 Mega 的 16-groups/CTA 路径保持原状。保留 O-projection 布局优化与最初三项优化，保留本文件中的历史实验数据。

**因果解释补充：** 回退不等于证明此项无效或必然使服务变慢。同输入数值回归、量化 kernel 1.3855→0.5993 ms 与 GPU busy 42.4078→41.6817 ms 支持明确的局部收益；不同请求组成窗口中周期仅 46.7683→46.7401 ms，不能精确归因到 CPU overlap/等待，也不能凭一轮 +2.91% TPOT 认定是此 kernel 导致退化。保守回退的依据是当时未可靠证明 512 端到端收益，而不是否认 GPU 加速。

停止 D 的完整 PGID 98542，确认进程组与 GPU context 清空、旧 SHM 清理 helper 退出后，用原 launcher/config 启动新 D PGID 115392，runtime `runtime-quant-rollback-20261013`。主/草稿 checkpoint 的 115/285 个索引 shard 在 P、D 分别通过 O_RDONLY 检查；运行代码的量化 SHA256 为 `529c0070af31a918224025113161205ca0f5919576c6b9cb353d41a534d9c907`，O-projection 为 `974ef4dc67245cada95ac00fddb06156a6b4f0cbd893bbd20cadfe442eaa41ea`。P/master 未重启，模型、启动参数、镜像、依赖与持久 cache 不变。

首次正式回放有 5 个请求在收到 HTTP 状态前失败，日志保留为空的异常字符串，507/512 成功；该轮排除出性能对比，原始摘要保存在 `summary-original.json`。脚本仅补充异常类型/cause 记录，保持 payload、timeout、并发与采样不变后再次回放，512/512 全部成功，无 profiler。沿用同一组预热后的归档 16K token IDs，每请求 4096 输出；这仍是受控 decode 回放，不是新的冷缓存平台压测。

| 输入/并发 | 成功 | 当前 TPOT (ms) | 当前 TTFT (ms) | 当前 output TPS | 整轮耗时 (s) |
|---|---:|---:|---:|---:|---:|
| 16K / 512 | 512/512 | 11.3009 | 10801.1 | 30,912.7 | 67.8412 |

该结果没有证明回退使整体速度显著提升；恢复此前已确认有效的 O-projection 路径，是对未证明 512 收益的量化候选做保守回退。当前 TPOT 相对归档 vLLM 9.8407 ms 仍高 14.84%，整轮 output TPS 相对 26,675.9 高 15.88%；整轮吞吐较高不等于逐 token 输出更快。前述 profile 差额排查仍适用，不能把回退当作解决剩余 TPOT 差距。

量化既有 32 项 GPU 回归通过，black/flake8 通过；内、外部文本与真实 image_url 请求均 HTTP 200、`2`/`Red`，注册与健康均为 1P/1D。P/master 44 个进程及 2 个 detached SHM helper、D 27 个进程及 1 个 helper 的 nofile 均 1048576；最终资源采样最大 FD 370/417，cgroup OOM 为 0，8 GPU 时钟均 1980/2619 MHz。新 D 启动后无 HTTP 500、CUDA/NCCL/WRITE-task 错误；旧 D 停止时 master 的 WebSocketDisconnect 保留在启动前历史日志。

日志与证据均在既有 `/mnt/vqa/wanzihao/bench-logs/speedup-1012-20261012/`，前缀 `quant-rollback-20261013-`：`launch.json`、`stop.json`、`decode.log`、`validation.log`、`repeat1.log`、`unit-tests.log`、`{p,d}-checkpoint-preflight.json`、`{p,d}-resource-validation.json`、`final-smoke.json`；有效逐请求记录位于 `controlled/rollback-16k-c512-quant-1013-repeat1/`，首轮失败证据位于 `controlled/rollback-16k-c512-quant-1013/`。

## 第二轮优化 ①：draft 词表 bias＋max 融合

按用户确认的顺序，先在既有 `local_vocab_topk.py` 中增加两阶段词表归约，在通用 DSpark `_sample_markov` 接入；DeepSeek-V4 复用该路径。直接读取 `[B,V]` 的 strided BF16 logits view 与 bias，在寄存器中转 FP32、相加、取最大值，只写分块最大值与最终 token ID，省去完整 `.float()`/scores 张量。每个 Markov 步仍顺序执行；同分优先最小 local token ID，TP 后续 winner gather 保持原路径。未更改 head 精度、模型权重或 MTP 配置。

GPU 回归 10 项通过：BF16/FP32 bias、小 batch/尾词表/真实 V=129280、三个 step 的 strided 输入、同分 ID、CUDA Graph 新输入 replay，以及依赖上一步 token 的完整三步 Markov。最大值与 ID 均与原 FP32 add＋torch.max 精确相同。H100 Graph 下测试 7 种 tile，选 BLOCK_V=256、BLOCK_B=8、4 warps；B16 原 44.65→12.26 µs，B64 原 197.19→40.89 µs，均含两阶段归约和输出分配。

部署前重新采集当前 `313dc8eb` 服务基线；仅停止 D PGID 115392、确认 GPU 清空，再用原 launcher/config 启动 PGID 132765，runtime `runtime-draft-vocab-1013`。P/master 未重启，原 nofile、模型/cache/batch/graph/时钟/Microbatch 设置保持一致。主表采用第一轮正式 before/after，profiler 关闭，每请求输出 4096 tokens，640/640 全部成功：

| 输入/并发 | 成功（前→后） | TPOT 前→后 (ms) | TPOT 降幅 | output TPS 前→后 | TPS 增幅 |
|---|---:|---:|---:|---:|---:|
| 16K / 128 | 128→128 | 6.9130 → 6.9247 | -0.17% | 16,629.1 → 16,184.3 | -2.68% |
| 16K / 512 | 512→512 | 11.1632 → 11.1830 | -0.18% | 33,981.9 → 35,038.7 | +3.11% |

TTFT 为 2800.6→2589.5 ms、10192.2→10205.7 ms。采集后额外 unprofiled 512 回放 512/512 成功，TPOT 11.2531 ms、整轮 75.4568 s，不替换主表。上述 TPOT 接近，**尚未证明端到端 TPOT 收益**，整轮 TPS 也受供给/尾部波动影响；本项保留的是数值已验证、实际 GPU 工作量减少的融合路径，不把局部倍率宣称为服务加速。

512 profile 保存 8 rank trace；实际 batch=512、每 rank=64。与现有 O-projection 后、同量化代码的 profile 比较，序列长 17331.8→17367.0，差 0.20%。8-rank 均值：draft GPU 区间并集 3.1861→2.7237 ms/周期，GPU busy 42.4078→41.7245，CPU execute 起点间隔 46.7683→46.3431。copy 从 13→10 次、0.4311→0.3203 ms；PyTorch reduction 从 7→4 次、0.4275→0.0884 ms。新增 partial/finish 各 3 次/周期，kernel duration 和 0.1228/0.0061 ms；确认三次 draft cast/max 消失。kernel 和、GPU 并集与起点间隔不可直接相加，profile 客户端延迟不进入主表。

**因果解释补充：** 上述 draft 与周期同时缩短只是不同请求组成窗口之间的观察，与收益传递相符，不能认定周期减少的 0.4252 ms 全由本项带来。同输入微基准与 cast/max 调用消失证明局部工作减少；正常服务约 1% 的收益尚未被可靠证明。D 的默认 `bs_balancer` 在等负载 rank 间 `random.choice`，固定 prompt seed 不固定 rank 分配；固定映射本身也会改变调度，而且不能固定 KV 到达、MTP 接受长度或 MoE 路由。不能只靠固定 rank 映射做精确归因。服务性能应以多轮完整回放为统计单位，不能把同一轮的 512 个共享 GPU 请求当成 512 个独立实验。

主表 MTP 平均 tokens/step 为 128 档 3.9901→3.9904、512 档 3.9849→3.9868。逐请求前 256 字符的全模型输出不完全一致；相同原代码、同 D 进程的两次基线也仅 369/512 相同，融合前后为 351/512，融合后两次为 347/512。因此本次不声称全模型输出 bit-exact；数值精确一致的结论限于上述 kernel 与三步 Markov 回归。

内/外部文本和真实 image_url 请求均 HTTP 200、`2`/`Red`，1P/1D 健康；D 27 个进程及 SHM helper 的 nofile 为 1048576，采样最大 FD 433，OOM=0，GPU 时钟 1980/2619 MHz。无运行期 HTTP 500/CUDA/NCCL/WRITE-task 错误；Kineto 的既知 profiler callback 诊断单独记录。

证据位于既有日志目录：`draft-vocab-1013-{baseline,after,benchmark}.log`、`draft-vocab-1013-benchmark.json`、`draft-vocab-1013-unit-tests-final.log`、`draft-vocab-1013-profile-summary.json`、`draft-vocab-1013-{launch,stop,deployment-plan}.json`、`draft-vocab-1013-final-smoke.json`、`controlled/{baseline,after,profile,hot}-16k-c*-draft-vocab-1013/`。本项记录并单独提交后，再做主模型 greedy 采样融合。

## 第二轮优化 ②：主模型 greedy token＋logprob 融合

在既有 `local_vocab_topk.py` 增加两阶段 argmax＋logsumexp，greedy 路径不再生成完整 softmax 概率矩阵，但仍返回选中 token 的 logprob；惩罚、invalid token 与 temperature 的顺序保持原样。随机采样继续原路径。相等 logits 选择最小 token ID，logprob 用最大值平移后的 −log(sum(exp(logit−max)))，避免巨大共同 offset 引起抵消。

新增 11 项 GPU 回归，连同上一项共 21 项通过；覆盖真实词表 B64/B256、小词表/尾块、温度与惩罚顺序、同分/1e30 offset、CUDA Graph replay。ID 精确一致，logprob 最大误差 9.5367e-7。3 次 Graph 计时中位数：B64 原 94.45→9.58 µs，B256 原 240.02→46.62 µs；eager 为 121.95→24.06、258.90→60.73 µs。局部结果不能直接当服务加速。

仅重启 D，正式 PGID 151375，runtime `runtime-greedy-fuse-1013`，原 launcher/config/env、模型/cache/batch/graph/MTP/Microbatch 保持原设置。首次启动因 8121 暂时无法 bind 退出，保存独立失败日志；确认无进程/GPU context、端口可 bind 后同脚本重试。后续 restart helper 加入端口可绑定检查。P/master 未重启，两节点主/草稿索引的 115/285 个 shard 均可读。

本项第一轮 before/after 原始结果全部保留。相同归档 16K token IDs、warm-prefix、4096 输出，profiler 关闭，640/640 全部成功：

| 输入/并发 | 成功（前→后） | TPOT 前→后 (ms) | TPOT 降幅 | output TPS 前→后 | TPS 增幅 |
|---|---:|---:|---:|---:|---:|
| 16K / 128 | 128→128 | 7.4829 → 6.8475 | 8.49% | 15,136.6 → 16,847.1 | +11.30% |
| 16K / 512 | 512→512 | 12.0921 → 11.4567 | 5.25% | 33,041.5 → 34,304.6 | +3.82% |

**该表的 8.49%/5.25% TPOT 差异不能归为本项因果收益。** 未改代码、未换 D 进程的 before 已比上一项采集前 6.9247/11.1830 ms 慢约 8%；额外重复 128 基线仍为 7.3643 ms。已检查时钟/资源与 profiler start/stop，未找到明确原因，不能断言是 profiler 残留。相对上一项采集前结果，本项 128 好 1.12%，512 **慢 2.45%**；正式退化窗口不隐藏。采集后额外 512 为 11.2826 ms，与上一项额外 11.2531 ms 仅差 0.26%，尚不能证明稳定 TPOT 收益。

512 profile：batch=512、每 rank=64，平均序列长 17367.0→17440.2（+0.42%）；8-rank 平均周期 46.3431→45.7510 ms。graph 外 sampling/copy GPU 并集 0.6477→0.4542 ms，完整 SoftMaxForward 从 1 次→0 次，greedy partial/finish 各 1 次，合计 0.05279 ms/周期。target 38.3555→38.6256 ms、draft 2.7237→2.7044、GPU busy 41.7245→41.7827；总 GPU busy 没有减少，通信波动抵消了采样节省。保留数值与真实采样工作量已验证的融合，不将其列为已证实的 512 端到端加速。

profile 512/512 成功，8 trace 留存，profile 客户端延迟排除。外部文本/真实图片均 HTTP 200、`2`/`Red`；P/master 46、D 28 个进程（含 SHM helper）nofile 均 1048576，最终最大 FD 370/433，压测中 D 曾达 2178/1048576，未触及限制。cgroup OOM=0，时钟 1980/2619 MHz，无新运行期 HTTP 500、CUDA/NCCL、WRITE-task 错误。

证据均在既有日志目录，前缀 `greedy-fuse-1013-`：`normal-results.json`、`profile-summary.json`、`unit-tests.log`、`benchmark.{log,json}`、`{baseline,after,baseline-repeat128}.log`、`{launch,stop,deployment-plan}.json`、`{p,d}-resource-validation.json`、`final-smoke.json`、`failed-startup-port-busy.log`，以及 `controlled/{baseline,after,profile,hot}-16k-c*-greedy-fuse-1013/`。后续各项使用上一项采集前正式 after 作为 before，保留额外窗口，避免再把异常偏慢基线算为优化收益。本项单独提交后继续 dense GEMM。

## 第二轮优化 ③：纯 Decode DeepGEMM 使用完整 SM 池

先使用当前安装的 DeepGEMM 实际 API（`fp8_gemm_nt(...,compiled_dims='nk')`），按相同 UE8M0/F8 block128 格式测试 N=4096、K=8192/4096/2048、M=16/64/256。118/132 SM 与 nk/mnk 共 36 个组合的 BF16 GEMM 输出精确一致；每组 100 次 Graph 调用、7 轮计时中位数，原始数据保留。正式只改 SM 选择，compiled_dims 继续 nk。

| GEMM（N=4096） | M64：118→132 SM (µs) | M256：118→132 SM (µs) | M256 降幅 |
|---|---:|---:|---:|
| wo_b, K=8192 | 12.010 → 10.769 | 25.738 → 18.076 | 29.77% |
| shared w13, K=4096 | 6.873 → 6.418 | 13.987 → 9.994 | 28.55% |
| shared w2, K=2048 | 4.470 → 4.315 | 8.585 → 6.314 | 26.45% |

在资源 owner `DistributeGroupManager._set_num_sms_for_deep_gemm` 增加纯 Decode + low-latency buffer 的分支，DeepGEMM pool 取设备实际 SM 数。`ep_num_sms` 与 `Buffer.set_num_sms` 的 14 SM 通信预算保留；普通混合模式、Prefill 继续原预留逻辑，不在热调用切换全局 setter。4 项 selector 回归通过。此设置会影响 D 的所有 DeepGEMM，包括 expert GEMM 和相关 metadata，不把全部收益只归到三个 dense GEMM。

仅停止原 D PGID 151375，确认 GPU 清空后原 launcher/config 启动 PGID 168678，runtime `runtime-dense-sms-1013`。P/master、权重、镜像、cache、batch/MTP/graph/Microbatch、1980/2619 MHz 时钟不变。部署清单首次缺少文件列表时在停服务前失败，已补齐并将旧 snapshot 单独归档；正式启动前检查文件列表/SHA 合约。

第一轮 512 有 17 个 pre-HTTP `ReadError(BrokenResourceError())`，495/512 成功；原始 case 留存并标记 invalid，不把 12.1216 ms 当性能数据。P/master/D 无 GPU、OOM、FD 或 HTTP 500 错误；监听 backlog=100，发现累计 ListenDrops，但没有该失败窗口的增量证据，不能认定具体根因。相同代码、进程、payload 重复 512 后全部成功。下表 before 为第二项采集前正式 after，after 为本项第一个全成功窗口（128 首轮、512 repeat1）：

| 输入/并发 | 成功（前→后） | TPOT 前→后 (ms) | TPOT 降幅 | output TPS 前→后 | TPS 增幅 |
|---|---:|---:|---:|---:|---:|
| 16K / 128 | 128→128 | 6.8475 → 6.4999 | 5.08% | 16,847.1 → 17,231.7 | +2.28% |
| 16K / 512 | 512→512 | 11.4567 → 11.2330 | 1.95% | 34,304.6 → 34,990.2 | +2.00% |

本项采集后额外 512 为 11.4313 ms、33,636.0 TPS、TTFT 10703.2 ms，第二项额外为 11.2826 ms；额外窗口没有重复主表的收益。保留所有窗口，**尚不能将 512 的约 2% 改善视为稳定因果收益**，不择优替换主表。

512 profile 两项均 512/512 成功、8 trace、实际 batch=512/每 rank64；序列长 17440.2→17423.6（−0.10%）。8-rank GPU 并集：

| 指标 (ms/周期) | 第二项后 | 第三项后 |
|---|---:|---:|
| 周期起点间隔 | 45.7510 | 44.3624 |
| GPU busy | 41.7827 | 40.1761 |
| target verification | 38.6256 | 37.0830 |
| draft | 2.7044 | 2.6519 |
| 其他 dense/indexer GEMM | 9.1481 | 8.3469 |
| expert masked GEMM | 12.3402 | 11.8173 |
| 通信/collectives | 9.5982 | 9.3223 |

三个 dense 家族的独立 kernel duration 和（不能与并集直接相加）：

| 家族 | 118 SM (ms/周期) | 132 SM (ms/周期) |
|---|---:|---:|
| wo_b | 1.2551 | 0.9013 |
| shared w13 | 0.7204 | 0.5278 |
| shared w2 | 0.4682 | 0.3654 |

每家族约 46 次/周期，所有 rank 的模板均确认从 118→132 SM；三家族合计节省约 0.65 ms，实际其他 GEMM 并集少 0.80 ms、expert GEMM 少 0.52 ms、GPU busy 少 1.61 ms、周期少 1.39 ms。通信并集没有增长；短窗口不能推广成全部负载都不会抢占通信。当前稳态 GPU busy 40.1761 ms 已接近归档 vLLM 40.2090 ms，但正常整请求 TPOT 仍有差距，二者不是同一个测量窗口，不能据此宣称 TPOT 已追平。

最终外部文本/真实图片 HTTP 200、`2`/`Red`，1P1D 健康；P/master 46、D 28 个进程含 helper，nofile 均 1048576、最大 FD 370/433，cgroup OOM=0，无新 HTTP 500/CUDA/NCCL/WRITE-task 错误。证据位于既有日志目录，`dense-sms-1013-{benchmark,normal-results,profile-summary}.json`、`unit-tests.log`、`{after,repeat1}.log`、`{launch,stop,deployment-plan,valid-case-map}.json`、`{p,d}-resource-validation.json`、`final-smoke.json`、`controlled/{after,after-repeat1,profile,hot}-16k-c*-dense-sms-1013/`。本项单独提交后，继续 CPU 采样参数准备。

## 第二轮优化 ④：CPU 采样参数单次打包传输

在原 `_get_post_sample_tensors` 中读取同一批动态字段，将五个 32-bit 字段与一个 dense bool 字段放进 21×B bytes pinned buffer：req_idx、temperature、top_p、top_k、length penalty、EOS mask。复用原 `PinMemTensorManager` 四缓冲轮换，一次 H2D 后以 dtype view 取各字段，保留顺序、连续性和 lifetime；invalid token 与 EOS ID 路径保持原样。不缓存可变采样参数，不删除 verify/sync event，不改调度策略或 Microbatch。

2 项新增 GPU 回归验证混合 dtype/布局/单 storage、invalid prefix、重复行、运行时更新与旧结果存活，连同 greedy 回归共 13 项通过。独立相同字段传输测试，每次调用后 synchronize，300 轮 CPU 提交/完成中位数，数据逐值一致：

| B | CPU 提交：六次→打包 (µs) | 完成：六次→打包 (µs) |
|---|---:|---:|
| 16 | 57.86 → 29.55 | 64.55 → 36.31 |
| 64 | 63.18 → 34.84 | 69.81 → 41.30 |
| 256 | 85.13 → 56.93 | 91.42 → 63.17 |

B256 每次 CPU 提交减少约 28.20 µs，是约 44 ms 周期的 0.064%，不能以整轮百分比波动替代这个局部结论。仅重启 D，PGID 190582、runtime `runtime-sampling-pack-1013`，原 launcher/config/env、权重/cache、batch/MTP/graph/Microbatch、SM132 与时钟不变，P/master 未重启。

512 第一轮有 1 个 pre-HTTP ReadError，511/512；原始 case 标为 invalid，11.2332 ms 不纳入性能比较。同代码/进程/载荷 repeat1 全部成功。before 为第三项采集前正式有效 after，after 为本项第一个完整成功窗口，profiler 关闭、相同归档 16K/4096 tokens：

| 输入/并发 | 成功（前→后） | TPOT 前→后 (ms) | TPOT 降幅 | output TPS 前→后 | TPS 增幅 |
|---|---:|---:|---:|---:|---:|
| 16K / 128 | 128→128 | 6.4999 → 6.7123 | -3.27% | 17,231.7 → 16,699.2 | -3.09% |
| 16K / 512 | 512→512 | 11.2330 → 11.1434 | 0.80% | 34,990.2 → 33,344.3 | -4.70% |

128 的正式窗口变慢约 3.27%，512 低约 0.80%；采集后额外 512 为 11.3569 ms，第三项额外为 11.4313 ms。**不认定上述整轮差异是本项带来的收益或退化**，也不能证明 CPU overlap 的因果份额。本项确认局部 CPU 提交开销下降，尚未确认服务 TPOT 加速；保留减少 pinned/API 工作、数值已验证的实现，性能总结不将其计为端到端加速。历史量化回退仍遵从用户先前指令，其局部收益与不确定性已在对应节补充说明。

512 profile 512/512 成功、8 trace、实际 batch512/每 rank64；序列长 17423.6→17341.3（−0.47%）。8-rank 平均 GPU busy 40.1761→40.2436 ms，周期 44.3624→44.7220 ms，graph 外 sampling/copy 并集 0.4425→0.4411 ms。不能以此声称周期改善。捕获窗口中的 cudaMemcpyAsync CPU API 次数/周期 62.31→59.38，CPU duration 和 0.3538→0.3350 ms；aten::to duration 和 0.5145→0.4092 ms。这些包含其他复制、CPU profiler 开销与不同请求组成，不能直接等于本项六次→一次的五次差额或壁钟节省。

内/外部文本和真实图片 HTTP 200、`2`/`Red`，1P1D 健康；P/master46、D28 进程含 helper 的 nofile 均 1048576，最终最大 FD370/433，OOM=0、时钟1980/2619 MHz，无新 HTTP 500/CUDA/NCCL/WRITE-task 错误。证据在既有日志目录，`sampling-pack-1013-{benchmark,normal-results,profile-summary}.json`、`unit-tests.log`、`{after,repeat1}.log`、`{launch,stop,deployment-plan,valid-case-map}.json`、`{p,d}-resource-validation.json`、`final-smoke.json`，以及 `controlled/{after,after-repeat1,profile,hot}-16k-c*-sampling-pack-1013/`。本项记录并单独提交后，继续 shared expert activation＋量化融合。

## 第二轮优化 ⑤：shared expert 激活＋FP8 量化融合

沿既有 MMWeight → config 选择的 QuantizationMethod 调用链增加 `mm_swiglu/apply_swiglu`。普通 backend 保持原 SiLU＋mm；DeepGEMM FP8 block128 在已有通用 `moe_silu_and_mul.py` 融合激活与 UE8M0 量化，直接送入原 GEMM。保留 clamp、既有 GELU 选择、两处 BF16 rounding、eps=1e-4、FP8 dtype 和列主序 TMA-aligned FP32 scale，不硬编码模型的 backend。默认 fused kernel 为 4 groups/CTA、4 warps；它不恢复已经回退的普通量化 8-groups/CTA。

21 项新增 GPU 回归覆盖 M1/7/64/256、clamp、GELU、零/微小值、FP8 字节和 scale、CUDA Graph replay、真实 shared-w2 和带 bias 的 NoQuant backend。量化字节/scale 与 BF16 GEMM 输出精确一致；连同 11 项 greedy、2 项参数打包为 34 项通过，O-projection 10 项也通过，black/flake8 通过。完整 owner 路径 Graph 计时的三轮中位数：

| shared w2（N=4096,K=2048） | 原激活＋量化＋GEMM (µs) | 融合后 (µs) | 降幅 |
|---|---:|---:|---:|
| M16 | 7.801 | 6.090 | 21.93% |
| M64 | 7.787 | 6.195 | 20.44% |
| M256 | 12.327 | 8.568 | 30.49% |

仅重启 D，停止 PGID 190582、确认 GPU context 清空后，原 launcher/config 启动 PGID 208695，runtime `runtime-silu-quant-1013`；P/master、模型/cache/MTP/batch/graph/Microbatch/时钟不变。首次 preflight 的直接 quant 包导入命中既有循环依赖，同样导入在上一项 runtime 也失败；停旧服务前已改为验证实际模型 caller，没有改包初始化或依赖。实际 caller、MMWeight、量化 backend、kernel 的导入路径与方法签名均已留存。

512 预热首轮 504/512、repeat1 496/512；均为 pre-HTTP ReadError，失败 ID 分别 242–249、163–178。两轮成功集合覆盖全部 512 个 archived prompt，逐请求核实至少预热成功一次后开始正式回放；两份失败预热记录均保留，不计入性能对比，也不伪称单轮预热全成功。服务无 HTTP 500/CUDA/OOM/FD 错误，连接断开的具体根因未确定。没有修改 HTTP 传输、payload、并发或启动参数来隐藏失败。

正式 128/512、带 profiler 的 512 与采集后额外 512 全部成功，每请求输出 4096 tokens。before 为第四项正式有效 after；主表使用本项第一轮正式结果：

| 输入/并发 | 成功（前→后） | TPOT 前→后 (ms) | TPOT 降幅 | output TPS 前→后 | TPS 增幅 |
|---|---:|---:|---:|---:|---:|
| 16K / 128 | 128→128 | 6.7123 → 6.5685 | 2.14% | 16,699.2 → 17,392.4 | +4.15% |
| 16K / 512 | 512→512 | 11.1434 → 11.1926 | -0.44% | 33,344.3 → 35,146.2 | +5.40% |

128 TPOT 低约 2.14%，512 高约 0.44%；额外 512 为 11.3715 ms，第四项额外为 11.3569 ms。没有证明本项稳定改善整轮 TPOT，128 的单轮改善也不作为因果倍率。正式 TTFT 为 2548.3/9921.4 ms。

512 profile：8 rank、实际 batch512/每 rank64，平均序列长 17341.3→17279.8（−0.35%）。独立 SiLU 约 46 次/周期全部消失，普通量化从 253.04→207.02 次，新融合约 46.02 次，确认实际路径替换成功：

| 指标 (ms/周期，8-rank 均值) | 第四项后 | 第五项后 |
|---|---:|---:|
| 周期起点间隔 | 44.7220 | 44.9392 |
| GPU busy 并集 | 40.2436 | 39.8426 |
| target verification | 37.1415 | 36.7657 |
| draft | 2.6637 | 2.6397 |
| 普通 FP8 quant kernel duration 和 | 1.3864 | 1.2237 |
| 独立 SiLU kernel duration 和 | 0.1085 | 0.0000 |
| 融合 SiLU＋quant kernel duration 和 | 0.0000 | 0.1055 |

量化＋激活的 kernel duration 和共少约 0.166 ms/周期；GPU busy 少 0.401 ms，但通信并集也少约 0.287 ms，不能把全部 busy 变化归到融合。周期反而多 0.217 ms；这与之前回退量化的结论采用同一标准：局部 GPU 工作减少已有证据，整轮收益与抵消原因仍未精确归因。kernel duration 和、区间并集、周期不相加。

最终 1P1D 健康，外部 10.210.6.10:10418 文本/真实 image_url 均 HTTP 200、回答 `2`/`Red`。P/master46、D28 进程含 helper 的 nofile 全为1048576，最终最大 FD370/433，额外回放中 D 达2179，OOM=0，时钟1980/2619 MHz，未出现新的 HTTP 500/CUDA/NCCL/WRITE-task 错误。

所有原始日志在 `/mnt/vqa/wanzihao/bench-logs/speedup-1012-20261012/`，本项前缀 `silu-quant-1013-`：`normal-results.json`、`profile-summary.json`、`{prototype,full-w2}-benchmark.json`、`unit-tests.log`、`o-proj-tests.log`、`{after,repeat1,formal-continuation}.log`、`{launch,stop,deployment-plan,import-contract,warm-coverage}.json`、`{p,d}-resource-validation.json`、`d-resource-during-hot.json`、`final-smoke.json`；case 在 `controlled/{after,profile,hot}-16k-c*-silu-quant-1013/`，失败预热单独留存。

## PR 1623：DSv4 跨 DP checkpoint 共享（2026-10-13）

取 PR 的三个功能提交（原始 `71c8ee01`、`4511fdb2`、`216c7427`），未引入其上游基线中的 GLM5.3 或依赖变更。沿用已有 StateCacheManager/LayerCache，在同节点的对应 TP shard 之间共享 DSv4 大、小 continuation checkpoint 及引用计数；目标 DP 接收 checkpoint ID 并保留引用，不再重复分配或通过 Gloo 传输 checkpoint 字节。GPU compressed history 的复制、源 radix pin、恢复完成后的同步和临时小页释放顺序保持原合同。模型 warmup 的 free_all 不重置其他 DP 持有的 checkpoint。

五组回归合计 **40 passed**，包括双 GPU 的生产 `_transfer_dsv4_source_data`、实际 save/restore，以及源 DP 释放后的目标引用存活检查。2304 token 用例的 Gloo 数据为 **18432 字节 token-slot 索引 + 16 字节 checkpoint ID**；checkpoint 本体留在共享 pinned CPU 内存。该字节数是回归用例，不代表服务端 TTFT/TPOT 收益。日志：`/mnt/vqa/wanzihao/bench-logs/pr1623-dsv4-dp-fetch-20261013/candidate-tests-r4.log`。服务重启和同 seed 的 LLM Bench 结果待记录；下方既有 vLLM 对照均早于此改动。

首次端到端检查发现 pinned 共享池的注册边界故障：原 helper 每 128 MiB 独立 `cudaHostRegister`，DSv4 checkpoint 的 7749120 字节步长使索引 17 的槽位跨注册段，GPU→CPU `copy_` 抛出 `CUDA invalid argument`，P supervisor 随后退出。这轮服务/性能数据不能作为优化收益。新增 GPU 回归已在修复前复现同一异常；共享 LayerCache 现在向原 helper 提供完整 cache cell 的字节对齐，helper 取其与系统页大小的最小公倍数并在约 128 MiB 的粒度内分段，确保每个 checkpoint 的所有子视图落在单个注册段中。没有填充、改变 checkpoint ABI 或修改 CPU 大缓存的默认注册粒度。修复后上述五组回归共 **41 passed**，包括该槽位的 GPU→CPU→GPU 完整字节往返。前后日志分别为 `registration-boundary-before.log`、`registration-fix-tests.log`，均在同一 PR 日志目录。

## 五项完成后的 vLLM 对照（2026-10-13）

当前 LightLLM 为全部五项叠加，普通量化仍为用户要求回退后的原路径。vLLM 0.31.0 取归档 `hot-16k-c{128,512}-1011`；同一两台 H100、同自定义主/draft 检查点、MTP3，不能同时启动两套服务，因此不是同期随机化 A/B。这里比较的是相同归档 token IDs、warm-prefix、greedy、4096 输出的 text decode 诊断，不是原 vision LLM Bench 任务、冷缓存或不同随机 seed 的端到端测试。当前数据不足以证明约 1% 的单项服务收益。

| 输入/并发 | vLLM TPOT (ms) | LightLLM TPOT (ms) | TPOT 差异 | vLLM output TPS | LightLLM output TPS | TPS 差异 |
|---|---:|---:|---:|---:|---:|---:|
| 16K / 128 | 7.2066 | 6.5685 | -8.85% | 12,264.4 | 17,392.4 | +41.81% |
| 16K / 512 | 9.8407 | 11.1926 | +13.74% | 26,675.9 | 35,146.2 | +31.75% |

LightLLM 额外 512 为 11.3715 ms、34,463.4 TPS，TPOT 相对归档 vLLM 高15.55%；保留额外窗口，不以最快一轮代替主表。当前 512 正式 TTFT 9921.4 ms，归档 vLLM 17737.0 ms。

短 profile 的 GPU busy 为当前 LightLLM39.8426、vLLM40.2090 ms，周期44.9392/43.4372 ms；实际 batch512/504、平均序列长17279.8/17541.6。GPU busy 接近不能证明整请求 TPOT 接近，也不能把带 profiler 的忙碌时间直接除 tokens 解释正常请求。尚未确认剩余差距全部来自 CPU、接收率或某个调度策略。

独立采样的客户端供给曲线有明显差异。下表均以“已收到首字、尚未完成”计数，条件均值只统计至少一个 active 请求的采样区间；它不是 GPU batch，且不能给出各因素的因果份额：

| 512 窗口 | 首字最早→最晚 (ms) | 客户端 active 条件均值 | active≥480 的时间 (s) |
|---|---:|---:|---:|
| 归档 vLLM | 3808.3 → 39543.6 | 288.4 | 16.39 |
| 当前 LightLLM 正式 | 8078.0 → 13070.2 | 471.5 | 40.70 |
| 当前 LightLLM 额外 | 8310.4 → 12984.2 | 473.4 | 40.61 |

LightLLM 维持接近满负载的时间更长，vLLM 首字到达更分散，这支持“整个生命周期的负载过程不同”这一观察，与整轮 TPS 高而平均 TPOT 仍慢的现象相符，仍是待验证的解释；不能据此证明具体调度策略贡献了多少。由请求 decode 时长之和/整轮 wall 算出的均值与 TPOT×TPS 存在代数关系，只作一致性校验，不重复当作独立因果证据。上述可复查数据保存为 `silu-quant-1013-vllm-final-comparison.json`。

下一步若要判断小优化的端到端收益，应交错多轮 A/B，以完整回放为实验单位并比较整轮分布；同一归档 prompt 的预热/命中状态须匹配，冷缓存另做独立实验。固定 rank 映射会改变调度且无法固定所有工作量，不作为单独的精确归因办法。本轮没有恢复旧量化、改变 rank 映射、开启 Microbatch 或改动 P/master 服务。

### 补充：vLLM 服务端的整轮负载与 TPOT 分组

继续读取原已归档 `occupancy.json` 与 Prometheus 定义，未修改服务或重跑。vLLM D 的 `vllm:num_requests_running` 帮助文本为“Number of requests in model execution batches”；8 engine 求和的时间加权条件均值为288.56（排除零 running 的采样区间），与独立客户端计数接近。采样约9.8 s时各 rank8–12、19.7 s时27–32、30.7 s时63–64，证明“客户端并发512”并不意味着整轮 D 一直满 batch。此项仍是 scheduler 采样状态，不是每个 GPU step 的实际输入/graph padding；不与 LightLLM router admission gauge 直接当同口径相减。

逐请求按 TTFT 从小到大排序，每组128个、全部16K输入/4096输出；以下是同轮内的描述性分组，不是128个独立性能实验或同 batch 的严格对照：

| TTFT 排序四分组 | vLLM 平均 TTFT (ms) | vLLM 平均 TPOT (ms) | LightLLM 平均 TTFT (ms) | LightLLM 平均 TPOT (ms) |
|---|---:|---:|---:|---:|
| 1 | 7911.4 | 9.1682 | 8998.0 | 11.3299 |
| 2 | 15262.6 | 9.8757 | 9836.9 | 11.4518 |
| 3 | 23505.3 | 10.1589 | 11048.5 | 11.4180 |
| 4 | 24268.8 | 10.1601 | 12277.3 | 11.2861 |

vLLM 的9.8407 ms包含最小TTFT组的9.1682 ms，其余三组约9.8757–10.1601 ms；LightLLM各组约11.2861–11.4518 ms。结合服务端running曲线，更支持负载/到达时序是候选原因之一；分组仍受请求组成、rank映射、序列状态影响，不能把组间差直接归因给batch，或声称它解释了全部约14%的差距。逐step按相近batch/序列长/每周期产出对齐的证据仍缺失。原始派生数据为`silu-quant-1013-vllm-scheduler-running.json`、`silu-quant-1013-ttft-tpot-cohorts.json`。
