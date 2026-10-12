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
