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
