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
