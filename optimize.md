# B300 V4.1 FP4 MegaMoE 优化

2026-11-10 开始。原版 DeepSeek-V4.1-Flash-1017，B300 单机 4 卡 P + 4 卡 D，分别 DP4/EP4，NIXL 传输。P 的 batch/chunk 为 8192，主 GPU KV 容量 6400 万 token/rank，CPU KV 512 GiB。D 的 batch 上限为 1024，DSpark 3 步动态验证。每轮在同一数据集上按 100% 采样和 1× 速率回放，任务进入 running 后计时 20 分钟并取消，记录缓存命中率、成功数和错误分类。保留同一转发入口和运行参数，除非本轮实验明确测试这些参数。

回放任务设置 `clear_cache_before_run=false`。只切 D 的轮次沿用了 P 已有的 prompt cache。表里的「冷缓存」表示 P 刚重启，P 的 GPU、CPU 和磁盘 KV 缓存均重新初始化。各轮的预热历史和 D 端缓存状态不同，整机指标和跨轮百分比均为运行观测。

B300 实测验证了磁盘缓存隔离。基线、R06、R09、R08 和 R11 都传入同一个 `--disk_cache_dir /mnt/afs-agentpro/lightllm_disk_cache`，实际 worker 初始化日志分别记录了以 `fd915821c0514991_0`、`55e6af6e19a64167_0`、`fe652b02293644ef_0`、`370dc7d645064988_0`、`98d1ed49ef234bdf_0` 结尾的独立子目录。R11 worker 的 `DISK_CACHE_NAME` 未设置，检查其打开的 40 个磁盘缓存文件均位于 R11 子目录。相同父目录下的旧文件没有进入 R11 的缓存实例。启动代码默认用每次生成的服务 UUID 选子目录，显式设置 `DISK_CACHE_NAME` 会覆盖该命名规则。

## 起点

旧任务于 13:48:19 UTC 开始，14:28:38 发出取消信号，14:29:15 完成取消。最终有 26,240 条成功，汇总记录 3,639 条失败和 575 条取消时仍在途的请求。表中的统计含前期 SSH FD 耗尽与发送拥塞，作为排障记录保留。[旧任务报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=546be0c4-d260-4c1b-a728-40e69fd8797a)

| 轮次 | OTPS P50 (token/s) | TPOT P50/P90 (ms) | TTFT P50/P90 (ms) | 输入 token 缓存命中率 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| 初始回放 | 53.16 | 18.81 / 34.85 | 3946.8 / 37558.7 | 79.40% | 含转发故障和取消在途请求，排障用 |

四路压缩 SSH 转发启用后，取 14:08:51 至 14:11:02 的 1,566 条 Bench 记录，成功请求的 OTPS P50 为 57.43，TPOT P50/P95 为 17.40/37.16 ms，TTFT P50/P95 为 3184/161049 ms。有 49 条响应头超时、31 条 HTTP 422、10 条 HTTP 400。同期服务端日志中，输入不超过 8K 的请求 TTFT P95 为 8.02 s，输入超过 32K 的为 414.85 s。两组 D 单轮验证间隔估算的中位数均约 47 ms，尚无 kernel 级归因。按 D 计数器抽样，实际草稿接收率 73.8%，每轮推进 2.65 token。D 主 KV 仅用 30%–33%，独立的小页快照池曾满。D 的小页申请失败累积 153 次，P 为 41 次，先查分配交接和实际回退频率，再量化其性能收益。

原始日志与指标快照位于 [起点任务目录](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/546be0c4-d260-4c1b-a728-40e69fd8797a)。

## 干净链路基线

20 分钟任务从 14:30:19 UTC 开始，自动取消时间为 14:50:19 UTC。沿用原模型、P/D 参数和四路压缩转发，使用新 seed `e67Q1`。第一份进度快照有 5,028 条已处理、4,870 条成功。GPU 30 秒采样中，D 的 SM 活跃率中位数 76%，Tensor Core 活跃率 13%，DRAM 活跃率 20%，每卡 NVLink 收发约 13.5 GiB/s。GPU busy 中位数 95.5%，下一轮用 phase timing 和 kernel trace 区分验证、草稿和其他开销。[基线报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=37b72844-73ad-40fd-8e16-85c1546e4b3c) · [基线原始记录](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-baseline20/37b72844-73ad-40fd-8e16-85c1546e4b3c)

| 轮次 | 成功 / 失败 | OTPS P50 (token/s) | TPOT P50/P90 (ms) | TTFT P50/P90 (ms) | 成功输出 (token/s) | 输入缓存命中率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20 分钟基线 | 26,222 / 1,022 | 27.66 | 36.15 / 57.98 | 2867.2 / 5310.5 | 15,451 | 93.43% |
| R01，仅 D 小页交接 | 30,440 / 944 | 21.38 | 46.77 / 78.75 | 2565.5 / 4622.1 | 17,470 | 96.22% |
| R02，D 原生 32×32 FP8 GEMM | 32,027 / 988 | 23.20 | 43.10 / 65.84 | 1698.0 / 3980.3 | 20,460 | 98.03% |
| R04/R05，D packed 量化与启动重置 | 31,769 / 1,192 | 24.52 | 40.79 / 60.09 | 1905.0 / 5110.3 | 21,104 | 97.98% |
| R06，P 小页交接，P 重启后的冷缓存 | 12,678 / 1,069 | 76.61 | 13.05 / 26.46 | 3053.8 / 39361.1 | 7,727 | 67.51% |
| R06 暖机复跑 | 23,255 / 2,520 | 48.61 | 20.57 / 39.37 | 3689.7 / 19708.1 | 15,391 | 87.54% |
| R09，跳过共享候选，P 重启后的冷缓存 | 13,049 / 1,081 | 80.41 | 12.44 / 25.75 | 3575.6 / 21618.8 | 8,086 | 68.85% |
| R09 暖机复跑 | 26,441 / 1,227 | 48.22 | 20.74 / 40.59 | 2797.9 / 6272.3 | 17,125 | 90.10% |
| R08，P 原生 FP8 线性层，P 重启后的冷缓存 | 21,621 / 971 | 58.52 | 17.09 / 29.88 | 2092.5 / 33849.4 | 13,762 | 71.29% |
| R08 暖机复跑 | 32,142 / 997 | 23.24 | 43.03 / 64.43 | 1621.7 / 3595.8 | 20,311 | 93.68% |
| R07，D 路由写入融合 | 35,064 / 1,073 | 28.04 | 35.67 / 52.60 | 1751.1 / 3619.8 | 23,102 | 97.60% |
| R10，D MegaMoE 118 → 146 SM | 34,061 / 1,030 | 27.45 | 36.43 / 53.42 | 1617.5 / 3481.8 | 23,543 | 98.28% |
| R11，P MegaMoE 146 SM，P 重启后的冷缓存 | 22,285 / 1,017 | 65.40 | 15.29 / 26.51 | 2134.3 / 36806.1 | 14,313 | 71.95% |
| R11 第一轮暖机 | 34,189 / 1,045 | 26.45 | 37.80 / 56.31 | 1811.5 / 3815.5 | 22,738 | 93.62% |
| R11 第二轮暖机 | 33,772 / 1,063 | 26.28 | 38.05 / 56.25 | 1598.9 / 3566.1 | 22,704 | 98.12% |

成功输出按 18,541,555 个 token 除以 1200 s 计算。取消时 835 条在途，独立于失败数。20 分钟内 P/D 小页分配告警为 614/439 条。D 有 2 条独立 KV 等待超时，另外有请求取消清理后的迟到 WRITE。原转发 FD 故障未复现。按服务端输出至少 65 token 的 21,753 条请求计，TPOT P50 为 34.68 ms，每轮推进 P50 为 2.25 token，折算的验证间隔 P50 约 75.87 ms。这个值含执行和调度，后续用 D 的 GPU phase timing 拆分。

## R01 小页 checkpoint 槽交接

把小页淘汰和申请合到现有 radix/状态池路径。全局池仍有空槽时优先分配空槽。池满时，在分配锁内确认本 rank 的最旧候选只剩一份引用，直接交给新请求，省去放回空闲队列再竞争。缓存树已经 pin 住的节点不参与淘汰，跨 rank 或异步拷贝额外持有的引用也会阻止交接。池大小不变。用两份进程共享元数据的 allocator 验证了交接前后引用数以及其他 rank 无法再次分配同一槽。

现有回归套件对照为 111 通过、7 失败，改后为 112 通过、7 失败。相同 7 项都是原有视觉前缀匹配断言，新增小页独占交接测试通过。B300 D 的改动已放在独立源码副本 `/opt/lightllm-b300/runs/1p1d-v4.1-256K-B300-fp4-megamoe-20261110/opt-r01-checkpoint/source`，基线结束后仅重启 D。下一轮保留原 P 与转发，记录 D 小页失败次数、缓存命中率、TTFT/TPOT、成功输出 token/s。D 同时每 64 轮采样一次验证、提交和草稿阶段，20 分钟对比期间不开 kernel trace。

15:18:21 UTC 开始，运行 20 分钟后取消。D 的小页失败由 439 降到 0，仍用旧路径的 P 在相同窗口内有 1,657 条。成功输出吞吐从 15,451 升到 17,470 token/s（+13.1%），单请求 OTPS P50 从 27.66 降到 21.38（−22.7%）。本轮缓存命中率高 2.79 个百分点，成功请求数量与输入长度分布也变了，不能用这轮的 OTPS 差值归因于小页分配。该改动的可验证收益是消除这一轮 D 的小页分配失败。原每 64 轮计时没有产出样本，之后修正双推理线程共用采样状态的问题。 [R01 报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=e70f26b7-9623-4826-a048-2260f6992b3b) · [服务端计数](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r01-checkpoint20/e70f26b7-9623-4826-a048-2260f6992b3b/service-r01.json)

## R02 SM100 原生 32×32 FP8 线性层

在 B300 上比较现有 Triton 与 DeepGEMM 原生 block-scaled FP8 GEMM，计时含激活量化和 Graph 回放。5 种 D 线性层形状和 6 个行数共 30 组，激活 FP8 字节一致，BF16 输出最大相对 RMSE 为 0.0093%。算子加速 1.74–11.48 倍，中位数 4.06 倍。代表形状如下。

R01 结束后，另做 128 并发固定输入、每条 384 输出 token 的短 trace，128/128 成功，客户端 OTPS P50 40.28 token/s。四个 D rank 的 kernel 累计时间中，FP4 MegaMoE 占约 43%–46%，32×32 Triton GEMM 占约 31%–32%，FP8 激活量化约 6%，FlashMLA 约 3%。此处百分比来自固定形状的 kernel 时间求和，回放整机 TPOT 另测。 [Trace 统计](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r02-mxfp8-20/r01-kernel-trace-summary.jsonl)

| M × N × K | Triton (µs) | DeepGEMM (µs) | 加速 |
| --- | ---: | ---: | ---: |
| 384 × 1792 × 5120 | 90.17 | 20.58 | 4.38× |
| 384 × 4608 × 5120 | 124.98 | 22.60 | 5.53× |
| 384 × 5120 × 2304 | 59.52 | 14.40 | 4.13× |
| 384 × 25600 × 6144 | 618.07 | 60.54 | 10.21× |

在现有 `Quantcfg` 注册 `fp8w8a8-b32-deepgemm`，本轮显式选用。复用 DeepGEMM 的权重分配和加载路径，将原始 32×32 UE8M0 指数一次展开到原生 kernel 所需的行粒度并打包。保留 checkpoint 权重字节、指数及激活动态量化的 1e-4 下限。实际加载原版 1017 的 WQ-A、WQ-B 和共享专家 W2，走注册表、权重加载和 Graph 后，权重字节一致，最大相对 RMSE 为 0.0038%。384 行的 WQ-B 和共享 W2 分别从 139.32 降到 23.13 µs、从 60.47 降到 14.44 µs。原版 FP4 routed experts 仍选择原有 MegaMoE。首次试验触发 SGL packed 量化的固定 eps 断言，改用已有 LightLLM Triton packed 入口保留原下限。端到端对照待 R01 完成后运行。 [形状微测](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-baseline20/mxfp8-linear-probe.log) · [真实权重验证](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-baseline20/mxfp8-registered-probe.log)

R02 从 16:12:24 UTC 回放 20 分钟后取消，P 未重启。相对 R01，成功输出吞吐 **+17.1%**，OTPS P50 **+8.5%**，TTFT P50 **−33.8%**，输入缓存命中率上升 1.81 个百分点。D 小页申请失败和独立 KV 等待超时均为 0，P 仍有 2,059 条小页分配告警。D 每 rank 374 个在线计时样本，每轮平均约 340 个 verify row，target 35.5 ms、draft 2.6 ms、KV commit 0.16 ms、GPU round 43.9 ms、wall round 52.2 ms。 [R02 报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=910fc006-d465-42d9-a089-083ee289018b) · [在线计时](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r02-mxfp8-20/910fc006-d465-42d9-a089-083ee289018b/service-timing-r02.json)

同样 128 并发的短 trace，按 target MoE kernel 调用数归一，线性 GEMM 的 GPU 时间由约 281 降至 64 µs，所有 kernel 的累计时间由约 874–914 降至 621–640 µs。带 profiler 的客户端 OTPS P50 从 40.28 降到 36.15，短 trace 的客户端数据有波动。GPU 时间按同一调用数口径比较，线上端到端结果取上面的 20 分钟回放。 [Trace 对照](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r04-packed20/r01-r02-trace-diff.jsonl)

## R03 已预热配置跳过输出克隆

R01 命中已有 autotune 配置时，D 的主模型 Graph 捕获仍用了 914.3 s。非阻塞 Python 采样命中 Triton GEMM 的 `kernel_warmup → _mutate_args_clone`。原实现会先复制输出张量，再检查该配置是否预热过。将检查移到复制前，保留首次调用与失败重试。B300 上用 384×25600 BF16 输出做 1,000 次已预热调用，五次中位耗时从 10.08 ms 降到 2.13 ms，约 4.74×；autotuner 单测 54 项通过。该变更单独保留，完整启动对照仍待测。已有 autotune 缓存含源码指纹，整机比较需要区分是否复用历史配置。 [启动调用栈](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r01-checkpoint20/startup-python-stack.log) · [微测](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-baseline20/warmup-clone-r03.log)

## R04 Packed UE8M0 激活量化

复用现有 Triton packed 分支。较大 batch 由每 CTA 打包 4 个 scale group 改为 32 个，减少 launch grid 和 scale 写入开销。较小输入继续按 4 个 group 打包。对 74 组 group-32 形状及 24 组 group-128 形状，覆盖行优先和 TMA 列优先 scale、部分尾列及极小激活，FP8 激活和 packed scale 都逐字节一致。384×5120 的两种 layout 均约从 10.27 降到 4.11 µs，8192×5120 从 171.97 降到约 32.76 µs。8 行以内中位速度持平。R02 单独测原生 GEMM，R04/R05 在 D 增量部署 packed 分块。 [Group 32 微测](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r02-mxfp8-20/packed-quant-r04.log) · [Group 128 微测](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r02-mxfp8-20/packed-quant-group128.log)

## R05 Graph 预热的 KV allocator 重置

R02 Graph 捕获期间对 D rank 0 做 15 秒非阻塞 CPU 采样，584 个初始化线程样本里有 511 个停在 `KvCacheAllocator.free_all`，占 87.5%。每捕获一个形状都会调用它。旧实现重建含约 9,162 万个 Python int 的列表，再复制回已锁页的 CPU tensor。

改为用 `torch.arange(out=...)` 原地重置。在 B300 上使用 91,620,352 个 pinned int32 索引，3 次中位耗时 **5.46 s → 81.9 ms**，约 **66.6×**。独立进程峰值 RSS 从约 5.04 GiB 降到 1.96 GiB。完整序列、锁页属性、指针、分配游标和共享余量均验证一致。下一次 D 重启一并验证这项启动改动，20 分钟回放只计服务就绪之后的吞吐和延迟。 [CPU 栈采样](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r02-mxfp8-20/startup-python.folded) · [重置微测](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r02-mxfp8-20/allocator-reset-r05.log)

R04/R05 实际重启对照中，D rank 0 主模型 Graph **1039.5 → 73.4 s**，草稿 Graph **247.0 → 3.3 s**，两项合计 **21 分 27 秒 → 1 分 17 秒**。主权重加载及 Engram 映射均约 376 s。同一次重启也部署了 R04 的 packed kernel。启动数据是组合结果，重置微测和 CPU 栈可独立验证 allocator 开销。文本、图片、SSE 验收通过后，R04/R05 于 16:45:06 UTC 开始回放。 [实际启动日志](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r04-packed20/startup-r02-r04.log) · [回放报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=0bcd8594-8494-4656-8c5b-6d02f18a73b1)

17:05:06 UTC 发出取消信号，平台于 17:05:45 确认。相对 R02，成功输出 **+3.1%**，单请求 OTPS P50 **+5.7%**，TPOT P50 降 **5.4%**。TTFT P50/P90 升 **12.2% / 28.4%**，失败从 988 增至 1,192，另有 413 条取消时在途。输入缓存命中率基本持平。D 各 rank 404 个在线样本，平均每轮约 317 个 verify row（R02 约 340），target 约 33.7 ms、draft 2.5 ms、KV commit 0.14 ms、GPU round 41.7 ms、wall round 49.2 ms。批量变小，需固定形状 trace 区分 kernel 提速与负载差异。P/D 分配告警 2,375/0，未观察到迟到 WRITE。 [回放汇总](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r04-packed20/0bcd8594-8494-4656-8c5b-6d02f18a73b1/timebox.log) · [在线计时](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r04-packed20/0bcd8594-8494-4656-8c5b-6d02f18a73b1/service-timing-r04.json)

固定 128 并发的 R04 短 trace 有 128/128 条成功，客户端 OTPS P50 为 37.37（R02 为 36.15）。按 target MoE 调用数归一，四卡平均激活量化 GPU 时间 **20.69 → 12.22 µs（−41.0%）**，线性 GEMM 约 64.4 µs 持平，target MegaMoE **345.8 → 356.5 µs**。GPU kernel 总时间基本持平，target MegaMoE 仍占约 56%，单独看量化收益大于整轮收益。对 3,700 条 Bench 请求分段抽样，失败样本主要是 HTTP 422、HTTP 400 和连接类报错，其中工具 schema 35 条、developer role 33 条、连接类 32 条。该分布为抽样，不将平台的 1,192 条失败全部归到 P 或 KV。 [Trace 对照](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r06-p-checkpoint20/r02-r04-trace-diff.jsonl) · [失败样本](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r04-packed20/0bcd8594-8494-4656-8c5b-6d02f18a73b1/request-error-sample.json)

## R06 P 侧小页 checkpoint 槽交接

P 和 D 各自有一池由本机 4 个 DP rank 共享的小页 checkpoint，`512 × 4 = 2048` 槽。每槽约 3.07 MiB，每侧占约 6.13 GiB 的 `/dev/shm`。请求结束后，checkpoint 仍可挂在前缀缓存树上。跨 rank 前缀复用和异步拷贝还可能增加引用。因此空闲槽数可以为 0，而请求池仍能接收新请求。

R04/R05 开始后截至 16:48:50 UTC，两侧空闲槽均为 0。D 2048 个槽均为单引用，窗口内申请告警 0 条。P 有 1805 个单引用槽、243 个多引用槽，窗口内申请告警 553 条。引用分布是瞬时快照，不能按它给每条失败归因。拟先单独在 P 部署 R01 原子交接，保持 8192 batch/chunk、量化路径、槽数和 D 参数。候选源码和原 launcher 已在 B300 暂存，尚未切换。回放对比 P 分配告警、缓存命中率和 TTFT。P 重启会改变缓存温度，需要分开记录。

R06 切换了 P 的 5 个 checkpoint 相关源码文件，原 P launcher 和 32×32 Triton 线性层保持一致。仅回收已退出的 P 进程组创建且无附着者的 SysV 段。四个 PD worker 注册、11K 文本、图片和外部 SSE 均通过，cgroup 未出现 OOM。20 分钟回放于 17:25:48 UTC 开始。 [R06 报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=a4f42b78-81b5-475c-8fd7-74c293cf6279) · [部署和验收](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r06-p-checkpoint20/restart-and-replay.log)

17:45:48 UTC 发出取消信号，17:46:07 UTC 进入终态。P 分配告警 **2,375 → 0**，D 保持 0。P 刚重启，输入缓存命中率 **97.98% → 67.51%**。同一窗口 D 每轮平均 verify row 约 **317 → 77**，成功输出吞吐低 **63.4%**，TTFT P90 从 5.11 s 升至 39.36 s，低 D 并发下单请求 OTPS 显著升高。D 有 406 条等待 WRITE 到期日志，对应 14 个不同请求，P/D 有 123/77 条迟到 WRITE 相关日志。出现真实传输超时，且前缀命中和 D 负载差异很大，这一轮用于定位 P 的回收和冷启动瓶颈，端到端收益待同配置暖机复跑。 [冷缓存回放原始汇总](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r06-p-checkpoint20/a4f42b78-81b5-475c-8fd7-74c293cf6279/timebox.log)

保留已暖机的 R06 P 和 R04 D，外部 SSE 复验通过后用新 seed `6AdQ3` 于 17:48:12 UTC 开始 20 分钟复跑。因本机 `/mnt/vqa` 底盘已满，追加日志保存到 B300 的同名 `/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r06-warm-control20/`。本轮结束后再切换 D，先看 P 暖机后是否仍有 WRITE 超时。 [暖机对照报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=92fda2be-da44-407a-8ec6-358028b10e0a)

18:08:12 UTC 发出取消信号，18:08:42 UTC 进入终态。暖机后成功输出比冷缓存约高 99%，仍低于 R04/R05。D 还有 3 个不同请求等 WRITE 到期，P/D 分别有 288/151 条迟到 WRITE 相关日志。本轮 P checkpoint 告警 **6,874**，推翻了冷缓存轮 0 告警能代表满负载的假设。结束时 P 空闲槽为 0，其中 2,031 个槽仅有一份引用，17 个槽有多份引用。该快照能解释为什么全池满时仍值得继续找可交接槽，不能用来推断每条失败的归因。暂缓 R07 的 D 切换。

## R07 MegaMoE FP8×FP4 路由写入融合

本地 vLLM `5df63a7203` 的 `prepare_megamoe_inputs` 和 SGLang `af5731cca6` 的 `mega_moe_pre_dispatch` 在量化时写入 Top-K 专家编号和权重。LightLLM 的 FP8×FP8 路径已这么做，FP8×FP4 的 packed UE8M0 分支仍在量化后各做一次 `copy_`。复用现有量化入口和路由输出参数，由每行第一个 packed CTA 写入路由，保留量化公式、尺度和输出 dtype。候选尚未部署，先在两轮回放之间检查输出字节及 CUDA Graph 下的三次 launch 对一次 launch。 [微测脚本](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r07-mega-stage/packed-mega-stage-probe.py)

在 NVIDIA B300（SM103）验证 91 组形状，覆盖空批、尾列、两种 packed scale 布局、2/6 路路由和原 FP8×FP8 分支。激活、scale、专家编号与权重均与原路径一致，输出缓冲区尾部未被改写。按每组 128 次 CUDA Graph 回放交替计时，代表形状如下。这里只计入量化和路由写入，MegaMoE 主 kernel 时间另测。 [B300 微测](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r07-mega-stage/stage-probe.log)

| 行数 × 5120 | 原路径 (µs) | 融合 (µs) |
| --- | ---: | ---: |
| 32 | 3.622 | 1.587 |
| 128 | 3.969 | 1.892 |
| 384 | 4.817 | 2.612 |
| 1024 | 6.877 | 4.658 |
| 8192 | 34.244 | 32.265 |

R08 暖机轮确认无 WRITE 等待超时后，保留 R08 的 P 和 master，仅重启 D 部署路由融合。11K 文本、图片及外部 SSE 通过。新一轮沿用相同的 D 量化、DSpark、batch 和 Graph 参数。 [R07 回放报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=a1a69796-c88e-413e-9682-7fdc92dfe12f)

R07 从 20:25:30 UTC 运行 20 分钟后发出取消信号，20:46:12 UTC 进入终态。比 R08 暖机轮，成功输出吞吐 **+13.7%**、OTPS P50 **+20.7%**、TPOT P50 **−17.1%**，TTFT P50 **+8.0%**。P 命中率同时上升 3.92 个百分点，D 平均 verify row 从约 330 升至 404，端到端增益包含负载变化。P/D checkpoint 告警 773/0，未观察到 WRITE 超时或迟到 WRITE。固定 128 并发短 trace 128/128 成功，归一后 kernel 累计时间 634.3 → 630.5 µs，带 profiler 的客户端 OTPS P50 37.37 → 36.64。路由融合的直接收益以逐字节对照和算子微测为准。

## R09 跳过被共享的最老 checkpoint

R06 只把本 rank 最老的 checkpoint 候选传给 allocator。如果该候选仍有其他 rank 的引用，即使后面的槽可以回收也会返回失败。R09 保留空闲队列优先。在同一把锁下按 LRU 继续找第一个仅有一份引用的候选，保持共享和传输中的槽不动。空闲队列有槽时不会扫描。

用真实的两份共享 allocator 元数据和 radix 树复现。池满且最老槽引用数为 2、下一槽引用数为 1 时，R06 返回 `None`，R09 原子交接下一槽。检查共享槽引用数不变、被交接槽从树节点解除关联、其他 rank 无法抢到槽，以及空闲队列始终优先。验证日志在 B300 的 `/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r09-p-reuse20/checkpoint-handoff-regression.log`。接下来仅重启 P，分别用 20 分钟冷缓存和暖机回放与 R06 对照。

R09 的 P 已通过四个 PD worker 注册、11K 文本、图片和外部 SSE 验收。D 保留 R04/R05。冷缓存任务于 18:30:29 UTC 开始，计时结束后先收集终态和服务计数，再用新 seed 开始暖机轮。 [R09 冷缓存报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=95bdfb10-c085-43a7-aae7-c9ae7bfe2f02)

R09 冷缓存轮于 18:50:29 UTC 取消。比 R06 冷缓存轮，成功输出吞吐增加 4.6%，TTFT P90 降 45.1%，TTFT P50 升 17.1%。两轮 P checkpoint 告警都为 0。D 仍有 11 个不同请求 WRITE 等待到期，P/D 迟到 WRITE 相关日志为 298/157 条。冷缓存数据继续作为排障对照。保留同一 P/D 的暖机轮于 18:51:02 UTC 开始。 [冷缓存报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=95bdfb10-c085-43a7-aae7-c9ae7bfe2f02) · [暖机报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=1ba0592c-9869-40eb-8b4b-4d00935e974a)

R09 暖机轮于 19:11:02 UTC 发出取消信号，19:11:34 UTC 进入终态。P checkpoint 告警 **6,874 → 0**，D 仍为 0。相对 R06 暖机轮，成功输出吞吐 **+11.3%**，TTFT P50/P90 降 **24.2% / 68.2%**，OTPS P50 约持平（48.61 → 48.22），缓存命中率提高 2.56 个百分点。D 仍有 5 个不同请求 WRITE 等待到期，迟到 WRITE 相关日志 P/D 为 763/435 条。回收改动在暖机压力下消除了槽申请告警。两轮仍存在传输故障，端到端差值作为排障数据保留。

## R08 P 侧原生 32×32 FP8 线性层

R09 消除了 checkpoint 分配告警，D 仍有 WRITE 等待到期。下一轮仅切换 P 的线性层为已在 D 验证的 `fp8w8a8-b32-deepgemm`，带上 R04 packed 量化分块，保留 R09 回收、原版 FP4 routed experts、8192 batch/chunk 以及 D 的参数和源码。候选 launcher 相比 R09 仅新增一行量化参数。用重启后的冷缓存轮和暖机轮分别和 R09 比较，记录未命中输入下的 TTFT、吞吐和 WRITE 到期请求数。

R08 已在 P 日志确认原生 FP8 线性层和 FP4 routed experts 的选择。四个 PD worker、11K 文本、图片及外部 SSE 均通过，cgroup 未出现 OOM。冷缓存回放于 19:31:43 UTC 开始。原始记录在 B300 的 `/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r08-p-native20/`。 [冷缓存报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=ec070bdd-0f7b-451e-8688-125e966a804a)

R08 冷缓存轮于 19:51:43 UTC 取消。相对 R09 冷缓存轮，成功输出吞吐 **+70.2%**，成功数 13,049 → 21,621，TTFT P50 **−41.5%**、P90 **+56.6%**。平台统计的未命中输入 token 从 6,153 万升至 1.386 亿，D 平均每轮 verify row 从约 80 增到 185，解释了相同 D 路径下 TPOT P50 的升高。两边 checkpoint 告警均为 0，D WRITE 超时请求 11 → 4，P/D 迟到 WRITE 日志为 147/99 条。冷缓存轮仍含传输超时。保持 P/D 的暖机轮随后开始。 [暖机报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=0eaa8606-a44f-4666-bbc9-4ba2a41f1812)

R08 暖机轮于 20:12:28 UTC 发出取消信号，20:13:09 UTC 进入终态。比 R09 暖机轮，成功输出吞吐 **+18.6%**，TTFT P50/P90 **−42.0% / −42.7%**。D 平均 verify row 约 330（R09 暖机约 262），OTPS P50 48.22 → 23.24，缓存命中率提高 3.58 个百分点。P 有 176 条 checkpoint 告警，D 为 0。P/D 均未观察到 WRITE 超时或迟到 WRITE，且 B300 cgroup 无 OOM。与此前吞吐最高的 R04/R05 比，输出低 3.8%，TTFT P50/P90 分别低 14.9% / 29.6%，缓存命中率低 4.30 个百分点。该轮作为 R07 的基线，保留 P，单独切 D 的 MegaMoE 路由融合。

## R10 MegaMoE 的 SM 预算

R04 D 的固定并发 trace 中 target MegaMoE 约占 kernel 时间 56%，模板使用 118 个 SM。当前纯 FP8×FP4 MegaMoE 启动日志显示没有 legacy 低延迟 buffer 或 prefill workspace，但 SM 预算仍从设备总数扣除 DeepEP 的理论预留。SGLang 的 SM100 MegaMoE 显式保留少量 SM 给 clustered launch 的驻留余量，本地默认保留 2 个。先在 R07 回放结束后，用独立进程跑实际 EP4 target 形状，比较 118 和更大的 SM 预算。每档独立分配对称 buffer，覆盖 64–1024 行及均匀和偏斜路由，检查 BF16 输出。之后再决定是否调整纯 MegaMoE 路径的预算。探针和输出保存在 B300 的 `/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r10-mega-sms/`。

B300 实测设备 148 个 SM。按 118/128/136/146 四档做 56 组检查，BF16 最大相对 RMSE 为 0。118 → 146 时，64–1024 行的均匀路由中位耗时下降 **16.1%**，偏斜路由下降 **13.8%**。384 行的均匀和偏斜路由分别是 **404.82 → 339.17 µs**、**716.10 → 628.27 µs**。改用已有的 `has_legacy_moe_layer` 判断是否需要预留 DeepEP SM。纯 FP8×FP4 MegaMoE 留一对 CTA 的驻留余量，混合 legacy 层和已有 SM90 路径沿用原预算规则。只在 D 切换这一改动，P 保留 R08。

P 的 8192 行微测也使用原有路由复制。118 → 146 时，均匀路由 **2,033.55 → 1,934.64 µs**，偏斜路由 **6,577.26 → 6,152.29 µs**，20 组输出最大相对 RMSE 为 0。D 的 146 SM 服务完成验收后于 21:07:16 UTC 开始回放。新 SM 预算首次 JIT 后，rank 0 主模型和草稿 Graph 分别耗时 338.6 s 和 76.1 s，R07 用旧缓存时为 72.1 s 和 3.8 s。 [D 回放报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=8826f8d3-0d95-4b78-8db1-a3e27b19c26d)

R10 于 21:27:16 UTC 发出取消信号，21:27:57 UTC 确认终态。对 R07，成功输出吞吐 **+1.9%**，OTPS P50 **−2.1%**，TPOT P50 **+2.1%**，TTFT P50/P90 **−7.6% / −3.8%**。缓存命中率增加 0.68 个百分点。D 在线 target 平均约 31.37 ms（R07 约 36.00 ms），每轮 verify row 约 341（R07 约 404），不能直接用在线相减估算同形状收益。P/D checkpoint 告警为 482/0，均无 WRITE 等待超时或迟到 WRITE。

固定 128 并发 trace 的 4 个 rank 均确认 target 模板使用 146 SM，128/128 请求成功。按 target MoE 调用数归一，target 平均 GPU 时间 **358.1 → 301.2 µs（−15.9%）**，全部 kernel 累计 **630.5 → 575.4 µs（−8.7%）**。带 profiler 的客户端 OTPS P50 为 132.35，R07 为 36.64，波动远大于算子耗时差，保留为观测值。原始 trace 对照在 B300 的 `opt-r10-mega-sms/r07-r10-trace-diff.jsonl`。

## R11 P 侧 MegaMoE 146 SM

沿用 R10 的 SM 预算改动和原生 FP8 线性层，P 的 launcher 保持与 R08 一致，D 保留 R10。只有在 R10 已终止、固定 128 并发验证通过且服务端无 WRITE 等待超时时切换 P。对照 R08 分别记录 20 分钟冷缓存和暖机表现。

新 P 完成四个 PD worker 注册、11K 文本、图片及外部 SSE 验收。冷缓存轮先跑 20 分钟，取消并收集服务计数后开始暖机轮。记录在 B300 的 `/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r11-p-mega-sms20/`。 [R11 冷缓存报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=0c958cf5-1763-495f-9c52-a940bb31d673)

R11 冷缓存轮于 22:07:59 UTC 发出取消信号，22:08:30 UTC 进入终态。相对 R08 冷缓存轮，成功输出吞吐 **+4.0%**，TTFT P50/P90 **+2.0% / +8.7%**，输入缓存命中率高 0.66 个百分点。D 也已从 R04 切到 R10，这组冷缓存差值包含 P 和 D 两侧变更。P/D checkpoint 告警均为 0，D 有 4 个不同请求等待 WRITE 到期，与 R08 冷缓存轮相同。暖机轮继续验证。 [R11 暖机报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=f9e187cc-2290-472d-9588-3b4fc15f5b45)

R11 第一轮暖机于 22:28:43 UTC 发出取消信号，22:29:25 UTC 进入终态。P/D 均无小页分配告警、WRITE 等待超时或迟到 WRITE。吞吐 22,738 token/s，输入命中率 93.62%，低于同一 D 版本的 R10 4.66 个百分点。保留服务和缓存再跑 20 分钟，以更接近的缓存水平对照只改 P 的增量效果。 [R11 第二轮暖机报告](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=c0e4461b-8095-406c-9459-9aa30c25e355)

R11 第二轮暖机于 22:49:54 UTC 发出取消信号，22:50:35 UTC 进入终态。与 R10 的总输入命中率仅差 0.16 个百分点，成功输出吞吐 **−3.6%**，TPOT P50 **+4.4%**，TTFT P50 **−1.1%**、P90 **+2.4%**。P/D 再次无小页分配告警、WRITE 等待超时或迟到 WRITE。两轮缓存历史不同，相近的总命中率无法确认命中层级和请求级未命中长度相同。P 的整机收益仍待同缓存状态复测。最终工作区仅在纯 FP8×FP4 的 decode 路径使用新 SM 预算，P、混合 legacy 路径及 SM90 继续走各自已有策略。用最终源码逻辑验证了 8 种 SM 预算场景。B300 当时保留已经暖机的 R11 P 和 R10 D，回放结束后的外部 SSE 验证通过，cgroup 无 OOM。

## 夜间对照结论

最高观测到的成功输出吞吐是 R10 的 **23,543 token/s**，基线为 **15,451 token/s**。观测差值为 **+52.4%**，同期输入缓存命中率从 **93.43% 升至 98.28%**，未命中输入占比从 **6.57% 降至 1.72%**。R11 第一轮暖机的总命中率接近基线，观测吞吐为 **22,738 token/s（+47.2%）**，但缓存历史仍未控制。现有回放无法量化代码单独带来的吞吐或延迟提升。固定形状的算子收益以各轮微测和 trace 为准。

复测 P 端时，基线与候选各自重启 P 即可重建 P 的 GPU、CPU 和磁盘 KV 缓存。沿用磁盘缓存父目录时，应确认 worker 使用新子目录。对照整条 PD 链路时也要控制 D 的初始缓存状态。冷启动测试在相同的服务验收流程后开始。暖缓存测试应从相同初始状态回放同一段预热流量，随后用相同的请求序列和时间安排测量 20 分钟，核对命中层级、未命中输入长度与完成请求分布。

R09 暖机时 P 的小页告警从 6,874 降为 0。R08、R07、R10 和 R11 暖机轮均未观察到 WRITE 等待超时或迟到 WRITE，P 重启后冷缓存轮仍有少量超时。服务上线验收覆盖 11K 文本、图片和外部 SSE。B300 保留同一回放的结果，vLLM/SGLang 对照限于代码路径。

## 诊断与部署

R01 运行期间每 64 轮的 CUDA event 采样未产出记录。两条推理线程共享同一 `DecodeTimeLog`，样本状态和 CUDA events 改为线程本地，保留既有的正常同步点。对应 CPU 测试 4 项通过，GPU 测试待运行。R02 初次重启时 D 的 NCCL 端口仍在 TIME_WAIT，启动校验拒绝。确认无旧 GPU 进程且 28841 和 8121 能普通 bind 后，用相同端口启动，重试前一并部署上述采样修复。 [启动及对照记录](/mnt/vqa/wanzihao/bench-logs/20261110/b300-1017-fp4-replay100/opt-r02-mxfp8-20)

## 2026-11-11 逐项复验

在 B300 单机 4P + 4D（DPEP4）上使用原版 FP4 DeepSeek-V4.1-Flash-1017，P 的 batch 和 chunk 都是 8192，保持 DSpark、模型和服务参数一致。验证顺序为原生 FP4 基线 → checkpoint 回收 → 原生 32×32 FP8 线性层 → packed 量化分块 → 路由融合 → D 的 MegaMoE SM 预算。每轮只加入对应提交，基线和候选均重启 P、D，核对 P 使用相同磁盘父目录下的新子目录及 D 的新进程命名空间，并通过同样的长文本、图片和外部 SSE 验收。100% 采样、1× 的 20 分钟回放在基线上出现大量 KV 传输超时，作为诊断保留。整机性能改用固定形状对照：24 并发长输入、128 并发短输入长生成，要求每个请求未命中缓存并完整生成指定长度，每种形状同并发预热一轮后测两轮。

| 提交 | 验证结果 |
| --- | --- |
| `9358062db` autotuner 热缓存跳过复制 | 新回归在父提交复现多余的输出 buffer 复制，修复后 autotuner 54 项测试通过。 |
| `cde71b7db` 推理线程独立采集 decode 计时 | 父提交有两项线程间状态冲突回归失败；修复后 B300 的 5 项测试通过，覆盖真实 CUDA Graph。 |
| `52d0a3030` 原地重置 pinned KV 索引 | B300，服务线程设置，91,620,352 个索引：重置中位 5.257 s → 28.658 ms。索引序列、存储指针、pinned 属性与分配器计数保持正确。 |
| `bb6d6618c` PD 代理及时转发 SSE | 实际 HAProxy 下 9 项测试通过，包含首个 SSE 片段在上游流结束前抵达。另有一项关闭流程的 mock 测试在父提交和候选均失败。 |

原始复验日志：B300 的 `/mnt/vqa/wanzihao/bench-logs/20261111/v41-opt-reverify/`。

首次基线在 D 启动自检中复现 Blackwell FlashMLA 的单页物理步长错误，未创建压测任务。独立提交 `703c6619f` 修正 SWA 和压缩 KV 的页步长：B300 单页原生算子由 `TMA_K_STRIDE` 断言失败变为通过，父提交的 3 个单页回归失败，修复后缓存视图和 workspace 共 18 项通过。各对照快照共用这项修复；不包含远端临时拆分大于 8192 query rows 的改动。

- `d4db3a171`：B300 原生 FP4 routed experts 和 DSpark 的兼容基础。模型选择和加载 8 项通过；4P+4D 的四个 master worker 全部注册，11,226 token 文本、图片和外部 SSE 通过。
- `82c4ad9a5`：checkpoint 槽位原子交接。B300 上两个共享 allocator 附件和真实 radix 树复现旧实现：最老槽仍被另一个 rank 引用时，解除树关联后分配仍失败。候选保留最老槽两份引用，接管下一槽，另一 rank 无法抢到；空闲槽始终优先。相关 CPU 测试从 124 通过到 125 通过，父子版本均有相同 7 项既存图片前缀断言失败。
- [独立冷启动基线](http://10.9.66.47:5174/spaces/default/env/default/reports?region=ms-01-tech-cci&taskId=1f8a34d0-07c3-46ab-a938-e1a08b96917b) 于 09:53:22（北京时间）开跑。新 P 与上次尝试使用同一磁盘缓存父目录、不同子目录；D 使用新 IPC 命名空间。复用先前验收过的 11,226 token 文本和 294 token 图片请求，首次响应的 `cached_tokens` 均为 0。该轮保留 Triton 32×32 FP8 线性层、未回收的 checkpoint 路径及原有 118 SM 预算。

基线回放终态：12,283 成功，1,070 失败，取消时有 699 个请求在途。D 有 353 条 KV 等待超时日志，P/D 有 355/183 条迟到 WRITE；无新增 cgroup OOM。此轮成功输出为 7,136 token/s（到终态的总输出除以总时长），TTFT P50/P90 为 2.68/33.83 s，TPOT P50 18.70 ms，输入缓存命中率 68.40%。这些数值仅用来描述过载基线。

| 提交 | B300 独立算子结果 |
| --- | --- |
| `4b8339866` 原生 32×32 FP8 线性层 | 3 个实际权重、4 种行数，最大 BF16 相对 RMSE 4.22e−5。32、128、384、8192 行的算子中位加速分别为 2.04×、4.75×、5.31×、7.95×。使用 `--quant_type fp8w8a8-b32-deepgemm` 启用，routed experts 仍用原生 FP4。 |
| `bbdb7ee22` packed UE8M0 激活量化 | 74 组 group32 和 24 组 group128 的 FP8 值和 packed scale 逐值相同。K=5120 的 TMA 布局，384 行 10.26 → 4.07 µs，8192 行 171.97 → 32.81 µs。 |
| `21bd60143` MegaMoE 路由融合 | 91 组比较的激活、scale、路由 ID/权重及输出 buffer 尾部均一致。K=5120、topk=6，384 行 4.80 → 2.59 µs，8192 行 34.23 → 32.23 µs。 |
| `b6785e0de` D 的纯 FP8×FP4 SM 预算 | 四卡 EP、56 组测试输出一致。118 → 146 SM，均匀/偏斜路由的算子中位耗时下降 15.82%/13.19%。8 种选择检查中，只有纯 FP8×FP4 D 改变预算。 |

固定形状的长输入为 24 并发，每条 4,647 输入 token、24 输出 token；长生成是 128 并发，每条 37 输入 token、384 输出 token。以下取两个测量轮的中位数，计时不含同并发预热。所有请求的 `cached_tokens=0`，所有输出均到达指定长度，KV 等待超时、迟到 WRITE、HTTP 5xx 和 OOM 需要为零。

| 版本（逐项累加） | P 测试 TTFT P50 (ms) | D 测试 TPOT P50 (ms) | D 测试输出 (token/s) |
| --- | ---: | ---: | ---: |
| 原生 FP4 基线 | 1528.47 | 9.056 | 10705.53 |
| + checkpoint 槽位原子交接 | 1547.71 | 8.995 | 9393.40 |
| + 原生 32×32 FP8 线性层 | 811.64 | 6.691 | 13218.48 |
| + packed UE8M0 量化分块 | 775.86 | 6.488 | 13747.61 |
| + MegaMoE 路由融合 | 785.69 | 6.755 | 13754.14 |
| + D 端 MegaMoE 146 SM | 846.53 | 5.909 | 15912.62 |

全部代码提交叠加后的 CPU 回归：202 通过、464 跳过，另有与父提交相同的 7 项图片前缀断言失败。最终分支的推理改动与 B300 最终快照逐文件校验一致，保留的 PD master 正在使用已经验证的 `option http-no-delay`。

checkpoint 固定对照全部成功，未观察到 KV 等待超时、迟到 WRITE、HTTP 5xx 或 OOM。长输入 TTFT P50 +1.26%，长生成 TPOT P50 −0.67%，长生成批次首 token P50 为 590.01 → 812.20 ms，整批输出吞吐 −12.26%。该固定负载不以填满 checkpoint 池为目标。基线固定对照在 1× 回放停止、请求排空后执行，候选在上线验收后执行；两者测量请求的命中数都是 0，其他缓存条目占用可能不同。checkpoint 的收益以已复现的槽位可用性修复为准。

原生 32×32 FP8 线性层对 checkpoint 版本：长输入 TTFT P50 −47.56%、P90 −46.71%；长生成 TPOT P50 −25.61%、P90 −24.63%，整批输出吞吐 +40.72%。同形状、同并发预热一轮后测两轮，均完整成功、缓存命中为 0，无传输错误和 OOM。独立实际权重算子的相对 RMSE 和耗时记录见上表。

packed 量化对原生 32×32 FP8 线性层版本：长输入 TTFT P50 −4.41%、P90 −4.14%；长生成 TPOT P50 −3.03%、P90 −3.37%，整批输出吞吐 +4.00%，长生成 TTFT P50 +3.12%。两轮均完整成功、命中为 0，无传输错误和 OOM。这是两轮固定负载下的小幅增量。

路由融合对 packed 版本：长输入 TTFT P50 +1.27%、P90 +0.21%；长生成 TPOT P50 +4.12%、P90 +5.52%，整批输出吞吐 +0.05%，长生成 TTFT P50 +10.67%。全部请求成功、命中为 0，无传输错误和 OOM。独立量化/路由 staging kernel 有收益，当前两轮服务测量没有显示吞吐加速，延迟还出现上升，不能将该算子收益直接换算为整机收益。

D 的 SM 预算对路由融合版本：长生成 TPOT P50 −12.52%、P90 −13.61%，整批输出吞吐 +15.69%；长输入 TTFT P50 +7.74%，P 配置及预算相同。两轮固定请求全部成功、命中为 0，无传输错误和 OOM。随后单独抓 D trace，128/128 请求完成，四个 rank 各 3,920 次 target kernel 调用均使用 146 SM；trace 不参与上表的计时。

全量累加相对原生 FP4 基线：固定长输入 TTFT P50 1528.47 → 846.53 ms（−44.62%），固定长生成 TPOT P50 9.056 → 5.909 ms（−34.75%），输出 10,705.53 → 15,912.62 token/s（+48.64%）。这些是固定负载对照；1× 数据集回放的过载基线不用于推导生产吞吐。

checkpoint 候选第一次启动的 P 在 AFS 只剩 1.6 TiB 时拒绝创建 5 TiB 磁盘 KV，未进入推理、无 cgroup OOM。用退出的 P 服务 ID、日志、进程状态和打开文件检查确认归属后，回收两套本轮基线的 5 TiB 缓存分片，AFS 恢复到约 11.5 TiB 可用。仅重新启动失败的 P，D 保持原版本和启动参数。后续每次退出旧 P 后，启动脚本只清理上一轮已退出的专用缓存目录。
