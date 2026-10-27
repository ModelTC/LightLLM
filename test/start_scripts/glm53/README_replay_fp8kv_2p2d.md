# Replay8 / FP8KV 四机 2P2D 启动

本入口整理自 2026-10-27 已运行的 1M 上下文部署。它只启动指定角色，不跨机器停止进程，不预热、不发模型请求，也不创建评测任务。

## 代码分支

- GPU：`codex/replay-fp8kv-20261027`。当前服务加载的推理代码提交为 `1e08c1dbdb8d8f70f2d6c930f036e46250356dfc`。
- CPU：`codex/replay-fp8kv-runtime-20261027`。当前服务加载的提交为 `adfa40d726b0db5db46bf8424f22509a3d4375a9`。
- 本次整理只增加启动脚本、说明和运行产物忽略规则；不改变推理代码和正在运行的服务。

GPU 分支包含 Replay8、FP8KV、Gloo DP 控制、Decode DeepGEMM 满 SM、保留 GLM 语义的 fused-W2 量化、动态长度编译修复、Hopper FlashMLA 兼容，以及最新 cache-aware 修复。CPU 分支保留对应调度修复和 CPU 运行依赖布局。

cache-aware 对匹配前缀至少 65,536 字符、预计匹配率至少 80% 且候选在线的请求保留原 P/DP；其他请求比较预计新增冷计算成本。共享负载计数使用同一估算值，并在首 token、异常和取消路径释放。字符前缀估计不等于实时 KV 驻留状态。

## 角色与 node_rank

| 角色 | 机器 | 组内 node_rank | NCCL 主机 / 端口 | API 端口 |
|---|---|---:|---|---:|
| cpu | 10.119.16.95 | 不适用 | 不适用 | 60011 |
| p0 | 10.119.25.88 | 0 | 10.119.25.88:29501 | 8019 |
| p1 | 10.119.25.80 | 1 | 10.119.25.88:29501 | 8019 |
| d0 | 10.119.25.76 | 0 | 10.119.25.76:29502 | 8121 |
| d1 | 10.119.25.72 | 1 | 10.119.25.76:29502 | 8121 |

这里的 2P2D 指四台 GPU 机器。每个 P 或 D 组分别使用 `--nnodes 2 --tp 16 --dp 16 --enable_ep_moe`，每台 8 张 H100，共 32 卡；不是一个 `nnodes=4` 的组。

脚本接受 `p0/p1/d0/d1`，自动写入对应的 `--node_rank 0/1`，无需再手工传 `node0/node1` 或 `node_rank`。直接调用底层 API server 时，当前两机并行组仍需要正确的 rank；两台都省略会默认成 0。

已对照 `support_ds4` 的 `1e509707c4fff7e58952499a30067da2acfa656b` 与 `nsx_support_ds4` 的 `8cf1f7e48df7dd27a2f432cf9522ab1331c1b5db`：`StartArgs.node_rank` 默认 0，Router 使用它计算全局 rank，多机 EP 示例仍显式传 0/1。每台独立 `nnodes=1` 时可以都用默认 0，但那是不同的并行拓扑。

## 使用

先在相应机器、相应分支预览命令：

```bash
bash test/start_scripts/glm53/glm53_pd_2p2d.sh cpu --dry-run
bash test/start_scripts/glm53/glm53_pd_2p2d.sh p0 --dry-run
bash test/start_scripts/glm53/glm53_pd_2p2d.sh p1 --dry-run
bash test/start_scripts/glm53/glm53_pd_2p2d.sh d0 --dry-run
bash test/start_scripts/glm53/glm53_pd_2p2d.sh d1 --dry-run
```

`--dry-run` 只打印角色、源码目录、日志路径和完整命令，不写运行文件、不检查网络、不导入模型。

实际启动时，在每台机器分别去掉 `--dry-run`。先启动 CPU Master，再成对启动 P0/P1 和 D0/D1；不要等 rank 0 服务完全就绪后才启动 rank 1。入口在前台运行，可由现有进程监督器或终端会话管理；它不负责自动重启。

入口带每角色文件锁，启动前拒绝已占用的 API 端口，不会替你停止旧进程。

## 已保留的实参

- 1,048,576 上下文；MTP 3，`eagle_with_att`；FP8 权重 `fp8w8a8-b128-deepgemm`；KV 为 `fp8kv_dsa`，SSM 为 BF16。
- Replay8：`cache_len=8`、`compute_mode=precompute`、`fold_method=raw`；小页 2048，linear attention cache 64。
- P：batch tokens 4096、chunk 2048、请求容量 128、显存比例 0.75、关闭 CUDA Graph；CPU 缓存 1024 GiB/台，磁盘缓存 10240 GiB/台。
- D：batch tokens 1024、请求容量 256、显存比例 0.85、KV 上限 2,800,000 tokens/卡、graph batch 16。启用 profiler 接口不会自动开始采集。
- CPU：4 workers、cache-aware、资源等待 200 秒、`2p2d`；GPU 传输等待 600 秒；Schema 上限 65536；GLM47 工具解析和 GLM45 推理解析。
- Master/P 握手 415/430 秒由对应代码分支实现。

P 启动时自动分配的实际 GPU KV 容量曾核验为 2,380,171 tokens/卡，CPU 缓存 5046 个 16384-token 页/台；它是启动测量值，不是新增固定上限。

## 路径和依赖

默认路径与本次部署一致：

```text
CPU_SOURCE=/opt/lightllm-pd-master-60011-20261026/tuned
GPU_SOURCE=/mnt/niushengxiao/LightLLM-replay-fp8kv-20261027
CPU 日志：/opt/lightllm-pd-master-60011-20261026/cache-aware-20261027-restart/logs/cpu.log
GPU 日志：$GPU_SOURCE/outputs/cache-aware-1m-20261027-restart/logs/{p0,p1,d0,d1}.log
PID：对应 RUN_DIR/pids/<角色>.pid
```

换目录时使用 `CPU_SOURCE`、`GPU_SOURCE`、`CPU_MODEL_DIR`、`MODEL_DIR`、`RUN_DIR` 等环境变量覆盖。`PD_MASTER_IP/PORT` 和四个角色 IP 也可覆盖，但同一组机器必须使用一致的地址配置。

运行依赖不打包进 Git。默认 `PYTHONPATH` 保留已验证的 CPU deps、GPU `runtime-deps-complete`、Cutlass 与 cubin 路径。换环境时用 `CPU_EXTRA_PYTHONPATH` / `GPU_EXTRA_PYTHONPATH` 指向已准备好的兼容依赖。各 GPU 主机应可见 8 张 GPU。

历史 `DISK_CACHE_NAME=1p1d-p0` 被保留以兼容现有缓存命名空间；该字符串不决定 P/D 拓扑，拓扑由启动参数决定。

日志、模型、缓存、trace、隔离依赖和 API Key 不随本脚本提交。
