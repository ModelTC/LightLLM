# glm53_optim：Replay8 / FP8KV / 四个独立 PD Master

`glm53_optim` 基于 `support_ds4` 的 `1e509707c4fff7e58952499a30067da2acfa656b`。
移植来源为 `codex/replay-fp8kv-20261027`，以 `cb40809f` 为来源快照。
该脚本不修改正在运行的服务，不包含停止其他进程、预热或发起评测的操作。

## 控制面

```text
客户端 :60011 -> HAProxy -> PD Master 0 :60111
                        -> PD Master 1 :60112
                        -> PD Master 2 :60113
                        -> PD Master 3 :60114
                              |
                      Config Server :60088
                              |
                   P0/P1 :8019、D0/D1 :8121
```

四个 Master 分别运行独立的 `api_server --run_mode pd_master --httpserver_workers 1`，
各自有进程监督、请求 ID 生成器、日志和指标端口。P/D 通过 Config Server 发现并连接全部
Master。缓存历史和待 Prefill 工作量通过 `support_ds4` 的 metadata relay 跨 Master 同步。
HAProxy 只负责 HTTP/SSE 分流，`cpu` 角色不会进入 `httpserver_workers=4` 的内置启动流程。

四个 Master 的 `/metrics` 是各自的统计；按 Master 求和即可，不要再叠加 P/D 的同一请求。
metadata 负载快照每 250ms 发布、超过 3 秒视为未知，属于调度估计，不是全局原子预占。

## 代码与依赖

CPU 默认源码路径：`/opt/lightllm-pd-master-60011-20261026/glm53_optim`。
GPU 默认源码路径：`/mnt/niushengxiao/LightLLM-glm53-optim-20261027`。
需要在这两个目录准备此分支的同一版本；本次仅交付分支和脚本，不自动替换线上源码或部署。

首次准备独立源码时，使用空的新目录：

```bash
git clone --single-branch --branch glm53_optim https://github.com/ModelTC/LightLLM.git /path/to/new/LightLLM
```

脚本提供本机已验证环境的路径默认值；使用其他目录时覆盖 `CPU_SOURCE` / `GPU_SOURCE`。
权重、tokenizer、HAProxy、CUDA、Triton、FlashMLA、DeepGEMM、DeepEP、NIXL 等运行依赖
不打包进 Git。GPU 默认继续引用原部署的 `runtime-deps-complete` 等隔离依赖目录；
用 `GPU_RUNTIME_ROOT` 或 `GPU_EXTRA_PYTHONPATH` 可以覆盖。CPU 依赖使用
`/opt/lightllm-pd-master-60011-20261026/deps`，也可设置 `CPU_EXTRA_PYTHONPATH`。

不要把新分支的 `PYTHONPATH` 和旧源码混排。启动前可执行 `--dry-run` 核对源码目录、命令和日志路径，
该模式不加载模型、不检查端口、不写运行目录。

## 启动

先在 CPU 主机启动控制面，再在四台 GPU 主机各运行一个角色。CPU 启动需要 Bash >=4.3、
`flock` 和 `haproxy`；可用 `HAPROXY_BIN` 指定二进制路径。
`cpu` 会先确认六个控制端口均空闲；端口被占用时退出，绝不停止占用它的进程。

```bash
# CPU 10.119.16.95；在此分支的源码目录执行
bash test/start_scripts/glm53/glm53_pd_2p2d.sh cpu --dry-run
bash test/start_scripts/glm53/glm53_pd_2p2d.sh cpu

# GPU 各自主机，在同版本源码目录执行
bash test/start_scripts/glm53/glm53_pd_2p2d.sh p0   # 10.119.25.88
bash test/start_scripts/glm53/glm53_pd_2p2d.sh p1   # 10.119.25.80
bash test/start_scripts/glm53/glm53_pd_2p2d.sh d0   # 10.119.25.76
bash test/start_scripts/glm53/glm53_pd_2p2d.sh d1   # 10.119.25.72
```

`cpu` 前台监督 Config Server、四个 Master 和代理，任一子进程退出即清理本次启动的其余控制进程。
Ctrl-C 只结束它启动的控制进程，不操作 GPU 服务或其他部署。也可分别运行 `config`、
`master0`、`master1`、`master2`、`master3`、`proxy` 角色，由外部进程管理器监督。

`p0/p1` 是一个 16 卡 EP 组，`d0/d1` 是另一个 16 卡 EP 组，每台 8 卡。
底层仍使用 `nnodes=2` 和组内 `node_rank=0/1`；脚本根据角色填写，操作者无需手传 rank。
独立 Master 的发现机制与 GPU 的分布式 rank 是两个不同层次。

主要 GPU 参数保持：1M 上下文、MTP3、BF16 SSM、Replay8/precompute/raw、`fp8kv_dsa`、
P `running_max_req_size=128` / D `256`（每台总槽位；8 个本地 DP 分别为 16 / 32）、
P `.75` / D `.85`、D 每卡 KV 上限 `2800000`、P 每台 CPU 缓存 `1024 GiB`、
16384-token 大页 / 2048-token 小页、`linear_att_cache_size=64`、GLM47 工具解析 / GLM45 推理解析。
P 的自动 KV 容量应以新分支实际启动日志为准，旧分支的自动容量不能当作本分支已验证值。
CPU 资源等待 200 秒、Master/P 外层握手 415/430 秒、GPU KV 传输 600 秒；Schema 上限 65536。
`DISK_CACHE_NAME=1p1d-p0` 沿用原缓存命名，不控制部署拓扑。

## Cache-aware 策略和测量

- 抽样前缀匹配超过 50% 才作为候选，并校验 P/DP 的当前进程代次；已重启的同地址节点不复用旧提示。
- 匹配字符数至少 65536，且匹配比例至少 80%，优先保留原 P/DP，即使另一个 P 暂时空闲。
- 其余请求比较已有预计未缓存工作量加本次冷计算成本。预占、共享负载发布和释放使用同一字符估算值。
- 首 token、异常、取消都会释放预占；metadata 暂不可用时，长前缀仍保留在线节点亲和，其余按 GPU 上报负载回退。
- 新 namespace `sampled_chars_v2_uncached_prefill` 防止新旧估算单位混用。

这是历史位置亲和，不是 KV 驻留证明。实际 token 缓存命中由 P/D 回传元数据判断，日志包含：

```text
pd_cache_selection origin_req_id:... p_node:... candidate:... reason:... estimated_hit_rate:... reserved_prefill_chars:...
pd_dispatch origin_req_id:... block_req_id:... p_node:... d_node:... segment:...
pd_cache_result origin_req_id:... p_node:... prompt_tokens:... actual_cache_tokens:... actual_hit_rate:... ttft_ms:...
```

新增测量只记录数字、时间、原因以及请求/节点 ID，不记录 prompt、采样 key 或生成内容。
按时间窗口汇总时，以 `sum(actual_cache_tokens) / sum(prompt_tokens)` 计算 token 加权实际命中，
按 `p_node` 保留 DP 维度，用 `origin_req_id` 去重。不把关闭流的 segment timing 自动当作完整成功。
无请求样本时报告无样本；业务失败、取消、维护期错误分别统计。

## 移植范围和验证边界

移入 ReplaySSM 的 GDN/KDA、状态提交与 checkpoint/PD 传输接口、FP8KV MLA/index KV
存储和读写、Hopper FlashMLA 适配、动态长度编译复用，以及保持 GLM clamp 语义的共享 FFN
激活与 FP8 输入量化融合。保留 `support_ds4` 已有的 Gloo 控制、Decode 满 SM、传输页隔离、
NIXL 恢复和节点可用性过滤；没有用旧版本覆盖这些实现。

CPU 测试覆盖四 Master 的 metadata 同步、长前缀亲和、代次失效、待计算量释放、首 token
缓存测量、超时，以及独立控制面使用假 P/D 协议节点的启动/转发/退出。
CUDA 数值回归保留在 `unit_tests/models/glm5_next/test_fp8_kv.py`、
`unit_tests/common/test_glm_swiglu_quant_mm.py` 和 ReplaySSM 测试中。
本次迁移没有启动真实 32 卡模型或复跑性能评测，性能和实际显存容量仍需部署后实测。

独立控制面集成测试可单独运行，使用临时端口、微型 tokenizer 和假 P/D，无真实模型加载：

```bash
CUDA_VISIBLE_DEVICES= LIGHTLLM_JSON_SCHEMA_MAX_LENGTH=65536 \
  python -m pytest -q unit_tests/server/test_glm53_independent_masters.py
```

Schema 长度会改变跨进程采样参数结构的布局，测试驱动和所有服务实例必须一致。
构建镜像继续保留 support_ds4 的原生依赖版本，未沿用旧 FP8 补丁中的 FlashMLA 镜像回退。
