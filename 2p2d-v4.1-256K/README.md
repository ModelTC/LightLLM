# V4.1 FP8：双机 P + 双机 D，256K

基于 `support_ds4_v41`。本目录 launcher/config 为部署命令的唯一来源，每个参数一行；部署时复制原文件并核对 SHA256，不另拼启动命令。

| 主机 | 角色 | rank | HTTP |
| --- | --- | --- | --- |
| 10.119.30.78 | P + PD master | 0 | 8019 / 60011 |
| 10.119.30.56 | P | 1 | 8019 |
| 10.119.29.251 | D | 0 | 8121 |
| 10.119.29.227 | D | 1 | 8121 |

每个 P/D 组为 DPEP16、nnodes2；每台注册 master，master mode=2p2d。P 使用 DeepEP V2，D 使用 legacy/IBGDA。主/草稿权重均为 `/mnt/vqa/models/DeepSeek-V4.1-Flash-1017-FP8`，DSpark3，max_req_total_len262144，mem_fraction0.75，KV自动profile。P batch/chunk8192、CPU cache1024GiB/主机、disk cache5120GiB/主机；D batch1024、running_max_req_size768/主机（96/rank）、graph batch96/rank，target验证图最多384行。内核和磁盘缓存保留节点本地目录。

运行目录见 config.txt。冻结源码放 `$RUN_DIR/src`，本目录脚本放 `$RUN_DIR/deploy`。先核对 GPU/旧进程/SysV/端口空闲、模型分片可读，再启动 master，随后四台并发启动；每个 launcher 在相同 shell 中设置 PATH/PYTHONPATH/nofile1048576。完整验收需要四节点健康、master2P2D就绪、实际文本/图片请求、KV/state传输和所有长期进程FD检查。

各机器执行：

```bash
bash "$RUN_DIR/deploy/launch_pd_master.sh"  # 仅 .30.78，先启动
bash "$RUN_DIR/deploy/launch_prefill.sh" 0  # .30.78
bash "$RUN_DIR/deploy/launch_prefill.sh" 1  # .30.56
bash "$RUN_DIR/deploy/launch_decode.sh" 0   # .29.251
bash "$RUN_DIR/deploy/launch_decode.sh" 1   # .29.227
```

本轮部署到上述四台节点；启动与验收进度追加到仓库根目录 test.md。

D节点使用隔离目录 `$RUN_DIR/deps/deepep-pr696` 中的历史自编译DeepEP PR696 wheel，不修改全局安装；wheel SHA256=e4981f5645c23b3d08d61eb6409503ec6f3a0c20a5eb3005eec1dd45f729ea9a。当前全局2.1.0+60d4403在D双机首次legacy dispatch复现illegal memory access；隔离依赖下16rank dispatch/combine、数值及graph检查通过。P仍使用全局DeepEP V2。

本轮外部入口为 `http://10.210.6.10:11046`，映射H100-msw的8020，再由本目录haproxy.cfg转发至P0的60011；10418/10066不修改。

P 开启 `--enable_dp_prompt_cache_fetch`，在同一节点的 DP rank 间复用前缀缓存；D 开启 `--mtp_dynamic_verify`，按置信度动态选择主模型验证行，draft 最大步数为3。P/D 的 `--mtp_step` 均为3：P 仅提交 draft KV，但当前 P/D 原样传输的 compressor ring 尺寸依赖该值，必须同步修改。D 的 dispatch capacity 为96×(3+1)=384，按 PR696 legacy layout 估算 RDMA buffer 约5.71GiB/卡，不包含其他 MoE 临时张量和 graph pool。

## 启动与退出优化

- D 设置 `--graph_split_batch_size 16`，grow 仍为16。DSpark3 动态验证下，每 rank 的 target/draft 档位由144/36减为84/21，共105档；最大384/288行不变。部分 batch 的 padding 增加，需在下一轮实际部署验证启动时间、显存、TPOT 与 TPM。
- Engram 构造只建立 CPU 表；单次加载者完成真实权重写入后，各 rank 才注册并创建 GPU view，取消每 rank 的前置逐页扫描。NUMA interleave 沿用已有开关，在首次写入之前设置；不会自动开启 HugeTLB。
- P 的 CPU KV 注册提前与模型初始化、加载重叠。模型进入 autotune/Graph 预热前等待完整注册，仍以全部1 TiB可用为就绪条件。这不是后台分阶段扩容，也不改变全局 page ID/容量或推理期搬运路径。
- V4.1 EP safetensors 加载在 `get_tensor()` 前过滤非本 rank 专家，保留初始 EPLB 冗余副本；dense、共享专家和 Engram 权重不被过滤。
- `LIGHTLLM_AUTOTUNE_CACHE_DIR` 与 `LIGHTLLM_ENGRAM_CACHE_DIR` 使用节点本地持久目录。Autotune 按 PyTorch/CUDA、Triton、GPU、kernel/config 源码区分，每节点 leader 原子保存调优结果；Engram CPU 元数据按 tokenizer 内容、相关配置和实现校验，并通过文件锁只生成一次。两者均不复用进程 CUDA Graph 或请求 KV。
- `model_startup` 日志按 rank、target/draft 拆分对象创建、KV容量、定制状态、权重/映射、host就绪、调优、预热、Graph与长度检查；记录 host wall time，不额外同步 GPU。注册/注销、Engram 填表和 CPU KV 触页另有计时。

正常退出应向每个 launcher 发 SIGTERM，并同时停止同一 DPEP 组的所有节点。当前非 overlap DP worker 在已有 control collective 中同步停止推理，join 推理线程并同步设备后，逐段注销 host memory；随后保持 GPU IPC owner 存活，launcher 先停止并等待传输等使用者，再结束 worker。此握手只确认 host memory 的安全注销，不代表新增了完整的 NIXL 请求 drain；传输进程仍由 launcher 终止。握手超时、未完成初始化、不支持该协议的 backend 和 SIGINT 都保留强制退出路径并报告剩余 PID。

日志区分 `serving_stopped`、`host_memory_released`、`workers_exited`。仍有活进程时不删除磁盘缓存、不报告退出成功，launcher 返回失败；共享内存仍由现有独立清理进程按 service namespace 回收。`workers_exited` 不应被当作 SysV 物理页已回收的单独证据，重启前仍需检查 attachment、GPU进程和 cgroup。

大池清理按每个 worker 的注销进展计时：只有成功的 `cudaHostUnregister` 才刷新进度，连续180秒没有进展才进入强制退出，避免总耗时超过180秒就打断仍在工作的清理线程。`host_memory_cleanup` 分别记录停止推理、GPU同步、注销阶段；每个注册对象另记字节数、分段数与注销耗时。主线程为 zombie 时仍检查其余线程，驱动清理线程未退出前不宣布完成、不提前清理 launcher 共享内存或释放权重租约。这些修改不保留1 TiB CPU KV池，也不保证注销或物理回收耗时下降；权重常驻的收益不能覆盖这段成本。

本轮只提交代码和配置，不重启当前服务。HOST_NUMA/FABRIC、CPU KV 分阶段扩容或 staging、处理后权重 artifact 尚未实现；这些需要单独容量、传输及整机验证，不能用局部测试代替。

## 可选的权重常驻

`--weight_cache_dir` 默认关闭。目前支持 CUDA SM90 上的不可变 V4.1 BF16/DeepGEMM FP8 权重及可选 DSpark；不支持 RL 更新、EPLB、memory saver 或其他量化方法。owner 通过现有模型/量化工厂加载处理后的 GPU 权重，并保留已填好的共享 Engram CPU 表。worker 使用 FakeTensor 构造权重骨架后绑定同卡 CUDA IPC tensors，保留 WeightPack、别名、stride 和加载状态；不会先分配整份 GPU 副本，也不会重读或再次处理权重。推理路径直接读 GPU tensors，没有逐请求 owner RPC。

首次启用前先停止旧服务并确认显存可用；已有 owner 的暖重启只停止 worker，保留 owner。owner 是独立于 API launcher 的进程组，必须先启动并等待每张卡打印 `resident_weights_ready`。使用同一份经过核对的 launcher，改变 Python 入口即可，避免另拼权重参数；冻结源码必须包含本功能。例如在每个 P 节点分别执行，rank 参数仍按原拓扑填写：

```bash
export LIGHTLLM_WEIGHT_CACHE_DIR=/root/.cache/lightllm-dsv41-opt/resident
# 独立会话中启动 owner；D 节点使用 launch_decode.sh。
LIGHTLLM_SERVER_MODULE=lightllm.server.weight_cache bash launch_prefill.sh 0
# owner 全部就绪后，在另一会话启动或重启服务 worker。
export LIGHTLLM_WEIGHT_CACHE_DIR=/root/.cache/lightllm-dsv41-opt/resident
bash launch_prefill.sh 0
```

目录必须位于本地文件系统、由运行用户独占（0700）；owner 与 worker 需共享 IPC、目录并具有可见的进程身份。每卡通过物理 GPU UUID 选择 socket。checkpoint 内容清单/文件元数据、TP/DP/rank、量化、MTP、PyTorch/CUDA/Triton 及权重布局源码不匹配时直接拒绝，不能一边保留旧 owner、一边退回磁盘加载出另一份权重。只修改推理方法或 kernel 可以继续复用；修改权重构造、加载或布局后需要停掉旧 owner 并重新预载。

```bash
python -m lightllm.server.weight_cache --weight_cache_dir "$LIGHTLLM_WEIGHT_CACHE_DIR" --weight_cache_action status
# 先停止相应服务的全部 worker，再归还 owner 的 GPU/CPU 资源。
python -m lightllm.server.weight_cache --weight_cache_dir "$LIGHTLLM_WEIGHT_CACHE_DIR" --weight_cache_action stop
```

status 给出 owner PID、GPU 权重字节数、Engram 共享表及活跃 worker PID。存在租约时 stop/SIGTERM 会拒绝释放；连接断开后还会等待实际 worker 退出，避免将 EOF 当作 CUDA context 已清理。owner 异常断开会令关联 worker 失败退出，不能继续重放旧 Graph。启动失败留下的 namespace/socket 应先核对进程与共享映射，再处理，不能自动清除不明或仍在使用的资源。

重启预检只允许 status 验证过的 owner 保留；旧 worker、请求 KV、传输进程仍须退出。owner 会常驻显存和 Engram 物理页，完全归还机器时必须另行停止。owner 的额外 CUDA context 开销由实际空闲显存计入 KV profile；target/draft 权重已全部常驻，因此不会再重复预留 MTP 权重。通信、CUDA Graph、请求状态以及 1 TiB CPU KV 池仍需初始化和注册，权重常驻不等于整个服务几秒就绪。
