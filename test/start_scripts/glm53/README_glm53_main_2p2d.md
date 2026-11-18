# glm53_main：四个独立 PD Master 与共享缓存元数据

该分支基于 `main` 的 `608d4f02113d75be8f1abbb9dff0fd0c7b6ac614`，
移植 `glm53_optim` 的独立 Master / ConfigServer 元数据同步实现。
每台机器运行一个独立 PD Master，每个 Master 使用一个 HTTP worker。
ConfigServer 为四个 Master 转发有序缓存提示和负载快照；请求、逐 token 输出和 KV 传输沿用主干路径。
主干单机多 worker 的 RPYC 共享树和统计服务也继续可用。

## 四机角色

| 机器 | 控制面 | GPU 角色 |
| --- | --- | --- |
| `10.119.17.96` | ConfigServer、Master 0 | `p0` |
| `10.119.16.240` | Master 1 | `p1` |
| `10.119.16.90` | Master 2 | `d0` |
| `10.119.31.85` | Master 3 | `d1` |

P0/P1 构成 TP16/DP16 的 Prefill 组，D0/D1 构成独立的 TP16/DP16 Decode 组，
每台使用 8 张 GPU。`p0/p1/d0/d1` 指定组内 `node_rank`。
四台模型目录默认相同：

```text
/nvme/sensenova-flash-320b-v22-v28-average-fp8-e8m0
```

脚本保留源分支的 1M 上下文、BF16 SSM 和 FP8 KV 设置。Prefill 使用 MTP1/native，
Decode 使用 MTP3/Replay8 precompute。
使用主干当前接口：显式指定 BF16 模型 dtype，FP8 DSA 的 PD page size 为 32768，
page num 为 2。Replay 配置仅包含 `cache_len` 和 `compute_mode`。

## 启动入口

在各机器 `/lightllm` 下设置相同的 ConfigServer 可达地址；使用服务 DNS 时可替换此地址：

```bash
export CONFIG_SERVER_HOST=10.119.17.96
```

在第一台的独立终端运行 ConfigServer：

```bash
bash test/start_scripts/glm53/glm53_pd_2p2d.sh config
```

在四台各运行一次 `master`，默认监听各自的 60011 端口：

```bash
bash test/start_scripts/glm53/glm53_pd_2p2d.sh master
```

然后在对应机器的其他终端分别运行 `p0`、`p1`、`d0`、`d1`：

```bash
bash test/start_scripts/glm53/glm53_pd_2p2d.sh p0
```

`master0` 到 `master3` 也是可用角色，区别在日志名；在同一主机启动多实例时必须显式设置不同的 `MASTER_PORT`。
ConfigServer 对等端使用 `CONFIG_SERVER_HOST`，本地监听地址由 `CONFIG_SERVER_BIND_HOST` 控制，默认 `0.0.0.0`。
默认 ConfigServer 端口为 60088。可通过 `MODEL_DIR`、`MODEL_NAME`、`CPU_MODEL_DIR`、
`CPU_SOURCE`、`GPU_SOURCE`、`PYTHON_BIN`、`RUN_DIR` 等环境变量覆盖路径。
默认日志位于 `/var/log/lightllm/logs`；`--dry-run` 仅输出命令，不创建文件或启动进程。

## 同步与选点语义

- 四个 Master 必须使用相同的模型路径、模型名和采样 schema，才能进入同一个共享 namespace。
- 负载快照每 250ms 发布，超过 3 秒视为未知；保留本 Master 的即时计数，避免重复累加自己的快照。
- 前缀缓存提示带顺序号、观察时间和 P/DP 进程代次。重连回放缺失事件；无法补齐时清空旧树。
- 共享选点按预计未缓存字符数估算 Prefill 工作量，首 token、异常和取消均释放相同预占值。
- 长前缀保留在线 P/DP 的历史位置亲和；元数据失效时其余请求回退到 GPU 上报负载。
- 元数据队列和历史记录均有上限；慢订阅者重连，发布缓存提示不会阻塞请求处理。

缓存提示代表历史位置，不保证 KV 仍驻留；共享负载是调度估计，不是全局原子预占。

## 控制面验证

```bash
CUDA_VISIBLE_DEVICES= LIGHTLLM_JSON_SCHEMA_MAX_LENGTH=65536 \
  python -m pytest -q \
  unit_tests/server/test_pd_shared_cache.py \
  unit_tests/server/test_glm53_independent_masters.py \
  unit_tests/server/test_pd_cache_aware.py \
  unit_tests/server/test_pd_local_selection.py \
  unit_tests/server/test_prompt_cache_tree_service.py \
  unit_tests/server/test_pd_node_stats.py
```

独立 Master 端到端测试启动一个真实 ConfigServer、四个真实 CPU Master 和假 P/D，
覆盖注册、独立入口转发、请求 ID 唯一性与关闭；共享测试覆盖真实 WebSocket、重连、
进程重启隔离、过期快照和队列溢出。真实 GPU 模型推理需要单独验证。
