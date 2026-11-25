# 在四台新 H100 节点部署 DeepSeek-V4.1-Flash-1017-FP8

这套服务需要四台各有 8 张 H100 的节点。两台组成一个 16-rank 的 Prefill 组，两台组成一个 16-rank 的 Decode 组。PD master 与 P0 同机。客户端访问 P0 的 60011 端口。

| 节点 | 运行进程 | 组内 node rank | 服务端口 |
| --- | --- | --- | --- |
| P0 | PD master 和 Prefill | P rank 0 | 60011、8019 |
| P1 | Prefill | P rank 1 | 8019 |
| D0 | Decode | D rank 0 | 8121 |
| D1 | Decode | D rank 1 | 8121 |

目录名保留了历史的 `256K`。当前默认上下文是 **1,048,576 tokens**。P 和 D 都用 DeepEP V2、NIXL 和 DSpark5。P 用 `hrrn_resume`，chunk 为 8192。D 用动态验证和 FlashInfer 采样，每台最多 1024 个请求，每 rank 最多 128 个请求、768 行验证输入。P/D 的 `mem_fraction` 分别为 0.75/0.77。三端使用 `last_chunk`，P 调度到最后一个 prefill chunk 时才申请 D 端 KV。

## 1. 准备机器、模型和运行环境

- 配置四个 SSH 别名 `v41-p0`、`v41-p1`、`v41-d0`、`v41-d1`，分别指向四台实际运行服务的环境。可在 SSH 配置里指定用户、端口和跳板。另记下四台服务网卡的内网 IP，供节点相互连接。
- 四台使用相同的已验证 V4.1 H100 运行环境。launcher 调用 `/opt/conda/bin/python`，依赖包括匹配的 CUDA、NCCL、NVSHMEM、DeepEP V2、DeepGEMM、NIXL、Humming、FlashInfer 和 P 的 `light_mem`。如已验证环境另有 `deps/` 覆盖包，也要按相同版本部署到相应节点。模型权重和源码归档需要单独准备。
- 在四台的相同绝对路径挂载转换好的 **1017-FP8 b128** checkpoint，主模型和 DSpark 草稿均从该目录读取。保留 tokenizer、encoding、视觉配置和完整权重分片。
- 四台之间需具备可用的 NCCL、DeepEP 和 NIXL 网络。除表中 HTTP 端口外，P 组用 28793、D 组用 28901 建组；PD master 的多个 HTTP worker 还会发布各自的连接端口。容器网络须能让对端直连这些地址。
- 每台 P 的 CPU KV 配额是 **512 GiB**，另需给模型加载、Engram、进程和系统留内存。每台 P 的磁盘缓存配额是 **4608 GiB**。将 `--disk_cache_dir` 放在有足够可用空间的本地盘；新部署使用独立、可写的缓存目录。

## 2. 在控制机准备一份部署副本

以下控制机命令在同一个 Bash 会话运行。起点是包含本 README 和三个 launcher 的目录。四台节点统一从 `6f3cb3c4b` 打包源码。这个版本包含延迟分配 D 端 KV 的 `last_chunk` 策略，以及 prefill 和 candidate indexer 的 Triton 内核复用优化。

```bash
set -euo pipefail
REPO=/mtc/wanzihao/lightllm-ds4-v41
DEPLOY=$(mktemp -d)
cp config.txt launch_pd_master.sh launch_prefill.sh launch_decode.sh "$DEPLOY/"
SOURCE_COMMIT=$(git -C "$REPO" rev-parse --verify '6f3cb3c4bab009c12056304c02d2a1ec63ebe8b8^{commit}')
git -C "$REPO" archive --format=tar.gz -o "$DEPLOY/source.tar.gz" "$SOURCE_COMMIT"
```

编辑 **`$DEPLOY/config.txt`**。下表中的 P0、P1、D0、D1 都替换成服务内网 IP，SSH 别名只用于登录。

| 配置 | 新集群填写值 |
| --- | --- |
| `PD_MASTER_IP`、`PREFILL_NCCL_HOST` | P0 IP |
| `DECODE_NCCL_HOST` | D0 IP |
| `LIGHTLLM_FIXED_DECODE_HOSTS` | `D0_IP,D1_IP`，用逗号分隔 |
| `NO_PROXY` | 将旧四机 IP 替换成新四机 IP，保留 localhost、127.0.0.1 和原有代理例外 |
| `MODEL_DIR` | 四机均可读的 checkpoint 绝对路径；`MTP_DRAFT_MODEL_DIR` 继续引用它 |
| `RUN_DIR` | 为使用下面的命令，将开头按角色选择旧目录的整个 `case ... esac` 替换为一行 `export RUN_DIR="/srv/lightllm-v41"`。保留下方配置角色环境变量的第二个 `case` |
| `SOURCE_DIR`、`LOG_DIR` | 分别使用现有的 `$RUN_DIR/source`、`$RUN_DIR/service` |

在 **`$DEPLOY/launch_prefill.sh`** 中设置新机实际的 `--disk_cache_dir`。确认运行账号能写 `/srv/lightllm-v41`、缓存及编译目录。默认使用 1M 时清除外部遗留的 `MAX_REQ_TOTAL_LEN` 覆盖。

## 3. 分发文件并做启动前检查

在控制机执行。下面选用新的运行目录；如果该目录已有服务，先选另一个空目录，并同步修改配置和下文命令。

```bash
cd "$DEPLOY"
sha256sum config.txt launch_pd_master.sh launch_prefill.sh launch_decode.sh source.tar.gz > SHA256SUMS
for host in v41-p0 v41-p1 v41-d0 v41-d1; do
    ssh "$host" 'test ! -e /srv/lightllm-v41 && mkdir -p /srv/lightllm-v41/{launch,source,service,deps}'
    scp config.txt launch_*.sh source.tar.gz SHA256SUMS "$host:/srv/lightllm-v41/launch/"
    ssh "$host" 'cd /srv/lightllm-v41/launch && sha256sum -c SHA256SUMS && tar -xzf source.tar.gz -C ../source'
done
```

在每台节点上运行下面的预检。P0/P1 设 `ROLE=prefill`，D0/D1 设 `ROLE=decode`。在 P 上还要对实际磁盘缓存挂载点执行 `df -h`。检查 GPU 没有旧服务占用、相关端口空闲、内存限额及剩余量足够；共享内存段先确认归属和 attachment。

```bash
set -euo pipefail
ROLE=prefill  # D0、D1 改为 decode
source /srv/lightllm-v41/launch/config.txt "$ROLE"
ulimit -Sn 1048576
nvidia-smi
ss -lnt
free -h
cat /proc/sysvipc/shm
if test -r /sys/fs/cgroup/memory.max; then
    cat /sys/fs/cgroup/memory.{current,max,events}
fi
/opt/conda/bin/python - "$ROLE" <<'PY'
import json, os, sys
from pathlib import Path
import lightllm, torch, deep_ep, deep_gemm
from nixl._api import nixl_agent
from lightllm.server.api_cli import make_argument_parser

source = Path(os.environ["SOURCE_DIR"]).resolve()
assert Path(lightllm.__file__).resolve().parent.parent == source
assert "--pd_decode_allocation_policy" in make_argument_parser().format_help()
assert torch.cuda.device_count() == 8 and callable(deep_ep.ElasticBuffer)
if sys.argv[1] == "prefill":
    import humming
    from light_mem import PyLocalCacheService
else:
    import flashinfer
    assert callable(flashinfer.sampling.top_k_top_p_sampling_from_probs)
model = Path(os.environ["MODEL_DIR"])
cfg = json.loads((model / "config.json").read_text())
text = cfg.get("text_config") or cfg
assert text["max_position_embeddings"] >= int(os.environ["MAX_REQ_TOTAL_LEN"])
assert text["dspark_block_size"] >= 5
index = json.loads((model / "model.safetensors.index.json").read_text())
shards = set(index["weight_map"].values())
for name in [*shards, "tokenizer.json", "encoding/encoding.py"]:
    with (model / name).open("rb") as f:
        assert f.read(1), name
print("OK", source, model, len(shards), "shards")
PY
```

## 4. 先启动 master，再并发启动四个 P/D 节点

在控制机执行。先让 P0 的 master 提供 `/get_server_info`；完整的 `/health` 要等 P/D 都注册。各角色日志独立，路径在各自机器的 `/srv/lightllm-v41/service/`。

```bash
ssh v41-p0 'cd /srv/lightllm-v41 && (nohup setsid bash launch/launch_pd_master.sh </dev/null >service/master.log 2>&1 &)'
ssh v41-p0 'for i in {1..60}; do
    curl --noproxy "*" -fsS -m 2 http://127.0.0.1:60011/get_server_info >/dev/null && exit 0
    sleep 2
done
tail -n 40 /srv/lightllm-v41/service/master.log
exit 1'

pids=()
for spec in 'v41-p0 prefill 0' 'v41-p1 prefill 1' 'v41-d0 decode 0' 'v41-d1 decode 1'; do
    read -r host role rank <<< "$spec"
    ssh "$host" "cd /srv/lightllm-v41 && (nohup setsid bash launch/launch_${role}.sh $rank </dev/null >service/${role}${rank}.log 2>&1 &)" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
```

SSH 命令返回表示启动命令已发出。可以用 `ssh v41-d0 'tail -f /srv/lightllm-v41/service/decode0.log'` 看 D 的权重加载、autotune 和 CUDA Graph 预热；P0 对应 `prefill0.log`。若某个 rank 提前退出，先看该节点最早的错误和 cgroup 的 `oom_kill` 计数。

## 5. 验收节点和实际请求

在控制机检查四个本地 HTTP 服务，再看 master 的拓扑。预期各节点都返回 HTTP 200，master 的 `registered_prefill_nodes` 和 `registered_decode_nodes` 均为 2，`pd_nodes_ready`、`pd_nodes_healthy`、`inference_healthy` 均为 true。

```bash
for host in v41-p0 v41-p1; do ssh "$host" 'curl --noproxy "*" -fsS http://127.0.0.1:8019/health'; done
for host in v41-d0 v41-d1; do ssh "$host" 'curl --noproxy "*" -fsS http://127.0.0.1:8121/health'; done
ssh v41-p0 'curl --noproxy "*" -fsS http://127.0.0.1:60011/health'
```

在 P0 执行一次真实对话。确认返回了多个生成 token，并检查同一时间的 P/D 和 master 日志，确认 KV/state 交接与后续 decode 成功。若提供视觉服务，再用一个已知内容的小图片测试真实 `image_url` 请求；如另有对外代理，通过客户实际访问的地址重复验收。

```bash
curl --noproxy '*' -fsS --max-time 120 http://127.0.0.1:60011/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"DeepSeek-V4.1-Flash-FP8","messages":[{"role":"user","content":"请依次写出从一到十。"}],"max_tokens":64,"stream":false}'
```

交付前核对每台长期运行的 API、router、detokenizer 和 KV 传输进程。用 `ps -eo pid,ppid,args` 定位 PID，从 `/proc/<PID>/limits` 查看 `Max open files`（soft 应为 1048576），用 `ls /proc/<PID>/fd | wc -l` 查看当前使用数。若出现 HTTP 500、缺失 WRITE 任务、CUDA/NCCL/NIXL 错误或 OOM，保留本次日志并检查对应节点。
