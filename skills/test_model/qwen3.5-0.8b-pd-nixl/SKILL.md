---
name: test-model-qwen3.5-0.8b-pd-nixl
description: >-
  LightLLM Qwen3.5-0.8B PD disaggregation over NIXL gsm8k: pd_master on 8089,
  prefill on 8001, decode on 8002. Supports TP1 and TP2 runs by setting
  TP / PREFILL_CUDA_DEVICES / DECODE_CUDA_DEVICES. Qwen3.5 has linear-attention
  state transfer; use --pd_kv_page_size 2048 and --pd_kv_page_num 16.
  lm_eval hits pd_master URL. Requires UCX/RDMA env, nvidia_peermem
  check, curl warmup before lm_eval, registration wait in pd_master.log, and
  summary.txt. Includes optional repeated-prompt decode cache probe for linear-att
  page-boundary behavior.
---

# Qwen3.5-0.8B **PD 分离（NIXL）** 本地 GSM8K 评测

**测试标识**：同一 **`MODEL_DIR`（Qwen3.5-0.8B）** 下拆三条 `api_server` 进程：
**`pd_master`**、**`prefill`**、**`decode`**。评测和 warmup 只访问
**`pd_master` 的 HTTP 端口 `8089`**。

Qwen3.5 与 Qwen3-8B 的关键差异：

| 项 | Qwen3.5-0.8B NIXL PD 要点 |
|---|---|
| linear-att 状态 | PD 传输除了 KV page，还会传 `linear_att_state` 特殊页 |
| NIXL page size | 建议固定 **`--pd_kv_page_size 2048`**；`1024` 可能不足以容纳 linear-att 状态 |
| page num | 建议 **`--pd_kv_page_num 16`** 起步，避免 page 池过大导致显存压力 |
| cache 判断 | repeated prompt 可能只在 prefill 侧命中，decode 侧不一定 decode-only 命中 |

## 日志目录

每轮使用独立 `LOG_DIR`，至少保留：

- `summary.txt`
- `pd_master.log`
- `prefill.log`
- `decode.log`
- `curl_warmup.log`
- `eval_gsm8k.log`

建议命名：

```bash
export LOG_DIR="/mtc/wzj/lightllm_dev2/LightLLM/test/benchmark/static_inference/log/qwen35_pd_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
```

## 启动前检查

1. **模型目录**：优先使用 `MODEL_DIR=/mtc/models/Qwen3.5-0.8B`；不存在时再改成本机实际路径。
2. **端口**：确认 `8089`、`8001`、`8002` 空闲。
3. **显卡**：TP1 需要 prefill/decode 各 1 张卡；TP2 需要 prefill/decode 各 2 张卡，互不重叠。
4. **代理**：启动服务和评测前清空 `http_proxy` / `https_proxy`；评测设置 `no_proxy`。
5. **UCX/RDMA**：prefill/decode 启动前设置 `UCX_NET_DEVICES`、`UCX_TLS`。本机若默认 UCX 打到 `mlx5_8` 报 `Address not valid`，可显式使用 `mlx5_0:1` 到 `mlx5_7:1`。
6. **nvidia_peermem**：运行本目录的 `check_nvidia_peermem.sh`，结果写入 `summary.txt`。
7. **MPS**：如需更稳定的高并发/传输性能，可在启动服务前开启 NVIDIA MPS，并把开启状态写入 `summary.txt`。

## 变量配置

### TP2 推荐配置

```bash
export MODEL_DIR=/mtc/models/Qwen3.5-0.8B
export MODEL_NAME='qwen/Qwen3.5-0.8B'
export TP=2
export PREFILL_CUDA_DEVICES='0,1'
export DECODE_CUDA_DEVICES='2,3'
export PD_KV_PAGE_SIZE=2048
export PD_KV_PAGE_NUM=16
export PD_MASTER_IP="$(hostname -I | awk '{print $1}')"
export HOST="${PD_MASTER_IP}"
```

### TP1 快速验证配置

```bash
export MODEL_DIR=/mtc/models/Qwen3.5-0.8B
export MODEL_NAME='qwen/Qwen3.5-0.8B'
export TP=1
export PREFILL_CUDA_DEVICES='4'
export DECODE_CUDA_DEVICES='5'
export PD_KV_PAGE_SIZE=2048
export PD_KV_PAGE_NUM=16
export PD_MASTER_IP="$(hostname -I | awk '{print $1}')"
export HOST="${PD_MASTER_IP}"
```

### UCX 示例

按本机拓扑调整，不要盲目照抄其它机器：

```bash
export UCX_NET_DEVICES='mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1'
export UCX_TLS=rc,cuda,gdr_copy
```

## 启动命令

先写入基础信息：

```bash
export http_proxy=
export https_proxy=
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,${PD_MASTER_IP}

{
  echo "MODEL_DIR=${MODEL_DIR}"
  echo "MODEL_NAME=${MODEL_NAME}"
  echo "TP=${TP}"
  echo "PREFILL_CUDA_DEVICES=${PREFILL_CUDA_DEVICES}"
  echo "DECODE_CUDA_DEVICES=${DECODE_CUDA_DEVICES}"
  echo "PD_KV_PAGE_SIZE=${PD_KV_PAGE_SIZE}"
  echo "PD_KV_PAGE_NUM=${PD_KV_PAGE_NUM}"
  echo "PD_MASTER_IP=${PD_MASTER_IP}"
  echo "HOST=${HOST}"
  echo "UCX_NET_DEVICES=${UCX_NET_DEVICES-}"
  echo "UCX_TLS=${UCX_TLS-}"
} | tee "${LOG_DIR}/summary.txt"

bash skills/test_model/qwen3.5-0.8b-pd-nixl/check_nvidia_peermem.sh >> "${LOG_DIR}/summary.txt" 2>&1
```

### 1. 启动 `pd_master`

```bash
nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --run_mode pd_master \
  --host "${PD_MASTER_IP}" \
  --port 8089 \
  >> "${LOG_DIR}/pd_master.log" 2>&1 &
```

等待 `8089` listen 后再启动节点。

### 2. 启动 `prefill`

```bash
LOADWORKER=18 CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_DEVICES}" \
nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --run_mode prefill \
  --tp "${TP}" \
  --dp 1 \
  --host "${HOST}" \
  --port 8001 \
  --disable_cudagraph \
  --pd_master_ip "${PD_MASTER_IP}" \
  --pd_master_port 8089 \
  --pd_kv_page_size "${PD_KV_PAGE_SIZE}" \
  --pd_kv_page_num "${PD_KV_PAGE_NUM}" \
  >> "${LOG_DIR}/prefill.log" 2>&1 &
```

### 3. 启动 `decode`

```bash
LOADWORKER=18 CUDA_VISIBLE_DEVICES="${DECODE_CUDA_DEVICES}" \
nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --run_mode decode \
  --tp "${TP}" \
  --dp 1 \
  --host "${HOST}" \
  --port 8002 \
  --pd_master_ip "${PD_MASTER_IP}" \
  --pd_master_port 8089 \
  --pd_kv_page_size "${PD_KV_PAGE_SIZE}" \
  --pd_kv_page_num "${PD_KV_PAGE_NUM}" \
  >> "${LOG_DIR}/decode.log" 2>&1 &
```

## 就绪判定

不要只看端口。必须等待 `pd_master.log` 同时出现：

```text
mode: prefill ... registed
mode: decode ... registed
```

可用命令：

```bash
rg 'mode: prefill .* registed|mode: decode .* registed|ERROR|Traceback|Exception' "${LOG_DIR}/pd_master.log" "${LOG_DIR}/prefill.log" "${LOG_DIR}/decode.log"
```

## Warmup

`lm_eval` 前必须先打一次 `pd_master`：

```bash
curl -sS -w "\nhttp_code:%{http_code}\n" -X POST "http://${PD_MASTER_IP}:8089/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"warmup\",\"max_tokens\":16,\"temperature\":0}" \
  | tee "${LOG_DIR}/curl_warmup.log"
```

期望 `http_code:200`。失败时先查 `pd_master.log` / `prefill.log` / `decode.log`，不要直接跑全量评测。

## GSM8K 评测

默认并发和 batch 使用 64，避免高并发掩盖关键问题；压测时再提高。

```bash
export http_proxy=
export https_proxy=
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,${PD_MASTER_IP}

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 \
lm_eval --model local-completions \
  --model_args "model=${MODEL_NAME},base_url=http://${PD_MASTER_IP}:8089/v1/completions,num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer=${MODEL_DIR}" \
  --tasks gsm8k \
  --batch_size 64 \
  --confirm_run_unsafe_code \
  >> "${LOG_DIR}/eval_gsm8k.log" 2>&1
```

提取结果：

```bash
rg -n 'flexible-extract|strict-match|exact_match|Traceback|ERROR|can not find waiting WRITE task|has_error=True' \
  "${LOG_DIR}/eval_gsm8k.log" "${LOG_DIR}/pd_master.log" "${LOG_DIR}/prefill.log" "${LOG_DIR}/decode.log" \
  | tee -a "${LOG_DIR}/summary.txt"
```

参考正常结果：

| 场景 | 参考精度 |
|---|---|
| TP1 NIXL PD | `flexible-extract exact_match ~= 0.332`，`strict-match exact_match ~= 0.327` |
| TP2 NIXL PD | `flexible-extract exact_match ~= 0.331`，`strict-match exact_match ~= 0.328` |

## 可选：decode-only cache 命中探针

这个探针用于确认重复 prompt 是否在 decode 节点全命中。Qwen3.5 的 linear-att cache 以
`linear_att_hash_page_size` 为粒度，默认 `512`。历史观察显示：

- prefill 侧会按 512 token 粒度逐步命中，例如 513 的第二次可命中 512。
- decode 侧可能仍为 `gpu cache hit: False`、`gpu_prompt_cache_len:0`。
- 只要 decode 未全命中，仍会出现 `recv WRITE request from prefill` 和 `linear_att_state` 传输。

### 简单重复 prompt

在同一套服务生命周期内连续请求两次相同 prompt：

```bash
PROMPT_FILE="${LOG_DIR}/repeat_prompt.txt"
python3 - <<'PY' "${MODEL_DIR}" "${PROMPT_FILE}"
from transformers import AutoTokenizer
import sys
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
target = 2049
s = "Qwen3.5 linear attention cache boundary probe. "
unit = " Repeatable cache probe sentence."
while len(tok.encode(s, add_special_tokens=False)) < target:
    s += unit
open(sys.argv[2], "w").write(s)
print(len(tok.encode(s, add_special_tokens=False)))
PY

for i in 1 2; do
  curl -sS -X POST "http://${PD_MASTER_IP}:8089/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":$(python3 -c 'import json,sys; print(json.dumps(open(sys.argv[1]).read()))' "${PROMPT_FILE}"),\"max_tokens\":4,\"temperature\":0}" \
    > "${LOG_DIR}/repeat_${i}.json"
  sleep 2
done
```

### 判定信号

```bash
rg -n 'gpu cache hit:|recv WRITE request from prefill|start WRITE to decode node|linear_att_state|trans task ret success' \
  "${LOG_DIR}/prefill.log" "${LOG_DIR}/decode.log" \
  | tee -a "${LOG_DIR}/summary.txt"
```

decode-only 全命中的期望信号：

| 日志 | 期望 |
|---|---|
| `decode.log` | `gpu cache hit: True` |
| `decode.log` | `gpu_prompt_cache_len` 接近 `prompt_tokens` 或至少 `input_len - cur_kv_len <= 1` |
| `decode.log` | 不再出现真实 `recv WRITE request from prefill` |
| `prefill.log` | 不再出现对应请求的 `start WRITE to decode node` |

如果 decode 仍是 `gpu cache hit: False gpu_prompt_cache_len:0`，则说明没有进入 decode-only 命中路径。

## 常见问题

| 现象 | 处理 |
|---|---|
| `NIXL_ERR_BACKEND` / `uct_iface_open(rc_verbs/mlx5_8:1) failed: Address not valid` | 显式设置可用 `UCX_NET_DEVICES`，例如避开 `mlx5_8/9` |
| `digest sent was rejected` | 多为快速重启后的共享内存 / multiprocessing authkey 残留；清理端口和残留 `lightllm::...` worker 后重启 |
| `can not find waiting WRITE task` | 检查 NIXL notify key、abort 日志、以及 `pd_io_struct.py` 中 key 是否包含进程本地 `req_idx` |
| 1024 page size 失败 | Qwen3.5 linear-att state 页可能放不下；使用 `--pd_kv_page_size 2048` |
| 第二次同 prompt 仍走 WRITE | 可能是 decode 侧没有建立可复用 cache，或 linear-att 尾块状态无法全命中 |

## 收尾

结束后释放本轮服务：

```bash
fuser -k 8089/tcp 8001/tcp 8002/tcp || true
```

如仍有显存占用，检查残留 worker：

```bash
ps -eo pid,ppid,stat,cmd | rg 'lightllm::|api_server|hypercorn'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

