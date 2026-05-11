---
name: test-model-qwen3-8b-pd-nixl
description: >-
  LightLLM Qwen3-8b PD disaggregation over NIXL gsm8k: pd_master on 8089, nixl_prefill on 8001,
  nixl_decode on 8002, tp 2 each. Assign four GPUs via nvidia-smi then export
  PREFILL_CUDA_DEVICES / DECODE_CUDA_DEVICES (no fixed card IDs; no complex shell automation).
  UCX_NET_DEVICES and TLS for RDMA per cluster. lm_eval hits pd_master URL. HOST vs
  PD_MASTER_IP when co-located. Before lm_eval, must POST one completion via curl to
  pd_master for warmup verification. Requires LOG_DIR, MODEL_DIR, proxy cleared, no_proxy,
  summary.txt. Use for PD NIXL-style separation tests.
---

# Qwen3-8B **PD 分离（NIXL）**（`pd_master` + `nixl_prefill` + `nixl_decode`）本地 GSM8K 评测

**测试标识**：同一 **`--model_dir`**（Qwen3-8B）下拆 **三条** `api_server` 进程——**调度/入口（`pd_master`）**、**`nixl_prefill` 节点**、**`nixl_decode` 节点**；评测 **`lm_eval`** 只访问 **`pd_master` 的 HTTP 端口（8089）**。与 **NCCL 版 PD**（`prefill` / `decode`）区分之处在于 **`--run_mode`** 与 **prefill/decode 前须配置 UCX/RDMA 环境变量**。

**端口约定**：**`pd_master`：`8089`**；**prefill：`8001`**；**decode：`8002`**。启动与就绪探测须覆盖这三处（以及日志中的 PD 注册/报错信息）。

**绑定 IP（`HOST` / `PD_MASTER_IP`）**：各进程的 **`--host`** 表示 **本服务监听绑定的 IP**。当 **`pd_master`、`nixl_prefill`、`nixl_decode` 部署在同一台机器上时**，三者使用的绑定 IP **相同**：可 **`export HOST="${PD_MASTER_IP}"`**；**`lm_eval` 的 `base_url` 仍指向 `pd_master`**。

整轮产物落在**同一日志目录**，写入 **`summary.txt`** 与各进程日志；**不要**写聚合启动脚本，按「启动说明」逐条手动启动并在后台落盘。

## 日志目录（含 `summary.txt`）

- 每次评测先选定或新建**一个日志目录**（例如带时间戳或任务名），与其它测试轮次分开。
- **三个 `api_server` 的标准输出/错误**分别写入该目录，建议命名：**`pd_master.log`**、**`prefill.log`**、**`decode.log`**（文件名可沿用习惯，与 NCCL 测试一致便于对比）。
- **`summary.txt` 固定放在该日志目录下**，汇总：三台进程的启动参数摘要、端口与就绪情况、**UCX 配置要点**、`lm_eval` 关键结果、失败原因与最终结论。
- **`eval_gsm8k.log`**：`lm_eval` 终端输出；**`curl_warmup.log`**：测试前 **`curl`** 打 **`pd_master`** 的留档（建议）；**`summary.txt`** 仍为**总览结论**。

## 启动说明

本节包含：启动前检查 → 可变项说明 → 显卡分配 → UCX → **按顺序**三条完整 server 命令 → **curl warmup** → 评测命令。

### 启动前检查

1. **显卡**：prefill / decode 各需 **2 张物理 GPU**（**`--tp 2`**），共 **4 张互不重复**。**不要写死卡号**：先 **`nvidia-smi`**（见「显卡分配」），再 **`export PREFILL_CUDA_DEVICES`**、**`DECODE_CUDA_DEVICES`**。
2. **端口**：**`8089`、`8001`、`8002`** 均须未被占用。
3. **网络 / IP**：**`HOST`** 与 **`PD_MASTER_IP`** 约定同 NCCL PD skill；单机三进程 **`export HOST="${PD_MASTER_IP}"`**。
4. **代理**：启动 **任一 server 前**将 **`http_proxy` / `https_proxy` 置空**；评测使用 **`no_proxy`**（见评测命令）。
5. **RDMA / UCX**：prefill 与 decode 进程在启动 Python 前须设置 **`UCX_NET_DEVICES`**（及可选 **`UCX_LOG_LEVEL`**、**`UCX_TLS`**），取值依赖本机 **`ibv_devinfo`** 与机房拓扑（见「UCX / RDMA」）；**不要**默认照抄他机上的设备名或排除列表。

### 启动服务的命令模板（可变项）

| 可变项 | 含义 |
|--------|------|
| `LOG_DIR` | 本轮日志根目录；`export LOG_DIR=…`。 |
| `MODEL_DIR` | **`--model_dir`**；`lm_eval` 的 **`tokenizer` 须与此路径一致**。 |
| `PD_MASTER_IP` | **`pd_master` 的 `--host`**；**`lm_eval` 的 `base_url` 主机**。 |
| `HOST` | **`nixl_prefill` / `nixl_decode` 的 `--host`**。同机时 **`export HOST="${PD_MASTER_IP}"`**。 |
| `PREFILL_CUDA_DEVICES` | prefill 的 **`CUDA_VISIBLE_DEVICES`**（两张物理索引）；**`nvidia-smi` 后 export**。 |
| `DECODE_CUDA_DEVICES` | decode 的 **`CUDA_VISIBLE_DEVICES`**；与 prefill **四卡互不重复**。 |
| `UCX_NET_DEVICES` | UCX 使用的 HCA 列表，形如 `mlx5_0:1,mlx5_1:1`；**按本机 `ibv_devinfo` 与规划填写**。 |
| `UCX_LOG_LEVEL` / `UCX_TLS` | 常见为 **`info`** 与 **`rc,cuda,gdr_copy`**；可按环境调整。 |

开跑前导出示例：

```bash
export LOG_DIR='〈日志根目录〉'
export MODEL_DIR='〈Qwen3-8B 模型目录〉'
export PD_MASTER_IP='〈本机绑定 IP〉'
export HOST="${PD_MASTER_IP}"
export UCX_NET_DEVICES='〈按 ibv_devinfo 填写，逗号分隔 port :1〉'
export UCX_LOG_LEVEL=info
export UCX_TLS=rc,cuda,gdr_copy
```

### 显卡分配（`nvidia-smi` + 人工/Agent 决策，不用复杂脚本）

与 **qwen3-8b-pd-nccl** skill 相同：**prefill**、**decode** 各 **2** 张 GPU，共 **4** 张互不重复。

1. 执行 **`nvidia-smi`**（可选用 `--query-gpu=index,name,memory.used,memory.free --format=csv`）。
2. 由执行者选定哪 2 张给 prefill、哪 2 张给 decode（不重叠）。
3. **`export PREFILL_CUDA_DEVICES='…','…'`**、**`export DECODE_CUDA_DEVICES='…','…'`**。
4. 将选卡依据记入 **`summary.txt`**。

**禁止**：为选卡编写 **awk / mapfile / 长段 bash** 自动化；以 **`nvidia-smi` 事实 + 明确决策**为准。

### UCX / RDMA（NIXL 传输）

- **`UCX_NET_DEVICES`**：须覆盖本进程要用的 **RDMA 设备**；是否排除某些 HCA（例如数据面网卡）由**本机拓扑**决定，在 **`summary.txt`** 中写明依据。
- **`UCX_TLS`**：常见 **`rc,cuda,gdr_copy`**；若环境不支持再按报错调整。

### 1）启动 `pd_master`（须最先就绪监听）

```bash
export http_proxy=
export https_proxy=

nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --run_mode pd_master \
  --host "${PD_MASTER_IP}" \
  --port 8089 \
  >> "${LOG_DIR}/pd_master.log" 2>&1 &
```

### 2）启动 `nixl_prefill` 节点

**须在 `pd_master` 就绪后**再启动。启动前已完成 **`nvidia-smi` 决策**并 **`export PREFILL_CUDA_DEVICES`**，且已设置 **UCX**。

```bash
export http_proxy=
export https_proxy=

LOADWORKER=18 CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_DEVICES}" \
nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --run_mode nixl_prefill \
  --tp 2 \
  --dp 1 \
  --host "${HOST}" \
  --port 8001 \
  --disable_cudagraph \
  --pd_master_ip "${PD_MASTER_IP}" \
  --pd_master_port 8089 \
  >> "${LOG_DIR}/prefill.log" 2>&1 &
```

（若需显式传入 UCX，可在同一 shell 中于本块之前 **`export UCX_NET_DEVICES`** 等；**`nohup` 会继承当前 shell 的环境变量**。）

### 3）启动 `nixl_decode` 节点

启动前 **`export DECODE_CUDA_DEVICES`**，并确保 **UCX** 已设置。

```bash
export http_proxy=
export https_proxy=
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,${PD_MASTER_IP}

LOADWORKER=18 CUDA_VISIBLE_DEVICES="${DECODE_CUDA_DEVICES}" \
nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --run_mode nixl_decode \
  --tp 2 \
  --dp 1 \
  --host "${HOST}" \
  --port 8002 \
  --pd_master_ip "${PD_MASTER_IP}" \
  --pd_master_port 8089 \
  >> "${LOG_DIR}/decode.log" 2>&1 &
```

### 测试前 curl warmup（**须执行**，再走 `lm_eval`）

NIXL PD 链路在首次真实推理前易出现冷启动与传输路径问题。**在跑 `lm_eval` 正式评测之前**，必须先对 **`pd_master`** 的 **`/v1/completions`** 发 **至少一次** HTTP 请求，确认返回 **2xx** 且响应体含正常 completion（再走长评测）。

1. **时机**：**`nixl_prefill` 与 `nixl_decode` 均已启动**，且日志显示已与 **`pd_master`** 建立 PD 链路后再执行（可与端口 listen、日志轮询结合判断）。
2. **代理**：执行 **`curl` 前**同样 **`export http_proxy=` / `export https_proxy=`**；若评测机对 **`PD_MASTER_IP`** 走代理会失败，可对本次 shell 设置 **`no_proxy`**（与下文 `lm_eval` 一致，须包含 **`${PD_MASTER_IP}`**）。
3. **记录**：将 **`curl` 使用的命令、HTTP 状态码、若失败则错误摘要** 写入 **`summary.txt`**；成功后再启动 **`lm_eval`**。

示例（**`model` 与 `lm_eval` 中 `model` 字段保持一致**，一般为 **`qwen/qwen3-8b`**；可按需改 **`prompt` / `max_tokens`**）：

```bash
export http_proxy=
export https_proxy=
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,${PD_MASTER_IP}

curl -sS -w "\nhttp_code:%{http_code}\n" -X POST "http://${PD_MASTER_IP}:8089/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen/qwen3-8b\",\"prompt\":\"warmup\",\"max_tokens\":32}" \
  | tee "${LOG_DIR}/curl_warmup.log"
```

- 期望 **`http_code:200`**（或环境约定的成功码）；非 2xx 时先查 **`pd_master.log` / `prefill.log` / `decode.log`**，**不要**直接开大批量 `lm_eval`。
- 可将 **`curl` 输出**保留为 **`curl_warmup.log`**（如上），便于与 **`eval_gsm8k.log`** 对照。

### 评测命令（**curl warmup 成功后**执行）

**`base_url` 指向 `pd_master`**；**`tokenizer` 与 `MODEL_DIR` 一致**：

```bash
export http_proxy=
export https_proxy=

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 \
no_proxy=localhost,127.0.0.1,0.0.0.0,::1,${PD_MASTER_IP} \
lm_eval --model local-completions \
  --model_args "{\"model\":\"qwen/qwen3-8b\", \"base_url\":\"http://${PD_MASTER_IP}:8089/v1/completions\", \"max_length\": 16384, \"tokenized_requests\": false, \"tokenizer\":\"${MODEL_DIR}\"}" \
  --tasks gsm8k --batch_size 64 --confirm_run_unsafe_code \
  >> "${LOG_DIR}/eval_gsm8k.log" 2>&1
```

- 若需 **`lm_eval` 侧**再跑一次小样本，可加 **`--limit 1`**；**不能替代**上文 **`curl` warmup**。
- 若需先用代理下载 `lm_eval` 缓存，见「执行约定」。

## 执行约定

**模型目录**：首轮可 **`export MODEL_DIR=/mtc/models/qwen3-8b`**；路径报错时由用户提供本机 **`MODEL_DIR`**。

1. **启动顺序**：**`pd_master`** → **`nvidia-smi` + export 四卡** → **设置 UCX** → **`nixl_prefill`** → **`nixl_decode`** → **`curl` warmup（须成功）** → **`lm_eval`**。
2. **不要用 health 接口**作为唯一依据；结合 **端口 listen** 与 **`pd_master.log` / `prefill.log` / `decode.log`**。
3. **约每 20 秒**查看日志直至就绪或报错；异常写入 **`summary.txt`**。
4. **`summary.txt`**：记录启动摘要、**`PREFILL_CUDA_DEVICES` / `DECODE_CUDA_DEVICES`** 与选卡依据、**`UCX_NET_DEVICES` 等**、**`curl` warmup 结果（或 `curl_warmup.log` 路径）**、评测关键输出、最终结论。
5. **结束后**关闭 **`pd_master`、`nixl_prefill`、`nixl_decode`** 相关进程。
6. 当用户说明是压测的时候，将lmeval 的 --batch_size 修改为 500 
7. 发现 connetion to pd_master has error 错误的时候，可以先容忍一会，这种网络状态错误有时是可以自行恢复的。

## 输出文件

- **`summary.txt`**、**`pd_master.log`、`prefill.log`、`decode.log`**、**`curl_warmup.log`（建议）**、**`eval_gsm8k.log`** 均落在**同一 `LOG_DIR`**。
