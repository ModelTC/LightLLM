---
name: test-model-qwen3.5-0.8b-gsm8k-scenarios
description: >-
  LightLLM Qwen3.5-0.8B GSM8K multi-scenario regression: five isolated runs (baseline
  api_server, prefill cudagraph, linear-attention cache flags, CPU cache plus linear-att,
  disk cache with LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH). Each scenario uses api_server
  tp 2 port 8089, then lm_eval local-completions gsm8k batch 500. Scenarios 4 and 5 run
  lm_eval twice for cache warm hit. Per-scenario LOG_DIR, server.log, eval logs, summary.txt.
  GPUs from nvidia-smi; port listen readiness not health; clear proxies and set no_proxy.
  Default MODEL_DIR HuggingFace hub snapshot path; default DISK_CACHE_DIR /mtc/test/tmp/
  for scenario 5; ask user for paths if missing or not writable.
---

# Qwen3.5-0.8B **多场景 GSM8K 回归**

覆盖五种 **`api_server` 配置**：**基线**、**Prefill CUDA Graph**、**Linear-Attention 缓存参数**、**CPU Cache 与 Linear-Att 组合**、**Disk Cache（含环境变量）**。每种配置单独起服务并完成 **`lm_eval`**，互不混用日志。

**测试标识**：同一 **`MODEL_DIR`**（Qwen3.5-0.8B 权重）下，按场景顺序执行：**启动 `api_server` → 确认端口监听与服务日志正常 → 执行 `lm_eval`**。场景 **4、5** 在相同服务配置下 **`lm_eval` 连续执行两次**（缓存预热与命中后行为/耗时对照）。

**端口**：**`8089`**（默认，与 **`PORT`** 一致）。

**算力**：**`--tp 2`**，需要 **2 张物理 GPU**。

**评测**：**`lm_eval`**，**`--tasks gsm8k`**，**`--batch_size 500`**，**`model`** 为 **`qwen/Qwen3.5-0.8B`**。**`--model_dir` 与 `model_args` 中的 `tokenizer` 必须为同一目录路径（即 `MODEL_DIR`）**。

## 场景总览

| # | 名称 | `api_server` 相对上一场景的增量 | `lm_eval` |
|---|------|----------------------------------|-----------|
| 1 | 基线 | **`--model_dir` / `--tp 2` / `--port`** | 1 次 |
| 2 | Prefill CUDA Graph | **`--enable_prefill_cudagraph`** | 1 次 |
| 3 | Linear-Attention 参数 | **`--linear_att_cache_size 10`**、**`--linear_att_hash_page_size 256`**、**`--linear_att_page_block_num 2`**、**`--max_total_token_num 270000`** | 1 次 |
| 4 | CPU Cache + Linear-Att | 在场景 3 同类参数基础上增加 **`--enable_cpu_cache`**、**`--cpu_cache_storage_size 128`**（**`--max_total_token_num` 仍为 `270000`**） | **2 次** |
| 5 | Disk Cache | **`LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH=128`**；**`--linear_att_cache_size 128`** 等一组参数，及 **`--enable_cpu_cache`**、**`--enable_disk_cache`**、**`--disk_cache_dir`** 等（见下文命令块） | **2 次** |

## 日志目录（含 `summary.txt`）

- **每个场景**使用独立 **`LOG_DIR`**（建议绝对路径，并带场景编号或时间戳）。
- **`api_server`**：标准输出与标准错误写入 **`"${LOG_DIR}/server.log"`**（推荐使用 **`nohup … >> … 2>&1 &`**）。
- **`lm_eval`**：默认写入 **`"${LOG_DIR}/eval_gsm8k.log"`**；同一服务下的第二次评测写入 **`"${LOG_DIR}/eval_gsm8k_run2.log"`**（场景 4、5）。
- **`summary.txt`**：记录本场景完整启动参数、**`lm_eval` 命令摘要**、端口与日志就绪情况、两轮评测目的与结论（若适用）、异常与最终结论。

## 启动前检查

1. **显卡**：执行 **`nvidia-smi`**，按占用选定 2 张卡后 **`export CUDA_VISIBLE_DEVICES='i,j'`**（**不要写死卡号**）。
2. **端口**：每轮启动前确认 **`8089`**（或当前 **`PORT`**）未被占用（例如 **`ss -tlnp`**、**`lsof -i :8089`**）；上一轮结束后须 **终止对应 `api_server` 进程** 再启动下一轮。
3. **`MODEL_DIR`**：见 **「路径约定」**；启动前执行 **`test -d "${MODEL_DIR}"`**；若路径无效或服务报路径类错误，按该节处理。
4. **`DISK_CACHE_DIR`（仅场景 5）**：见 **「路径约定」**；须为可写目录；先 **`mkdir -p "${DISK_CACHE_DIR}"`**，仍不可写则按该节处理。
5. **代理**：每次启动 **`api_server`** 或执行 **`lm_eval`** 之前，将 **`http_proxy` / `https_proxy` 置空**；执行 **`lm_eval`** 时配置 **`no_proxy`**（见评测命令块）。

## 路径约定（`MODEL_DIR` 与 `DISK_CACHE_DIR`）

**原则**：下列 **默认路径** 与历史 **`test/acc/test_qwen3.5.sh`** 一致，可作为首轮试跑起点。若目录不存在、不可读、权重加载失败，或磁盘缓存路径不可写，**不得在未向用户确认的情况下反复更换路径盲试**；应 **向用户询问** 本机可用的 **`MODEL_DIR` / `DISK_CACHE_DIR` 绝对路径**，在收到答复后更新环境变量，并在 **`summary.txt`** 中记录最终采用的路径。

### `MODEL_DIR`

- **`--model_dir`** 与 **`lm_eval` 的 `tokenizer` 字段** 必须为 **同一字符串**（本 skill 中均记为 **`MODEL_DIR`**）。
- **默认路径（HuggingFace Hub 本地缓存示例）**：以下 **`snapshots/` 下目录名随实际下载版本可能变化**；若本机不存在该路径，则默认值不适用，须改用本机真实路径或询问用户。

```bash
export MODEL_DIR='/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'
```

- **备选**：若权重部署在统一管理目录（例如 **`/mtc/models/Qwen3.5-0.8B`**），且 **`test -d`** 通过，可 **`export MODEL_DIR='/mtc/models/Qwen3.5-0.8B'`**。

### `DISK_CACHE_DIR`（仅场景 5：`--disk_cache_dir`）

- **默认路径**：

```bash
export DISK_CACHE_DIR='/mtc/test/tmp/'
```

- 场景 5 启动 **`api_server`** 前执行 **`mkdir -p "${DISK_CACHE_DIR}"`**。若父目录不可创建、无写权限或不符合运维规范，**向用户询问** 合适的可写目录后再 **`export`**。

## 可变项

| 变量 | 含义 |
|------|------|
| `LOG_DIR` | 当前场景的日志根目录（**`server.log` / `eval_gsm8k*.log` / `summary.txt`**）。 |
| `MODEL_DIR` | **`--model_dir`**；**`lm_eval` 的 `tokenizer`**。默认路径与失败时处理见 **「路径约定」**。 |
| `PORT` | HTTP 端口，默认 **`8089`**。 |
| `BIND_URL_HOST` | **`lm_eval` 中 `base_url` 的主机名**；本机常用 **`127.0.0.1`** 或 **`localhost`**。 |
| `CUDA_VISIBLE_DEVICES` | 2 个物理 GPU 索引（逗号分隔）。 |
| `DISK_CACHE_DIR` | 场景 5 的 **`--disk_cache_dir`**。默认 **`/mtc/test/tmp/`**；不可写或不可用时见 **「路径约定」**。 |

**开跑前导出示例**（按需修改引号内路径）：

```bash
export LOG_DIR='〈场景日志目录，每场景换新目录〉'
export MODEL_DIR='/root/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'
export DISK_CACHE_DIR='/mtc/test/tmp/'
export PORT=8089
export BIND_URL_HOST='127.0.0.1'
# export CUDA_VISIBLE_DEVICES='6,7'
```

## 服务就绪判定

**不要使用 HTTP health 接口作为唯一就绪依据**。应结合：**`PORT` 是否处于 LISTEN 状态**、**`server.log` 是否出现致命错误**；可约 **每 20 秒** 查看一次日志直至可接受评测或确认失败。

## `lm_eval` 命令模板（单次）

服务就绪后执行；场景 **4、5** 在**同一 `api_server` 生命周期内**再执行一次，并将第二次输出重定向到 **`eval_gsm8k_run2.log`**。

```bash
export http_proxy=
export https_proxy=

export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,${BIND_URL_HOST}

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 \
lm_eval --model local-completions \
  --model_args "{\"model\":\"qwen/Qwen3.5-0.8B\", \"base_url\":\"http://${BIND_URL_HOST}:${PORT}/v1/completions\", \"max_length\": 16384, \"tokenizer\":\"${MODEL_DIR}\"}" \
  --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code \
  >> "${LOG_DIR}/eval_gsm8k.log" 2>&1
```

第二次（场景 4、5）：将末尾重定向改为 **`>> "${LOG_DIR}/eval_gsm8k_run2.log" 2>&1`**，并在 **`summary.txt`** 中说明两次运行的目的（例如缓存预热与命中后对照）。

## 各场景 `api_server` 命令模板

以下命令块仅列出 **`api_server` 参数差异**。实际执行时须在 **`export http_proxy=`、`export https_proxy=`** 之后，按仓库其它 acc 测试惯例自行补全：**`LOADWORKER=18 CUDA_VISIBLE_DEVICES=…`**、**`nohup`**、以及 **`>> "${LOG_DIR}/server.log" 2>&1 &`**。

### 场景 1：基线

```bash
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --port "${PORT}"
```

### 场景 2：Prefill CUDA Graph

```bash
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --port "${PORT}" \
  --enable_prefill_cudagraph
```

### 场景 3：Linear-Attention 参数

```bash
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --port "${PORT}" \
  --linear_att_cache_size 10 \
  --linear_att_hash_page_size 256 \
  --linear_att_page_block_num 2 \
  --max_total_token_num 270000
```

### 场景 4：CPU Cache + Linear-Attention（`lm_eval` 两次）

```bash
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --port "${PORT}" \
  --linear_att_cache_size 10 \
  --linear_att_hash_page_size 256 \
  --linear_att_page_block_num 2 \
  --max_total_token_num 270000 \
  --enable_cpu_cache \
  --cpu_cache_storage_size 128
```

### 场景 5：Disk Cache（环境变量 + `lm_eval` 两次）

在 **`python`** 进程前设置 **`LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH=128`**（与历史脚本一致，使子进程继承该变量）：

```bash
LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH=128 \
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --port "${PORT}" \
  --linear_att_cache_size 128 \
  --linear_att_hash_page_size 256 \
  --linear_att_page_block_num 32 \
  --max_total_token_num 270000 \
  --enable_cpu_cache \
  --cpu_cache_storage_size 32 \
  --enable_disk_cache \
  --disk_cache_storage_size 512 \
  --disk_cache_dir "${DISK_CACHE_DIR}"
```

## 执行约定

1. **场景顺序**：按 **1 → 2 → 3 → 4 → 5** 执行；每步 **先停止上一场景的 `api_server`**，使用 **新的 `LOG_DIR`**。
2. **`MODEL_DIR` 与 `DISK_CACHE_DIR`**：遵循 **「路径约定」**；**`summary.txt`** 记录最终采用的路径。
3. **场景 4、5**：在同一服务配置下 **`lm_eval` 执行两次**，并保留两次日志文件以便对比。
4. **收尾**：全部场景结束后，确保 **`api_server` 进程已结束**，释放 GPU 与端口。
5. **失败处理**：将错误摘要写入 **`summary.txt`**，并在对话中给出关键日志信息。