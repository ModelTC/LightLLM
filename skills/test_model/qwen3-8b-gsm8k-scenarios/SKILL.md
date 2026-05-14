---
name: test-model-qwen3-8b-gsm8k-scenarios
description: >-
  LightLLM Qwen3-8B GSM8K multi-scenario regression: seven isolated api_server configs
  (baseline, vllm-fp8w8a8 quant, tpsp mix, tpsp with dp2 and dp prefill balance, cpu cache,
  int8kv on top of cpu cache, disk cache with LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH).
  Each scenario then lm_eval gsm8k batch 500. Scenarios 5–7 run lm_eval twice for cache
  hit. Per-scenario LOG_DIR, server.log, eval logs, summary.txt. Default MODEL_DIR
  /mtc/models/qwen3-8b; DISK_CACHE_DIR /mtc/test/tmp/ for scenario 7; ask user if paths
  invalid. Fixed HTTP port 8089 (not configurable). nvidia-smi GPUs, port listen not health,
  clear proxies and no_proxy.
---

# Qwen3-8B **多场景 GSM8K 回归**

同一 **`MODEL_DIR`（Qwen3-8B 权重）** 下，按 **七种 `api_server` 配置** 依次各跑一轮：**启动服务 → 端口与日志就绪 → `lm_eval`**。场景 **5、6、7** 在相同服务配置下 **`lm_eval` 连续执行两次**（缓存预热与命中后效率/精度对照，与历史脚本注释一致）。

**端口**：**固定 `8089`**（与 **`--port`**、**`lm_eval` 的 `base_url`** 一致；**不作为环境变量**）。

**评测**：**`lm_eval`**，**`tasks gsm8k`**，**`batch_size 500`**，**`model`：`qwen/qwen3-8b`**。**`tokenizer` 与 `MODEL_DIR` 须为同一目录路径**。

## 场景总览

| # | 名称 | `api_server` 要点 | `lm_eval` |
|---|------|-------------------|-----------|
| 1 | 基线 | **`--tp 2`**，无额外开关 | 1 次 |
| 2 | FP8 量化 | **`--quant_type vllm-fp8w8a8`**（在场景 1 基础上） | 1 次 |
| 3 | TP-SP 混合 | **`--enable_tpsp_mix_mode`**（**`--tp 2`**，无 **`--dp`**） | 1 次 |
| 4 | TP-SP + DP2 + DP prefill 均衡 | **`--tp 2 --dp 2`**、**`--enable_tpsp_mix_mode`**、**`--enable_dp_prefill_balance`** | 1 次 |
| 5 | CPU Cache | **`--tp 2 --dp 2`**，**`--max_total_token_num 200000`**，**`--enable_cpu_cache`**，**`--cpu_cache_storage_size 128`**，**`--cpu_cache_token_page_size 128`** | **2 次** |
| 6 | CPU Cache + INT8 KV | 在场景 5 基础上增加 **`--llm_kv_type int8kv`** | **2 次** |
| 7 | Disk Cache | **`LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH=128`**；**`--tp 2 --dp 2`**，**`--max_total_token_num 200000`**，CPU cache 与 **disk cache** 一组参数（见命令块） | **2 次** |

**算力说明**：历史脚本使用 **`CUDA_VISIBLE_DEVICES`** 指向 **2 张卡**；场景 **4–7** 含 **`--dp 2`**。执行前须结合 **`nvidia-smi`** 与 **LightLLM 对 tp/dp 的资源说明** 确认本机 GPU 数与映射是否满足；不满足时 **向用户询问** 正确启动方式，**不要盲试**。

## 日志目录（含 `summary.txt`）

- **每个场景**使用独立 **`LOG_DIR`**。
- **`api_server`** → **`"${LOG_DIR}/server.log"`**（推荐 **`nohup … >> … 2>&1 &`**）。
- **`lm_eval`**：第一次 **`"${LOG_DIR}/eval_gsm8k.log"`**；第二次（场景 5–7）**`"${LOG_DIR}/eval_gsm8k_run2.log"`**。
- **`summary.txt`**：本场景完整启动参数、**`lm_eval` 摘要**、端口与日志就绪情况、两轮评测说明（若适用）、结论与失败原因。

## 启动前检查

1. **显卡**：**`nvidia-smi`** 后 **`export CUDA_VISIBLE_DEVICES`**；**不要写死卡号**；**`--tp` / `--dp`** 与卡数须匹配本机规范。
2. **端口**：每轮前确认 **`8089`** 空闲；上一轮结束后 **终止 `api_server`** 再启下一轮。
3. **`MODEL_DIR`**：见 **「路径约定」**；**`test -d "${MODEL_DIR}"`**。
4. **`DISK_CACHE_DIR`（仅场景 7）**：见 **「路径约定」**；**`mkdir -p`** 后须可写。
5. **代理**：**`api_server` / `lm_eval` 前** 置空 **`http_proxy` / `https_proxy`**；**`lm_eval`** 配置 **`no_proxy`**（见评测块）。

## 路径约定（`MODEL_DIR` 与 `DISK_CACHE_DIR`）

- **`MODEL_DIR`**：**首轮试跑默认** **`/mtc/models/qwen3-8b`**。若目录不存在或加载失败，**向用户询问** 本机正确路径；**`--model_dir` 与 `lm_eval` 的 `tokenizer` 保持一致**。
- **`DISK_CACHE_DIR`（场景 7）**：**默认** **`/mtc/test/tmp/`**；不可写或不存在时 **向用户询问** 可写目录；**`summary.txt`** 记录最终路径。

## 可变项

| 变量 | 含义 |
|------|------|
| `LOG_DIR` | 当前场景日志根目录。 |
| `MODEL_DIR` | **`--model_dir`**；**`lm_eval` 的 `tokenizer`**。 |
| `BIND_URL_HOST` | **`base_url` 主机**；常用 **`127.0.0.1`**。 |
| `CUDA_VISIBLE_DEVICES` | 由 **`nvidia-smi`** 决定；与 tp/dp 组合须匹配环境。 |
| `DISK_CACHE_DIR` | 场景 7 的 **`--disk_cache_dir`**；默认 **`/mtc/test/tmp/`**。 |

**开跑前导出示例**：

```bash
export LOG_DIR='〈本场景日志目录〉'
export MODEL_DIR='/mtc/models/qwen3-8b'
export DISK_CACHE_DIR='/mtc/test/tmp/'
export BIND_URL_HOST='127.0.0.1'
# export CUDA_VISIBLE_DEVICES='6,7'
```

## 服务就绪判定

**不要使用 HTTP health 作为唯一依据**。结合 **`8089` 是否 LISTEN** 与 **`server.log`**；可约 **每 20 秒** 查看一次直至可评测或确认失败。

## `lm_eval` 命令模板（单次）

```bash
export http_proxy=
export https_proxy=

export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,${BIND_URL_HOST}

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 \
lm_eval --model local-completions \
  --model_args "{\"model\":\"qwen/qwen3-8b\", \"base_url\":\"http://${BIND_URL_HOST}:8089/v1/completions\", \"max_length\": 16384, \"tokenizer\":\"${MODEL_DIR}\"}" \
  --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code \
  >> "${LOG_DIR}/eval_gsm8k.log" 2>&1
```

场景 **5–7** 第二次：将重定向改为 **`>> "${LOG_DIR}/eval_gsm8k_run2.log" 2>&1`**。

## 各场景 `api_server` 命令模板

以下省略 **`export http_proxy=` / `export https_proxy=`**、**`LOADWORKER=18`**、**`CUDA_VISIBLE_DEVICES`**、**`nohup`** 与 **`>> "${LOG_DIR}/server.log" 2>&1 &`**；实际执行时与其它 acc skill 一致自行补全。

### 场景 1：基线

```bash
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --port 8089
```

### 场景 2：FP8 量化（`vllm-fp8w8a8`）

```bash
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --port 8089 \
  --quant_type vllm-fp8w8a8
```

### 场景 3：TP-SP 混合

```bash
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --port 8089 \
  --enable_tpsp_mix_mode
```

### 场景 4：TP-SP + DP2 + DP prefill 均衡

```bash
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --dp 2 \
  --port 8089 \
  --enable_tpsp_mix_mode \
  --enable_dp_prefill_balance
```

### 场景 5：CPU Cache（`lm_eval` 两次）

```bash
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --dp 2 \
  --port 8089 \
  --max_total_token_num 200000 \
  --enable_cpu_cache \
  --cpu_cache_storage_size 128 \
  --cpu_cache_token_page_size 128
```

### 场景 6：CPU Cache + INT8 KV（`lm_eval` 两次）

```bash
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --dp 2 \
  --port 8089 \
  --max_total_token_num 200000 \
  --enable_cpu_cache \
  --cpu_cache_storage_size 128 \
  --cpu_cache_token_page_size 128 \
  --llm_kv_type int8kv
```

### 场景 7：Disk Cache（`lm_eval` 两次）

与历史脚本一致：在 **`python`** 前加 **`LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH=128`**。

```bash
LIGHTLLM_DISK_CACHE_PROMPT_LIMIT_LENGTH=128 \
python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --dp 2 \
  --port 8089 \
  --max_total_token_num 200000 \
  --enable_cpu_cache \
  --cpu_cache_storage_size 64 \
  --cpu_cache_token_page_size 128 \
  --enable_disk_cache \
  --disk_cache_storage_size 256 \
  --disk_cache_dir "${DISK_CACHE_DIR}"
```

## 执行约定

1. **顺序**：**1 → 7** 严格递增；每步 **新 `LOG_DIR`**，**先停旧服务**。
2. **场景 5–7**：**`lm_eval` 各执行两次**，并在 **`summary.txt`** 说明 run1 / run2 目的。
3. **`MODEL_DIR` / `DISK_CACHE_DIR`**：遵循 **「路径约定」**。
4. **收尾**：全部结束后释放进程、端口与 GPU。
