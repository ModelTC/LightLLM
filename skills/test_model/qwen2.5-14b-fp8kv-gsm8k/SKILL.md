---
name: test-model-qwen2.5-14b-fp8kv-gsm8k
description: >-
  LightLLM Qwen2.5-14B-Instruct GSM8K with FP8 KV cache quantization: either fp8kv_sph
  (per-head calibration JSON) or fp8kv_spt (per-tensor calibration JSON). Single api_server
  tp 2 fixed HTTP port 8089 (not configurable), lm_eval local-completions. Assign GPUs via nvidia-smi then export
  CUDA_VISIBLE_DEVICES. Before starting api_server, cwd must be LightLLM repo root; pass
  --kv_quant_calibration_config_path as the repo-relative path from the table row that matches
  --llm_kv_type (fp8kv_sph with per-head JSON only; fp8kv_spt with per-tensor JSON only; no absolute path,
  no REPO_ROOT/CALIB_JSON shell concatenation). If default MODEL_DIR path is missing or
  load fails with path errors, ask the user for the correct MODEL_DIR. LOG_DIR,
  summary.txt, port listen checks (not health), no_proxy, background server with log redirect.
  Two variants documented in one skill.
---

# Qwen2.5-14B-Instruct **FP8 KV Cache（`fp8kv_sph` / `fp8kv_spt`）** GSM8K 评测

**测试标识**：同一 **`Qwen2.5-14B-Instruct`** 权重下，用 **`api_server`** 跑 **单机 TP=2**；通过 **`--llm_kv_type`** 区分两种 **FP8 KV 量化形态**，每种形态对应 **不同的标定 JSON**（**per-head** vs **per-tensor**）。**每一轮只选其中一种形态**跑通：先起服务，再 **`lm_eval`**。

| 形态 | `--llm_kv_type` | 标定配置（相对 LightLLM 仓库根目录） |
|------|-----------------|--------------------------------------|
| **SPH**（per-head） | **`fp8kv_sph`** | **`test/advanced_config/fp8_calibration_per_head/test_kv_cache_calib_per_head_qwen2.5_14b.json`** |
| **SPT**（per-tensor） | **`fp8kv_spt`** | **`test/advanced_config/fp8_calibration_per_tensor/test_kv_cache_calib_per_tensor_qwen2.5_14b.json`** |

**配对规则（必守）**：**`--llm_kv_type` 与 `--kv_quant_calibration_config_path` 必须取自上表同一行**。 **`fp8kv_sph` 只能** 搭配 **per_head** 标定 JSON；**`fp8kv_spt` 只能** 搭配 **per_tensor** 标定 JSON。只改其一会导致启动或运行时报错。**不要**在命令里写 **`--llm_kv_type "${LLM_KV_TYPE}"`** 却**固定**另一条 `--kv_quant_calibration_config_path`（二者会漂移）；应像下文 **按形态分块**：每一块内两条参数**字面一致、成对出现**。

**端口**：**固定 `8089`**，**不可改**（与脚本一致；**`--port`** 与 **`lm_eval` 的 `base_url`** 均须为 **`8089`**）。**`--tp 2`** 需要 **2 张 GPU**。

整轮产物落在**同一日志目录**：**`summary.txt`**、**`server.log`**、**`eval_gsm8k.log`**；**不要**写复杂聚合启动脚本，按下面块**手动**或复制为独立命令执行。

## 日志目录（含 `summary.txt`）

- 每次评测新建或选定 **`LOG_DIR`**（建议带任务名与时间戳，例如 `…/qwen25_fp8kv_sph_〈时间〉` 与 `…/qwen25_fp8kv_spt_〈时间〉` **分开**，便于对比两种形态）。
- **`api_server`** 标准输出/错误 → **`"${LOG_DIR}/server.log"`**（后台 **`nohup … >> … 2>&1 &`**）。
- **`lm_eval`** → **`"${LOG_DIR}/eval_gsm8k.log"`**。
- **`summary.txt`**：本轮 **`--llm_kv_type`（`fp8kv_sph` / `fp8kv_spt`）**、启动命令摘要、端口检测结果、**`lm_eval` 要点**、失败原因与结论。

## 启动前检查

1. **显卡**：**`--tp 2`**，需 **2 张物理 GPU**。**不要写死卡号**：先 **`nvidia-smi`**，再 **`export CUDA_VISIBLE_DEVICES='i,j'`**。
2. **端口**：**`8089`** 未被占用（**`ss -tlnp`** / **`lsof -i :8089`**）。
3. **标定文件与 KV 形态**：**`--kv_quant_calibration_config_path`** 须为 **与本轮 `--llm_kv_type` 上表同一行** 的相对路径（**不要**写成磁盘绝对路径；**不要** `fp8kv_sph` 配 per_tensor 或 `fp8kv_spt` 配 per_head）。启动 **`python -m lightllm.server.api_server` 时，shell 当前目录须已是仓库根**（先由 Agent **`cd` 到检出根** 再执行 `nohup`；或在一行里 **`cd '…根…' && nohup python …`**）。确认 `os.path.exists` 意义下该相对路径可读。**禁止** `export CALIB_JSON="${REPO_ROOT}/…"` 这类环境变量拼接。
4. **模型目录 `MODEL_DIR`**：启动前确认路径存在（例如 **`test -d "${MODEL_DIR}"`**）且内含权重；默认可用 **`/mtc/models/Qwen2.5-14B-Instruct`**。若默认不存在、或服务 / 日志出现 **找不到模型目录、权重文件缺失、路径类加载失败** 等，**不要盲换路径重试**：**向用户询问**本机正确的 **`MODEL_DIR` 绝对路径**，待用户回复后更新 **`export MODEL_DIR=…`**，并在 **`summary.txt`** 中记录最终采用的路径；**`--model_dir` 与 `lm_eval` 的 `tokenizer` 必须始终为同一字符串**。
5. **代理**：启动 server 前 **`export http_proxy=`**、**`export https_proxy=`**；评测时设置 **`no_proxy`**（见评测命令）。

## 可变项

| 变量 | 含义 |
|------|------|
| `LOG_DIR` | 本轮日志目录（绝对路径）。 |
| `MODEL_DIR` | **`--model_dir`** 与 **`lm_eval` 的 `tokenizer`**，须为**同一路径**。默认试跑 **`/mtc/models/Qwen2.5-14B-Instruct`**；**不可用或报错时向用户询问**正确目录后再 `export`（见「执行约定」与启动前检查第 4 条）。 |
| `LLM_KV_TYPE` | 即 **`--llm_kv_type`**：**`fp8kv_sph`**（上表 **SPH / per-head**）或 **`fp8kv_spt`**（上表 **SPT / per-tensor**）；本轮只选其一；**须与下一行的标定文件同表同行成对**。 |
| 标定 JSON（**相对仓库根**） | **`--kv_quant_calibration_config_path`**：仅允许为上表中 **与当前 `LLM_KV_TYPE` 同一行** 的那一个相对路径；依赖 **`cd` 到仓库根** 后的 cwd，**不要**写绝对路径，勿用 **`${REPO_ROOT}/…`** 拼接；**禁止**与 **`--llm_kv_type` 交叉混用**（见上文配对规则）。 |
| `CUDA_VISIBLE_DEVICES` | 两张卡，**`nvidia-smi` 后 export**。 |

**标定路径写法**：**`--kv_quant_calibration_config_path`** 里 **只写相对路径**，且 **必须是上表与本轮 `--llm_kv_type` 同一行的那一个**。由 Agent 保证 **`nohup` 所在进程的工作目录为 LightLLM 根**（先 `cd` 再启动，或与 `nohup` 写在同一行的 `cd … &&`）。

## 启动服务（后台 + 日志）

**不要用 health 接口**判断就绪；以 **端口 listen**（**`8089`**）结合 **`server.log`** 为准；约 **每 20 秒**看日志直至就绪或报错。

以下为 **两种成对配置**，**每次整段复制其一**；**不要**混用「变量展开的 `--llm_kv_type` + 写死的标定路径」以免与上表不一致。

### 形态 A：**SPH**（`fp8kv_sph` + per-head 标定）

```bash
export http_proxy=
export https_proxy=

export LOG_DIR='〈本轮日志目录〉'
export MODEL_DIR='/mtc/models/Qwen2.5-14B-Instruct'

# 〈LightLLM 仓库根〉由 Agent 改为本机检出目录的绝对路径，仅用于 cd
cd '〈LightLLM 仓库根〉'

LOADWORKER=18 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --port 8089 \
  --llm_kv_type fp8kv_sph \
  --kv_quant_calibration_config_path test/advanced_config/fp8_calibration_per_head/test_kv_cache_calib_per_head_qwen2.5_14b.json \
  >> "${LOG_DIR}/server.log" 2>&1 &
```

### 形态 B：**SPT**（`fp8kv_spt` + per-tensor 标定）

```bash
export http_proxy=
export https_proxy=

export LOG_DIR='〈本轮日志目录〉'
export MODEL_DIR='/mtc/models/Qwen2.5-14B-Instruct'

cd '〈LightLLM 仓库根〉'

LOADWORKER=18 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --port 8089 \
  --llm_kv_type fp8kv_spt \
  --kv_quant_calibration_config_path test/advanced_config/fp8_calibration_per_tensor/test_kv_cache_calib_per_tensor_qwen2.5_14b.json \
  >> "${LOG_DIR}/server.log" 2>&1 &
```

（**`--kv_quant_calibration_config_path`** 均为 **相对仓库根**；**不要**写成绝对路径。）

- **`lm_eval` 的 `base_url`**：本 skill 约定 **`http://127.0.0.1:8089/v1/completions`**（**端口固定**，评测与 **`no_proxy`** 均按 **`127.0.0.1`**）；**`api_server` 须 `--port 8089`**（默认不显式 **`--host`** 时一般为 `0.0.0.0`，本机访问用 **`127.0.0.1`** 即可）。

## 评测命令（服务就绪后）

**`tokenizer` 与 `MODEL_DIR` 对齐**（与其它 test_model skill 一致）：

```bash
export http_proxy=
export https_proxy=

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 \
no_proxy=localhost,127.0.0.1,0.0.0.0,::1 \
lm_eval --model local-completions \
  --model_args "{\"model\":\"Qwen/Qwen2.5-14B-Instruct\", \"base_url\":\"http://127.0.0.1:8089/v1/completions\", \"max_length\": 16384, \"tokenizer\":\"${MODEL_DIR}\"}" \
  --tasks gsm8k --batch_size 64 --confirm_run_unsafe_code \
  >> "${LOG_DIR}/eval_gsm8k.log" 2>&1
```

## 执行约定

### 模型目录（`MODEL_DIR`）

- 不同机器上路径不同。**首轮**可 **`export MODEL_DIR=/mtc/models/Qwen2.5-14B-Instruct`**（与命令模板一致）。
- **在启动 `api_server` 之前**：若该路径**不存在**，或启动后日志明确为 **模型路径 / 权重 / 文件不存在** 等问题，**停止盲试**，**向用户询问**当前环境下 **Qwen2.5-14B-Instruct** 的实际目录绝对路径；用户给出后更新 **`export MODEL_DIR='…用户提供的绝对路径…'`**，并保证后续 **`--model_dir`** 与 **`lm_eval` 的 `tokenizer`** 使用该同一变量；将最终采用的 **`MODEL_DIR`** 写入 **`summary.txt`**。

1. **两种形态分两轮测**：先完整跑 **形态 A（SPH）**（含 **`summary.txt`**），再换 **`LOG_DIR`** 并完整使用 **形态 B（SPT）** 启动块（**`--llm_kv_type fp8kv_spt` 与 per-tensor 标定路径须同时来自上表 SPT 行**）；不要混在同一 **`server.log`** 里，也不要只改 **`--llm_kv_type`** 而不换标定 JSON。
2. **端口**：确认 **`8089`** 进入 **LISTEN** 后再跑 **`lm_eval`**（**勿改端口**）。
3. **结束后**：关闭 **`api_server`** 进程，释放 GPU 与端口。
4. **错误**：将摘要写入 **`summary.txt`**，并在对话中说明关键日志行。
