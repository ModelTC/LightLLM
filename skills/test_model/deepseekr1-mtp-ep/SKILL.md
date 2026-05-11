---
name: test-model-deepseekr1-mtp-ep
description: >-
  Runs LightLLM DeepSeek-R1 EP MoE + MTP (EAGLE) server variants and GSM8K lm_eval
  against localhost. Requires each full run to use a dedicated log directory: persist every
  api_server process log under that tree (per-variant subdirectories recommended),
  write the consolidated summary to summary.txt in that same log directory, and keep artifacts
  separated from other test runs. Use when running DeepSeek-R1 MTP EP accuracy workflows
  or when the user asks to run these four server configurations one-by-one with logged results.
---

# DeepSeek-R1 MTP + EP MoE 串行评测流程

按固定顺序依次启动四种 `api_server` 配置；每次待服务就绪后执行 `lm_eval`。整轮评测须落在**同一日志目录**内归档日志与最终结论（见「日志目录」）；具体操作见「启动说明」。

## 日志目录（含 `summary.txt`）

- 每次完整评测（四种变体串行）先选定或新建**一个日志目录**（例如带时间戳或任务名），与其它测试轮次分开，便于区分管理。
- **所有 `api_server` 进程的标准输出/错误**须写入该目录下文件（建议每种变体单独子目录，如 `variant_01_baseline/`、`variant_02_tpsp_mix/`；或同级命名 `server_01_baseline.log` 等，团队任选其一，保持可追溯）。
- **`summary.txt` 固定放在该日志目录下**，汇总整轮测试：各变体启动参数摘要、`lm_eval` 关键结果、失败原因与最终对比；**不再**把「最终总结」散落在当前工作目录或其它路径。
- `lm_eval` 终端输出也要有单独的日志文件（如 `eval_gsm8k.log`），**`summary.txt`** 仍承担**总览结论**角色。

## 启动说明

本节包含：启动前检查 → 启动服务的命令模板（可变项说明）→ 四种完整 server 命令 → 评测命令。

### 启动前检查

开跑前先确认资源可用；**不满足则先清理相关进程，再进入后续变体**。

1. **显卡占用**：用 `nvidia-smi`（或与集群一致的占用查看方式）检查目标 GPU 是否被无关任务占满；若有冲突进程，结束后再启动本评测。
2. **端口**：服务固定 **`8089`**；用 `ss -tlnp`、`lsof -i :8089` 等确认**无进程监听**该端口；若已被占用，查出 PID 并结束占用进程后再启动。

### 启动服务的命令模板（可变项）

下列命令中出现的可变项含义如下（其余为固定写法）：

| 可变项 | 含义 |
|--------|------|
| `LOG_DIR` | 本轮评测日志目录，建议**绝对路径**；执行前 `export LOG_DIR=…`。 |
| `MODEL_DIR` | 主模型目录，对应 `--model_dir`；与 `lm_eval` 的 `tokenizer` 必须一致。 |
| `MTP_DRAFT_DIR` | MTP 草稿模型目录，对应 `--mtp_draft_model_dir`。 |
| `server_*.log`、`eval_*.log` | 仅文件名示例，可按变体重命名。 |

开跑前在同一 shell 中导出三类路径（将引号内整段替换为本机绝对路径；**勿写死下文未给出的机器路径**）：

```bash
export LOG_DIR='〈日志根目录〉'
export MODEL_DIR='〈主模型目录，对应 --model_dir〉'
export MTP_DRAFT_DIR='〈MTP 草稿目录，对应 --mtp_draft_model_dir〉'
```

首次试跑可用的**默认路径组合**见「执行约定」；与当前环境不符时再改为用户提供的目录。

### 四种 server 启动命令（按顺序逐个测）

每条 **单独** 跑完「启动 → 等就绪 → 评测 → 写入日志目录下的日志 → 停服务」再进入下一条，不要并行多个 server。以下为**可直接执行**的后台启动形式（已含 `nohup` 与日志重定向）；若暂时不需落盘，可自行去掉 `nohup`、`>> … 2>&1 &` 并在前台调试。命令中 **`${MODEL_DIR}`、`${MTP_DRAFT_DIR}`** 须已由上文 `export` 赋值。

#### 变体 1：基线（EP + MTP）

```bash
LOADWORKER=18 NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
nohup python -m lightllm.server.api_server \
  --enable_ep_moe --model_dir "${MODEL_DIR}" --tp 8 --dp 8 --port 8089 \
  --max_total_token_num 60000 --graph_max_batch_size 16 --batch_max_tokens 6000 \
  --max_req_total_len 56000 \
  --mtp_mode eagle_with_att --mtp_draft_model_dir "${MTP_DRAFT_DIR}" --mtp_step 2 \
  >> "${LOG_DIR}/server_01_baseline.log" 2>&1 &
```

#### 变体 2：`--enable_tpsp_mix_mode`

```bash
LOADWORKER=18 NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
nohup python -m lightllm.server.api_server \
  --enable_ep_moe --model_dir "${MODEL_DIR}" --tp 8 --dp 8 --port 8089 \
  --max_total_token_num 60000 --graph_max_batch_size 16 --batch_max_tokens 6000 \
  --max_req_total_len 56000 \
  --mtp_mode eagle_with_att --mtp_draft_model_dir "${MTP_DRAFT_DIR}" --mtp_step 2 \
  --enable_tpsp_mix_mode \
  >> "${LOG_DIR}/server_02_tpsp_mix.log" 2>&1 &
```

#### 变体 3：prefill / decode microbatch overlap

```bash
LOADWORKER=18 NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
nohup python -m lightllm.server.api_server \
  --enable_ep_moe --model_dir "${MODEL_DIR}" --tp 8 --dp 8 --port 8089 \
  --max_total_token_num 60000 --graph_max_batch_size 16 --batch_max_tokens 6000 \
  --max_req_total_len 56000 \
  --mtp_mode eagle_with_att --mtp_draft_model_dir "${MTP_DRAFT_DIR}" --mtp_step 2 \
  --enable_prefill_microbatch_overlap --enable_decode_microbatch_overlap \
  >> "${LOG_DIR}/server_03_overlap.log" 2>&1 &
```

#### 变体 4：overlap + `--enable_dp_prefill_balance`

```bash
LOADWORKER=18 NUM_MAX_DISPATCH_TOKENS_PER_RANK=256 \
nohup python -m lightllm.server.api_server \
  --enable_ep_moe --model_dir "${MODEL_DIR}" --tp 8 --dp 8 --port 8089 \
  --max_total_token_num 60000 --graph_max_batch_size 16 --batch_max_tokens 6000 \
  --max_req_total_len 56000 \
  --mtp_mode eagle_with_att --mtp_draft_model_dir "${MTP_DRAFT_DIR}" --mtp_step 2 \
  --enable_prefill_microbatch_overlap --enable_decode_microbatch_overlap \
  --enable_dp_prefill_balance \
  >> "${LOG_DIR}/server_04_overlap_dp_balance.log" 2>&1 &
```

### 评测命令（每个变体各执行一次）

服务就绪后执行（本地回环走代理时用 `no_proxy` / `NO_PROXY` 排除本机）。**`model_args` 中 `tokenizer` 必须与本次 server 的 `--model_dir`（即 **`${MODEL_DIR}`**）为同一字符串路径**。以下为带日志落盘的**完整命令**（`--model_args` 使用双引号以便展开 **`${MODEL_DIR}`**）：

```bash
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 \
no_proxy=127.0.0.1,localhost,::1 \
lm_eval --model local-completions \
  --model_args "{\"model\":\"deepseek-ai/DeepSeek-R1\", \"base_url\":\"http://localhost:8089/v1/completions\", \"max_length\": 16384, \"tokenizer\":\"${MODEL_DIR}\"}" \
  --tasks gsm8k --batch_size 32 --confirm_run_unsafe_code \
  >> "${LOG_DIR}/eval_gsm8k.log" 2>&1
```

- **`LOG_DIR`**：与启动服务一节相同；若仅调试不重定向，去掉 `\` 续行及最后的 `>> "${LOG_DIR}/eval_gsm8k.log" 2>&1` 即可在前台查看输出。
- **`MODEL_DIR`**：须与 server 启动命令中的 `--model_dir` 一致；路径随环境变化时的默认试跑与向用户确认见「执行约定」。
- 若环境需要，可同时设置 `NO_PROXY=127.0.0.1,localhost,::1`（或与团队约定一致的列表）。

## 执行约定（不要额外写“专用启动脚本”）

**模型与 MTP 目录（随环境变化）**：`MODEL_DIR`（主模型）、`MTP_DRAFT_DIR`（MTP 草稿）在不同机器上路径不同。**首轮试跑**可先用下列默认组合（与本文档常见部署对应；若本机不存在则跳过默认、直接执行下一步「向用户确认」）：

```bash
export MODEL_DIR=/mtc/models/DeepSeek-R1
export MTP_DRAFT_DIR=/mtc/models/DeepSeek-R1-NextN
```

若按默认路径 **export** 后仍无法启动服务，或日志中出现**明确的模型路径 / 权重加载 / 文件不存在**等错误，**不要反复盲试**：根据日志判断为路径问题时，**请用户提供**当前环境下实际的主模型目录与 MTP 草稿目录，更新 `export MODEL_DIR=…`、`export MTP_DRAFT_DIR=…` 后再执行（且保证 **`MODEL_DIR` 与 `lm_eval` 的 `tokenizer` 仍为同一路径**）。

1. **后台启动 server**：用 shell 后台或终端任务跑当前变体的 `python -m lightllm.server.api_server ...`，**并将该进程输出重定向到本轮日志目录下的日志文件**（见上文「日志目录（含 summary.txt）」）；排查问题时 tail 该文件，而不是依赖未落盘的终端缓冲。
2. **不要用 health 接口** 判断就绪；改为探测 **端口 8089 是否处于 listen**（例如 `ss -tlnp` / `lsof -i :8089` 等，与系统一致即可）。
3. **等待启动**：若端口未就绪，约 **每 20 秒** 查看一次**该变体对应的服务日志文件**，区分仍在启动还是已报错退出；报错则写入日志目录下的 `summary.txt`（或先写变体日志再在该汇总文件中引用）并停止该变体，不要继续盲等。
4. **维护 `summary.txt`**：位于**日志目录**；随进度追加每个变体的标记块——**本条使用的完整启动命令**（或等价摘要）、**端口检测结果**、**lm_eval 关键输出**；全部结束后在该文件内写**最终汇总**（各配置成败、指标对比或失败原因）。可与用户口头摘要对照，但以日志目录中 **`summary.txt`** 为归档准绳。
5. **变体之间**：停止上一进程的 server，再启动下一变体（避免端口占用）。
6. **全部完成后**：确认日志目录下的 **`summary.txt`** 已包含完整最终总结；原始 server / eval 日志保留在同目录（或子目录）中备查。

## 输出文件

- **`summary.txt`**：仅位于**本轮日志目录**，作为整次四变体测试的**最终总结**文档。
- **服务与评测日志**：全部落在**同一日志目录**（建议按变体分子目录或分文件名），不得与未指定目录混写。
