---
name: test-model-deepseekr1-mtp-tp
description: >-
  DeepSeek-R1 MTP-TP test: LightLLM api_server with MTP (EAGLE) draft, tensor parallel
  only (--tp 8, no --dp, no EP MoE), plus GSM8K lm_eval on localhost. Distinct from the
  MTP-EP-TPDP skill which uses --tp 8 --dp 8 and EP MoE. Requires a dedicated log directory,
  summary.txt, tokenizer aligned with MODEL_DIR. Use for TP-only MTP gsm8k accuracy runs.
---

# DeepSeek-R1 **MTP–TP**（仅张量并行 `--tp 8`，无 DP / 无 EP）本地 GSM8K 评测

**测试标识**：并行方式为 **`--tp 8` 单路 TP**，不包含 **`--dp`** 与 **`--enable_ep_moe`**。用于与 **MTP–EP–TPDP**（`--tp 8 --dp 8` + EP MoE）流水线区分。

启动一组 `api_server`（`eagle_with_att` MTP），待就绪后对同一进程执行一次 `lm_eval`（任务 `gsm8k`）。全过程产物落在**同一日志目录**（见「日志目录」）；命令与流程见「启动说明」。

## 日志目录（含 `summary.txt`）

- 先选定或新建**一个日志目录**（例如带时间戳或任务名），与其它测试轮次分开。
- **`api_server` 的标准输出/错误**写入该目录下文件（示例文件名 `server_mtp_tp.log`；可按团队习惯改名或分子目录）。
- **`summary.txt` 固定放在该日志目录下**，写入本轮启动参数摘要、`lm_eval` 关键结果与简要结论。
- `lm_eval` 终端输出建议单独落盘（如 `eval_gsm8k.log`）；**`summary.txt`** 仍为整次任务的**总览结论**。

## 启动说明

本节包含：启动前检查 → 启动服务的命令模板（可变项说明）→ 一条完整 server 命令 → 评测命令。

### 启动前检查

开跑前先确认资源可用；**不满足则先清理相关进程，再启动**。

1. **显卡独占**：用 `nvidia-smi` 检查 **8 张 GPU 均无其它推理任务占用**（显存应基本空闲）；若有冲突进程，结束后再启动。本评测 `--tp 8` 需占满 8 卡，勿与其它 `api_server` 同卡混跑。
2. **端口独占**：服务固定 **`8089`**；用 `ss -tlnp`、`lsof -i :8089` 等确认 **无进程监听** 该端口；若已被占用，结束占用进程后再启动。

### 启动服务的命令模板（可变项）

下列符号与 EP–TPDP 版评测共用含义：

| 可变项 | 含义 |
|--------|------|
| `LOG_DIR` | 本轮评测日志目录，建议**绝对路径**；执行前 `export LOG_DIR=…`。 |
| `MODEL_DIR` | 主模型目录，对应 `--model_dir`；与 `lm_eval` 的 `tokenizer` 必须一致。 |
| `MTP_DRAFT_DIR` | MTP 草稿模型目录，对应 `--mtp_draft_model_dir`。 |

开跑前在同一 shell 中导出路径（引号内替换为本机绝对路径）：

```bash
export LOG_DIR='〈日志根目录〉'
export MODEL_DIR='〈主模型目录，对应 --model_dir〉'
export MTP_DRAFT_DIR='〈MTP 草稿目录，对应 --mtp_draft_model_dir〉'
```

首次试跑可用的**默认路径组合**见「执行约定」。

### 一条 server 启动命令（后台落盘）

以下为 **MTP–TP** 固定形态：**`--tp 8`**，**无 `--dp`**。可直接执行的后台形式（已含 `nohup` 与日志重定向）；调试时可去掉 `nohup` 与 `>> … 2>&1 &` 改前台。**`${MODEL_DIR}`、`${MTP_DRAFT_DIR}`、`${LOG_DIR}`** 须已由上文 `export` 赋值。

`--mem_fraction` 使用 **0.65**（较 0.75 更省显存，MTP 加载主模型与草稿时不易 OOM）。

```bash
LOADWORKER=18 \
nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" --tp 8 --port 8089 \
  --mem_fraction 0.65 --batch_max_tokens 6000 \
  --mtp_mode eagle_with_att --mtp_draft_model_dir "${MTP_DRAFT_DIR}" --mtp_step 2 \
  >> "${LOG_DIR}/server_mtp_tp.log" 2>&1 &
```

### 评测命令（服务就绪后执行一次）

本地回环需排除代理：`no_proxy` / `NO_PROXY`。**`tokenizer` 与 `--model_dir`（`${MODEL_DIR}`）须为同一路径**。以下为带日志落盘的**完整命令**：

```bash
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 \
no_proxy=127.0.0.1,localhost,::1 \
lm_eval --model local-completions \
  --model_args "{\"model\":\"deepseek-ai/DeepSeek-R1\", \"base_url\":\"http://localhost:8089/v1/completions\", \"max_length\": 16384, \"tokenizer\":\"${MODEL_DIR}\"}" \
  --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code \
  >> "${LOG_DIR}/eval_gsm8k.log" 2>&1
```

- **`LOG_DIR`**：与 server 一节一致；若仅调试不重定向，可去掉末尾 `>> "${LOG_DIR}/eval_gsm8k.log" 2>&1`。
- **`MODEL_DIR`**：与 server 的 `--model_dir` 一致；默认试跑与用户确认路径见「执行约定」。
- 若环境需要，可同时设置 `NO_PROXY=127.0.0.1,localhost,::1`。

## 执行约定（不要额外写“专用启动脚本”）

**模型与 MTP 目录（随环境变化）**：`MODEL_DIR`、`MTP_DRAFT_DIR` 在不同机器上路径不同。**首轮试跑**可先使用：

```bash
export MODEL_DIR=/mtc/models/DeepSeek-R1
export MTP_DRAFT_DIR=/mtc/models/DeepSeek-R1-NextN
```

若默认路径不存在或服务报错指向路径/权重加载失败，**请用户提供**本机实际目录并更新两个 `export`；**保持 `MODEL_DIR` 与 `lm_eval` 中 `tokenizer` 一致**。

1. **后台启动 server**：将 `api_server` 输出重定向到日志目录下文件（见「日志目录」）；排查时用 `tail` 查看该日志。
2. **不要用 health 接口** 判断就绪；改为探测 **端口 8089 是否 listen**（例如 `ss -tlnp` / `lsof -i :8089`）。
3. **等待启动**：端口未就绪时约 **每 20 秒** 查看服务日志，区分仍在启动或已报错；路径类错误按上文向用户确认目录。
4. **维护 `summary.txt`**：记录完整启动命令摘要（须能看出 **`--tp 8`、无 `--dp`**）、端口检测结果、`lm_eval` 关键输出与最终结论。
5. **全部完成后**：确认 **`summary.txt`** 完整；server / eval 原始日志保留在同一日志目录备查。

## 输出文件

- **`summary.txt`**：位于**本轮日志目录**，作为本次 **MTP–TP** 评测的**最终总结**。
- **服务与评测日志**：与 **`summary.txt`** 落在**同一日志目录**。
