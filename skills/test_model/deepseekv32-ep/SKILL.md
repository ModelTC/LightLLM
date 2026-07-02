---
name: test-model-deepseekv32-ep
description: >-
  Runs LightLLM DeepSeek-V3.2 EP MoE gsm8k: api_server with --tp 8 --dp 8 --enable_ep_moe,
  tool_call_parser deepseekv32, reasoning_parser deepseek-v3, graph_max_batch_size 32,
  mem_fraction 0.8, LOADWORKER 14, port 8000 aligned with lm_eval base_url. Requires a
  dedicated log directory, api_server and eval logs, summary.txt consolidated report.
  lm_eval uses tokenizer_backend=null (server-side tokenization) because local
  transformers does not recognize model_type deepseek_v32. Distinct from R1 MTP/Base
  flows. Use for V3.2 EP MoE gsm8k accuracy on LightLLM.
---

# DeepSeek-V3.2 **EP**（`--tp 8`、`--dp 8`、`--enable_ep_moe`）本地 GSM8K 评测

**测试标识**：本流程针对 **DeepSeek-V3.2**，启用 **EP MoE**（**`--enable_ep_moe`**）与 **TP+DP**（**`--tp 8 --dp 8`**），并包含 **`tool_call_parser deepseekv32`**、**`reasoning_parser deepseek-v3`**、**`graph_max_batch_size 32`**、**`mem_fraction 0.8`** 等与推理栈相关的参数。与 **Base–R1**、**MTP–TP / MTP–EP**（R1 系列）区分。

**监听端口**：`api_server` 与 `lm_eval` 的 **`base_url` 必须使用同一端口**；本流程固定为 **`8000`**（下文 server 命令含 **`--port 8000`**，评测 URL 为 `http://localhost:8000/v1/completions`）。

启动一组 `api_server`，待端口就绪后执行一次 `lm_eval`（任务 **`gsm8k`**，`batch_size` **500**）。整轮产物须落在**同一日志目录**内归档日志与 **`summary.txt`**（见「日志目录」）；具体操作见「启动说明」。

## 日志目录（含 `summary.txt`）

- 每次评测先选定或新建**一个日志目录**（例如带时间戳或任务名），与其它测试轮次分开，便于区分管理。
- **所有 `api_server` 进程的标准输出/错误**须写入该目录下文件（示例同级命名 **`server_v32_ep.log`**；也可分子目录，团队任选其一，保持可追溯）。
- **`summary.txt` 固定放在该日志目录下**，写入本轮启动参数摘要、`lm_eval` 关键结果、失败原因或简要对比；**不再**把「最终总结」散落在当前工作目录或其它路径。
- `lm_eval` 终端输出也要有单独的日志文件（如 **`eval_gsm8k.log`**）；**`summary.txt`** 仍承担**总览结论**角色。

## 启动说明

本节包含：启动前检查 → 启动服务的命令模板（可变项说明）→ 一条完整 server 命令 → 评测命令。

### 启动前检查

开跑前先确认资源可用；**不满足则先清理相关进程**，再启动服务与评测。

1. **显卡占用**：用 `nvidia-smi`（或与集群一致的占用查看方式）检查目标 GPU 是否被无关任务占满；若有冲突进程，结束后再启动本评测（本配置为 **TP+DP**，需足够 GPU 资源）。
2. **端口**：服务固定 **`8000`**（与下文 `lm_eval` 的 `base_url` 端口一致）；用 `ss -tlnp`、`lsof -i :8000` 等确认**无进程监听**该端口；若已被占用，查出 PID 并结束占用进程后再启动。

### 启动服务的命令模板（可变项）

下列命令中出现的可变项含义如下（其余为固定写法）：

| 可变项 | 含义 |
|--------|------|
| `LOG_DIR` | 本轮评测日志目录，建议**绝对路径**；执行前 `export LOG_DIR=…`。 |
| `MODEL_DIR` | 主模型目录，对应 `--model_dir`；与 `lm_eval` 的 `tokenizer` 必须一致。 |
| `server_*.log`、`eval_*.log` | 仅文件名示例，可按任务重命名。 |

开跑前在同一 shell 中导出路径（将引号内整段替换为本机绝对路径；**勿写死下文未给出的机器路径**）：

```bash
export LOG_DIR='〈日志根目录〉'
export MODEL_DIR='〈主模型目录，对应 --model_dir〉'
```

首次试跑可用的**默认 `MODEL_DIR`** 见「执行约定」；与当前环境不符时再改为用户提供的目录。

### 一条 server 启动命令（后台落盘）

本条为 **DeepSeek-V3.2 EP** 固定形态：**`LOADWORKER=14`**，**`--tp 8 --dp 8 --enable_ep_moe`**，**`--port 8000`**，以及 **`tool_call_parser` / `reasoning_parser` / `graph_max_batch_size` / `mem_fraction`** 等与脚本一致的参数。以下为**可直接执行**的后台启动形式（已含 `nohup` 与日志重定向）；若暂时不需落盘，可自行去掉 `nohup`、`>> … 2>&1 &` 并在前台调试。命令中 **`${MODEL_DIR}`、`${LOG_DIR}`** 须已由上文 `export` 赋值。

```bash
LOADWORKER=14 \
nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" --tp 8 \
  --graph_max_batch_size 32 \
  --tool_call_parser deepseekv32 \
  --mem_fraction 0.8 \
  --reasoning_parser deepseek-v3 \
  --dp 8 --enable_ep_moe \
  --port 8000 \
  >> "${LOG_DIR}/server_v32_ep.log" 2>&1 &
```

### 评测命令（服务就绪后执行一次）

服务就绪后执行（本地回环走代理时用 `no_proxy` / `NO_PROXY` 排除本机）。**`base_url` 中的端口须为 `8000`，与 `api_server` 的 `--port` 一致。** 以下为带日志落盘的**完整命令**：

```bash
HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 \
no_proxy=127.0.0.1,localhost,::1 \
lm_eval --model local-completions \
  --model_args '{"model":"deepseek-ai/DeepSeek-V3.2", "base_url":"http://localhost:8000/v1/completions", "tokenizer_backend":null, "eos_string":"<｜end▁of▁sentence｜>"}' \
  --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code \
  >> "${LOG_DIR}/eval_gsm8k.log" 2>&1
```

> **为什么用 `tokenizer_backend=null` 而非 `tokenizer=${MODEL_DIR}`**：`local-completions` 默认会用 `transformers.AutoTokenizer.from_pretrained(${MODEL_DIR})` 在本地加载 HF tokenizer，但当前环境的 **transformers 不识别 `model_type: deepseek_v32`**（`KeyError: 'deepseek_v32'` → rope `AttributeError`），评测在加载 tokenizer 阶段即崩溃，根本跑不到推理。设 **`tokenizer_backend=null`** 后 lm_eval 不再本地加载 tokenizer，直接把 **prompt 文本**发给 server，由 lightllm 服务端用真正的 deepseek_v32 tokenizer 分词——更贴合实际且无需本地 HF 适配。`eos_string` 显式给出 DeepSeek 的结束符以消除 “Cannot determine EOS string” 告警（gsm8k 本身也带 stop 序列）。`tokenized_requests` 会被自动关闭、不再做 context 长度校验（gsm8k 5-shot prompt 很短，无需截断）。
> 若哪天升级了能识别 `deepseek_v32` 的 transformers，可改回 `"tokenizer":"${MODEL_DIR}"` 形式（届时 tokenizer 须与 `--model_dir` 同一路径）。

- **`LOG_DIR`**：与启动服务一节相同；若仅调试不重定向，去掉 `\` 续行及最后的 `>> "${LOG_DIR}/eval_gsm8k.log" 2>&1` 即可在前台查看输出。
- **tokenizer**：本命令用 `tokenizer_backend=null`，评测端不再依赖 `MODEL_DIR` 下的 HF tokenizer（分词在 server 端完成），故 `MODEL_DIR` 路径变化不影响评测命令；server 启动命令中的 `--model_dir` 仍按「执行约定」处理。
- 若环境需要，可同时设置 `NO_PROXY=127.0.0.1,localhost,::1`（或与团队约定一致的列表）。

## 执行约定（不要额外写“专用启动脚本”）

**模型目录（随环境变化）**：`MODEL_DIR` 在不同机器上路径不同。**首轮试跑**可先用下列默认（与本文档常见部署对应；若本机不存在则跳过默认、直接执行下一步「向用户确认」）：

```bash
export MODEL_DIR=/mtc/models/DeepSeek-V3.2
```

若按默认路径 **export** 后仍无法启动服务，或日志中出现**明确的模型路径 / 权重加载 / 文件不存在**等错误，**不要反复盲试**：根据日志判断为路径问题时，**请用户提供**当前环境下实际的主模型目录，更新 `export MODEL_DIR=…` 后再执行（且保证 **`MODEL_DIR` 与 `lm_eval` 的 `tokenizer` 仍为同一路径**）。

1. **后台启动 server**：用 shell 后台或终端任务跑 `python -m lightllm.server.api_server ...`，**并将该进程输出重定向到本轮日志目录下的日志文件**（见上文「日志目录（含 summary.txt）」）；排查问题时 **tail** 该文件，而不是依赖未落盘的终端缓冲。
2. **不要用 health 接口** 判断就绪；改为探测 **端口 8000 是否处于 listen**（例如 `ss -tlnp` / `lsof -i :8000` 等，与系统一致即可）。
3. **等待启动**：若端口未就绪，约 **每 20 秒** 查看一次**服务日志文件**，区分仍在启动还是已报错退出；报错则写入日志目录下的 **`summary.txt`**（或先写服务日志再在 `summary.txt` 引用）并停止，不要继续盲等。
4. **维护 `summary.txt`**：位于**日志目录**；记录**本条使用的完整启动命令**（须能看出 **`--tp 8`、`--dp 8`、EP MoE**）、**端口检测结果**、**`lm_eval` 关键输出**；全部结束后在该文件内写**最终汇总**（是否成功、主要指标或失败原因）。可与用户口头摘要对照，但以日志目录中 **`summary.txt`** 为归档准绳。
5. **全部完成后**：确认日志目录下的 **`summary.txt`** 已包含完整最终总结；原始 server / eval 日志保留在同目录（或子目录）中备查。

### 服务启动 OK 判定经验（本流程补充）

- **不要只看“主进程在不在”**：`python -m lightllm.server.api_server` 进程存活不代表可用；必须至少满足“`8000` 已 listen”再进入评测。
- **长时间加载不等于失败**：DeepSeek-V3.2 EP 首次加载可能持续数分钟。若日志持续出现 `Loading model weights ...` 进度推进，视为“仍在启动”，继续按 20 秒间隔观察。
- **判定“启动 OK”建议三要素**：① `8000` 端口监听；② 服务日志无新的 `OutOfMemoryError`/Traceback；③ 用一条最小请求（如 1 条 completions/chat 请求）拿到 200 或有效响应，再跑 `lm_eval`。
- **出现 OOM 要先清残留再重试**：一旦日志出现 `torch.OutOfMemoryError`，先结束该轮 `api_server` 及其派生进程（含 `hypercorn`/`lightllm::...` 子进程），确认 `8000` 释放后再重启，避免“旧进程占资源导致假失败”。
- **重试优先调启动参数而非盲等**：若 OOM 发生在权重加载阶段，优先降低加载/显存压力（例如使用更保守的 `mem_fraction`），并在 `summary.txt` 记录“失败参数 -> 重试参数 -> 结果”。

## 输出文件

- **`summary.txt`**：仅位于**本轮日志目录**，作为本次 **DeepSeek-V3.2 EP** 评测的**最终总结**文档。
- **服务与评测日志**：全部落在**同一日志目录**（建议按任务命名文件或分子目录），不得与未指定目录混写。
