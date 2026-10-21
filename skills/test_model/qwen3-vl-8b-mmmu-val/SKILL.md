---
name: test-model-qwen3-vl-8b-mmmu-val
description: >-
  LightLLM Qwen3-VL-8B-Instruct: api_server tp 2 on port 8089, then lmms-eval CLI
  (python -m lmms_eval, model openai_compatible, tasks mmmu_val, batch_size 900)
  with OPENAI_API_BASE pointing at LightLLM OpenAI-compatible /v1. Restore https_proxy for Hub
  while no_proxy includes 127.0.0.1. Requires lmms-eval install, OPENAI_API_KEY placeholder,
  LOG_DIR and MODEL_DIR, nvidia-smi GPU choice, pipefail with tee, summary.txt. No wrapper
  script; use command line only.
---

# Qwen3-VL-8B-Instruct **MMMU 验证集（`mmmu_val`）** 评测

**测试标识**：先在本机启动 **`lightllm.server.api_server`**（**Qwen3-VL-8B-Instruct**，**`--tp 2`**，HTTP **`8089`**）；服务就绪后，在已安装 **`lmms-eval`** 的环境中直接执行 **`python3 -m lmms_eval`**（**`openai_compatible`**，任务 **`mmmu_val`**），通过环境变量 **`OPENAI_API_BASE`** 指向 **`api_server`** 的 OpenAI 兼容前缀（**含 `/v1`**）。

**依赖（评测侧）**：**`lmms-eval`**（版本示例）：

```text
git clone --branch v0.3.3 --depth 1 https://github.com/EvolvingLMMs-Lab/lmms-eval.git
pip install -e lmms-eval/
```

执行 **`python3 -m lmms_eval`** 的 Python 环境须已安装上述包；**不要求**在 LightLLM 仓库根目录下执行（除非你的数据或配置依赖 **`cwd`**）。

## 日志目录（含 `summary.txt`）

- 选定 **`LOG_DIR`**（绝对路径建议带时间戳）。
- **`api_server`** → **`"${LOG_DIR}/server.log"`**（推荐 **`nohup`** 后台）。
- **`lmms_eval`** 的 **`--output_path`**：**建议 `"${LOG_DIR}/lmms_eval_out"`**；控制台输出可 **`tee`** 到 **`"${LOG_DIR}/lmms_eval_console.log"`**。
- **`summary.txt`**：模型路径、**`OPENAI_API_BASE`**、完整 **`lmms_eval` 命令**、端口检测结果、输出目录路径、失败原因。

## 启动前检查

1. **显卡**：**`--tp 2`** → **2 张物理 GPU**；先 **`nvidia-smi`**，再 **`export CUDA_VISIBLE_DEVICES`**（**不要写死**）。
2. **端口**：**`8089`** 未被占用。
3. **`MODEL_DIR`**：**`api_server --model_dir`** 与 **`--model_args` 里的 `model_version=`** 须为**同一 Qwen3-VL-8B-Instruct 权重路径**（默认示例 **`/mtc/models/Qwen3-VL-8B-Instruct`**；不存在时向用户询问本机路径）。
4. **`lmms-eval` 已安装**且 **`python3 -m lmms_eval`** 可用。
5. **代理**：启动 **`api_server` 前**清空 **`http_proxy` / `https_proxy`**；跑 **`lmms_eval` 前**将 **`no_proxy`** 设为包含本机 **`127.0.0.1`**（见下文评测块）；**若需从 Hugging Face Hub 拉取 `lmms-lab/MMMU`，评测阶段应恢复可用的 `https_proxy`（或等价镜像）**，否则清空代理后可能出现 **`ConnectionError: Couldn't reach 'lmms-lab/MMMU' on the Hub`**。

## 可变项

| 变量 | 含义 |
|------|------|
| `LOG_DIR` | 本轮日志与 **`lmms_eval --output_path`** 父目录。 |
| `MODEL_DIR` | **`api_server --model_dir`**；**`--model_args` 中 `model_version=`** 与之相同。 |
| `PORT` | 默认 **`8089`**。 |
| `BIND_URL_HOST` | 与 **`OPENAI_API_BASE`** 主机一致；本机常用 **`127.0.0.1`**。 |
| `OPENAI_API_BASE` | 形如 **`http://${BIND_URL_HOST}:${PORT}/v1`**（**末尾含 `/v1`**）。 |
| `OPENAI_API_KEY` | 占位即可（常用 **`lightllm123`**）；若服务端校验密钥，与用户环境对齐。 |
| `CUDA_VISIBLE_DEVICES` | 两张卡。 |

## 启动 `api_server`

**不要用 health 作为唯一依据**；以 **端口 listen** + **`server.log`** 为准；可约 **每 20 秒**查看日志。

```bash
export http_proxy=
export https_proxy=

export LOG_DIR='〈日志目录〉'
export MODEL_DIR='/mtc/models/Qwen3-VL-8B-Instruct'
export PORT=8089

LOADWORKER=18 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --port "${PORT}" \
  >> "${LOG_DIR}/server.log" 2>&1 &
```

## 运行 `lmms_eval`（服务就绪后，仅命令行）

设置 **`OPENAI_API_*`** 与代理后，直接 **`python3 -m lmms_eval`**（**`timeout` 可选**，例如单次上限 **3600 秒**）：

```bash
# 若启动 api_server 时曾清空代理，请先保存并在评测前恢复 Hub 代理，例如：
#   export ORIG_HTTPS_PROXY="${https_proxy-}"
#   export http_proxy=; export https_proxy=
#   … 启动 api_server …
#   export https_proxy="${ORIG_HTTPS_PROXY}"

export BIND_URL_HOST='127.0.0.1'
export PORT=8089
export OPENAI_API_BASE="http://${BIND_URL_HOST}:${PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-lightllm123}"
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,${BIND_URL_HOST}

export LOG_DIR='〈与上文同一日志目录〉'
export MODEL_DIR='/mtc/models/Qwen3-VL-8B-Instruct'

mkdir -p "${LOG_DIR}/lmms_eval_out"

timeout 3600 python3 -m lmms_eval \
  --model openai_compatible \
  --model_args "model_version=${MODEL_DIR},tp=1" \
  --tasks mmmu_val \
  --batch_size 900 \
  --log_samples \
  --log_samples_suffix openai_compatible \
  --output_path "${LOG_DIR}/lmms_eval_out" \
  2>&1 | tee "${LOG_DIR}/lmms_eval_console.log"
```

说明：**`model_args` 中的 `tp=1`** 为 **`lmms_eval` / `openai_compatible` 侧参数**，与 **`api_server` 的 `--tp 2`** 不同；**不要**混用含义。

若环境无 **`timeout`** 命令，可去掉 **`timeout 3600`**。

## 执行约定

1. **顺序**：**`api_server` 就绪** → 再 **`lmms_eval`**。
2. **`model_version` 与 `MODEL_DIR` 必须一致**。
3. 超时或失败将摘要写入 **`summary.txt`**。
4. 结束后关闭 **`api_server`**，释放 GPU 与端口。
