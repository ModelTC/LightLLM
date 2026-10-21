---
name: test-model-qwen3-vl-8b-vit-sep-mode
description: >-
  LightLLM Qwen3-VL-8B-Instruct visual separation (ViT sep / proxy): three processes in
  order—config_server on 8090; internal Redis on 6000; visual_only with visual_rpyc 8091
  and afs_image_embed_dir; normal api_server tp 2 port 8089 with visual_use_proxy_mode.
  After HTTP /v1/models on normal, lmms_eval mmmu_val (openai_compatible, batch 900,
  OPENAI_API_BASE http://HOST:8089/v1); restore https_proxy for Hub while no_proxy includes 127.0.0.1.
  lmms_eval_out, console log, mmmu_acc in summary. pipefail for tee exit code.
---

# Qwen3-VL-8B-Instruct **视觉分离（`visual_only` + `normal` + `config_server`）**

**测试标识**：按顺序启动 **三条** `api_server` 进程——**`config_server`**（配置与元数据；**进程内部会启动 Redis 服务，并通过 `--config_server_visual_redis_port`（默认 `6000`）对外暴露**）、**`visual_only`**（独立视觉 / ViT 侧）、**`normal`**（主 LLM，**`--visual_use_proxy_mode`** 经 config 访问视觉侧）。**`6000` 不是本机另行安装的 `redis-server`**，勿与系统包管理器里的 Redis 混为一谈。本流程验证 **ViT 分离 + AFS 嵌入目录** 的联调，并在 **`normal` 就绪后须强制跑通 MMMU 验证集 `mmmu_val`**（**`lmms_eval` + `openai_compatible`**，命令与 **`skills/test_model/qwen3-vl-8b-mmmu-val/SKILL.md`** 评测块一致；仅 **`api_server` 拓扑** 为本文的 **visual 分离三进程**）。

**端口约定（固定，与脚本一致）**：

| 用途 | 端口 |
|------|------|
| **`config_server`** | **`8090`**（**`--config_server_port`**） |
| **Redis（由 `config_server` 内部启动并对外暴露）** | **`6000`**（**`--config_server_visual_redis_port`**；与系统独立安装的 Redis 无关） |
| **`visual_only` RPyC** | **`8091`**（**`--visual_rpyc_port`**） |
| **`normal` HTTP** | **`8089`**（**`--port`**） |

**算力**：**`visual_only`** 默认 **1 张 GPU**；**`normal`** 默认 **`--tp 2`** → **2 张 GPU**；**三组进程不得争抢同一物理 GPU**（脚本示例为 **visual：`0`**，**LLM：`6,7`**；实际以 **`nvidia-smi`** 选定）。

## 依赖

- **Python 环境**：与运行 **`lightllm.server.api_server`** 的虚拟环境一致。
- **`mmmu_val` 评测（必须）**：已安装 **`lmms-eval`**，**`python3 -m lmms_eval`** 可用（安装示例见 **`skills/test_model/qwen3-vl-8b-mmmu-val/SKILL.md`**）；未完成 **`mmmu_val`** 则本轮 **不算通过**。
- **`6000` 端口**：由 **`config_server` 在启动后内部拉起 Redis 并监听**；**无需**、也**不应**依赖「事先在本机 **`apt install redis-server`** 并独占 **`6000`**」——若系统已有其它服务占用 **`6000`**，须释放或改 **`--config_server_visual_redis_port`**（**`visual_only` / `normal` 须同步同一端口参数**）。

## 日志目录（含 `summary.txt`）

- 选定 **`LOG_DIR`**，三条进程日志建议：**`"${LOG_DIR}/config_server.log"`**、**`"${LOG_DIR}/visual_only.log"`**、**`"${LOG_DIR}/normal.log"`**（**`nohup … >> … 2>&1 &`**）。
- **`summary.txt`**：三条命令摘要、各端口 listen 情况、**`MODEL_DIR` / `AFS_IMAGE_EMBED_DIR`** 最终取值；**`mmmu_val` 必记**：**`OPENAI_API_BASE`**、完整 **`lmms_eval` 命令**、**`lmms_eval_console.log`** 与 **`lmms_eval_out`** 路径、**`mmmu_acc`**（见下文「精度」）；失败时写清原因与结论。
- **`lmms_eval_console.log`**（**必须**）：**`lmms_eval`** 终端输出（**`tee`**）。
- **`lmms_eval_out/`**（**必须**）：**`--output_path`** 下的 **`*_results.json`**、**`*_samples_mmmu_val.jsonl`**（**`--log_samples`** 生成样本日志）。

## 启动前检查

1. **端口**：本机 **`8090`、`6000`、`8091`、`8089`** 未被其它进程占用（**`8090` / `6000` 在 `config_server` 启动后由其占用**）。
2. **`config_server` 已就绪**：**`8090`** 与 **`6000`** 均已 **LISTEN**（表明 **HTTP 配置面** 与 **内部 Redis 暴露面** 已起来），再启动 **`visual_only`**。
3. **`MODEL_DIR`**：指向 **Qwen3-VL-8B-Instruct**；默认试跑 **`/mtc/models/Qwen3-VL-8B-Instruct`**；不存在或加载失败时 **向用户询问** 本机路径。
4. **`AFS_IMAGE_EMBED_DIR`**：**`--afs_image_embed_dir`** 指向的目录须存在或可创建；默认试跑 **`/mtc/afs/vit_embed_dir`**；不可用时 **向用户询问**。
5. **显卡**：**`visual_only`** 与 **`normal`** 的 **`CUDA_VISIBLE_DEVICES`** **不得重叠**；先 **`nvidia-smi`** 再 **`export`**（**不要写死**示例卡号）。
6. **`CONFIG_SERVER_HOST`**：各进程 **`--config_server_host`** 须能访问 **`config_server` 的 HTTP 服务**。**三进程同机**时常见为 **`0.0.0.0`**（与历史脚本一致）；多机时改为 **对端可达的 IP**，并保证防火墙放行 **`config_server` 端口（8090）**、**其内部 Redis 暴露端口（6000）**、**`8091`** 等。
7. **代理**：每条 **`python`（LightLLM）** 前 **`export http_proxy=`**、**`export https_proxy=`**（与仓库其它 acc 测试一致）。**`lmms_eval` 拉取 `lmms-lab/MMMU` 时**若需经企业代理访问 Hugging Face，应在 **`no_proxy` 已包含 `127.0.0.1`** 的前提下**恢复 `https_proxy`（或等价镜像）**；否则易出现 Hub **`ConnectionError`**。仅当本机已有完整离线缓存且确认 **`datasets` 可纯离线命中**时，才可全程无代理。
8. **`lmms-eval`**：**`python3 -m lmms_eval --help`** 可执行；否则无法完成必跑 **`mmmu_val`**。

## 可变项

| 变量 | 含义 |
|------|------|
| `LOG_DIR` | 本轮日志根目录。 |
| `MODEL_DIR` | **`--model_dir`**（三条中涉及模型的命令一致）。 |
| `AFS_IMAGE_EMBED_DIR` | **`--afs_image_embed_dir`**；**`visual_only` 与 `normal` 须一致**。 |
| `CONFIG_SERVER_HOST` | **`--config_server_host`**；同机试跑常用 **`0.0.0.0`**。 |
| `VISUAL_CUDA_DEVICES` | **`visual_only`** 的 **`CUDA_VISIBLE_DEVICES`**（1 张卡）。 |
| `LLM_CUDA_DEVICES` | **`normal`** 的 **`CUDA_VISIBLE_DEVICES`**（**`--tp 2`** → 2 张卡）。 |
| `AFS_EMBED_CAPACITY` | **`--afs_embed_capacity`**；默认 **`250000`**；调试替换逻辑时可改为较小值（例如 **`100`**），见「调试提示」。 |
| `ORIG_HTTP_PROXY` / `ORIG_HTTPS_PROXY` | 在清空代理启动 LightLLM **之前**备份（见 **`lmms_eval` 命令块**），评测阶段恢复以便访问 Hugging Face Hub。 |

**开跑前导出示例**：

```bash
export ORIG_HTTP_PROXY="${http_proxy-}"
export ORIG_HTTPS_PROXY="${https_proxy-}"
export LOG_DIR='〈日志目录〉'
export MODEL_DIR='/mtc/models/Qwen3-VL-8B-Instruct'
export AFS_IMAGE_EMBED_DIR='/mtc/afs/vit_embed_dir'
export CONFIG_SERVER_HOST='0.0.0.0'
export AFS_EMBED_CAPACITY=250000
# export VISUAL_CUDA_DEVICES='0'
# export LLM_CUDA_DEVICES='6,7'
```

## 服务就绪判定

**不要使用 HTTP health 作为唯一依据**。依次确认：**`config_server` 已占用 `8090` 与 `6000`（内部 Redis）** → **`8091`（`visual_only` RPyC）** → **`8089`（`normal`）** 的 **LISTEN** 状态，并结合各 **`*.log`**；可约 **每 20 秒** 查看日志直至就绪或报错。

## 启动命令（须按顺序）

以下块前均须 **`export http_proxy=`**、**`export https_proxy=`**；生产式跑法请自行加 **`nohup`** 与 **`>> "${LOG_DIR}/….log" 2>&1 &`**。

### 1）`config_server`（最先）

```bash
python -m lightllm.server.api_server \
  --run_mode config_server \
  --config_server_host 0.0.0.0 \
  --config_server_port 8090 \
  --config_server_visual_redis_port 6000
```

若仅需绑定到 **`CONFIG_SERVER_HOST`**，将 **`--config_server_host`** 改为 **`"${CONFIG_SERVER_HOST}"`**（须与 **`visual_only` / `normal` 中的 `--config_server_host`** 指向同一可达地址）。

### 2）`visual_only`（**`config_server` 已在 8090 / 6000 就绪后**）

**`--visual_rpyc_port 8091`** 为 **visual_only 模式必需**，供其它服务调用本机视觉推理接口。

```bash
CUDA_VISIBLE_DEVICES="${VISUAL_CUDA_DEVICES}" python -m lightllm.server.api_server \
  --run_mode visual_only \
  --host 0.0.0.0 \
  --config_server_host "${CONFIG_SERVER_HOST}" \
  --config_server_port 8090 \
  --config_server_visual_redis_port 6000 \
  --model_dir "${MODEL_DIR}" \
  --visual_dp 1 \
  --visual_tp 1 \
  --afs_image_embed_dir "${AFS_IMAGE_EMBED_DIR}" \
  --afs_embed_capacity "${AFS_EMBED_CAPACITY}" \
  --visual_rpyc_port 8091
```

**`--host`** 为 **本进程监听地址**；与 **`--config_server_host`** 含义不同：后者为 **config_server 的可达地址**。

### 3）`normal`（visual 就绪后）

```bash
CUDA_VISIBLE_DEVICES="${LLM_CUDA_DEVICES}" python -m lightllm.server.api_server \
  --run_mode normal \
  --model_dir "${MODEL_DIR}" \
  --tp 2 \
  --port 8089 \
  --config_server_host "${CONFIG_SERVER_HOST}" \
  --config_server_port 8090 \
  --config_server_visual_redis_port 6000 \
  --visual_dp 1 \
  --afs_image_embed_dir "${AFS_IMAGE_EMBED_DIR}" \
  --afs_embed_capacity "${AFS_EMBED_CAPACITY}" \
  --visual_use_proxy_mode
```

## **`mmmu_val` 评测（`normal` 已监听 `8089` 后，必须执行）**

在 **`config_server` / `visual_only` / `normal` 均就绪**、**`normal`** 仍占用 **`8089`** 时执行；**关停任一服务前须跑完本节**。评测流量只打 **`normal` HTTP**；**`OPENAI_API_BASE`** 须指向 **`http://〈可达主机〉:8089/v1`**（**末尾含 `/v1`**）。**`--model_args` 中 `model_version=` 与三进程共用的 `MODEL_DIR` 须为同一权重目录**。

```bash
export BIND_URL_HOST='127.0.0.1'
export PORT=8089
export OPENAI_API_BASE="http://${BIND_URL_HOST}:${PORT}/v1"
export OPENAI_API_KEY="${OPENAI_API_KEY:-lightllm123}"
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,${BIND_URL_HOST}

# 若启动 LightLLM 时清空了代理，此处恢复 Hub 代理（勿把 127.0.0.1 放进 ALL_PROXY）
export http_proxy="${ORIG_HTTP_PROXY:-}"
export https_proxy="${ORIG_HTTPS_PROXY:-}"

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

说明：**`model_args` 中的 `tp=1`** 为 **`lmms_eval` / `openai_compatible` 侧参数**，与 **`normal` 的 `--tp 2`** 不同。若环境无 **`timeout`**，去掉 **`timeout 3600`**。仅本地冒烟可在命令中加 **`--limit`**；**正式回归不得省略全量 `mmmu_val`**。

**精度（必须写入 `summary.txt`）**：最新 **`"${LOG_DIR}/lmms_eval_out"/*_results.json`** 中 **`results.mmmu_val["mmmu_acc,none"]`**（**0～1**）；无 **`jq`** 时打开 JSON 或对照 **`lmms_eval_console.log`** 末尾汇总表。

## 调试提示（可选）

- 将 **`AFS_EMBED_CAPACITY`** 设为较小值（例如 **`100`**）可更快触发 **嵌入目录替换 / 淘汰** 相关逻辑，便于缩短调试周期；正式回归再恢复 **`250000`**（或业务约定值）。

## 执行约定

1. **顺序**：**`config_server` → `visual_only` → `normal` → `mmmu_val`（`lmms_eval`）**，前三步不可颠倒；**`mmmu_val` 未完成不得视为本轮通过**。
2. **`mmmu_val`**：须在 **`normal` 就绪** 之后、**关停 `normal` 之前** 跑完（依赖 **`8089`** 与 **`OPENAI_API_BASE`**）。**`model_version` 与 `MODEL_DIR` 必须一致**。
3. **关停**：**`mmmu_val` 成功或失败均已落盘**（日志与 **`summary.txt`**）后，依次结束 **`normal`、`visual_only`、`config_server`**；**`config_server` 退出后，其内部在 `6000` 上暴露的 Redis 随之停止**，释放端口与 GPU。
4. **失败**：将摘要写入 **`summary.txt`**（含 **`lmms_eval` 退出码**、**`lmms_eval_console.log`** 末尾），并在对话中给出关键日志与端口状态。
