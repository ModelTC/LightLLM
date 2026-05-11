---
name: test-model-qwen3-8b-pd-nccl
description: >-
  LightLLM Qwen3-8b PD disaggregation gsm8k: pd_master on 8089, prefill on 8001, decode on
  8002, tp 2 each. Assign four GPUs by running nvidia-smi and deciding prefill/decode pairs
  (no fixed card IDs; no complex shell automation). lm_eval hits pd_master URL. HOST vs
  PD_MASTER_IP when co-located. Requires LOG_DIR, MODEL_DIR, proxy cleared, no_proxy, summary.txt.
  Use for PD NCCL-style separation tests.
---

# Qwen3-8B **PD 分离**（`pd_master` + `prefill` + `decode`）本地 GSM8K 评测

**测试标识**：同一 **`--model_dir`**（Qwen3-8B）下拆 **三条** `api_server` 进程——**调度/入口（pd_master）**、**prefill 节点**、**decode 节点**；评测 **`lm_eval`** 只访问 **`pd_master` 的 HTTP 端口（8089）**，由调度转发到 PD 链路。与单机单进程 Base–TP、MTP 等流程区分。

**端口约定**：**`pd_master`：`8089`**；**prefill：`8001`**；**decode：`8002`**。启动与就绪探测须覆盖这三处（以及日志中的 PD 注册/报错信息）。

**绑定 IP（`HOST` / `PD_MASTER_IP`）**：各进程的 **`--host`** 表示 **本服务监听绑定的 IP**（与其它集群「逻辑 hostname」概念区分时，此处一律按 **绑定地址** 理解）。当 **`pd_master`、`prefill`、`decode` 部署在同一台机器上时**，三者使用的绑定 IP **相同**：此时可只做一次赋值 **`export HOST="${PD_MASTER_IP}"`**（或先将本机对外/LAN IP 赋给 **`PD_MASTER_IP`**，再 **`export HOST="${PD_MASTER_IP}"`**），保证 **`pd_master` 的 `--host`** 与 **prefill/decode 的 `--host`** 一致；**`lm_eval` 的 `base_url` 仍指向 `pd_master`**，故 **`PD_MASTER_IP`** 也同时作为评测 URL 中的主机名。

整轮产物落在**同一日志目录**，写入 **`summary.txt`** 与各进程日志（见「日志目录」）；**不要**写聚合启动脚本，按「启动说明」逐条手动启动并在后台落盘。

## 日志目录（含 `summary.txt`）

- 每次评测先选定或新建**一个日志目录**（例如带时间戳或任务名），与其它测试轮次分开。
- **三个 `api_server` 的标准输出/错误**分别写入该目录，建议命名：**`pd_master.log`**、**`prefill.log`**、**`decode.log`**（或分子目录 `pd_master/`、`prefill/`、`decode/`）。
- **`summary.txt` 固定放在该日志目录下**，汇总：三台进程的启动参数摘要、端口与就绪情况、`lm_eval` 关键结果、失败原因与最终结论。
- **`eval_gsm8k.log`**：`lm_eval` 终端输出；**`summary.txt`** 仍为**总览结论**。

## 启动说明

本节包含：启动前检查 → 可变项说明 → 显卡分配 → **按顺序**三条完整 server 命令 → 评测命令。

### 启动前检查

开跑前先确认资源与环境可用；**不满足则先清理占用端口的进程或释放 GPU**，再按顺序启动。

1. **显卡**：prefill / decode 各需 **2 张物理 GPU**（**`--tp 2`**），共 **4 张互不重复**的卡。**不要写死卡号**：先 **`nvidia-smi`**（见下文「显卡分配」），由执行者根据占用与集群情况选定 **prefill 两张、decode 两张**，再 **`export PREFILL_CUDA_DEVICES`**、**`DECODE_CUDA_DEVICES`** 后启动。
2. **端口**：**`8089`、`8001`、`8002`** 均须未被监听（`ss -tlnp`、`lsof -i :端口` 等）；若被占用，结束占用进程后再启动。
3. **网络 / IP**：**`HOST`** 为 **prefill / decode 的服务绑定 IP**；**`PD_MASTER_IP`** 为 **`pd_master` 的 `--host`**，且与 **`lm_eval` 访问地址**一致。**单机三进程同机时**：**`HOST` 与 `PD_MASTER_IP` 取同一值**（见上文「绑定 IP」）；多机分发时再按各节点真实监听地址分别设置。
4. **代理**：启动 **任一 server 前**将 **`http_proxy` / `https_proxy` 置空**（见各命令块前 `export`）；避免代理干扰本地 PD 通信。**评测阶段**使用 **`no_proxy`** 排除本机（见评测命令）；若需先用代理下载 `lm_eval` 缓存，见「执行约定」。

### 启动服务的命令模板（可变项）

| 可变项 | 含义 |
|--------|------|
| `LOG_DIR` | 本轮日志根目录，建议**绝对路径**；`export LOG_DIR=…`。 |
| `MODEL_DIR` | 模型目录，对应三条命令中的 **`--model_dir`**；`lm_eval` 的 **`tokenizer` 须与此路径一致**。 |
| `PD_MASTER_IP` | **`pd_master` 进程 `--host`** 所使用的 **绑定 IP**；同时也是 **`lm_eval` 里 `base_url` 的主机部分**（评测客户端访问 pd_master 的地址）。 |
| `HOST` | **`prefill` / `decode` 进程 `--host`** 所使用的 **绑定 IP**（本服务监听地址）。**与 `pd_master` 同机时**：与 **`PD_MASTER_IP` 相同**，可 **`export HOST="${PD_MASTER_IP}"`**。 |
| `PREFILL_CUDA_DEVICES` | **prefill** 的 **`CUDA_VISIBLE_DEVICES`**，形如 `a,b`（两张物理卡索引）；由 **`nvidia-smi`** 判断后 **`export`**。 |
| `DECODE_CUDA_DEVICES` | **decode** 的 **`CUDA_VISIBLE_DEVICES`**，形如 `c,d`；与 prefill **四卡互不重复**。 |
| `pd_master.log` 等 | 文件名示例，可改名。 |

开跑前导出（引号内替换为本机实际值）：

```bash
export LOG_DIR='〈日志根目录〉'
export MODEL_DIR='〈Qwen3-8B 模型目录〉'
export PD_MASTER_IP='〈本机绑定 IP：pd_master --host，且供 lm_eval 访问〉'
# 单机：prefill/decode 与 pd_master 同机时，绑定同一 IP
export HOST="${PD_MASTER_IP}"
# 多机：若 prefill/decode 监听地址不同，再单独 export HOST='〈该机上绑定 IP〉'
```

首次试跑可用的**默认 `MODEL_DIR`** 见「执行约定」。

### 显卡分配（`nvidia-smi` + 人工/Agent 决策，不用复杂脚本）

约束：**prefill**、**decode** 各 **2 张物理 GPU**（**`--tp 2`**），共 **4 张互不重复**；**不要**默认写死 `0,1` / `2,3`。

1. **查看占用**：在启动 **prefill** 之前（**`pd_master` 已起来之后**即可），执行 **`nvidia-smi`**，需要时可带列表输出便于比对：  
   `nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv`  
   或先看总览再决定。
2. **选定四卡**：由**执行本 skill 的 Agent（或操作者）**根据上述输出、机器上其它任务占用、是否需避开某几张卡等因素，**自行决定**哪 **2** 张给 **prefill**、哪 **2** 张给 **decode**（两组索引不得重叠）。
3. **写入环境变量**：在同一 shell 中 **`export`**（示例数值仅作格式说明）：

```bash
export PREFILL_CUDA_DEVICES='〈物理索引1〉,〈物理索引2〉'
export DECODE_CUDA_DEVICES='〈物理索引3〉,〈物理索引4〉'
```

4. **记录**：把最终 **`PREFILL_CUDA_DEVICES`**、**`DECODE_CUDA_DEVICES`** 及当时 **`nvidia-smi` 要点**记入 **`summary.txt`**。

**禁止**：不必编写 **awk / mapfile / 长段 bash** 自动选卡脚本；以 **`nvidia-smi` 事实 + 明确决策**为准。

### 1）启动 `pd_master`（须最先就绪监听）

每条命令前清空代理；以下为 **可直接执行** 的后台形式（含 **`nohup`** 与重定向）。若调试可去掉 `nohup` 与 `>> … &`。

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

### 2）启动 `prefill` 节点

**须在 pd_master 已监听且日志无致命错误后再启动**（见「执行约定」）。启动本命令前须已完成 **`nvidia-smi` 决策并 `export PREFILL_CUDA_DEVICES=…`**（见「显卡分配」）。

```bash
export http_proxy=
export https_proxy=

LOADWORKER=18 CUDA_VISIBLE_DEVICES="${PREFILL_CUDA_DEVICES}" \
nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --run_mode prefill \
  --tp 2 \
  --dp 1 \
  --host "${HOST}" \
  --port 8001 \
  --disable_cudagraph \
  --pd_master_ip "${PD_MASTER_IP}" \
  --pd_master_port 8089 \
  >> "${LOG_DIR}/prefill.log" 2>&1 &
```

### 3）启动 `decode` 节点

启动前须已完成 **`export DECODE_CUDA_DEVICES=…`**（见「显卡分配」）。

```bash
export http_proxy=
export https_proxy=

LOADWORKER=18 CUDA_VISIBLE_DEVICES="${DECODE_CUDA_DEVICES}" \
nohup python -m lightllm.server.api_server \
  --model_dir "${MODEL_DIR}" \
  --run_mode decode \
  --tp 2 \
  --dp 1 \
  --host "${HOST}" \
  --port 8002 \
  --pd_master_ip "${PD_MASTER_IP}" \
  --pd_master_port 8089 \
  >> "${LOG_DIR}/decode.log" 2>&1 &
```

### 评测命令（prefill / decode 已与 pd_master 建立 PD 链路后执行）

**`base_url` 指向 `pd_master`**：`http://${PD_MASTER_IP}:8089/v1/completions`。以下为带日志落盘的**完整命令**（`--model_args` 使用双引号以展开变量；**`tokenizer` 与 `MODEL_DIR` 一致**）：

```bash
export http_proxy=
export https_proxy=

HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=0 \
no_proxy=localhost,127.0.0.1,0.0.0.0,::1,${PD_MASTER_IP} \
lm_eval --model local-completions \
  --model_args "{\"model\":\"qwen/qwen3-8b\", \"base_url\":\"http://${PD_MASTER_IP}:8089/v1/completions\", \"max_length\": 16384, \"tokenized_requests\": false, \"tokenizer\":\"${MODEL_DIR}\"}" \
  --tasks gsm8k --batch_size 500 --confirm_run_unsafe_code \
  >> "${LOG_DIR}/eval_gsm8k.log" 2>&1
```

- **`no_proxy`**：须包含本机与 **`PD_MASTER_IP`**（及脚本中的 `0.0.0.0`、`::1` 等），避免评测流量误走 HTTP 代理。
- 若环境需要，可同时设置 **`NO_PROXY`** 与 **`no_proxy`** 一致。
- **`tokenized_requests`: `false`** 与脚本一致。
- 调试可不重定向：去掉末尾 `>> "${LOG_DIR}/eval_gsm8k.log" 2>&1`。

## 执行约定（不要额外写“专用启动脚本”）

**模型目录**：**首轮试跑**可先：

```bash
export MODEL_DIR=/mtc/models/qwen3-8b
```

无法启动或路径类报错时，**请用户提供**本机实际 **`MODEL_DIR`**；保持 **`tokenizer` 与 `--model_dir` 同路径**。

**`lm_eval` 与代理 / 缓存**：若评测依赖首次下载缓存，可先**保留代理**单独跑一次 `lm_eval` 完成缓存下载，再**清空代理**并按上文 **`no_proxy`** 跑正式评测（与脚本注释一致）。

1. **启动顺序**：先 **`pd_master`** → 再 **`nvidia-smi` 决策并 `export PREFILL_CUDA_DEVICES` / `DECODE_CUDA_DEVICES`** → 再 **prefill** → 再 **decode**；不要颠倒。每一步将输出重定向到 **`LOG_DIR`** 下对应日志。
2. **不要用 health 接口** 作为唯一依据；改为：**端口 listen**（8089 / 8001 / 8002）并结合日志判断是否已与 pd_master 建立 PD 链路或是否报错。
3. **等待 / 轮询**：若端口未就绪或链路未建立，约 **每 20 秒** 查看 **`pd_master.log`、`prefill.log`、`decode.log`**，区分仍在启动还是已报错；异常写入 **`summary.txt`** 并停止后续步骤。
4. **维护 `summary.txt`**：记录三条启动命令摘要（或等价）、**本次 `PREFILL_CUDA_DEVICES` / `DECODE_CUDA_DEVICES` 及选卡依据（`nvidia-smi` 要点）**、各端口检测结果、`lm_eval` 关键输出；结束后写**最终汇总**。
5. **测试结束后**：**关闭本次启动的所有相关进程**（`pd_master`、prefill、decode），释放端口与 GPU。
6. **错误记录**：启动或评测失败时，将错误摘要记入 **`summary.txt`**，并在对话中说明关键信息。

## 输出文件

- **`summary.txt`**：位于**本轮日志目录**，作为本次 PD 分离评测的**最终总结**。
- **服务与评测日志**：**`pd_master.log`、`prefill.log`、`decode.log`、`eval_gsm8k.log`** 均落在**同一日志目录**。
