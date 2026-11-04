---
name: test-model-glm5.3-flash-fp8kv-scenarios
description: >-
  GLM-5.3 Flash dynamic fp8kv_dsa end-to-end regression: full GSM8K with matched
  BF16 KV baselines for both normal and MTP configurations,
  native MTP, sparse prefix reuse, CPU reload, real request pause/resume, and P/D
  transfer. Supports glm5_next and glm5_next_text. No static calibration.
---

# GLM-5.3 Flash 动态 FP8 KV 回归

继承 [通用规则](../SKILL.md)。使用本机模型目录；`model_type` 可以为 `glm5_next`
或 `glm5_next_text`，计算 dtype 为 BF16，`kv_lora_rank=512`。
本流程测试动态 `fp8kv_dsa`，不需要校准文件；BF16 KV 用作逐配置精度对照。

## 准备

确认模型目录、8 张 Hopper GPU 的占用和所选端口。每个场景独立日志目录，保存完整命令、
server.log、评测日志、原始响应和 summary.txt。只清理本轮启动的进程。
服务就绪须同时检查端口监听、日志没有致命错误，并用一个实际生成请求 warmup。

```bash
export MODEL_DIR=/mnt/mtc/models/GLM-5.3-Flash
export HF_ALLOW_CODE_EVAL=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export http_proxy= https_proxy=
export no_proxy=127.0.0.1,localhost,::1
export NO_PROXY="$no_proxy"
export LIGHTLLM_LOG_LEVEL=debug
export LOADWORKER=8
```

## 场景矩阵

| 场景 | 配置 | 验证 |
|---|---|---|
| BF16 KV baseline | TP4，`llm_kv_type=None`，page_size=16 | 完整 GSM8K 1319 题，作为普通 FP8 的对照 |
| FP8 | TP4，`llm_kv_type=fp8kv_dsa`，page_size=16 | 完整 GSM8K 1319 题，稀疏 prefix cache |
| BF16 KV MTP baseline | BF16 KV，原生 MTP step=2，Prefill CUDA Graph | 完整 GSM8K，作为 FP8 MTP 的对照 |
| FP8 MTP | FP8，原生 MTP step=2，Prefill CUDA Graph | 完整 GSM8K、MTP 验证与接受日志 |
| 暂停恢复 | FP8 MTP；小 KV 容量 | 真实 pause→recover→完成，输出与串行参考对照 |
| FP8 CPU Cache | 小 KV 容量，hybrid 大页和 CPU cache | GPU 淘汰后真实 CPU 命中、输出一致 |
| FP8 P/D | P TP4 + D TP4，两端启用 FP8 | 四种 K-pool 尾部长度，输出与 normal FP8 对照 |

## 普通服务与完整 GSM8K

按场景设置 `CUDA_VISIBLE_DEVICES`、`PORT`、`NCCL_PORT`、`KV_TYPE` 和 `LOG_DIR`。
BF16 KV baseline 使用 `KV_TYPE=None`，FP8 使用 `KV_TYPE=fp8kv_dsa`：

```bash
python -m lightllm.server.api_server \
  --model_dir "$MODEL_DIR" --model_name glm53 --disable_vision \
  --host 127.0.0.1 --port "$PORT" --nccl_port "$NCCL_PORT" --tp 4 \
  --data_type bfloat16 --llm_kv_type "$KV_TYPE" --page_size 16 \
  --max_total_token_num 65536 --max_req_total_len 8192 \
  --running_max_req_size 32 --batch_max_tokens 4096 --chunked_prefill_size 512 \
  --graph_max_batch_size 32 --graph_max_len_in_batch 8192 \
  --linear_att_cache_size 8 --linear_att_hash_page_size 512 \
  > "$LOG_DIR/server.log" 2>&1
```

MTP 场景追加 `--mtp_mode eagle_with_att --mtp_step 2 --mtp_draft_model_dir "$MODEL_DIR"`。
完整 GSM8K 的 MTP 场景同时追加 `--enable_prefill_cudagraph --prefill_cudagraph_max_handle_token 1024`。
服务后台启动，就绪并 warmup 后执行：

```bash
lm_eval --model local-completions \
  --model_args "{\"model\":\"glm53\",\"base_url\":\"http://127.0.0.1:${PORT}/v1/completions\",\"tokenizer_backend\":null,\"eos_string\":\"<|endoftext|>\",\"num_concurrent\":16,\"max_gen_toks\":512,\"max_retries\":3,\"timeout\":600}" \
  --tasks gsm8k --batch_size 16 --confirm_run_unsafe_code --log_samples \
  --output_path "$LOG_DIR/gsm8k" > "$LOG_DIR/eval.log" 2>&1

```

精度验收必须满足：

1. 普通 FP8 对比普通 BF16 KV；FP8 MTP 对比相同 MTP step 和 Prefill CUDA Graph 配置的 BF16 KV。
   已有结果仅在以下条件全部一致且原始产物齐全时可以复用，否则补跑。不能用普通 BF16 代替 MTP baseline。
2. 每对实验使用相同模型权重、计算 dtype、TP、GPU 型号、缓存容量、page_size、分块及 graph 参数、
   软件版本；除 KV 类型与服务端口、日志路径等运行位置外，配置一致。BF16 指 KV dtype，模型权重量化方式不变。
3. 数据集版本、5-shot 样例、随机种子、贪心解码参数、输出上限、stop sequences 和评测并发完全一致。
   不使用 `--limit`；双方必须各有 1319 个唯一 doc_id，无缺失、重复或空响应。
4. 用 `(doc_id, filter)` 配对保存的 samples，逐题核对原始输入、标准答案和对应 hash。
   从原始响应重新提取答案并计算 strict/flexible 分数，确认逐题得分及均值与评测报告一致。
   检查缺少最终答案、输出截断和请求失败，保留具体样本及原因。
5. 汇总每对 BF16/FP8 的正确题数、accuracy、差值（百分点），以及仅 BF16 正确、仅 FP8 正确的题数。
   同时保留双方完整命令、日志、results 和 samples。未指定允许的精度下降阈值时报告实测差值，
   不能仅凭评测进程成功结束宣称精度无损或通过精度门槛。

以下功能场景单独验证缓存和运行时状态，不替代上述数据集精度对照，也不要求另跑 BF16 功能场景。
在普通 FP8 和 FP8 MTP 服务上执行缓存用例，覆盖 2304～2307 token，
超过 index_topk=2048 并包含四种 K-pool 尾部长度：

```bash
python test/acc/test_glm53_fp8kv.py \
  --url "http://127.0.0.1:$PORT" --model-dir "$MODEL_DIR" \
  --output "$LOG_DIR/cache.json"
```

## 暂停恢复与 CPU 回载

独立重启小容量 normal 服务：将上面的 `max_total_token_num` 改为 4096，
`max_req_total_len=3072`、`running_max_req_size=8`、`graph_max_batch_size=8`、
`batch_max_tokens=1024`、`chunked_prefill_size=257`、`linear_att_cache_size=8`、
`linear_att_hash_page_size=128`，增加 `--linear_att_page_block_num 2 --router_token_ratio 1.0`。
使用 FP8 MTP 配置，执行公共规则要求的暂停恢复用例。

```bash
python test/acc/test_request_pause_resume.py \
  --url "http://127.0.0.1:$PORT" --model-dir "$MODEL_DIR" \
  --server-log "$LOG_DIR/server.log" --input-tokens 257 --output-tokens 512 \
  --concurrency 8 --output "$LOG_DIR/pause-resume.json"
```

FP8 CPU 场景再增加 `--enable_cpu_cache --cpu_cache_storage_size 16 --cache_placement_strategy legacy`。
CPU 测试的淘汰请求总长度应大于 GPU KV 容量，且小于 CPU 容量：

```bash
python test/acc/test_glm53_fp8kv.py \
  --url "http://127.0.0.1:$PORT" --model-dir "$MODEL_DIR" \
  --cpu-cache --server-log "$LOG_DIR/server.log" --output "$LOG_DIR/cpu-cache.json"
```

必须在目标回载请求的完成日志中看到 `cpu_prompt_cache_len > 0`，仅配置 CPU cache 不算覆盖。

## P/D

使用 `test/start_scripts/glm53/glm53_pd_1p1d.sh` 的 P/D 配置，设置
`LLM_KV_TYPE=fp8kv_dsa`，给 P、D、master 分别保存日志。两端参数须一致；
FP8 transfer page 使用 `pd_kv_page_size=32768`，16384-token 的 FP8 页不足以容纳完整 hybrid 状态。
P/D 节点使用实际网卡 IP；master、P、D 的 `max_req_total_len` 必须一致。
与 normal FP8 对照时保持 page_size、计算 dtype 和 MTP 配置一致。

通过 master 运行 `test_glm53_fp8kv.py`，并增加
`--reference <normal-FP8场景的cache.json>`。检查 P/D 日志证明真实传输，保留结果。
每个场景完成后停止所属服务，最终确认 GPU 和端口释放。
