---
name: test-model-common
description: >-
  Common override guidance for all skills/test_model sub-skills. Applies to
  LightLLM model accuracy/speed tests that use lm_eval or lmms_eval, especially
  local-completions GSM8K runs.
---

# Test Model 通用覆盖规则

本目录下所有子 skill 默认继承这些规则。若子 skill 中的命令与这里冲突，优先按这里执行；
只有在用户明确要求在线拉取数据/模型，或本地缓存缺失时，才临时关闭对应离线变量。

## lm_eval 启动加速

`lm_eval` 每次新进程启动都会加载 task、dataset、tokenizer 和 HuggingFace 相关模块。
实测 `local-completions + gsm8k --limit 1` 时，默认在线探测模式会在 tokenizer/dataset
初始化阶段等待很久；强制使用本地缓存后，启动耗时明显下降。

执行所有 `lm_eval` 精度测试时，默认在命令前加：

```bash
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export http_proxy=
export https_proxy=
export no_proxy=localhost,127.0.0.1,0.0.0.0,::1,${HOST:-127.0.0.1},${PD_MASTER_IP:-127.0.0.1}
export NO_PROXY="${no_proxy}"
```

然后再执行子 skill 中的 `lm_eval` 命令，例如：

```bash
lm_eval --model local-completions \
  --model_args "model=${MODEL_NAME},base_url=${BASE_URL},num_concurrent=64,max_retries=3,tokenized_requests=False,tokenizer=${MODEL_DIR}" \
  --tasks gsm8k \
  --batch_size 64 \
  --confirm_run_unsafe_code
```

## 使用前检查

- 先确认对应数据集和 tokenizer 已经在本地缓存中；如果离线模式报缓存缺失，再切回在线模式补齐缓存。
- 精度评测前仍然要先做一次 `curl` warmup，确认服务端已经可用。
- 如果只是压测吞吐，不要用 `lm_eval`；使用轻量 benchmark client，避免 `lm_eval` 的 task/dataset/metric 初始化成本。
- 记录结果时要把是否启用了离线缓存写入 summary/log，方便比较不同轮次。

## 已验证现象

在本机 Qwen3.5-0.8B 普通服务上，`lm_eval --limit 1` 实测：

| 模式 | 耗时 |
|---|---:|
| 默认在线探测 | 约 123s |
| 离线缓存模式 | 约 20s |

因此，除非有明确理由，`skills/test_model` 下的 `lm_eval` 测试都应默认启用离线缓存变量。

## Paged KV 全量回归

对 paged KV 的修改，先运行 `python -m pytest unit_tests -ra`，再执行本目录全部子 skill 的
完整精度评测。GSM8K 应包含 1319 题，MMMU validation 应包含 900 题；保留
`--log_samples` 结果，核对题数、空响应、服务端错误与精度，不能用 `--limit` 替代全量。

普通端到端场景增加 `--page_size 16`；CPU/disk cache 场景增加 `--page_size 128`，
并保留子 skill 要求的连续两轮精度评测。CPU cache 的页大小须为物理 page size 的整数倍。

额外验证 CPU cache 在 GPU 前缀被逐出后的真实回载。以下三组分别启动独立服务：

| 模型 | KV 类型 | page_size | CPU cache 页大小 | 额外参数 |
|------|---------|-----------|------------------|----------|
| Qwen3-8B | 默认 | 16 | 128 | 无 |
| Qwen3-8B | int8kv | 128 | 128 | `--llm_kv_type int8kv` |
| Qwen3.5-0.8B | 默认 | 128 | 512 | `--linear_att_cache_size 10 --linear_att_hash_page_size 256 --linear_att_page_block_num 2` |

每组使用 `--tp 2 --enable_cpu_cache --cpu_cache_storage_size 16 --enable_prompt_logprobs`
以及 `--max_total_token_num 32768 --max_req_total_len 16384 --chunked_prefill_size 257`；
设置表中 `--page_size` 和 `--cpu_cache_token_page_size`。257 特意不整除物理页大小，
用于覆盖 chunk 跨页边界。按子 skill 的方法检查服务日志、端口，并完成真实请求 warmup。

仓库目前没有自动执行 GPU 前缀逐出后真实 CPU cache 回载的探针。服务就绪后，
用以下人工步骤记录请求参数、响应与服务日志，不能把连续两轮精度评测当作已覆盖逐出回载：

1. 对固定提示词使用 greedy 采样，记录 GPU 常驻前缀参考的 32 个输出 token 和 logprob。
2. 用其他提示词填充 KV cache，确认目标 GPU 前缀已被逐出且其 CPU cache 仍可用。
3. 再次请求目标提示词，确认日志中有实际 CPU cache 命中；另测仍有部分 GPU 前缀命中的情况。
   参考与回载请求须使用相同的 prefill 起点，避免 chunk/query 形状改变造成的 BF16 舍入差异。
   32 个输出 token 必须相同，logprob 最大绝对差不超过 0.02。
4. 请求 prompt logprobs，检查禁用缓存复用后仍返回全部输入位置的有效结果。

Qwen3.5 服务的 `--linear_att_hash_page_size` 保持为 256。现有
`unit_tests/server/router/model_infer/mode_backend/test_multi_level_kv_cache.py`
覆盖后端缓存任务状态和引用释放；它使用模拟传输，不能证明真实 GPU↔CPU KV 逐值一致性。
未完成上述人工验证或逐值传输校验时，在结果中明确列为验证缺口。
