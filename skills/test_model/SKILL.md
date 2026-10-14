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
