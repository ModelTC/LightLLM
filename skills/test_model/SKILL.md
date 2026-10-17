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

## 按修改范围选择端到端回归

- 先记录修改涉及的模型、注意力后端、缓存布局和调度路径，再选择相应子 skill；无需重跑未受影响的模型。
- GLM-5.3 Flash 动态 `fp8kv_dsa` 使用 [专用回归流程](glm5.3-flash-fp8kv-scenarios/SKILL.md)，包含完整 GSM8K、MTP、GPU/CPU 缓存与 P/D 传输。
- 精度评测须记录数据集完整样本数、空响应数和结果；冒烟请求不能替代完整数据集评测。

## 请求暂停与恢复

涉及 KV 分配、缓存布局、hybrid 状态或 MTP 的修改，还须测试实际调度暂停与恢复。
使用 `test/acc/test_request_pause_resume.py`，连接开启 debug 日志且 KV 容量较小的 normal 服务。
该用例先保存串行输出，再用短请求降低调度器的输出长度估计，并发提交较长的生成请求。
`min_new_tokens=max_new_tokens` 保证工作量；保留 `ignore_eos=False`，让调度器使用正常的长度估计。

验收必须同时满足：

1. 日志中同一个压力请求先出现 `infer paused req id`，再出现 `infer recover paused req id`。
2. 通过 `X-Request-Id` 将内部请求 ID 对应到测试请求，所有请求正常完成且 token 数量完整。
3. 暂停恢复后的输出 token IDs 与串行参考一致；差异需要保留并排查，不能直接忽略。
4. 未触发暂停应判失败，调整容量或负载后重测；不能用排队等待、HTTP 重试或 RL 服务暂停接口代替。
5. MTP 或 hybrid 模型须包含相应启用配置，覆盖 KV 与运行时状态恢复。日志、输出、启动参数归档到本轮目录。
