---
name: lightllm-profiler-control
description: LightLLM profiler 使用说明。用于需要启动或停止 LightLLM 的 torch_profiler / nvtx profiling 功能时，尤其是查看 --enable_profiling、/profiler_start、/profiler_stop 的使用方法。
---

# LightLLM Profiler 使用说明

## 使用场景

当用户需要使用 LightLLM profiler 功能时使用本 skill，包括：

- 启动服务时打开 profiler 能力。
- 通过 HTTP 接口控制 profiler start / stop。

## 启动方式

服务启动时增加 `--enable_profiling`：

```bash
python -m lightllm.server.api_server \
  --model_dir /path/to/model \
  --enable_profiling torch_profiler
```

支持值：

- `torch_profiler`：启用 PyTorch profiler，trace 默认写入 `./trace`，也可通过 `LIGHTLLM_TRACE_DIR` 指定目录。
- `nvtx`：启用 NVTX range，配合 NVIDIA Nsight Systems 等外部工具采集。

未设置 `--enable_profiling` 时，`/profiler_start` 和 `/profiler_stop` 会返回未启用提示。

## HTTP 控制接口

启动 profiler：

```bash
curl http://127.0.0.1:8000/profiler_start
```

停止 profiler：

```bash
curl http://127.0.0.1:8000/profiler_stop
```

端口 `8000` 替换为服务启动时的 `--port`。
