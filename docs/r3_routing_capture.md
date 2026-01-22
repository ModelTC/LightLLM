# R3: Routing Capture for MoE Models / MoE 模型路由捕获

## Overview / 概述

R3 (Rollout Router Replay) enables capturing MoE (Mixture of Experts) routing decisions during inference. This is useful for RL post-training to replay routing decisions.

R3（Rollout Router Replay）功能可在推理过程中捕获 MoE（混合专家）模型的路由决策。这对于强化学习后训练中重放路由决策非常有用。

## Usage / 使用方法

### 1. Enable on Server / 服务端启用

Start the server with `--enable_return_routed_experts`:

使用 `--enable_return_routed_experts` 参数启动服务：

```bash
python -m lightllm.server.api_server \
    --model_dir /path/to/moe_model \
    --enable_return_routed_experts \
    --tp 8
```

### 2. Request with return_routed_experts / 请求时启用

Add `return_routed_experts: true` in your request:

在请求中添加 `return_routed_experts: true`：

```python
import requests

response = requests.post("http://localhost:8000/generate", json={
    "inputs": "Hello, world!",
    "parameters": {
        "max_new_tokens": 100,
        "return_routed_experts": True  # Enable routing capture / 启用路由捕获
    }
})
```

### 3. Response Format / 响应格式

The final token's metadata will contain `routed_experts`:

最后一个 token 的 metadata 会包含 `routed_experts`：

```json
{
  "routed_experts": {
    "shape": [num_moe_layers, num_tokens, topk],
    "dtype": "int32",
    "data": "<base64 encoded>"
  }
}
```

### 4. Decode Routing Data / 解码路由数据

```python
import base64
import numpy as np

routing = response["routed_experts"]
data = np.frombuffer(base64.b64decode(routing["data"]), dtype=np.int32)
data = data.reshape(routing["shape"])
# data[layer_idx, token_idx, :] contains expert IDs for each token
# data[layer_idx, token_idx, :] 包含每个 token 选择的专家 ID
```

## Supported Models / 支持的模型

- DeepSeek v2/v3 (MoE)
- Qwen3-MOE
- Mixtral

## Memory Overhead / 内存开销

Memory usage: `num_moe_layers * max_tokens * topk * 4 bytes`

内存占用：`num_moe_layers * max_tokens * topk * 4 字节`

Example / 示例: DeepSeek v3 (58 MoE layers, 100K tokens, topk=8) ≈ 186 MB
