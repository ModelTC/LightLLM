π₀ / π₀.₅ VLA 推理
=================

LightLLM 支持 OpenPI π₀ 与 π₀.₅ 的 LeRobot checkpoint 推理。普通文本请求仍
使用原有 VLM backend；action 扩展由轻量 ``VLARequestLifecycle`` 和独立 action
expert 进程组承载。一次性请求可选择 text、action 或两者同时输出，控制循环还可
显式保留版本化的 VLM prefix，连续提交高频 action task。

结构设计
--------

请求仍走 LightLLM 的标准生命周期：

.. code-block:: text

   创建/更新 context：HTTP -> visualserver -> router -> 普通 VLM prefill
                                          -> PrefixContextRegistry 接管 prefix KV
   高频 action tick： HTTP ----------------> router -> ActionTask -> actionserver
                                                         |（复用 prefix KV）
                                                         `-> action output
   普通文本请求：      HTTP -> visualserver -> router -> 原有 prefill/decode/finish

没有单独的 VLA HTTP server 或 visual worker。``Pi0VLMModel`` 继承
``Gemma_2bTpPartModel``，直接复用 BaseModel 的
``ModelInput``、``InferStateInfo``、权重加载、TP、RoPE、RMSNorm、Gemma layer、
post layer 与 prefill attention backend。π₀ 前缀所需的 block-bidirectional
attention 只通过模型属性 ``prefill_causal=False`` 选择；普通模型默认仍为
``causal=True``。

``actionserver`` 与 visualserver/audioserver 一样，由 manager 和 TP model RPC
进程组成。首次 prefill 返回后，下一次 router 分类能观察到已同步完成的 KV，
``PrefixContextRegistry`` 随即接管物理 prefix page；通用 ``prefill_normal`` 无需
action 分支。worker 读取 router prefix 映射的不可变快照，且只写 context
预留并串行复用的 suffix scratch page。action 使用独立的逻辑
``req_to_token_indexs`` 表，因此 text decode 与 action suffix 不会相互覆盖。
CREATE/REPLACE 的首个成功 action 会在各 action rank 幂等注册完整的
``PrefixContextIdentity -> prefix/scratch mapping``。后续 REUSE 的热路径只传
versioned identity、最新 state/noise 和采样参数，不再通过 ZMQ/RPyC 重传
``O(prefix_len)`` mapping，也不再执行 target TP mapping gather 或 scratch 分配。
CLOSE 或旧版本 drain 后使用完整 identity 注销 worker-local mapping；物理 KV 的
所有权始终留在 router。

每个 ``PrefixContext`` 可排队多个短生命周期 ``ActionTask``，但同一 context
同一时刻只运行一个 task。只有所有 action worker 确认不再访问 KV，且所有
target TP rank 确认终态结果后，scratch 才能给下一 task 复用。prefix 仅在显式
CLOSE，或 copy-on-write REPLACE 的旧版本 drain 完成后释放。客户端必须把
``DELETE /v1/vla/contexts/{context_id}`` 作为正常资源回收路径，不能依赖自动
清理当前 context。
HTTP 输出消费是另一个独立生命周期步骤。若缺失安全 ACK，runtime 会保留 KV
并报告必须重启进程，而不会冒险复用仍可能被访问的 page。
CREATE/REPLACE 的新 handle 在公共 HTTP 响应 body 完成 ASGI send 前仍是 provisional owner；
即使普通 ``InferReq`` 已经走完 finish/filter，router 也保留一条常数大小的 owner
记录。响应成功发送后各 target rank 确认保留新 context；若断连、取消或响应后处理
失败，CREATE 会关闭新 context。对于 REPLACE，旧 current/KV 会保留到 ASGI send
完成；成功才提交新版本并在安全 drain 后退役旧版本，发送失败或 owner 被 discard
则回滚旧版本并释放新版本。完成 owner ACK 后才允许复用 ``ShmReq`` slot，避免留下
客户端拿不到 handle 的孤儿 KV。

action expert 保留独立模型和 denoise loop，但矩阵乘使用 LightLLM 的
``ROWMMWeight``、``COLMMWeight``、``QKVROWNMMWeight``、``KVROWNMMWeight`` 与
``Quantcfg``；attention、RoPE、RMSNorm 和 GELU 也使用已有 kernel。因此 TP shard、
all-reduce 和已有量化扩展点均可复用。

不再存在排他的 ``VLABackend``。普通 backend 仍负责模型拓扑、batch、sampling
和资源管理；``chunked_prefill/impl.py`` 保持语言模型流程，仅通用的
``_get_classed_reqs`` 生命周期钩子负责 poll、停放特殊 task、阻止不安全 pause，
以及在安全后复用原有 finish/filter。该边界也可继续承载生图等非文本输出。

启动
----

π₀ 会由 checkpoint 的 ``type`` 自动识别；使用普通 ``normal`` 模式即可：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python -m lightllm.server.api_server \
     --run_mode normal \
     --model_dir /path/to/pi0_base \
     --host 0.0.0.0 --port 8000 \
     --tp 1 \
     --data_type bfloat16 \
     --vla_tokenizer_path /path/to/paligemma_tokenizer.model \
     --vla_max_prefix_tokens 1024 \
     --enable_prefill_cudagraph \
     --prefill_cudagraph_max_handle_token 1024

π₀.₅ 只需把目录改成 ``/path/to/pi05_base``。

tokenizer 必须来自显式路径，或放在模型目录下的
``paligemma_tokenizer.model``、``tokenizer.model``、``tokenizer/`` 中。服务启动
不会访问硬编码 URL 下载 tokenizer；找不到本地资产时会直接报错。

action TP 自动跟随 VLM TP，并使用对应的同一组 GPU，因为 prefix KV 通过
CUDA IPC 共享。
视觉编码器仍使用通用参数 ``--visual_gpu_ids``、``--visual_tp`` 和
``--visual_dp``；可将 visualserver 放到其他 GPU。第一阶段已验证的 action runtime
组合为单机、``dp=1``、非 PD 分离，且不启用多级 KV cache、constraint decode、
prefill/decode 混合或 microbatch overlap。未验证组合会在启动时明确报错，
但不再被另一套 backend 选择永久阻断。

action dimension、horizon、denoise steps 和 state mode 默认从 checkpoint 读取，
也可通过 ``--vla_action_dim``、``--vla_action_horizon`` 和
``--vla_num_denoise_steps`` 覆盖。部署推荐显式使用
``--data_type bfloat16``；当前 LeRobot checkpoint 的 dtype 是 ``float32``，
省略该参数会继承 checkpoint 并以 FP32 运行。BF16 会用于 VLM、
Action Expert 和 KV cache，视觉编码器仍以 FP32 计算并在输出处转为 BF16。
FP32 主要用于严格数值回归和精度问题定位。
Action Expert 复用 LightLLM 标准 layer-infer 路径，并通过自己的 inference state
选择 bidirectional block attention。
bidirectional prefix 不进入 causal radix KV cache，因此 VLA 启动时会关闭 dynamic
prompt cache 和 chunked prefill；visualserver 的图像 embedding cache 仍会复用。

BF16 性能参考
-------------

以 NVIDIA H200、π₀、``batch=1``、3 张图像、``action_horizon=50`` 和
``num_denoise_steps=10`` 为例，在 10 次预热后采集 30 次变更图像输入的
HTTP 请求：

.. list-table::
   :header-rows: 1

   * - 指标
     - FP32 p50 / p95
     - BF16 p50 / p95
     - BF16 p50 加速
   * - HTTP 端到端延迟
     - 2286.2 / 2304.7 ms
     - 201.2 / 224.9 ms
     - 11.4x
   * - Action Expert GPU 时间
     - 1922.6 / 1923.3 ms
     - 90.5 / 93.6 ms
     - 21.2x

这是实际推荐部署组合的结果：BF16 自动选择 FA3/FlashInfer，FP32 使用
Triton，因此数据表示 dtype 与其默认 attention backend 的综合收益，
不是只替换 dtype 的 kernel 微基准。延迟会随 GPU、请求参数和负载变化。
表中的 ``90.5 ms`` 是完整 10-step Action Expert（约 ``9 ms/step``），不是 VLM
prefill；当前 action denoise loop 尚未纳入 CUDA Graph。持久 prefix 能去掉后续
tick 的 ViT/VLM prefill、KV gather 和 scratch 分配，但不会自动消除这 90 ms。
因此高频模式的端到端数据需要单独基准，不能用一次性请求的 ``201.2 ms`` 推算。

Action API
----------

请求发送到 ``POST /v1/vla/actions``。不带 context 字段时保留一次性语义，每次
处理一个 observation，字段如下：

* ``prompt``：任务文本；
* ``images``：按 checkpoint camera 名称给出的映射或已按顺序排列的 list；每张图可
  为 base64/data URL、HTTP(S) URL 或 ``H x W x 3`` 数组；
* ``state``：``[state_dim]`` 或 ``[1, state_dim]``；
* 可选 ``image_mask``；
* 可选 ``noise``，形状为 ``[1, action_horizon, action_dim]``；
* 可选 ``action_horizon``、``action_dim``、``num_denoise_steps`` 与 ``timeout``。
* ``outputs`` 可为 ``["action"]``、``["text"]`` 或
  ``["text", "action"]``。省略时保留该 endpoint 原有的 action-only 行为。

camera mapping 会按照 checkpoint 中的 base、left wrist、right wrist 顺序处理，
不依赖 JSON key 插入顺序。响应中的 ``actions`` 为
``[action_horizon, action_dim]``。

.. code-block:: python

   import base64
   import requests

   def encode_image(path):
       with open(path, "rb") as image_file:
           return base64.b64encode(image_file.read()).decode("ascii")

   body = {
       "request_id": "robot-1",
       "prompt": "pick up the block",
       "images": {
           "base_0_rgb": encode_image("base.jpg"),
           "left_wrist_0_rgb": encode_image("left.jpg"),
           "right_wrist_0_rgb": encode_image("right.jpg"),
       },
       "state": [0.0] * 32,
   }
   response = requests.post(
       "http://127.0.0.1:8000/v1/vla/actions", json=body, timeout=120
   )
   response.raise_for_status()
   actions = response.json()["actions"]

π₀ 的高频控制循环把低频 observation 更新与高频 action tick 分开。第一次请求
设置 ``persist_prefix=true``，执行低频 CREATE：完成 ViT/VLM prefill 并保存
prefix KV。响应会返回
``prefix_context={context_id, version, server_epoch}``：

.. code-block:: python

   first = requests.post(
       "http://127.0.0.1:8000/v1/vla/actions",
       json={**body, "persist_prefix": True, "context_id": "robot-1"},
       timeout=120,
   ).json()
   context = first["prefix_context"]

   # 高频 REUSE：每个 tick 都传 context identity 和最新 state；不再经过 ViT/VLM。
   tick = requests.post(
       "http://127.0.0.1:8000/v1/vla/actions",
       json={
           "context_id": context["context_id"],
           "context_version": context["version"],
           "server_epoch": context["server_epoch"],
           "state": [0.1] * 32,
       },
       timeout=120,
   ).json()

   requests.delete(
       f"http://127.0.0.1:8000/v1/vla/contexts/{context['context_id']}",
       params={
           "context_version": context["version"],
           "server_epoch": context["server_epoch"],
       },
       timeout=120,
   ).raise_for_status()

π₀ 的 continuous state 属于 action suffix，因此每次 REUSE 都携带最新 state，
同时复用同一份 prefix。服务对同一 context 按 FIFO 排队，并且任意时刻最多只有
一个 action task in-flight；客户端可以提前提交后续 tick，但应设置有限队列和
超时形成背压。当前语义是严格 FIFO，不会静默把中间 state 合并为 latest state；
如果输入频率高于 Action Expert 吞吐，排队延迟会增长，控制端通常应保持一个
outstanding tick。latest-state/coalescing 应作为后续显式队列策略实现。

π₀.₅ 会把离散 state 写入 bidirectional VLM prefix，因此 state 变化不能 REUSE，
必须使用 ``replace_prefix=true`` 执行 REPLACE，并携带旧 identity 以及完整
prompt/images/state。新版本 commit 后原子切换，旧版本等安全 ACK 后释放。
控制循环结束后必须显式发送 CLOSE；上例中的 ``DELETE`` 即为 CLOSE 请求。

兼容的 action-only 响应仍是上述裸 action 对象。包含 text 时返回聚合对象，例如：

.. code-block:: json

   {
     "request_id": "robot-1",
     "outputs": ["text", "action"],
     "text": "Grasping the block.",
     "finish_reason": "stop",
     "action_response": {
       "actions": [[0.1, 0.2]],
       "action_outcome": "success",
       "restart_required": false
     }
   }

action 输出只由专用 ``/v1/vla/actions`` 接口承载。通用 LightLLM
``/generate`` 和 ``/generate_stream`` 保持原有文本接口语义，不感知 action
事件。

每个 action task 仍使用标准 ``ShmReq`` 作为输出、取消、超时与最终回收的 owner。
持久 registry 只拥有版本化物理 prefix KV 和 context scratch，不复制 HTTP/session
调度器；task 安全完成后仍复用普通 LLM 的 finish/filter 路径。当前 REUSE 因此仍有
``ShmReq`` 创建、router poll，以及 worker 把缓存 index 写入 task-scoped logical row
的本地开销，只是不再跨进程重传或 TP gather ``O(prefix_len)`` mapping。若 action
CUDA Graph 后 denoise 显著变快，下一阶段应使用持久 logical row，并把 hot tick
提交抽成直接 action data plane；这不属于本分支已实现能力。

Normalization 与 adapter
------------------------

不提供 normalization 文件时，``state`` 与返回 action 被视为已经归一化。
``--vla_norm_config`` 可加载包含 ``state`` 和 ``action`` 的 JSON，每项支持
``identity``、``mean_std`` 或 ``quantiles``：

.. code-block:: json

   {
     "state": {"mode": "quantiles", "q01": [0.0], "q99": [1.0]},
     "action": {"mode": "quantiles", "q01": [0.0], "q99": [1.0]}
   }

``--vla_robot_adapter`` 支持例如
``{"relative_action_mask": [true, false]}``；scale、offset 与 clip 可通过
``--vla_action_postprocess_config`` 配置。

验证
----

独立精度 oracle 使用 Transformers/OpenPI 方程和真实 checkpoint，不复用 LightLLM
layer 实现，并通过与服务相同的 action-owned scoped KV view 安装 prefix/suffix
映射。float32 gate 为 action ``atol=rtol=2e-5``、prefix KV 最大绝对误差
``1e-3``、vision 最大绝对误差 ``2e-5``：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python test/acc/vla/run_openpi_lightllm_diff.py \
     --model-dir /path/to/pi0_base \
     --dtype float32 \
     --batch-size 2 --num-images 3 --action-horizon 50 --num-steps 10

BF16 的上述严格 FP32 gate 不适用于判定任务精度。在 ``batch=1``、
``action_horizon=4``、``num_denoise_steps=2`` 的固定合成输入快速对比中，
LightLLM BF16 相对 OpenPI FP32 oracle 的 π₀/π₀.₅ action 最大绝对偏差约为
``1.1e-2``，平均绝对偏差约为 ``2.5e-3``/``3.8e-3``。这一量级未显示
数值失稳，但它不能代替反归一化后的机器人单位误差、代表性数据集或
rollout success 验证。

服务启动后，可用 ``test/acc/vla/run_server_smoke.py`` 覆盖实际的
HTTP -> visualserver -> router/BaseModel -> actionserver 链路。
