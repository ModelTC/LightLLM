π₀ / π₀.₅ VLA 推理
=================

LightLLM 支持 OpenPI π₀ 与 π₀.₅ 的 LeRobot checkpoint 推理。普通文本请求仍
使用原有 VLM backend；自治 ``VLAServer`` 提供 action expert 推理，并通过
``ActivationChannel`` 消费 VLM KV activation。一次性请求输出 action，也可同时输出 text；控制循环还可
保留版本化的 VLM prefix，连续提交高频 state tick。

结构设计
--------

冷请求仍走 LightLLM 的标准 VLM 生命周期，同时异步提交给 ``VLAServer``：

.. code-block:: text

   HTTP（协议解析 + 分配 ShmReq）
      |-> visualserver -> router -> target prefill/decode -> 原有文本输出
      `-> VLAServer -> ActivationChannel -> model_rpc -> action 输出

   target post_handle -> 发布 kv_ready(rank mask, infer_req_idx)
   target finish/filter -> kv_consumer_mask 清零后才能回收 KV

   高频 REUSE：HTTP -> VLAServer -> model_rpc
   CLOSE/finish：VLAServer -> ActivationChannel 释放 lease -> 原 finish 回收 KV

没有单独的 VLA HTTP server 或 visual worker。HTTP 层只负责协议转换、
``ShmReq`` 分配、两路异步提交和结果汇合；图像与 action/state 预处理分别
由 visualserver 和 VLAServer 负责。``Pi0VLMModel`` 与
``Pi05VLMModel`` 分别注册各自 checkpoint 类型，并复用同一个 PaliGemma
BaseModel 实现。两者直接复用 BaseModel 的
``ModelInput``、``InferStateInfo``、权重加载、TP、RoPE、RMSNorm、Gemma layer、
post layer 与 prefill attention backend。π₀ 前缀所需的 block-bidirectional
attention 只通过模型属性 ``prefill_causal=False`` 选择；普通模型默认仍为
``causal=True``。

Action Expert 也在 ``models/pi0`` 与 ``models/pi05`` 中分别固定模型结构：π₀
始终使用 continuous-state token 和普通 RMSNorm；π₀.₅ 始终使用 prefix 中的离散
state、timestep condition 和 AdaRMSNorm。型号只在 worker 启动时选择一次，layer
热路径不再判断 ``is_pi05``，也不存在可选 ``condition`` 分支。两者统一走
``forward -> Pi0ActionInferStateInfo -> _context_forward``，CUDA Graph 复用 common
``CudaGraphRunner``，没有另一套 runtime/eager 流程。

``VLAServer`` 自己管理请求 deadline、取消、Prefix Context 和 Action TP model RPC，
不再依赖通用 GenerationManager。具体 Expert 类由 checkpoint 的
GenerationModelRegistry 注册项选择，ModelRegistry 只负责 Target VLM。对
ONESHOT/CREATE/REPLACE，target 各 TP rank 在 CUDA-safe 的 ``post_handle`` 边界发布
``kv_ready_rank_mask`` 和 ``infer_req_idx``；``ActivationChannel`` 在所有 rank
ready 后解析只读 ``KVActivation``。每个 action worker 通过 CUDA IPC 附着对应 rank
的 ``MemoryManager`` 和二维 request table，并在任务开始时一次性 gather 为本地
``DensePrefixKV``。从这一边界开始，action model 不再读取 target allocator 或 request
table。suffix 保持二维 packed 布局并使用固定 shape 的
QK/softmax/AV，最后通过 LSE 与 prefix attention 合并。这套逻辑封装在 common
``SplitPrefill`` attention backend 中，layer 只调用 infer state 的
``prefill_att``；Pi0/Pi0.5 只设置各自的 suffix mask policy。paged attention 的
scheduler metadata 在 10-step/layer 循环外只规划一次；action 侧不创建第二个
allocator 或 scratch page。BF16/FP16 生产路径只走 FA3；显式 FP32
精度实现隔离在 ``split_prefill/fp32.py``，不作为 BF16 的运行时 fallback。

CREATE 在首个 action 安全完成后激活 context；REPLACE 的新请求仍在
visualserver/router/VLM 做低频 prefill 时，旧 context 继续接收高频 tick；新 KV
进入 READY 后才冻结旧 context，新 action 成功后原子切换，失败则恢复旧 context。
REUSE 只把 versioned identity 和最新 state/noise 直接送到
``VLAServer``，不创建 ``ShmReq``，也不再经过 visualserver/router/VLM。
同一 context 的 tick 按 FIFO 串行；CLOSE 排在已接收任务之后。action worker
返回 CUDA-safe ACK 后，``VLAServer`` 通过 ``ActivationChannel`` 按
``(shm slot, request_id)`` 清除
共享 ``kv_consumer_mask`` 中自己的 bit，原有 finish/filter 路径随后自然回收 KV；
Router 和 target 推理循环都不接收 action release 命令。若 action worker
未确认 CUDA-safe，则保留 KV 并返回 ``restart_required``，不冒险复用 page。
客户端应把 ``DELETE /v1/vla/contexts/{context_id}`` 作为正常回收路径。

action expert 保留独立模型和 denoise loop，但矩阵乘使用 LightLLM 的
``ROWMMWeight``、``COLMMWeight``、``QKVROWNMMWeight``、``KVROWNMMWeight`` 与
``Quantcfg``；RoPE、RMSNorm 和投影仍走已有接口。单 KV head block attention
使用 common ``SplitPrefill`` backend，Pi 模型目录仅保留 51-token GELU tile 的小型
专用 kernel。因此 TP shard、all-reduce、attention backend 和已有量化扩展点均可复用。

不再存在 ``inference_runtime``、``PrefixContextRegistry`` 或 HTTP 与 Router 间的
owner ACK 控制面。
target 侧只有通用的 KV consumer 接口：``post_handle`` 发布 KV ready，原有
finish/filter/pause 在 ``kv_consumer_mask != 0`` 时延后物理 KV 回收。普通文本路径、Router 和 ``impl.py``
均无 action 分支。Neo 生图等消费侧复用同一个 ``KV_CONSUMER_GENERATION`` lease
和 dense gather 边界，但由独立 Image Generation Server 管理自己的流程。

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
``--visual_dp``；可将 visualserver 放到其他 GPU。第一阶段已验证的 action 服务
组合为单机、``dp=1``、非 PD 分离，且不启用多级 KV cache、constraint decode、
prefill/decode 混合或 microbatch overlap。未验证组合会在启动时明确报错，
但不再被另一套 backend 选择永久阻断。

action dimension、horizon 和 denoise steps 默认从 checkpoint 读取，
也可通过 ``--vla_action_dim``、``--vla_action_horizon`` 和
``--vla_num_denoise_steps`` 覆盖。部署推荐显式使用
``--data_type bfloat16``；当前 LeRobot checkpoint 的 dtype 是 ``float32``，
省略该参数会继承 checkpoint 并以 FP32 运行。BF16 会一致用于 Vision、
VLM、Action Expert 和 KV cache；不再在 BF16 服务中隐式保留 FP32 Vision。
FP32 主要用于严格数值回归和精度问题定位。
Action Expert 复用 LightLLM 标准 layer-infer 路径，并通过自己的 inference state
选择 bidirectional block attention。
bidirectional prefix 不进入 causal radix KV cache，因此 VLA 启动时会关闭 dynamic
prompt cache 和 chunked prefill；visualserver 的图像 embedding cache 仍会复用。

BF16 性能参考
-------------

以 NVIDIA H200、``batch=1``、``action_horizon=50`` 和
``num_denoise_steps=10`` 为例，Action Expert 在预热后测得：

.. list-table::
   :header-rows: 1

   * - 模型/输入
     - BF16 CUDA Graph p50
     - 说明
   * - π₀
     - 13.24 ms
     - 三图 prefix、50 action token、10 step
   * - π₀.₅
     - 19.07 ms
     - 三图 prefix、50 action token、10 step

Action Expert 把完整 10-step denoise loop 捕获为一次 CUDA Graph replay。
``--disable_cudagraph`` 会关闭 action graph；FP32 使用 eager，BF16/FP16 graph 不依赖
LLM 选择 FA3。首次遇到新的 ``(batch, horizon, steps)`` 会产生一次 capture 开销；
真实 prefix 长度和二维 page table 作为 graph input，attention 严格 mask 无效位置。
稳定延迟需在 warmup 后测量。

三图、``horizon=50``、10-step 的 Pi0 纯 GPU 分段 p50 为：Vision CUDA
Graph ``3.74 ms``、VLM prefill ``9.16 ms``、Action Expert ``13.24 ms``，分段
相加约 ``26.14 ms``。Vision graph 与 eager 逐元素一致；Pi visual adapter
还会把一个 observation 的 camera 作为一个 inline batch 发布，不走逐
item infer/store 队列。这两项都只对 Pi adapter 生效，其他 VLM 的
visualserver 队列和 LLM overlap 不变。

``policy_timing`` 会返回包含 RPC/后处理的 ``action_backend_ms``，以及它和 GPU
``action_expert_ms`` 之差 ``action_overhead_ms``。HTTP wall 还包含 JPEG、进程通信、
Visual/Router 调度和 KV 生命周期，不能用旧服务数字替代这组纯 GPU 基线。

`realtime-vla <https://github.com/dexmal/realtime-vla>`_ 的原始 ``benchmark.py``
在同一 H200 上实测 prompt 20、chunk 50 的 1/2/3 图 p50 为
``16.24/18.88/21.43 ms``；它公布的 4090 三图数字为 ``39.2 ms``。
该脚本测量的是固定 GPU tensor 和 shape，并把
Vision + VLM + 10-step AE 整体捕获为一张 CUDA Graph；不包含 HTTP、JPEG、
LightLLM 多进程和 Prefix KV 生命周期。它还对 RMS affine、action/time encoder、
QKV、RoPE 和 gated FFN 做了等价折叠/融合，并为细长 GEMM 手工调整 Triton tile。
详见 `Running VLAs at Real-time Speed <https://arxiv.org/html/2510.26742>`_。

切换到 dense snapshot 前，真实多进程 HTTP 服务的三图新 observation、已缓存图像和
Prefix REUSE wall p50 历史基线分别为 ``51.26/41.10/20.57 ms``；合入当前 gather
边界后需要重新测量。MPS 会把小 kernel 调度
成本带入 Action graph，REUSE wall 反而上升到约 ``22.6 ms``，因此不建议
对当前 Pi 路径开启。

Action Expert 不是只读一次“400 MB”。当前 checkpoint 实际加载 314.7M 参数，
即 629 MB BF16 权重；10 个 flow step 至少读取约 6.29 GB。H200 峰值带宽只给出
理想下界，``M≈50`` 的上千个小/细长 GEMM、层间全局依赖与 kernel 同步不会达到
连续 memcpy 的峰值带宽。原 ``22.72 ms`` 还包含 Python/进程热路径、重复
attention 规划和不合适的 kernel shape；当前 scheduler 从每个 layer-step 规划降为
整次 action 一次（profiler 约 ``0.0033 ms``）；Prefix KV 在任务开始时 gather
一次，之后 10 个 flow step 都复用同一份 dense snapshot；固定 suffix attention
和 action tile 则适配 50--51 token 形状。

Action API
----------

请求发送到 ``POST /v1/vla/actions``。不带 context 字段时保留一次性语义，每次
处理一个 observation，字段如下：

* ``prompt``：任务文本；
* ``images``：按 checkpoint camera 名称给出的映射或已按顺序排列的 list；每张图可
  为 base64/data URL、HTTP(S) URL 或 ``H x W x 3`` 数组；
* ``state``：``[state_dim]`` 或 ``[1, state_dim]``；
* 可选 ``domain``（也接受 LightX2V 的 ``domain_name``），用于选择环境相机映射、
  state/action 维度、归一化统计和机器人坐标转换；
* 可选 ``image_mask``；
* 可选 ``noise``，形状为 ``[1, action_horizon, max_action_dim]``；当前 checkpoint
  的固定模型宽度为 32，即使只返回较少机器人 action 维度，也要提供完整 32 维噪声；
* 可选 ``seed``，未显式提供 ``noise`` 时控制服务端噪声采样；
* 可选 ``action_horizon``、``action_dim``、``num_denoise_steps`` 与 ``timeout``。
* ``outputs`` 可为 ``["action"]`` 或 ``["text", "action"]``。省略时保留
  该 endpoint 原有的 action-only 行为。

camera mapping 会按 checkpoint 的 ``image_keys`` 排序并校验名称；兼容的 list
输入仍保留调用方顺序。响应中的 ``actions`` 为 ``[action_horizon, action_dim]``。

Pi 的三图约束与 Domain Adapter
-------------------------------

OpenPI 的 π₀/π₀.₅ 模型接口固定定义 ``base``、``left_wrist`` 和
``right_wrist`` 三个语义槽位，这不等于环境必须安装三台相机。训练和推理时
每个槽位都有 mask：LIBERO、DROID 只有两路有效视图时，缺失的第三路不会进入
LightLLM 的视觉 batch 和 Prefix KV；ALOHA、RobotWin 可使用三路。也就是说，
“三个槽位”是 checkpoint 输入 schema，“几台真实相机”是 domain contract。

``domain`` 只在模型外选择 I/O 变换，不会作为 learned domain id 输入 Pi 模型。
通用 Action node 不包含 benchmark 分支；以下轻量 profile 由 Pi 模型预处理器提供，
兼容 LightX2V 的环境命名：

.. list-table::
   :header-rows: 1

   * - domain
     - 环境相机 -> Pi 槽位
     - state/action
   * - ``libero``
     - ``agentview`` -> base，``wrist`` -> left wrist
     - 8 / 7
   * - ``droid`` / ``robolab``
     - exterior/``over_shoulder_left_camera`` -> base，wrist -> left wrist
     - 8 / 8
   * - ``aloha``
     - ``cam_high``、``cam_left_wrist``、``cam_right_wrist``
     - 14 / 14
   * - ``robotwin``
     - ``head_camera``、``left_camera``、``right_camera``
     - 14 / 14

ALOHA profile 还复现 OpenPI 官方的 joint sign flip 和双 gripper
linear/angular 坐标转换；左右 wrist 缺失时也可按 false mask 省略，只有
``cam_high`` 是必需视图。

例如 LightX2V 的 LIBERO observation 可直接发送环境字段名，不需要上层伪造第三张图：

.. code-block:: python

   body = {
       "domain": "libero",
       "prompt": "pick up the block",
       "images": {
           "agentview": encode_image("agentview.jpg"),
           "wrist": encode_image("wrist.jpg"),
       },
       "state": [0.0] * 8,
   }

模型 profile 只定义公开 I/O contract，不会让一个未在该 domain 上训练的 checkpoint
自动获得对应控制能力。实际 rollout 仍必须提供匹配 checkpoint 的 normalization
statistics 和机器人坐标语义。自定义环境通过 ``--vla_domain_config`` 添加，不需要
修改 HTTP、ActivationTransfer 或模型代码：

.. code-block:: json

   {
     "domains": {
       "my_robot": {
         "state_dim": 12,
         "action_dim": 10,
         "image_map": {"front": "base", "hand": "left_wrist"},
         "required_images": ["base", "left_wrist"],
         "norm_config": {
           "state": {"mode": "mean_std", "mean": [0.0], "std": [1.0]},
           "action": {"mode": "quantiles", "q01": [-1.0], "q99": [1.0]}
         },
         "robot_adapter": {"relative_action_mask": [true, true, false]},
         "action_postprocess_config": {"scale": 1.0, "clip": [-1.0, 1.0]}
       }
     }
   }

持久 context 的 domain 是 prefix contract 的一部分：CREATE/REPLACE 会记录它，
REUSE 可以省略并继承；显式传入不同 domain 会被拒绝，避免复用错误的相机 Prefix KV。

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

服务只在该 JSON 响应实际完成 ASGI 发送后，才把新 context 视为已交付；若客户端
在响应发送期间断开，服务会自动 CLOSE 这个尚未交付的 context，避免永久占用 KV。

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
prompt/images/state。新 VLM prefill 期间旧 context 仍可服务高频 tick；新 KV READY
后暂停接收旧版本 tick，新 action 安全完成后切换版本并释放旧 lease。
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

ONESHOT/CREATE/REPLACE 使用标准 ``ShmReq`` 启动 VLM；其中持久 context
保留该 request 的 KV lease。后续 REUSE/CLOSE 是直达 ``VLAServer`` 的
``ActionContextTask``，不创建 ``ShmReq``、不 poll router，也不触发 VLM。
target 的原有 finish/filter 仍负责最终回收，但必须等 consumer lease 释放。

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

上述三个全局参数用于无 ``domain`` 的兼容请求；domain profile 内同名字段会覆盖
全局值，从而允许同一个服务按环境选择不同 statistics 和 action 语义。

验证
----

独立精度 oracle 使用 Transformers/OpenPI 方程和真实 checkpoint，不复用 LightLLM
layer 实现，并按服务相同的二维 request table 顺序构造只读 Prefix KV。
float32 gate 为 action ``atol=rtol=2e-5``、prefix KV 最大绝对误差
``1e-3``、vision 最大绝对误差 ``2e-5``：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python test/acc/vla/run_openpi_lightllm_diff.py \
     --model-dir /path/to/pi0_base \
     --tokenizer-path /path/to/paligemma_tokenizer.model \
     --dtype float32 \
     --batch-size 2 --num-images 3 --action-horizon 50 --num-steps 10

``batch=2`` 默认构造一行三图、一行双图；增加 ``--valid-images 2`` 可让所有行
都使用双相机。H200 上默认 mixed ``batch=2`` 的 FP32 action 最大/平均绝对误差为
``1.37e-6``/``2.72e-7``，prefix KV 最大绝对误差 ``1.66e-4``。这证明省略
false-mask 图像与 OpenPI pad+mask 的数学语义一致。

BF16 部署组合可用下面的最终 action gate 复现：

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python test/acc/vla/run_openpi_lightllm_diff.py \
     --model-dir /path/to/pi0_base \
     --tokenizer-path /path/to/paligemma_tokenizer.model \
     --dtype bfloat16 --action-horizon 50 --num-steps 10 \
     --atol 0.015 --rtol 0.01 --vision-atol 0.9 --prefix-kv-atol 5.1 \
     --enable-action-cudagraph --benchmark-warmup 20 --benchmark-runs 100

H200 上使用真实 π₀ checkpoint、三图、``batch=2``、``action_horizon=50``、
``num_denoise_steps=10`` 的 FP32 action 最大/平均绝对误差为
``4.53e-6``/``1.78e-7``，prefix KV 最大绝对误差 ``1.66e-4``；``batch=1`` 全链路
BF16 action 最大/平均绝对误差为 ``1.26e-2``/``2.12e-3``。同形状 π₀.₅ BF16
最大/平均绝对误差为 ``1.38e-2``/``2.45e-3``。CUDA Graph 与对应 eager BF16 输出
逐元素一致，不增加数值
误差。BF16 深层 hidden/KV 的绝对误差会逐层累积，因此最终 action 是部署精度
gate，中间值用于诊断。这一数值回归仍不能代替反归一化后的机器人单位误差、
代表性数据集或 rollout success 验证。

π₀.₅ 的 BF16 Vision/Prefix 累积误差也更大；对应独立 gate 使用
``--atol 0.02 --rtol 0.01 --vision-atol 1.5 --prefix-kv-atol 8.0``。本次实测
Vision/Prefix 最大差为 ``1.4367/6.3961``，最终 action 仍是主验收值。

双相机 LIBERO 的 BF16 shape 会选择不同 kernel 数值路径；当前 action 最大/平均
绝对误差为 ``2.83e-2``/``4.42e-3``，高于三图使用的严格 ``0.015`` 绝对 gate，
但 CUDA Graph 与该 shape 的 eager 结果逐元素一致。因此双相机部署应以 FP32 oracle、
反归一化误差和真实 rollout 共同验收，不能直接沿用三图 gate。

以下也是切换 dense snapshot 前的 H200 多进程历史基线：在 ``horizon=50``、
10-step、每次使用新图像并完成预热后，
LIBERO 双相机 wall p50/p90 为 ``45.57/48.21 ms``，Action Expert p50 为
``15.35 ms``。同一个持久 Prefix 的 REUSE wall p50/p90 为 ``20.43/20.82 ms``，
Action backend/Expert p50 为 ``17.47/15.30 ms``，随后 CLOSE 成功释放 context。
RobotWin 三相机上次基线为 ``53.57/60.81 ms``。Domain profile 解析不进入 GPU
热路径。

2026-08-24 在隔离 H200 上完成了 dense snapshot 后的新一轮 π₀ BF16 测试。
``batch=1``、三图、774-token prefix、``horizon=50``、10-step、Action CUDA Graph，
warmup 5 次后采样 20 次，离线 stage p50/p90 分别为：Vision ``8.55/8.60 ms``、
VLM prefill ``9.10/11.17 ms``、Triton dense KV gather ``0.277/0.284 ms``、
Action Expert ``13.30/13.31 ms``。正常 HTTP 多进程链路使用不同图像时，wall
p50/p90 为 ``49.86/53.08 ms``，服务内部 total 为 ``41.47/42.40 ms``，Action
backend 为 ``18.60/18.91 ms``。启动与首次 CUDA Graph capture 未计入稳态样本。

逐层 Torch gather 至少发起 36 次 ``index_select``，真实模型隔离 p50 为
``0.643 ms``；融合为单个 Triton launch 后为 ``0.277 ms``。较早得到的
``4.98 ms`` gather/``79.34 ms`` wall 是所有 GPU 被另一套 TP8 服务打满时的共享卡
结果，不应作为 π₀ 性能基线。

服务启动后，可用 ``test/acc/vla/run_server_smoke.py`` 覆盖实际的
HTTP -> visualserver -> router/BaseModel -> ActivationChannel -> VLAServer 链路。
