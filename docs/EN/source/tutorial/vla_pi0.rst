π₀ / π₀.₅ VLA Inference
=======================

LightLLM supports inference for LeRobot exports of OpenPI π₀ and π₀.₅. Ordinary
text requests keep the existing VLM backend. An autonomous ``VLAServer`` runs
action-expert inference and consumes VLM Prefix KV through
``ActivationChannel``. Requests return actions
and may also return text, while control loops can retain a versioned VLM prefix
and submit high-frequency state ticks.

Architecture
------------

Cold requests retain the standard VLM lifecycle and are submitted to
``VLAServer`` concurrently:

.. code-block:: text

   HTTP (protocol parsing + ShmReq allocation)
      |-> visualserver -> router -> target prefill/decode -> existing text output
      `-> VLAServer -> ActivationChannel -> model_rpc -> action output

   target post_handle -> publish kv_ready(rank mask, infer_req_idx)
   target finish/filter -> reclaim KV only after kv_consumer_mask is clear

   hot REUSE: HTTP -> VLAServer -> model_rpc
   CLOSE/finish: VLAServer -> ActivationChannel release -> normal KV reclaim

There is no separate VLA HTTP server or visual worker. HTTP handles only wire
conversion, ``ShmReq`` allocation, asynchronous fan-out, and result joining;
visualserver and VLAServer own image and action/state preprocessing,
respectively. ``Pi0VLMModel`` and ``Pi05VLMModel`` register their checkpoint
types independently and share the same PaliGemma BaseModel implementation.
Both reuse BaseModel's ``ModelInput``, ``InferStateInfo``, weight loading, TP, RoPE,
RMSNorm, Gemma layers, post layer, and prefill attention backends. The required
block-bidirectional prefix attention is selected only by
``prefill_causal=False``; ordinary models retain the causal default.

The Action Expert also has fixed implementations under ``models/pi0`` and
``models/pi05``. Pi0 always uses a continuous-state token and ordinary RMSNorm;
Pi0.5 always uses discrete state in the prefix, a timestep condition, and
AdaRMSNorm. The worker selects the model once at startup, so layer hot paths
contain no ``is_pi05`` or optional-``condition`` branches. Both models follow
``forward -> Pi0ActionInferStateInfo -> _context_forward`` and reuse the common
``CudaGraphRunner`` rather than maintaining separate runtime/eager flows.

``VLAServer`` owns request deadlines, cancellation, Prefix Contexts, and Action
TP model RPCs; it no longer depends on a universal GenerationManager.
GenerationModelRegistry selects the concrete Expert class from the checkpoint
while ModelRegistry remains responsible only for the target VLM. For
ONESHOT/CREATE/REPLACE, every target TP rank publishes the generic
``kv_ready_rank_mask`` and ``infer_req_idx`` at the CUDA-safe ``post_handle``
boundary. ``ActivationChannel`` resolves a read-only ``KVActivation``
after all ranks are ready. Each action worker attaches its matching target
``MemoryManager`` and request table through CUDA IPC, then gathers a local
``DensePrefixKV`` once at task start. Beyond that adapter the action model no
longer reads the target allocator or request table. The suffix stays in a two-dimensional
packed layout and uses fixed-shape QK/softmax/AV before an LSE merge with prefix
attention. The common ``SplitPrefill`` attention backend owns this policy; layers
only call ``prefill_att`` through the inference state, while Pi0 and Pi0.5 select
their suffix masks. Scheduler metadata is planned once outside all ten steps and
layers. FP16/BF16 runs only through FA3. The explicit FP32 correctness path lives
in the isolated ``split_prefill/fp32.py`` module; it is not a runtime fallback for
BF16. The action side creates no second allocator or scratch page.

CREATE activates a context only after its first action completes safely. During
a REPLACE visualserver/router/VLM prefill, the old context keeps accepting
high-frequency ticks. It is frozen only when the new KV becomes READY; the
manager then switches atomically on success or restores the old context on
failure. REUSE sends only
the versioned identity, latest state/noise, and sampling options directly to
``VLAServer``: it creates no ``ShmReq`` and bypasses visualserver, router,
and the VLM. Ticks for one context run FIFO. CLOSE drains accepted work, then
``VLAServer`` asks ``ActivationChannel`` to clear its bit in the shared
``kv_consumer_mask`` using the
``(shm slot, request_id)`` identity after receiving a CUDA-safe worker ACK. The
normal finish/filter path then reclaims KV naturally. Router and the target
inference loop receive no action-release command. If an action worker cannot
confirm CUDA safety, the lease
stays pinned and the response sets ``restart_required`` rather than risking
reuse of live pages. Clients should use
``DELETE /v1/vla/contexts/{context_id}`` as the normal release path.

The action expert keeps its independent model and denoising loop, while using
LightLLM's ``ROWMMWeight``, ``COLMMWeight``, ``QKVROWNMMWeight``,
``KVROWNMMWeight``, and ``Quantcfg``. RoPE, RMSNorm, and projections retain the
existing interfaces. Single-KV-head block attention uses the common
``SplitPrefill`` backend; the Pi model directory keeps only the small 51-token
GELU tile kernel. TP sharding, all-reduce, attention backends, and existing
quantization extension points therefore remain available.

There is no ``inference_runtime``, ``PrefixContextRegistry``, or HTTP-to-Router
owner-ACK control plane.
The target exposes only generic KV-consumer hooks: ``post_handle`` publishes KV
readiness, and the existing finish/filter/pause path delays physical KV
reclamation while ``kv_consumer_mask != 0``. The ordinary text path, Router,
and ``impl.py`` contain no action branch. Neo media generation reuses the same
``KV_CONSUMER_GENERATION`` lease and dense-gather boundary, while its autonomous
Image Generation Server owns the media workflow.

Launch
------

The checkpoint ``type`` identifies π₀ automatically and selects ``VLAServer``,
so use normal mode:

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

Use ``/path/to/pi05_base`` for π₀.₅.

Tokenizer assets must be provided explicitly or stored under the model
directory as ``paligemma_tokenizer.model``, ``tokenizer.model``, or
``tokenizer/``. Server startup never downloads from a hard-coded URL and fails
fast when no local tokenizer is available.

Action TP automatically follows VLM TP and uses the corresponding GPUs because
prefix KV is shared through CUDA IPC. The visual encoder continues to
use the generic ``--visual_gpu_ids``, ``--visual_tp``, and ``--visual_dp``
options and may run on other GPUs. The first validated action-service matrix
requires a single node, ``dp=1``, no PD split, and no multi-level KV cache,
constraint decoding, mixed prefill/decode batching, or microbatch overlap.
Unsupported combinations fail during startup; they are not blocked by a
different backend selection and can be enabled independently after validation.

Action dimension, horizon, and denoising steps are read from the
checkpoint and can be overridden with ``--vla_action_dim``,
``--vla_action_horizon``, and ``--vla_num_denoise_steps``. For deployment,
explicitly using ``--data_type bfloat16`` is recommended. The current LeRobot
checkpoints declare ``float32``, so omitting this option inherits the checkpoint
dtype and runs in FP32. BF16 applies consistently to Vision, VLM, the Action
Expert, and KV cache; a BF16 service no longer keeps an implicit FP32 Vision
path. FP32 is
primarily useful for strict numerical regression and accuracy debugging.
The Action Expert follows LightLLM's standard layer-infer
path and selects bidirectional block attention through its own inference state.
A bidirectional prefix cannot enter the causal radix KV
cache, so VLA startup disables dynamic prompt caching and chunked prefill. The
ordinary visualserver embedding cache remains enabled.

BF16 performance reference
---------------------------

On an NVIDIA H200 with ``batch=1``, ``action_horizon=50``, and
``num_denoise_steps=10``, the warmed-up Action Expert measured:

.. list-table::
   :header-rows: 1

   * - Model/input
     - BF16 CUDA Graph p50
     - Notes
   * - π₀
     - 13.24 ms
     - three-image prefix, 50 action tokens, ten steps
   * - π₀.₅
     - 19.07 ms
     - three-image prefix, 50 action tokens, ten steps

The Action Expert captures the complete ten-step denoising loop as one CUDA
Graph replay. ``--disable_cudagraph`` disables it; FP32 uses eager, while
BF16/FP16 graph execution no longer depends on the LLM selecting FA3. The first
new ``(batch, horizon, steps)`` incurs capture cost. True prefix lengths and the
two-dimensional page table are graph inputs, and attention masks inactive
positions exactly.

For three images, ``horizon=50``, and ten steps, Pi0's pure-GPU p50 values are
Vision CUDA Graph ``3.74 ms``, VLM prefill ``9.16 ms``, and Action Expert
``13.24 ms``: ``26.14 ms`` when the stages are added. Vision graph output is
elementwise identical to eager output. The Pi visual adapter also publishes
one observation's cameras as one inline batch instead of using the per-item
infer/store queues. Both changes are Pi-only; every other VLM keeps the
existing visualserver queues and LLM overlap path.

``policy_timing`` returns RPC/postprocessing-inclusive ``action_backend_ms``
and its difference from GPU ``action_expert_ms`` as ``action_overhead_ms``.
HTTP wall time also includes JPEG, process transport, Visual/Router scheduling,
and KV lifecycle, so stale service measurements must not replace this pure-GPU
baseline.

The original `realtime-vla <https://github.com/dexmal/realtime-vla>`_
``benchmark.py`` measured ``16.24/18.88/21.43 ms`` p50 for 1/2/3 views on the
same H200 with prompt length 20 and chunk length 50; its published 4090
three-view result is ``39.2 ms``. That benchmark
uses fixed-shape GPU tensors and captures Vision + VLM + all ten AE steps in one
CUDA Graph; it excludes HTTP, JPEG, LightLLM multiprocess transport, and Prefix
KV lifecycle costs. It also folds or fuses RMS affine parameters, action/time
encoding, QKV, RoPE, and gated FFN, then manually tunes Triton tiles for skinny
GEMMs. See `Running VLAs at Real-time Speed
<https://arxiv.org/html/2510.26742>`_.

Before the dense-snapshot transition, historical three-view new-observation,
cached-image, and Prefix REUSE wall p50 values were ``51.26/41.10/20.57 ms``;
the service must be remeasured with the current gather boundary. MPS increases
small-kernel scheduling cost and raises REUSE wall time to about ``22.6 ms``, so
it is not recommended for the current Pi path.

The Action Expert does not read a single "400 MB" model once. The checkpoint
loads 314.7M parameters, or 629 MB of BF16 weights, and ten flow steps must read
at least about 6.29 GB. H200 peak bandwidth is only an ideal lower bound:
thousand-scale small/skinny GEMMs with ``M≈50``, layer dependencies, and kernel
synchronization cannot sustain memcpy bandwidth. The old ``22.72 ms`` path also
contained Python/process hot-path work, repeated attention planning, and
ill-shaped kernels. The current scheduler plans once per action instead of once
per layer-step (about ``0.0033 ms`` in the profiler). Prefix K/V is gathered
once at task start and the same dense snapshot is reused by all ten flow steps;
fixed suffix attention plus Pi-sized tiles match the 50--51-token shape.

Action API
----------

Send requests to ``POST /v1/vla/actions``. Without context fields, the endpoint
keeps its one-shot behavior and accepts one observation per request:

* ``prompt`` is the task text;
* ``images`` is a checkpoint-camera mapping or an already ordered list; each
  image can be base64/data URL, HTTP(S) URL, or an ``H x W x 3`` array;
* ``state`` has shape ``[state_dim]`` or ``[1, state_dim]``;
* optional ``domain`` (or LightX2V's ``domain_name`` alias) selects environment
  camera mapping, state/action dimensions, normalization, and robot coordinates;
* ``image_mask`` is optional;
* fixed ``noise`` is optional and has shape
  ``[1, action_horizon, max_action_dim]``; current checkpoints have fixed model
  width 32, even when fewer robot action dimensions are returned;
* optional ``seed`` controls server-side noise sampling when ``noise`` is omitted;
* horizon, dimension, denoising steps, and timeout may be overridden.
* ``outputs`` may be ``["action"]`` or ``["text", "action"]``. Omitting it
  preserves the endpoint's default action-only behavior.

Camera mappings are validated and reordered by the checkpoint's ``image_keys``;
the compatible list form still preserves caller order. Response ``actions``
has shape ``[action_horizon, action_dim]``.

Pi image slots and the Domain Adapter
-------------------------------------

The OpenPI π₀/π₀.₅ model interface defines three fixed semantic slots:
``base``, ``left_wrist``, and ``right_wrist``. This does not require every
environment to install three cameras. Each slot has a mask in the model
contract. For two-view LIBERO and DROID observations, LightLLM omits the absent
third view from both the visual batch and Prefix KV; ALOHA and RobotWin may use
all three. Three slots are a checkpoint schema, while the physical cameras are
a domain contract.

``domain`` selects only external I/O transforms; it is not a learned Pi domain
id. The common Action node has no benchmark branches; these lightweight
profiles are supplied by the Pi preprocessor and accept LightX2V environment
names:

.. list-table::
   :header-rows: 1

   * - domain
     - environment cameras -> Pi slots
     - state/action
   * - ``libero``
     - ``agentview`` -> base, ``wrist`` -> left wrist
     - 8 / 7
   * - ``droid`` / ``robolab``
     - exterior/``over_shoulder_left_camera`` -> base, wrist -> left wrist
     - 8 / 8
   * - ``aloha``
     - ``cam_high``, ``cam_left_wrist``, ``cam_right_wrist``
     - 14 / 14
   * - ``robotwin``
     - ``head_camera``, ``left_camera``, ``right_camera``
     - 14 / 14

The ALOHA profile also reproduces OpenPI's joint-sign flips and two gripper
linear/angular conversions. Either wrist view may be omitted with a false mask;
only ``cam_high`` is required.

A LightX2V LIBERO observation can therefore use environment field names without
fabricating a third image:

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

A model profile defines the public I/O contract only; it cannot make a
checkpoint control a domain on which it was not trained. Real rollouts still
need normalization statistics and robot-coordinate semantics matching that
checkpoint. Add a custom environment through ``--vla_domain_config`` without
changing HTTP, ActivationTransfer, or model code:

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

Domain is part of a persistent prefix contract. CREATE/REPLACE records it,
REUSE may omit and inherit it, and an explicit mismatch is rejected to prevent
using Prefix KV from the wrong camera layout.

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

For a high-frequency π₀ control loop, separate low-frequency observation
updates from high-frequency action ticks. Set ``persist_prefix=true`` on the
first request to perform a low-frequency CREATE: it runs ViT/VLM prefill and
retains the prefix KV. The response contains
``prefix_context={context_id, version, server_epoch}``:

The new context is considered delivered only after that JSON response finishes
its ASGI send. If the client disconnects while the body is being sent, the
server closes the undelivered context automatically instead of pinning its KV.

.. code-block:: python

   first = requests.post(
       "http://127.0.0.1:8000/v1/vla/actions",
       json={**body, "persist_prefix": True, "context_id": "robot-1"},
       timeout=120,
   ).json()
   context = first["prefix_context"]

   # High-frequency REUSE: send the latest state on every tick.
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

Pi0 continuous state belongs to the action suffix, so every REUSE carries the
latest state while retaining the same prefix. Tasks for one context execute
FIFO, with at most one action task in flight at a time. Clients may submit later
ticks before the previous response arrives, but should bound the queue and use
timeouts for backpressure. The current policy is strict FIFO and never silently
coalesces intermediate states into the latest state. If producers outrun the
Action Expert, queue latency grows; a control client should normally keep one
outstanding tick. A latest-state/coalescing policy belongs in a later, explicit
queue mode.

Pi0.5 discretizes state into the bidirectional VLM prefix, so a state change
cannot use REUSE. It must perform REPLACE with ``replace_prefix=true``, the old
identity, and complete prompt/images/state. The old context remains usable
during the new VLM prefill and stops accepting old-version ticks only after the
new KV is READY. The version switches after the new action completes safely,
then the old lease is released. When the control loop
ends, clients must send an explicit CLOSE; the ``DELETE`` request in the example
is that CLOSE operation.

The default action-only response remains the bare action object shown above.
Text-bearing requests return an aggregate object. For example:

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

Action output is exposed only by the dedicated ``/v1/vla/actions`` endpoint.
The ordinary LightLLM ``/generate`` and ``/generate_stream`` endpoints retain
their existing text-interface semantics and do not consume action events.

ONESHOT/CREATE/REPLACE use a standard ``ShmReq`` to run the VLM; a persistent
context retains that request's KV lease. Later REUSE/CLOSE operations are
``ActionContextTask`` messages sent directly to ``VLAServer``. They create
no ``ShmReq``, do not poll router, and do not run the VLM. The target's existing
finish/filter path still performs final reclamation, but only after the consumer
lease has been released.

Normalization and adapters
--------------------------

Without a normalization file, ``state`` and returned actions are treated as
already normalized. ``--vla_norm_config`` accepts ``state`` and ``action``
entries using ``identity``, ``mean_std``, or ``quantiles``. A robot adapter may
define ``relative_action_mask``; scale, offset, and clipping are accepted via
``--vla_action_postprocess_config``.

Those three global flags apply to compatible requests without ``domain``.
Fields with the same names in a domain profile override the globals, allowing
one service to select different statistics and action semantics per environment.

Validation
----------

The independent accuracy oracle uses Transformers/OpenPI equations and the
real checkpoint without reusing LightLLM layers. It follows the service's
two-dimensional request-table order when constructing immutable Prefix K/V.
Its float32 gates are action
``atol=rtol=2e-5``, prefix-KV maximum absolute error ``1e-3``, and vision
maximum absolute error ``2e-5``:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python test/acc/vla/run_openpi_lightllm_diff.py \
     --model-dir /path/to/pi0_base \
     --tokenizer-path /path/to/paligemma_tokenizer.model \
     --dtype float32 \
     --batch-size 2 --num-images 3 --action-horizon 50 --num-steps 10

``batch=2`` defaults to one three-view and one two-view row; add
``--valid-images 2`` to make every row two-view. On H200, the default mixed
``batch=2`` FP32 run produced
maximum/mean action error ``1.37e-6``/``2.72e-7`` and maximum Prefix-KV error
``1.66e-4``. This confirms that omitting a false-mask image is mathematically
equivalent to OpenPI's pad-and-mask representation.

The deployed BF16 action gate can be reproduced with:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python test/acc/vla/run_openpi_lightllm_diff.py \
     --model-dir /path/to/pi0_base \
     --tokenizer-path /path/to/paligemma_tokenizer.model \
     --dtype bfloat16 --action-horizon 50 --num-steps 10 \
     --atol 0.015 --rtol 0.01 --vision-atol 0.9 --prefix-kv-atol 5.1 \
     --enable-action-cudagraph --benchmark-warmup 20 --benchmark-runs 100

On an H200, a real π₀ checkpoint with three images, ``batch=2``,
``action_horizon=50``, and ``num_denoise_steps=10`` produced FP32 maximum/mean
action error ``4.53e-6``/``1.78e-7`` and maximum prefix-KV error ``1.66e-4``.
With ``batch=1``, end-to-end BF16 produced maximum/mean action error
``1.26e-2``/``2.12e-3``. Under that shape, π₀.₅ BF16 produced
``1.38e-2``/``2.45e-3``. CUDA Graph matches
its corresponding eager BF16 output elementwise and adds no numerical
error. Absolute hidden/KV drift grows through deep BF16 layers, so the final
action is the deployment accuracy gate and intermediate values are diagnostics.
This numerical regression still does not replace checks in denormalized robot
units, representative datasets, or rollout-success validation.

Pi0.5 also accumulates more BF16 Vision/Prefix error. Its standalone gate uses
``--atol 0.02 --rtol 0.01 --vision-atol 1.5 --prefix-kv-atol 8.0``; the measured
Vision/Prefix maxima are ``1.4367/6.3961``. Final action remains the primary
acceptance value.

The two-view LIBERO BF16 shape selects a different numerical kernel path. Its
current maximum/mean action error is ``2.83e-2``/``4.42e-3``, above the strict
three-view ``0.015`` absolute gate, while CUDA Graph remains elementwise equal
to eager for that shape. A two-view deployment must therefore combine the FP32
oracle with denormalized error and real rollout validation instead of reusing
the three-view gate unchanged.

The following is also a pre-dense-snapshot H200 baseline. After warmup, with
``horizon=50``, ten steps, and fresh images for each real multiprocess request,
two-view LIBERO wall p50/p90 was ``45.57/48.21 ms``
and Action Expert p50 was ``15.35 ms``. REUSE of one persistent Prefix measured
``20.43/20.82 ms`` wall p50/p90, with Action backend/Expert p50 at
``17.47/15.30 ms``; CLOSE then releases the context successfully. The previous
three-view RobotWin baseline is ``53.57/60.81 ms``. Domain profile resolution
adds no model computation to the GPU hot path.

On 2026-08-24, the post-dense-snapshot π₀ BF16 path was measured again on an
isolated H200. With batch 1, three images, a 774-token prefix, horizon 50, ten
denoising steps, and Action CUDA Graph, five warmups followed by 20 samples
gave offline stage p50/p90 values of ``8.55/8.60 ms`` for Vision,
``9.10/11.17 ms`` for VLM prefill, ``0.277/0.284 ms`` for Triton dense KV
gather, and ``13.30/13.31 ms`` for the Action Expert. The normal multiprocess
HTTP path with fresh images measured ``49.86/53.08 ms`` wall p50/p90,
``41.47/42.40 ms`` internal total, and ``18.60/18.91 ms`` Action backend.
Startup and first CUDA Graph capture are excluded from steady-state samples.

The former per-layer Torch gather launched at least 36 ``index_select``
operations and measured ``0.643 ms`` p50 in the isolated real-model run. One
fused Triton launch reduced it to ``0.277 ms``. Earlier ``4.98 ms`` gather and
``79.34 ms`` wall measurements were taken while another TP8 service saturated
all GPUs and must not be treated as a π₀ baseline.

With a server running, ``test/acc/vla/run_server_smoke.py`` covers the real
HTTP -> visualserver -> router/BaseModel -> ActivationChannel -> VLAServer route.
