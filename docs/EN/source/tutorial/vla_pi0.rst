π₀ / π₀.₅ VLA Inference
=======================

LightLLM supports inference for LeRobot exports of OpenPI π₀ and π₀.₅. Ordinary
text requests keep the existing VLM backend. A small ``VLARequestLifecycle``
and a separate action-expert process group provide one-shot outputs as well as
versioned persistent prefixes for high-frequency action tasks.

Architecture
------------

Requests retain the standard LightLLM lifecycle:

.. code-block:: text

   create/replace: HTTP -> visualserver -> router -> ordinary VLM prefill
                                      -> PrefixContextRegistry adopts prefix KV
   action tick:    HTTP ----------------> router -> ActionTask -> actionserver
                                                     | (reuse prefix KV)
                                                     `-> action output
   text request:   HTTP -> visualserver -> router -> existing prefill/decode/finish

There is no separate VLA HTTP server or visual worker. ``Pi0VLMModel``
subclasses ``Gemma_2bTpPartModel`` and
reuses BaseModel's ``ModelInput``, ``InferStateInfo``, weight loading, TP, RoPE,
RMSNorm, Gemma layers, post layer, and prefill attention backends. The required
block-bidirectional prefix attention is selected only by
``prefill_causal=False``; ordinary models retain the causal default.

Like visualserver and audioserver, ``actionserver`` consists of a manager and
TP model RPC processes. On the classifier pass after synchronized prefill,
``PrefixContextRegistry`` adopts the physical prefix pages; the generic
``prefill_normal`` contains no action branch. The worker reads an immutable
snapshot of the router-owned mapping and writes only context-owned, serially
reused suffix scratch pages. Its logical ``req_to_token_indexs`` table is independent from
the target table, so text decode and action suffix mappings do not
overwrite one another.
The first successful CREATE/REPLACE action idempotently registers the complete
``PrefixContextIdentity -> prefix/scratch mapping`` on every action rank.
Subsequent REUSE ticks send only the versioned identity, latest state/noise,
and sampling options; the hot path no longer retransmits an
``O(prefix_len)`` mapping over ZMQ/RPyC, gathers target-TP mappings, or allocates
scratch. CLOSE, or a drained old version, unregisters worker-local metadata by
the exact identity while physical KV ownership remains in the router.

One ``PrefixContext`` can queue multiple short-lived ``ActionTask`` objects,
with at most one running task per context. Scratch is reused only after every
action worker and target TP rank acknowledges safety. Prefix pages are released
only after an explicit CLOSE, or after a copy-on-write REPLACE version has
drained. Clients must use ``DELETE /v1/vla/contexts/{context_id}`` as the normal
resource-release path instead of relying on automatic cleanup for the current
context.
HTTP output consumption is a separate lifecycle step. A missing safe ACK keeps
the KV lease and reports that a process restart is required rather than risking
reuse of live pages.
A new CREATE/REPLACE handle remains provisionally owned until the public HTTP
response body completes its ASGI send. Even if the ordinary ``InferReq`` has already
passed finish/filter, the router retains a constant-size owner record. Every
target rank keeps the new context after a successful send; a failed send,
disconnect, cancellation, or response-processing failure closes a new CREATE
context. For REPLACE, the old current context and KV remain held until the ASGI
send completes. Success commits the new version and retires the old one after
safe drain; send failure or owner discard rolls back to the old version and
releases the new one. The reusable ``ShmReq`` slot is released only after that
owner ACK, preventing unreachable orphan KV.

The action expert keeps its independent model and denoising loop, while using
LightLLM's ``ROWMMWeight``, ``COLMMWeight``, ``QKVROWNMMWeight``,
``KVROWNMMWeight``, ``Quantcfg``, and existing attention, RoPE, RMSNorm, and
GELU kernels. TP sharding, all-reduce, and existing quantization extension
points therefore remain available.

There is no exclusive ``VLABackend``. The normal backend still owns model
topology, batching, sampling, and resource management. The language-model
``chunked_prefill/impl.py`` remains generic; small ``_get_classed_reqs`` hooks
poll and park extension tasks, prevent unsafe pause, and return safe requests to
the existing finish/filter path. This boundary can also host future non-text
branches such as image generation.

Launch
------

The checkpoint ``type`` identifies π₀ automatically and enables the action
runtime, so use normal mode:

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
options and may run on other GPUs. The first validated action-runtime matrix
requires a single node, ``dp=1``, no PD split, and no multi-level KV cache,
constraint decoding, mixed prefill/decode batching, or microbatch overlap.
Unsupported combinations fail during startup; they are not blocked by a
different backend selection and can be enabled independently after validation.

Action dimension, horizon, denoising steps, and state mode are read from the
checkpoint and can be overridden with ``--vla_action_dim``,
``--vla_action_horizon``, and ``--vla_num_denoise_steps``. For deployment,
explicitly using ``--data_type bfloat16`` is recommended. The current LeRobot
checkpoints declare ``float32``, so omitting this option inherits the checkpoint
dtype and runs in FP32. BF16 applies to the VLM, Action Expert, and KV cache;
the vision encoder computes in FP32 and casts its output to BF16. FP32 is
primarily useful for strict numerical regression and accuracy debugging.
The Action Expert follows LightLLM's standard layer-infer
path and selects bidirectional block attention through its own inference state.
A bidirectional prefix cannot enter the causal radix KV
cache, so VLA startup disables dynamic prompt caching and chunked prefill. The
ordinary visualserver embedding cache remains enabled.

BF16 performance reference
---------------------------

On an NVIDIA H200 with π₀, ``batch=1``, three images,
``action_horizon=50``, and ``num_denoise_steps=10``, 30 HTTP requests with
changing image inputs were measured after 10 warmup requests:

.. list-table::
   :header-rows: 1

   * - Metric
     - FP32 p50 / p95
     - BF16 p50 / p95
     - BF16 p50 speedup
   * - HTTP end-to-end latency
     - 2286.2 / 2304.7 ms
     - 201.2 / 224.9 ms
     - 11.4x
   * - Action Expert GPU time
     - 1922.6 / 1923.3 ms
     - 90.5 / 93.6 ms
     - 21.2x

This compares the recommended deployment stacks: BF16 auto-selects
FA3/FlashInfer while FP32 uses Triton. The result therefore includes both dtype
and its default attention backend, rather than being a dtype-only kernel
microbenchmark. Latency varies with the GPU, request shape, and system load.
The reported ``90.5 ms`` is the complete ten-step Action Expert loop (roughly
``9 ms/step``), not VLM prefill. The action denoising loop is not currently
captured by CUDA Graph. Persistent-prefix mode removes subsequent ViT/VLM
prefill, KV gather, and scratch allocation, but it does not itself remove those
90 ms; measure that mode separately rather than extrapolating from 201.2 ms.

Action API
----------

Send requests to ``POST /v1/vla/actions``. Without context fields, the endpoint
keeps its one-shot behavior and accepts one observation per request:

* ``prompt`` is the task text;
* ``images`` is a checkpoint-camera mapping or an already ordered list; each
  image can be base64/data URL, HTTP(S) URL, or an ``H x W x 3`` array;
* ``state`` has shape ``[state_dim]`` or ``[1, state_dim]``;
* ``image_mask`` is optional;
* fixed ``noise`` is optional and has shape
  ``[1, action_horizon, action_dim]``;
* horizon, dimension, denoising steps, and timeout may be overridden.
* ``outputs`` may be ``["action"]``, ``["text"]``, or
  ``["text", "action"]``. Omitting it preserves the legacy action-only
  behavior of this endpoint.

Camera mappings are processed in checkpoint order (base, left wrist, right
wrist), independent of JSON insertion order. Response ``actions`` has shape
``[action_horizon, action_dim]``.

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
identity, and complete prompt/images/state. The new version publishes
atomically, and the old version is released after safe ACK. When the control
loop ends, clients must send an explicit CLOSE; the ``DELETE`` request in the
example is that CLOSE operation.

The legacy action-only response remains the bare action object shown above.
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

Every action task still uses a standard ``ShmReq`` for output, cancellation,
timeout, and final recycling. The persistent registry owns only versioned
physical prefix KV and context scratch; it does not duplicate HTTP/session
scheduling. Safe tasks return to the ordinary LLM finish/filter path. REUSE
therefore still pays local ``ShmReq`` creation, router polling, and the worker
copy that installs cached indexes into a task-scoped logical row; it only
removes cross-process retransmission and target-TP gathering of the
``O(prefix_len)`` mapping. If action CUDA Graph makes denoising substantially
faster, the next step is a persistent logical row plus a direct hot-tick action
data plane. Those optimizations are not implemented by this branch.

Normalization and adapters
--------------------------

Without a normalization file, ``state`` and returned actions are treated as
already normalized. ``--vla_norm_config`` accepts ``state`` and ``action``
entries using ``identity``, ``mean_std``, or ``quantiles``. A robot adapter may
define ``relative_action_mask``; scale, offset, and clipping are accepted via
``--vla_action_postprocess_config``.

Validation
----------

The independent accuracy oracle uses Transformers/OpenPI equations and the
real checkpoint without reusing LightLLM layers. It installs prefix and suffix
indexes through the same action-owned scoped KV view as the server. Its float32 gates are action
``atol=rtol=2e-5``, prefix-KV maximum absolute error ``1e-3``, and vision
maximum absolute error ``2e-5``:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python test/acc/vla/run_openpi_lightllm_diff.py \
     --model-dir /path/to/pi0_base \
     --dtype float32 \
     --batch-size 2 --num-images 3 --action-horizon 50 --num-steps 10

The strict FP32 gate above is not a task-quality threshold for BF16. In a quick
fixed-synthetic-input comparison with ``batch=1``, ``action_horizon=4``, and
``num_denoise_steps=2``, LightLLM BF16 versus the OpenPI FP32 oracle produced
about ``1.1e-2`` maximum absolute action drift for π₀/π₀.₅ and about
``2.5e-3``/``3.8e-3`` mean absolute drift. This magnitude showed no numerical
instability, but it does not replace error checks in denormalized robot units,
representative datasets, or rollout-success validation.

With a server running, ``test/acc/vla/run_server_smoke.py`` covers the real
HTTP -> visualserver -> router/BaseModel -> actionserver route.
