# Attention Execution Design

This refactor separates persistent attention storage, static computation requirements, and batch execution. States own storage, backends implement computation for one phase, and sessions enforce execution contracts. Model layers describe their requirements, supply inputs, and consume results.

The aim is to keep changes for a new model or backend within its implementation, while defining capacity checks, resource reuse, graph inputs, and output handling once. Storage follows the existing [Attention Resources design (Chinese)](https://github.com/ModelTC/LightLLM/blob/refactor/attention-session-rewrite/docs/CN/source/framework/cache_manager_design.md).

## 1. Responsibilities and ownership

| Object | Holds | Responsibility |
| --- | --- | --- |
| `State` / `AttentionStateManager` | Persistent KV, Conv, and SSM storage and its layout | Allocate storage, expose layer views, checkpoint, restore, and transfer data; allocators and callers manage external page indices |
| `ModelAttention` | Backend groups, layer routes, adapters, workspace pools, and storage generation | Select backends, reserve resources, bind layers, and translate model inference state into execution inputs |
| `AttentionBackend` | An immutable `LayerSpec` and explicitly injected runtime dependencies | Plan and compute for one family, provider, and phase |
| `LayerAttention` | Layer ID, group ID, backend, and resource views | Route layer computation through the session; borrow State storage without owning its lifetime |
| `AttentionSession` | The current `AttentionBatch` and each group's `AttentionContext` | Validate ownership, phase, and capacity; manage workspace leases and output handling during execution |
| `CapturedGraph` | Captured decode inference states, outputs, workspaces, and provider hooks | Keep captured addresses valid and coordinate replay inputs, auxiliary work, and completion events |

Three data types describe distinct facts:

- `LayerSpec`: static requirements, including algorithm, phase, head dimensions, features, `CacheLayout`, and immutable family configuration. The backend holds it; layer bindings do not keep a second copy.
- `AttentionBatch`: request indices, lengths, positions, and other inputs for this execution. Tensors are borrowed from `InferStateInfo` / `ReqManager`.
- `AttentionContext`: metadata, capacity, and workspace references prepared by the backend for an execution or a graph capture instance. Contexts within a session share one batch.

`InferStateInfo` holds the session in one direction. The session receives only the batch, groups, capacity, and a workspace supplier; it does not reference the model or `InferStateInfo`. States return views defined by the storage module; adapters combine these views with weights into computation resources. ReplaySSM exposes tensors and configuration without giving the backend a State object with lifecycle operations. Backends can update KV/Conv/SSM through these views, while storage allocation, checkpointing, and request reclamation remain with their existing owners.

## 2. Initialization: describe, select, bind

```text
Layer describe_attention(phase) + State layout
                    |
              Family adapter
                    |
         (prefill_spec, decode_spec)
                    |
    ModelAttention groups by the full spec pair
                    |
    Select a prefill and decode backend per group
                    |
    Reserve workspace -> Allocate State storage
                    |
    Adapter borrows layer resources -> LayerAttention
```

Layers share backends and batch contexts only when their complete `(prefill_spec, decode_spec)` pairs match. Weights, logical layer IDs, and storage views remain in each layer's `LayerAttention`. Both phases of a layer must use the same physical `CacheLayout`, so decode can read what prefill writes.

All backends are constructed as `Backend(env, spec)`. `BackendEnv` explicitly supplies the device runtime, allocator, TP and MTP settings, and other runtime information, without a model configuration dictionary. Adapters translate GDN/KDA convolution and gate settings, MLA decomposition dimensions, and sparse index parameters into immutable configuration in `LayerSpec`, which participates in grouping. Layout parameters such as quantization group size come from `spec.cache`. Backend classes declare capabilities; the registry lists implementations and default priorities. Selection checks the environment, computation requirements, and serving constraints, then runs the corresponding probes. An explicit choice fails with an error when unusable; only `auto` tries other candidates.

Each layer selects its family adapter once and reuses it for description and binding. Dense and MLA have separate adapters. GDN and KDA share only the Conv/SSM layout description. Model-specific adapters, such as DeepSeekV4's, live in the model directory.

## 3. Execution: the caller enforces shared contracts

The model entry point uses one scope for preparation and cleanup:

```python
with model.attention.session(infer_state):
    output = model._token_forward(infer_state)
```

Ordinary execution follows this order. During decode capture, device metadata construction and computation become part of the graph:

```text
ModelAttention.session
  -> Check storage generation; build batch and capacity
  -> AttentionSession.prepare
      -> Validate batch; acquire workspace leases
      -> backend.allocate_context
      -> backend.plan_host                  # Planning before graph submission; may launch device work
  -> Model forward
      -> session.build_device_metadata      # Device metadata
      -> LayerAttention.forward per layer
          -> Session checks context ownership
          -> backend.forward                # Phase computation
          -> Session handles out
      -> model.attention.finish_decode      # Advance ReplaySSM cursors after all layers succeed
  -> Exit scope; finish leases
```

The backend execution interface consists of `allocate_context`, `plan_host`, `build_device_metadata`, and `forward`. Resource budgets and capture hooks are supplied as needed. Each instance implements either prefill or decode. Constructors and helpers specific to a phase live in that phase's class; shared classes retain only common definitions.

Layer execution and metadata construction require an active session scope and are rejected after it exits. Captured contexts can remain available for graph replay. Numerical probes use the same scoped entry point. The session checks context ownership and phase, and requires the backend to retain the requested capacity, workspace, and capture mode. `batch_size` and `token_num` must match the capacity contract; declared maximum sequence lengths must stay within its bounds. The model execution path handles padding and graph bucket selection.

When `out` is supplied, the session checks dtype, device, and the result's storage range. A backend may write directly into `out`. A result sharing storage must lie within a contiguous `out`, or be a prefix slice of a strided `out` with the same starting offset and strides; padding between rows is not part of the destination. If it returns an independent result, the session copies that result into `out` while preserving auxiliary outputs.

Workspaces are isolated by `(group_id, microbatch_index)`. Executions within a slot must be serialized; leases order completion events but do not provide a concurrency lock. If preparation, planning, or forward raises, `ExitStack` still finishes every acquired lease. Finishing a lease records a completion event for subsequent reuse; it does not free pooled tensors or imply that the GPU has finished executing.

## 4. CUDA Graph: stable addresses and validation before writes

During decode capture, all groups' contexts share one captured batch. Sequence capacity explicitly uses `graph_max_len_in_batch`, while the host planner receives the current actual length separately. This avoids inflating capture capacity to the serving maximum. Captured workspaces retain their tensor references and freeze their buffer bindings; providers remain responsible for internal address stability in shared objects; later pool growth does not replace addresses used by existing graphs.

Each `CapturedGraph` replay proceeds as follows:

1. Check storage generations and every microbatch lane's model role, backend groups, capacity, and input tensor presence, shapes, dtypes, devices, and strides for views at the same address. Build copy plans without writing inputs.
2. After all lanes pass validation, wait for the provider to finish using the previous metadata and acquire workspace leases.
3. Copy new inputs into captured addresses, call each group's `plan_host`, and submit the graph and provider auxiliary work.
4. Include auxiliary completion events when finishing leases, so subsequent reuse waits on all required work.

The hook ordering is covered by fake-provider tests; current providers do not implement auxiliary capture hooks. Failure recovery for a provider that submits auxiliary work needs provider-specific validation.

A validation failure in a later lane therefore leaves earlier lanes' inputs unchanged. This guarantee covers validation before writes; it does not provide transactional rollback after a copy or provider execution fails.

Each inference-state class declares `graph_input_names` for tensors outside `AttentionBatch`, including RoPE data and draft hidden inputs. Model integrations must extend this inventory for new replay inputs. Tensors produced inside the graph, such as DeepSeek-V4 C4 indices, are excluded and recomputed by captured nodes.

The request mapping table is shared storage and must retain its address during replay. A State storage generation change requires layer rebinding and graph recapture. Graph destruction waits for relevant device work, closes provider hooks, and then releases captured references.

Prefill follows a separate path: `PrefillGraphExecution` holds the segmented graph's inputs, outputs, and collector. Attention runs through callbacks between segments using the current batch's session. `build_device_metadata` runs before the segments, outside the decode capture flow described above. Validation of all lanes before writing also applies specifically to the `CapturedGraph` path.

GLM generates request-expansion metadata in its model inference state and passes `SparseQueryMetadata` to the indexer, without reading provider context internals. Decode input copying includes these tensors; prefill callbacks use the current batch's data. DeepSeek3.2 still reads ragged-index metadata from its sparse provider context; replacing that provider requires preserving this integration until it receives an explicit family contract.

The model advances each ReplaySSM cursor once after all participating layers return successfully. Ordinary and overlap decode use the same operation, which is included in graph capture. A layer failure prevents this step; earlier state writes are not transactionally rolled back.

## 5. Simplification decisions

| Previous complexity | Current design |
| --- | --- |
| Two base classes, `AttentionBackend` / `BaseAttentionBackend`, and constructor forwarding | One base class; remove `create()` and `configure_spec()` |
| Duplicate copies of spec, causal mode, dynamic verification, and other settings | Read from the backend's `spec`, avoiding copies that must stay synchronized |
| A single-method BatchBuilder class and the indirect `ContextSource` interface | Construct the batch on the model side and pass explicit data to the session |
| Separate prepare / forward / finish calls that callers must pair | One session scope handles normal and exceptional cleanup |
| Prefill and decode helpers and dependencies mixed in shared classes | Move them to their phase implementations; for example, GDN decode construction does not load a prefill kernel |
| GDN inheriting Dense and KDA inheriting GDN | Organize adapters around the Conv/SSM layout they actually share |

Family adapters remain because model fields and State resources differ. Sessions remain because all backends need the same lifecycle and validation rules. States already expose layouts and resource views, so no parallel storage-description interface is added.

## 6. Extension points and verification

For a new model, describe its layer requirements and reuse a family adapter where possible. Add a model-specific adapter only when resource binding differs. For a new provider, implement the relevant phase backend, declare its capabilities, and register it. For a new storage encoding, extend the State layout and resources and use `CacheLayout` for compatibility checks. See [Adding New Models](../models/add_new_model.md) for integration details.

Source entry points:

| Area | Files |
| --- | --- |
| Grouping, selection, and model adaptation | `lightllm/common/basemodel/attention.py`, `lightllm/common/basemodel/attention_adapters.py` |
| Static requirements and backend interface | `lightllm/common/attention/api/spec.py`, `lightllm/common/attention/api/backend.py` |
| Batch execution and workspaces | `lightllm/common/attention/session.py`, `lightllm/common/attention/api/workspace.py` |
| Graph replay | `lightllm/common/basemodel/graph.py`, `lightllm/common/basemodel/prefill_cuda_graph.py`, `lightllm/common/basemodel/infer_struct.py` |

Regression coverage focuses on rejecting incorrect context ownership and capacity, lease cleanup on failure, output handling, validation of all lanes before writes, captured addresses, and storage generations. Interface contract and workspace tests live in `unit_tests/common/attention/`; model binding and backend numerical tests live in `unit_tests/common/basemodel/attention/`. Model validation must also cover prefill/decode, graphs, MTP, and state restoration. Throughput and cross-host PD claims require their own runtime validation.
