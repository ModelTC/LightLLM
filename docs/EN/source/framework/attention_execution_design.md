# Attention Execution Design

Attention execution separates persistent storage from provider computation. The model runner
owns layer order, padding, collectives and graph buckets. `ModelAttention` selects phase backends
and binds State views to layers; each invocation uses an `AttentionSession`. Request and cache
owners remain responsible for acceptance, checkpoints, transfers and reclamation.

## Ownership

| Owner | Lifetime | Responsibility |
| --- | --- | --- |
| `AttentionStateManager` and concrete State classes | Model composition | Allocate KV, Conv and SSM storage; provide execution views and checkpoint/transfer operations |
| `ReqManager` | Shared main/draft request domain | Own request slots, token mappings, slot generations and MTP verification steps |
| `ModelAttention` | Model instance | Select and group phase backends, bind layer resources, reserve workspaces and run recurrent epilogues |
| `ModelBinding` | One weight/storage binding | Retain weights and State views; invalidate captured users when rebinding |
| `AttentionBackend` | One static spec and phase | Allocate contexts, prepare host metadata, build device metadata and compute attention |
| `AttentionWorkspace` | Model lane or capture | Own named scratch and provider objects; captured workspaces borrow scratch but keep private provider objects |
| `ExecutionScope` | Model instance | Serialize host submission, fence device reuse and retain borrowed inputs through completion |
| `AttentionSession` | One invocation and lane | Bind the batch to contexts, validate layer calls and track possible State mutation |
| `CapturedGraph` / `PrefillGraphExecution` | One graph bucket | Retain captured inputs, outputs, bindings and executable state |

Backend groups use the complete `LayerSpec`, including phase, family, layout, dtype and features.
Equal specs permit provider reuse; they do not permit sharing mutable request state or concurrently
using the same workspace. A `LayerAttention` binds a logical layer, its resources, backend and
family output contract. Providers receive borrowed tensors rather than request lifecycle methods.

The implementation is organized under `lightllm/common/attention/` for provider contracts and
sessions, `lightllm/common/basemodel/` for model orchestration and graphs, and
`lightllm/common/attention_state/` for storage and lifecycle operations.

## Initialization and memory admission

Serving, the static benchmark and the cache-continuation harness call
`initialize_model_runtime(models)` with the main model followed by its drafts. Initialization
finishes before cache publication or worker readiness:

1. Create one request domain, register every model's State, select backends, reserve named scratch
   and load cache calibration.
2. Allocate unpublished bootstrap storage large enough for startup captures and profiling.
3. Bind layers, autotune, initialize padding and capture the configured graph buckets.
4. Keep those graph pools resident while profiling all layers in eager mode. Cases cover cold
   prefill, a long cached query, many cached requests and maximum physical MTP decode rows.
   Configured overlap, retained outputs and sampling probabilities participate in the profile.
5. Drain execution, close graphs, unbind resources and release bootstrap storage.
6. Admit final KV capacity using the measured runtime reserve, fixed allocations, transfer reserve
   and provider-declared payloads that depend on KV capacity. Ranks agree on the smallest capacity.
7. Rebind and recapture at the final addresses, then profile again. Automatic sizing can reduce
   capacity and retry, up to four admission attempts. Explicit capacity fails if it exceeds the budget.

The profile aliases complete read-only history pages to the HOLD page. Every page touched by a
query, including a partially cached first page, has private writable storage. This permits long
context shapes without allocating the full request-by-context KV product during bootstrap.

Prefill scratch reservation uses the token budget, TP padding and a possible HOLD request. Decode
reservation uses physical MTP rows and batch alignment. Capture uses its own bucket envelope.
Sparse FP8 compaction and GLM row maps declare capacity-dependent payloads separately from named
scratch. `RuntimeMemory` records the measured peaks, reserves, attempted capacities and final capacity.

Graph admission is sealed after startup. Missing, closed or invalid buckets select eager execution
before attention preparation; serving does not capture new buckets on demand. Weight replacement
holds the model submission scope, drains execution, retires graphs and rebinds resources.

This is measured memory admission. It does not prove an upper bound for opaque provider allocations,
allocator fragmentation or every data-dependent MoE route. The memory fraction leaves headroom;
new providers and workload envelopes still need runtime qualification. Bootstrap profiling and
recapture add startup cost.

## Model invocation

The public `forward`, `microbatch_overlap_prefill` and `microbatch_overlap_decode` entries acquire
`ExecutionScope` before transferring inputs, padding or preparing model-specific metadata. Competing
host submissions fail before touching those inputs. Sessions, graph operations and both overlap
lanes join the current submission.

A session creates batch contexts and performs host preparation. Each layer callback then enters
`ModelAttention.layer_pass`: it builds device metadata once, runs the layers, and finishes the
recurrent decode epilogue after successful layer execution and overlap hooks. Capture warmup may
invoke this pass several times in one host scope. Prefill has no decode epilogue. The output head
runs after the attention pass and can still fail.

Model layers keep the same call shape:

```python
output = self.decode_attention.forward(state.attention, inputs, out=destination)
```

Each family adapter supplies the logical output shape. Before dispatch, the session checks the
provided destination's extent, dtype, device and overlap with inputs or persistent resources.
After dispatch, it checks the returned view and preserves auxiliary outputs such as recurrent gates.
A provider may fill the destination directly or return a tensor for the session to copy.

Sparse indexers use typed metadata. DeepSeek3.2 receives
`SparseIndexMetadata(ks, ke, lengths, ragged_mem_index)` through
`LayerAttention.sparse_index_metadata(session)`; it does not read a provider's private context.
Model-specific adapters stay with their models.

On submission exit, the scope records completion and retains borrowed inputs. Reuse on a different
stream waits for that completion. Providers must join auxiliary work to the calling stream before
returning. At most two unfinished submissions and two external readers are retained by default;
backpressure waits for the oldest completion. Host return means work was enqueued, not that the
device completed successfully.

## Graph capture and replay

Native decode graphs snapshot inference-state bindings and a private captured-attention record.
Reusing the original session does not replace the captured batch or contexts. Capture workspaces
keep their own mutable provider objects and retain borrowed scratch addresses when other buckets
grow the model pool.

`CaptureManifest` inventories captured values with explicit update policies:

| Policy | Examples | Replay behavior |
| --- | --- | --- |
| `copy` | Request lengths, rotary inputs, draft hidden inputs | Validate and copy into retained destinations |
| `shared` | Request mapping, weights, State views, scratch | Retain owners and preserve addresses until retirement |
| `generated` | Device metadata, DeepSeek-V4 C4 indices, outputs | Produce through the captured schedule |
| `constant` | Phase, role, capacity, static spec | Reject changes |
| `prepare` | Host planner metadata and actual KV length | Refresh before replay without replacing captured tensors |

Models declare extra tensors through `graph_input_names`, `graph_generated_names` and
`graph_shared_names`. Native capture rejects undeclared tensor fields. Binding-owned weights and
State views are inventoried once; replay checks binding validity rather than rescanning the weights.
Capture-local address and layout guards run before replay and after host preparation where needed.

`TensorCopyPlan` compiles destination relationships at capture. Replay validates all lanes before
any input copy: counts, presence, shape, dtype, device, destination identity and aliases. Identical
copies and consistent contiguous prefix views coalesce. Conflicting destination writes and copies
that destroy another source fail before mutation. Unchanged addresses remain in alias validation.

Replay then joins the model submission, copies inputs, prepares host metadata and launches the
executable. A failure after copying starts can invalidate the graph; validation does not provide
rollback of partially submitted GPU work.

Graph outputs are borrowed. Keeping a Python tensor reference does not preserve its values across
the next reuse of its memory pool. Consumers must finish before reuse, copy the needed values, or
register an external-stream completion event with `record_output_use(event)`. The consumer must also
wait for the producing stream. Reader registration orders later reuse across all buckets sharing
the model scope. Graphs sharing a pool outside the runner must share that scope as well.

Segmented prefill retains graph segments, inputs, outputs and its hidden collector in
`PrefillGraphExecution`. Attention callbacks use the current batch's session between segments.
Prefill shares copy-plan validation with decode but keeps this separate schedule. Prefill graph
microbatch overlap is unsupported and rejected at configuration time.

Capture snapshots and restores bounded State before the first live replay, including failed-capture
cleanup after draining. KV writes use the reserved token locations; this is not a full KV rollback.
Graph retirement rejects competing submissions, drains device work, destroys the executable and
releases references. Closing an already closed graph is harmless.

## Acceptance and failure

Attention pass completion, device completion, speculative acceptance and checkpoint publication
are separate milestones. ReplaySSM advances its full verification window after the layer pass;
MTP verification then removes the rejected tail through `commit_bounded_att_state`.

`RequestGenerationStamp` binds a target output to its role, request domain, request indices, MTP
indices, slot generations and verification step. Verification consumes the stamp once under the
model scope. A live request admits one unconsumed target output: another target decode or prefill
for that request fails until verification consumes it. Recycled slots, foreign managers, duplicate
results and delayed steps cannot commit current State. HOLD rows retain their padding semantics.

The possibly-mutating interval starts before model layers, since a layer can write KV before calling
the attention backend. An exception after that point, including an output-head or completion failure,
attaches `ExecutionFailure` and invalidates the shared local State manager. All roles then reject
execution, commit, checkpoint save and local export. There is no flag-reset recovery operation.

Failures before mutation can be corrected and retried. Failures after input copies or provider
preparation invalidate the affected executable. If completion cannot be established, the execution
scope retains borrowed inputs and quarantines itself. An escaped inference-thread failure invokes
the worker's fatal thread hook so the RPC process cannot remain alive with unusable execution state.

Recovery and distributed failure handling belong to the worker and request owners. A valid recovery
point must contain compatible KV and every required bounded State at the same accepted prefix.
Ordinary scope cleanup does not restore that prefix, and this design adds no automatic per-forward
snapshot or transparent retry after partial State mutation.

## Extending and validating the implementation

A provider implements its phase-specific capability checks, resource requirements, context
allocation, host planning, device metadata and computation. Declare capacity-dependent allocations
and host-prepared capture fields explicitly. Family adapters own tensor interpretation and output
contracts; State classes own representation, checkpoints and transfers.

Regression coverage lives in `unit_tests/common/attention/`, `unit_tests/common/basemodel/` and
`unit_tests/common/att_resources/`. Changes to these boundaries need rejection-before-mutation,
stream reuse, capture lifetime, state neutrality and failure coverage. Provider changes also need
numerical comparisons on supported devices. `test/att_resources/validate_model_cache.py` checks
restore followed by main/draft continuation with a real checkpoint.

Passing contract or kernel tests does not establish HTTP serving, cross-host transport, distributed
recovery or performance. Qualify those paths with their actual models, providers and workloads,
and retain the source revision and configuration with the results. For model integration steps,
see [Adding New Models](../models/add_new_model.md).
