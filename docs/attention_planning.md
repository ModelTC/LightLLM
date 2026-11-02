# Attention planning refactor

Status: implementation and acceptance are in progress. This document does not certify stages 0–6 complete.

| Stage | Implemented scope | Verification and remaining work |
| --- | --- | --- |
| 0 | Fixed source/environment and correctness, capacity and serving measurements | A stable tail-latency baseline on the shared host |
| 1 | Per-layer BF16 GQA/Triton specifications, joint selection and native binding | Covered by local GPU and serving checks |
| 2 | INT4/INT8/FP8 native formats, FA3/FlashInfer, calibration and Graph metadata | Broader quantized model quality and supported configuration coverage |
| 3 | Gemma4 heterogeneous widths and declared KV sharing | Full-model validation when compatible weights are available |
| 4 | Native copy plans, logical manifests and completion/retirement protocols | Cross-node transport; local CPU/DP/PD checks are separate evidence |
| 5 | Typed MLA/sparse/recurrent paths and complete hybrid recovery | Full DeepSeek V4 execution requires a compatible checkpoint/provider |
| 6 | Paired KV writes, shared transfer/page metadata and a native CPU path | Stable performance gate and agreement on second-hardware scope |

## Startup and execution

Models create their state requirements before storage allocation. `prepare_attention_runtime()` constructs candidates,
checks each layer against its persistent format, specializes providers by layer shape, and obtains workspace requirements.
Recurrent, sliding-window and compressor request states initially contain only meta tensors. `initialize_storage()`
budgets their fixed request storage together with native KV, fixed calibration, token bookkeeping, capacity-dependent checkpoint pools,
reserved pages, PD staging and provider workspaces in their CPU/CUDA domains. Shared workspaces count once; concurrent
lanes count separately. Workspaces are allocated after this decision. `bind_model_attention()` then binds native views,
calibration, writers and metadata slots. Batch preparation builds dynamic request metadata; layer execution consumes
its bound resources directly. Startup logs list each dense/MLA layer's resource, representation, storage row, writer,
selected providers, metadata slots and rejected candidates.
Fixed provider buffers follow the same order: FlashInfer page indices, NSA ragged indices, FA3 page tables and
DeepSeek V4 SWA prefill workspace are included before allocation. Compatible FlashInfer layer shapes share page-index
storage, with separate regions for concurrent lanes and separate keys for model roles.

Startup records allocated/reserved peaks for warmup, every decode/prefill Graph bucket and the maximum prefill check.
The final memory plan separates observed runtime cost from the remaining reserve. If automatic capacity exceeds its
budget, startup synchronizes, drops Graphs and bound views, releases unpublished storage/checkpoint pools, reallocates
and captures again. Explicit capacity is never silently reduced. Retries are bounded to three attempts. Single-rank OOM
can use this path; a multi-rank OOM fails rather than rebinding while a peer may still be executing a collective.
Autotune OOM propagates while restoring the layer count, tuning phase and request allocations.
Fixed request storage can remain allocated across retries; subsequent plans account for it through observed free
memory rather than reserving its bytes twice. Compressor HOLD slots retain their invalid gate initialization.
`plan_storage()` previews capacity without allocating native states. The static benchmark uses this preview to
resolve automatic batch/Graph limits before entering the same startup path as serving. The real-model cache validator
also uses shared startup. Standalone benchmarks initialize and clean their own shared-memory service identity.

`AttentionSpec` describes GQA semantics, including sharing, window, ALiBi, sinks and causal behavior. `MlaSpec` describes
projection and latent dimensions separately. `KernelOffer` validates a representation/provider pair. Joint selection
requires both prefill and decode to accept the same representation; an explicit backend choice is not silently changed.
The current model factory supplies the configured representation, while the planner can evaluate multiple alternatives.

`ResourceId(role, producer_layer, kind)` identifies content independently of local storage rows. Main and draft models
share the token index domain but have separate resources. Shared KV consumers cannot perform a second write. Disjoint
logical layer domains may use separate State instances with the same legacy format label. State member names remain a
compatibility interface, rather than the logical identity of an attention operation.

The `legacy` adapter still bridges existing State allocation/transport interfaces. GQA, MLA, sparse, compressed-sparse
and recurrent execution have separate typed inputs. State implementations retain allocation and native-layout
declarations; KV writer factories consume native views and parameters at binding time. Their execution does not query a State or recompute physical layer indices. The old
`write_full_att_kv` method remains a compatibility entry point.

Weight loading also has an immutable `WeightPlan`: checkpoint encoding is separate from the requested weight and
activation representations and the implementation. Existing quantization strings remain accepted. A `quant_type` YAML
entry may instead contain `weight_format: fp8_block128`, `activation_format: fp8_group128` and `implementation: triton`.
The explicit selection is preserved when reading a quantized checkpoint, and unsupported representation/conversion pairs
fail before loading. Native weight allocation, packing and activation quantization remain in the existing providers;
this is not a universal weight-layout conversion interface. `MMWeightTpl` retains its selected provider and
`WeightPack`; `FusedMoeWeight` retains its selected MoE implementation and native expert packs. Both execute directly
against those objects. The refactor reuses these existing binding boundaries instead of creating duplicate weight storage.
Cache fingerprints use effective quantization configuration and provider-declared numerical semantics. The block-FP8
Triton and vLLM implementations share their encoding and activation quantizer and declare the same content identity.
Other providers retain separate identities unless equivalence is explicitly declared; scale rounding remains semantic.
UE8M0 rounding is frozen in each model's `WeightPlan` before allocation. Loading a main/draft configuration does not
mutate the other model's environment; Linear, MoE dispatch and fused activation paths consume the bound setting.

## Batch metadata, Graph and workspace

Each provider supplies its metadata sharing key. FlashInfer shares request/page mappings across compatible layer shapes,
while native scheduling metadata remains specialized. Page counts, indptr and final-page lengths use one Triton kernel,
followed by index repacking. Concurrent lanes own separate fixed buffers and workspaces.
Dynamic index buffers cover actual KV lengths even when a synthetic warmup exceeds the configured context bound.
Within the configured bound they retain a fixed extent, so changed prefix lengths do not change captured buffer shapes.

Providers declare Graph buffer names and validate layout before updating captured buffers. `prepare_attention_capture()`
resets derived schedules after eager warmup so their GPU updates enter capture. Native CPU planning is refreshed explicitly
before replay where needed. Tests change request IDs, page contents, query values and sequence lengths between replays.
The runner and Graph must agree on the number of microbatches. Microbatch overlap requires a DP runner; the single-DP
runner is rejected before loading when either microbatch-overlap flag is set.

## Transport and recovery

`CopyPlan` binds local field mappings and copy callables at startup. PD manifests compare model/weight identity, logical
state/configuration, global wire fields and static KV calibration. Q calibration belongs to the logical computation and is
not part of the persistent KV digest. Different encodings or wire layouts currently fail negotiation; no implicit lossy
conversion is performed.

`TokenPageCopyPlan` binds arbitrary native payload/scale/zero-point fields to page layouts. CPU, DP and PD prepare one
`TransferBatch` of indices for all participating fields and states. Registered encodings do not need to implement those
three transport paths. Window restoration retains its dedicated ring kernels. Metadata caches use weak ownership
so an abandoned manager and its device allocations are not retained by process-global method caches.
Local token copies also bind the native field plans once, including dynamic scales and zero points.
DP peers bind native fields by logical resource and sharing relationship, then map them to each endpoint's physical rows.
Regrouping or reordering GQA/MLA/compressed arenas does not change the resource identity. Contiguous row runs retain one copy plan;
format, shard, KV calibration and complete bounded-state compatibility are checked before any destination write.
Bounded CPU/DP/PD plans bind runtime and checkpoint field views, TP slices and page offsets before executing transfers.
`BatchedStateView` retains the native buffers and accepted cursor; execution selects request/page slots without querying
the State or rebuilding field mappings. CPU bindings are invalidated when their page buffer or layout changes.
Window providers bind their native ring-conversion kernels separately. Bounded State classes retain allocation and
snapshot semantics and inherit compatibility transfer entry points. Raw DP ring copies require compatible geometry.
Packed group64/E8M0 MLA now uses the common native-field CPU/DP/PD paths as well. Its external pages contain the payload;
local/DP copies preserve complete physical rows, including padding. Regrouping tests include plain and packed MLA.
Compressed KV declares its indivisible physical block size in the native field contract. CPU/DP copies require whole
blocks; PD completes the final physical block inside the copy kernel without uploading a second index tensor. Both
compressed ratios and optional indexer bytes use the common plans, including different physical layer groupings.
CPU page binding accepts contiguous multidimensional page views and checks byte regions and host page indices before
submitting writes. All participating native plans validate batch constraints before any field is copied.
Full-attention State classes now declare their native fields and inherit the same local/CPU/DP/PD copy implementations.
PD wire version 2 orders full-attention and bounded resources by logical producer, independently of each endpoint's arena grouping.
Layer mappings place split or reordered local rows into that wire layout. Explicit format padding remains intact, and
static KV calibration digests belong to individual logical resources. CPU cache page layouts remain unchanged. CPU manifests retain zero-overlap compressor semantics without inventing snapshot bytes. Missing required layouts remain errors.
Transport descriptors cache fixed-size SHA256 fingerprints of the model and state manifests, keeping per-request PD
parameters below the existing 8 KiB slot limit independently of layer count. Full manifests remain available locally.

Transfer completion separates source reuse from destination readiness. Cancelled generations cannot publish completion,
and memory remains retired until device access ends. Pack/unpack failures record completion of already-submitted work;
uncertain device failures quarantine resources until the execution context is destroyed. Host page registration retains
storage until CUDA unregister completes.

Disk timeouts distinguish a successful LightMem abort (which guarantees no further cache-page access) from an abort or
completion-query failure. Unknown completion retains the task and its page leases, removes quarantined pages from lookup,
disables further disk submissions and recomputes the request prefix. Late completion cannot republish a quarantined page.
CPU pages have one writer: a second offload holds a reference but skips an existing `LOADING` page and cannot publish it.
Confirmed cancellation invalidates the pending hash immediately and waits for follower references before reuse. Unknown
completion leaves the page outside the LRU even when ordinary references drain. Block-aligned disk reads use unpublished
scratch pages for any overlap with a published CPU prefix, so a partial or late read cannot overwrite a concurrent reader.
Disk directories are checked against a model/weight/representation/calibration manifest before LightMem opens them.
Unidentified or incompatible existing files are preserved and rejected. The LightMem offline configuration used here
does not restore its index across process restarts; this manifest does not add cross-restart cache recovery.
Lookup failures before submission release their leases and fall back to prefix computation. Failures after submission
retain uncertain accesses; a completed read with an unavailable result can safely recycle its unpublished destinations.

GDN/KDA providers specialize to each declared recurrent state domain, with separate metadata slots and accepted
MTP cursor bindings. GDN supports different head dimensions across domains; KDA validates its decode kernel
requirement of 128 dimensions at binding time. Prefill Graph callbacks use the layer's metadata slot.

Hybrid recovery operates over KV plus bounded states. GDN/KDA checkpoint copies select the accepted speculative cursor
on the GPU. Their common Conv/SSM implementation also selects this cursor for DP and PD exports, including different
source/destination speculative widths; restoration normalizes the destination cursor to zero. Sliding-window and
compressor snapshots preserve the logical restore position. Restoring KV alone is not a
valid hybrid recovery point.

## Evidence and outstanding acceptance

The implementation baseline is `1d36f7a64`, including the GQA layout identity fix, applied over task starting commit
`9719c56c4`. Tests run on local H200 GPUs 4–7; GPUs 0–3 host unrelated workloads. CUDA is available outside the restricted
execution sandbox. The isolated validation environment uses Torch 2.11.0+cu130, Triton 3.6, SGL kernel 0.4.5,
FlashInfer 0.6.18, FlashMLA `5dceadbd9eabc9acfc547ef18ec8f7b14b980682` and DeepGEMM
`891d57b4db1071624b5c8fa0d1e51cb317fa709f`.

Evidence collected so far, with separate scopes:

- Qwen3.5-0.8B, BF16, Triton GQA/GDN, 4096 KV tokens: 10 serial greedy comparisons against the fixed baseline have
  identical complete results, including output tokens and log probabilities. This is not a task-quality evaluation.
- At 32768 KV tokens, the canonical 5-shot GSM8K evaluation over the remaining 1314 questions scored 444/1314 on the
  baseline and 445/1314 on the refactor (one invalid answer each). Extracted answers matched on 1303 questions.
- FP8 per-tensor KV + FlashInfer scored 443/1314 versus 445/1314 (two versus three invalid answers); 1230 extracted
  answers matched. A separate ten-pair serial comparison had identical tokens and log probabilities. Both used the same
  native KV calibration collected from startup warmup and the five reserved examples, not the evaluation questions.
  This small calibration sample is not a production quantization-quality claim. Qwen3.5 INT4 startup is unsupported in
  both the fixed baseline and refactor; it is not recorded as a passing full-model test.
- Real H200 numerical/Graph tests cover GQA, MLA, FP8 sparse and compressed attention. Targeted page metadata,
  identity and MLA tests passed 85 cases; other suites are being rerun after fixture/interface updates.
- The latest combined attention/native-field/Graph/transport/quantization/cache/startup regression passed
  **2883 cases, with 24 skipped**, after deferred allocation, static-entrypoint integration and page-index bounds checks.
  The final submission run took 262 seconds and included fixed Graph metadata extents when prefix lengths change. It covers real
  two-GPU NCCL/NIXL, per-model quantization, hybrid state restoration and startup replanning. Twenty skips require the
  optional vLLM top-k extension; four represent unsupported CPU-index/CUDA-page combinations. CUDA-index/host-page
  transfers passed. Lint and whitespace checks also passed.
  Earlier failures and fixes remain in the evidence file, including an obsolete transfer fixture that accessed
  unregistered CPU memory and a GLM FP8 constructor that inferred lane count from already allocated buffers.
  The latter now declares lane count independently of allocation. No numerical tolerance was relaxed.
- A fixed-batch comparison used real prefill and eight teacher-forced Graph decode steps, batch sizes 1/4, contexts
  128/2048 and two request/token mappings. Both BF16/Triton and FP8/FlashInfer matched the fixed baseline bit for bit
  across 180 complete vocabulary-logit rows per precision. This controls batch composition; it does not establish
  the cause of every earlier online FP8 difference or certify task quality. Its benchmark timing fields are placeholders
  and must not be used for performance claims.
  The initial FP8 baseline failed with an 8192-token warmup and 4096-token model context bound. A baseline with the
  larger context bound passed. The candidate page-index fix also passed the original warmup configuration with identical
  logits, including a final rerun after preserving the fixed Graph buffer extent. Both the failing run and corrected
  configuration are retained.
- The real-model cache validator restored a 128-token Qwen prefix through GPU pages, CPU pages, local DP copies and
  PD buffers. Each path matched all four subsequent Graph decode logits exactly and reclaimed its allocations.
  The standalone automatic-batch FP8 benchmark initialized a 4096-token cache and Graph buckets 1–4 successfully.
  This was entrypoint validation, not a latency comparison; Python reported one semaphore at shutdown.
- Six isolated real IPC tests passed without the previous shutdown warning after the consumer explicitly released imported
  resources before exiting. The weak-cache lifetime tests independently verify that metadata does not keep storage alive.
- A custom affine INT4 format with payload, scale and zero-point fields was encoded, copied through CPU/DP/PD and decoded
  without adding transport methods or scheduler branches. Its incompatible packing is rejected by the manifest.
- The 12-case native-copy benchmark preserved every byte. Bound submission took roughly 30–40% of legacy eager time in
  small/multiple-State cases, and about 98% for the largest host transfer, where bandwidth dominated. These are local
  transfer measurements, not end-to-end serving improvements.
- The weight quantization/manifest group passed 234 cases with its required device environment, including loading and executing
  the same FP8 GEMM through the legacy and structured selections with identical packed bytes, scales and outputs.
- The paired BF16 KV writer preserves bits and reduces launch count from two to one. Its microbenchmark shows a local
  improvement; this is not evidence of an equivalent service throughput gain.
- An interrupted broad regression run had 2486 passed, 47 skipped and 32 failures. Seven recurrent equivalence failures
  reproduced identically on the fixed baseline under Torch 2.11. The tolerances were not loosened. Remaining failures are
  tracked through targeted reruns; this interrupted run is not a passing suite.
- A first AIPerf scan at 128/64 tokens, concurrency 1/4/8, exposed a throughput difference at concurrency 4. Two controlled
  alternating repeats (240 requests, 36 warmups) measured baseline 1339/1318 and refactor 1349/1341 output tokens/s.
  The initial drop did not repeat, but tails remained noisy.
- A later same-H200 ABBA comparison at 32768 KV tokens measured short-context baseline 1251/1206 versus candidate
  1245/1242 tokens/s, and long-context baseline 1295/1275 versus candidate 1275/1260 tokens/s. These are two rounds
  per version, with 240 requests at 128/64 tokens and concurrency 4, or 120 requests at 4096/128 and concurrency 8.
  Short-run request p99 ranged from 235–391 ms on the baseline and 237–276 ms on the candidate. A formatter process
  pool overlapped one candidate round, and shared-host CPU activity was not controlled. This is provisional evidence,
  insufficient to certify a 1–2% regression threshold. Actual token-length distributions and all percentile values are
  preserved in [the evidence file](../benchmarks/evidence/attention_h200.json).
- A subsequent same-H200 ABBA run fixed server/client CPU affinity on NUMA node 1 and used 512 short / 384 long
  requests per round, each with 72 warmups. Short throughput was baseline 1268/1275 versus candidate 1346/1336
  tokens/s (mean +5.46%); long was 1311/1315 versus 1328/1315 (+0.62%). Mean p99 TTFT, per-request TPOT and
  request latency did not regress in either workload. CPUs remained shared with other host workloads; two rounds
  per version establish this workload result, not a universal performance guarantee. Full distributions are retained.
- Under a common 8 GiB device budget, all four ABBA rounds selected 316630 KV tokens and stayed below the limit.
  Short SLO goodput (TTFT 150 ms, TPOT 3 ms, 15 requests/s) was baseline 8.47/7.95 versus candidate 8.30/8.47
  requests/s, mean +2.17%. Long (TTFT 1000 ms, TPOT 5 ms, 7 requests/s) was 6.88/6.93 versus 6.91/6.80,
  mean -0.78%. Long request p99 increased 6.85% on average across two rounds; this remains an unresolved tail
  result, not a signed performance gate. No request errors occurred. The test imposed the same allocator ceiling
  after subtracting measured external allocations and a 512 MiB margin, and sampled device usage every 200 ms.
  These timings precede the subsequent writer and recurrent binding changes.
- With the latest writer/recurrent bindings, a reversed-order BAAB run used one client worker, 512 measured requests
  and 96 warmups per workload/round under the same 8 GiB budget. All rounds again selected 316630 tokens, with a
  sampled peak of 7253 MiB and zero request errors. Short SLO goodput was baseline 8.51/8.37 versus candidate
  8.34/8.42 requests/s (mean -0.70%); long was 6.95/6.77 versus 6.83/6.86 (-0.27%). Long request p99 was
  baseline 731/865 versus candidate 839/828 ms. Its mean increased 4.47%, while the baseline's own two rounds
  differed by 134 ms. This leaves the tail result unresolved; it does not establish either a causal regression or
  a passing performance gate. Percentile comparisons here use means of round percentiles, not pooled percentiles.
- After deferred request-state allocation, a fresh same-8-GiB capacity/output check selected 316630 tokens on the
  fixed baseline and 316144 on the candidate (-486, or -0.1535%). Sampled device peaks were 7275 and 7145 MiB,
  respectively, and all 12 complete token comparisons matched. The capacity difference equals the 5,971,968 bytes
  explicitly reserved for convolution state in this configuration; allocator-segment reuse is a possible explanation,
  not established by this run. This check did not measure a latency/SLO performance gate, and the earlier timings
  precede these allocation changes.
- In these AIPerf runs, 128 and 4096 are nominal client input lengths. Server-reported chat prompt lengths were
  137–140 and 4105–4108 tokens; completion usage was exactly 64 or 128 tokens. Both versions received the same
  generated inputs. Client-retokenized output counts and server usage are retained separately.
- Qwen3.5 CPU/disk recovery with 2048 KV tokens, 0.5 GiB CPU cache and 1 GiB disk cache produced identical token
  sequences across versions for all 28 requests. All 16 repeats matched their cold generations. Both versions recorded
  five CPU hits and two disk hits. This exercises complete KV/recurrent prefix restoration; failure paths have separate
  injection tests, and concurrent serving restoration still needs broader acceptance.
- Repeating that serving check with the directory manifest produced 28/28 cross-version token matches and 16/16
  cold/warm matches per version. Both had five CPU hits; baseline had two disk hits and candidate had one. Hit counts
  are observed outcomes of asynchronous offloading, not a guarantee from enabling cache flags.
  On the latest deferred-allocation version, a fresh directory again produced 28/28 cross-version matches and 16/16
  cold/warm matches, with five actual CPU hits, two disk hits and no server errors.
- Qwen3.5 DP=2 with shared KV and decode overlap produced 16/16 token matches against the fixed baseline. Four
  prompts were sent first to DP0 and then to DP1; all second requests hit a 768-token prefix with CPU/disk disabled.
  Four concurrent streaming requests were subsequently cancelled and confirmed aborted by the server; 8/8 subsequent
  generations matched the baseline. After native bounded field binding, the same scenario again passed 16/16 comparisons and 8/8 after four confirmed
  cancellations, with no server errors. This covers a small serving scenario, not all cancellation schedules or SLOs.
- The same DP/overlap setup with native Qwen3.5 MTP (`eagle_with_att`, three draft steps) passed 24/24 token
  comparisons against the matching baseline configuration, including cross-DP prefixes, concurrency and recovery after
  four confirmed stream cancellations. Request-level acceptance totals are recorded in the evidence file; they do not
  certify every possible per-step acceptance/rollback length.
  Repeating the same scenario after native writer and per-domain recurrent binding also passed 24/24 token
  comparisons. All four cross-DP requests hit 768-token prefixes; four cancellations were confirmed and eight
  subsequent generations matched. The server logged no errors before its intentional shutdown.
  The deferred-allocation version repeated all 24 comparisons successfully, including the same four cross-DP
  prefix hits and eight generations after four confirmed cancellations, with no server errors.
- Local Qwen3.5 NIXL/UCX PD serving with 8192 KV tokens, 2048-token transfer pages and prompt lengths
  127/129/2048/2051 matched the fixed normal-serving baseline in all 20 cold/repeated/concurrent/after-cancel checks.
  The latest wire/native-copy run recorded 31 KV and 25 recurrent-state transfer completions. All four stream
  cancellations were confirmed by both P and D; subsequent requests completed. This used local CUDA IPC/TCP,
  not cross-node RDMA, and preceded the later common Conv/SSM extraction.
- With the common Conv/SSM implementation, local PD with TP=1 prefill and TP=2 decode also passed all 20 comparisons
  against a saved fixed TP=2 normal-serving baseline. It recorded 31 KV/25 recurrent completions and four confirmed
  cancellations. After bounded resource regrouping, a real run exposed the per-request Manifest size limit; fixed-size
  fingerprints
  resolved it. The rerun again passed 20/20 token comparisons, 31 KV/25 recurrent completion notifications and four
  confirmed cancellations. This does not certify cross-node transport.
- After bounded native-field binding, TP1-to-TP2 PD again passed 20/20 complete token comparisons, 31 KV/25 recurrent
  transfer completions and four confirmed cancellations. This run used a 256-token checkpoint boundary. A preceding
  128-token-boundary run failed allocating a 1,245,708,288-byte pinned SSM tensor before binding. Pure PyTorch reproduced
  that failure on GPUs 4 and 5 without importing LightLLM; an alternative host-registration allocator did not resolve it.
  Increasing the interval reduced each pool allocation while retaining 8192 KV tokens. This changes checkpoint density;
  the underlying pinned-allocation limit remains unresolved. All failure logs are retained in the evidence file.
  The latest deferred-allocation version passed the same 20/20 comparisons with 31 KV/25 recurrent completions
  and four cancellations confirmed on P and both TP ranks of D. P/D logged no errors; the master logged four expected
  ASGI stream-cancellation errors. Its test processes exited, and the unrelated GPU6 baseline remained running.

Qwen3.5-35B-A3B BF16 MoE serving on one H200 also matched the fixed baseline in 20/20 complete token
comparisons: cold/repeated prompts, concurrency and recovery after two confirmed stream cancellations.
Prompt lengths were 17/127/129/513/2049 with 32 output tokens, Graph batch limit 4 and 8192 KV tokens.
Both servers recorded zero errors. This verifies a real MoE weight/attention path, not task quality or performance.

With a test-only remaining budget of 256 MiB CUDA and 2 GiB CPU, real Qwen3.5 MTP startup replanned from 11648 to
2924 KV tokens and recaptured its Graphs. All 12 serial/concurrent generations then matched the saved baseline tokens.
Another run injected one `torch.OutOfMemoryError` after main-model Graph capture, replanned from 11648 to 5996 tokens,
and again matched all 12 baseline generations. The injected exception did not exhaust physical H200 memory.
This validates the startup mechanism, not same-memory-budget service performance. The profiler observes PyTorch
transient peaks and persistent external allocations at stage boundaries; external allocations freed within a stage
still require provider bounds. Transfer staging is bounded by configured pages.

The CPU GQA provider exercises native views and runtime completion without CUDA. A standalone run with CUDA hidden
passed eight FP32/BF16 ragged prefill/decode checks. It repacked shared resources into different arenas and token slots,
preserved all transferred bytes and produced identical outputs after recovery; CUDA remained uninitialized.
This is a minimal real CPU path, not evidence for a second accelerator or full CPU model serving. Cross-node validation, broader quantized task quality,
and a stable performance gate remain open. The same-budget workload
comparison above covers capacity and measured SLO goodput, with its tail regression still unresolved. CPU/disk serving recovery
has the bounded evidence above; it does not certify every cancellation/eviction/overlap interleaving.

The local DeepSeek-V4-Flash checkpoint uses MXFP4 experts. Safetensors 0.4.5 could not read its E8M0 headers;
an isolated 0.7.0 installation reads all 46 shards. The fixed baseline then fails loading packed expert columns into
FP8 weights (2048 versus 4096). Native FP4 execution requires SM100 in the current provider. No full-model V4
comparison is certified on these H200s; compressed/SWA numerical and transport tests are separate evidence.

## Reproducing checks

Run unit tests with the repository-compatible environment and explicit visible devices. Relevant groups are
`unit_tests/common/att_resources`, `unit_tests/common/basemodel/attention`, the top-level basemodel Graph tests,
`unit_tests/common/test_deepseek4_paged_cache.py`, GLM5 tests, and PD mode-backend tests. Run network tests separately
with two visible GPUs and the required NIXL/UCX/OpenSSL runtime.

`python -m benchmarks.attention_plan` compares the paired KV writer against the previous two-kernel path. For serving
measurements keep model, tokenizer, precision, input/output token counts, KV capacity, Graph limits and load fixed;
record actual server token counts and latency percentiles. Compare both equal KV capacity and equal memory budget.
`python -m benchmarks.token_copy_plan` compares the legacy page path against native plans, including index preparation,
single/multiple states, CUDA/host page buffers, partial pages and byte equality.
`CUDA_VISIBLE_DEVICES='' python -m benchmarks.cpu_attention --output /tmp/cpu-attention.json` exercises bound GQA
and native page recovery without CUDA, including changed physical grouping and token placement.

With two ready cache-enabled servers, `python -m benchmarks.cache_recovery --baseline-url http://127.0.0.1:18891
--candidate-url http://127.0.0.1:18881 --tokenizer /path/to/model --output /tmp/cache-results.json` saves full responses
and checks cold/warm and cross-version token equality. Supply `--baseline-log` and `--candidate-log` with debug-level
server logs to record the CPU/disk hit lengths. Use a fresh `--request-tag` to distinguish log entries across runs.
Set GPU/CPU/disk capacity ratios so all desired tiers are actually exercised; merely enabling the flags is insufficient.

`python -m benchmarks.dp_cache_recovery` accepts the same URL/tokenizer/output arguments and requires a candidate with
at least two DP replicas and shared KV. It forces DP0 then DP1 prefix reuse, runs concurrent generations and, with
`--cancel-streams`, closes streams after their first event and verifies subsequent outputs. The JSON preserves prompts,
full responses and partial evidence on failure. Server abort logs must independently confirm cancellation.
