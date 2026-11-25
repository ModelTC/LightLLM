Logging guide
=============

Defaults and configuration
--------------------------

LightLLM uses Python logging with INFO service logs on stdout. Module loggers inherit
from the central ``lightllm`` logger. Text records use ``HH:mm:ss.SSS LEVEL [component] message``. P/D roles are included
in the component, such as ``[P/infer]`` or ``[D/http]``. Applicable known ranks are
shown only for multi-card, multi-DP or multi-node topologies, e.g. ``[P/infer tp=1]``.
Unknown ranks, hostname, PID and full module names are omitted from default text.
DEBUG, ERROR/CRITICAL and traceback-bearing records retain source file/line.
JSON retains full date/timezone, host, PID, module/component, topology and source. Each worker initializes its own process context.

No logging flags are required for a normal startup:

.. code-block:: bash

   python -m lightllm.server.api_server --model_dir /path/to/model

With the default configuration, service logs use INFO text output on stdout.
Engine statistics are reported every 10 seconds while active; enabling MTP also
enables its periodic statistics without another logging option. Successful request
summaries are sampled at 1%, failures are not sampled, and HTTP access logs are off.
PD nodes keep their model and topology arguments; they need no additional logging
flags either.

Customize logging with one parameter, using inline JSON:

.. code-block:: bash

   python -m lightllm.server.api_server --model_dir /path/to/model --log_config '{"format":"json","request_sample_rate":1}'

Or use a JSON file for a reusable configuration:

.. code-block:: bash

   python -m lightllm.server.api_server --model_dir /path/to/model --log_config test/logging.json

The example file ``test/logging.json`` enables JSON output and records every successful
request summary:

.. code-block:: json

   {
     "format": "json",
     "request_sample_rate": 1
   }

``--log_config`` accepts a JSON object inline or
from a file. Only include settings you want to change. Both forms use the same
validation and override the corresponding environment variables; omitted fields inherit
the environment, then defaults. Paths are relative
to the launcher's working directory. Unknown fields, invalid types and invalid values
fail startup before workers are launched.

The launcher propagates resolved settings through environment variables to spawned
workers and the HTTP subprocess. Child processes do not need to read a configuration file.
Existing ``LIGHTLLM_LOG_*`` environment settings remain supported.

.. list-table::
   :header-rows: 1

   * - JSON field
     - Environment variable
     - Default
   * - ``level``
     - ``LIGHTLLM_LOG_LEVEL``
     - ``info``
   * - ``format``
     - ``LIGHTLLM_LOG_FORMAT``
     - ``text``; alternatively ``json``
   * - ``module_levels``
     - ``LIGHTLLM_LOG_MODULE_LEVELS``
     - ``{}``; JSON mapping of module prefixes to levels
   * - ``request_sample_rate``
     - ``LIGHTLLM_LOG_REQUEST_SAMPLE_RATE``
     - ``0.01``; 1% of successful completions; failures are never sampled
   * - ``stats_interval``
     - ``LIGHTLLM_LOG_STATS_INTERVAL``
     - ``10`` seconds; must be positive
   * - ``http_level``
     - ``LIGHTLLM_LOG_HTTP_LEVEL``
     - ``warning``; Hypercorn error logger
   * - ``http_access``
     - ``LIGHTLLM_LOG_HTTP_ACCESS``
     - ``false``; a JSON boolean (environment: ``0`` or ``1``)
   * - ``http_exclude_paths``
     - ``LIGHTLLM_LOG_HTTP_EXCLUDE_PATHS``
     - ``["/health", "/healthz", "/readiness", "/liveness", "/metrics"]``; ``[]`` disables exclusions (environment: comma-separated paths)
   * - ``dir``
     - ``LIGHTLLM_LOG_DIR``
     - Unset; optional per-PID file output in addition to stdout
   * - ``max_bytes``
     - ``LIGHTLLM_LOG_MAX_BYTES``
     - ``104857600``; positive integer file rotation threshold
   * - ``backup_count``
     - ``LIGHTLLM_LOG_BACKUP_COUNT``
     - ``3``; positive integer number of rotation backups per PID

To enable DEBUG only for scheduling, set ``module_levels`` in the same file:

.. code-block:: json

   {
     "module_levels": {"lightllm.server.router": "DEBUG"}
   }

Sampling is deterministic by request ID and does not alter application RNG state.
Request summaries never enable prompt, output text, sampling parameter, or tensor dumps.
Set ``lightllm.server.httpserver.manager`` to DEBUG to see sampled request stage
timings and encoding lengths. For bounded PD KV transfer summaries, set
``lightllm.server.router.model_infer.mode_backend.pd`` to DEBUG.

Categories and severity
-----------------------

* Service logs: lifecycle, selected configuration/backends, state changes and failures.
* ``lightllm.stats``: node-local router snapshots of running/waiting/paused requests
  and KV utilization per local DP. All snapshots respect the configured interval;
  idle is reported only after a full interval without work and is not repeatedly logged. Mirrored multi-node TP slave routers do
  not duplicate snapshots. Queue and KV values are instantaneous; throughput uses deltas of engine progress
  counters over the reporting window. Use Prometheus for historical throughput and latency analysis.
* ``lightllm.requests``: completion metadata, including request ID, token counts,
  TTFT, elapsed time and cache/MTP statistics. Normal/P/D and PD master use the same
  base fields. P/D summaries describe processing within that role; segmented PD
  requests retain segment IDs and are not whole-client-request aggregates.
* ``lightllm.access``: HTTP method, path, status and duration, without queries,
  authentication headers or bodies. Application health/metrics paths are excluded
  by default; set ``LIGHTLLM_LOG_HTTP_EXCLUDE_PATHS`` to an empty string to include them.

Enabled request/access categories use INFO independently of service verbosity; explicit
module-level overrides can further filter them. Application access records come from
LightLLM's middleware rather than a second Hypercorn access logger. The configuration
server uses Hypercorn access; application path exclusions do not apply there. Custom
Hypercorn configuration can add its own outputs, so avoid duplicate configuration.
Python warnings use the central logger. Native library stdout/stderr and Hypercorn logs are not forcibly converted to LightLLM JSON.

Use DEBUG for request/batch/transfer internals and allocation details; INFO for startup,
final choices, readiness, periodic summaries and meaningful state transitions; WARNING
for unexpected recovery, retries or fallback; ERROR for an operation's final failure;
and CRITICAL when a process or service cannot continue reliably. Normal client
cancellation and idempotent cleanup belong at DEBUG. ``logger.exception`` emits ERROR
with a traceback, not a separate severity. Record a failure's traceback at its handling
boundary, avoiding adjacent error/exception messages and repeated stack traces on rethrow.

Shared model, backend and CUDA graph configuration is INFO on the TP group's leader
and DEBUG on peers. Never apply this filtering to worker-specific warnings/errors.
New events should use stable English messages and an ``event`` field, plus applicable
request/batch/transfer IDs. Guard expensive DEBUG arguments with ``isEnabledFor``;
parameterized formatting alone does not defer argument evaluation. Startup summaries
allowlist operational settings instead of dumping potentially sensitive arguments.

JSON contains one object per line with structured context and an ``exception`` field
for tracebacks. Text mode prefixes every physical message/traceback line.

Migration
---------

The default level remains INFO; detailed runtime diagnostics now use DEBUG. Completion summaries are enabled by default; HTTP access is opt-in.
File logging writes only ``process.<PID>.log`` and numbered rotation backups, without
creating ``default.log``. Forked workers switch to their own PID files. Backup limits
apply per PID; deployment tooling must expire historical PID files. Container deployments
should normally collect and archive stdout externally.

Correlation and non-interactive output
--------------------------------------

The HTTP entrypoint preserves ``X-Request-ID`` or generates one, and returns it in
the response header. Client IDs must contain at most 128 visible ASCII characters;
invalid IDs are replaced. ``http_request_id`` connects access and completion logs,
while ``request_id`` remains the internal inference ID. Correlation works even with
access logging disabled and is isolated across concurrent and streaming requests.
Source files and line numbers identify business callsites rather than logging helpers.

Weight loading and warmup replace progress bars with start/completion events on
non-interactive streams. JSON output and levels above INFO also disable progress bars.

P/D forwarding carries ``http_request_id`` in optional metadata; legacy messages
remain accepted. Summaries omit zero cache fields and include MTP fields only when
MTP is enabled. ``timing_scope`` identifies ``end_to_end``, ``prefill`` or ``decode``
measurement boundaries; node timings are not client end-to-end timings. Backend
probe warnings contain a short reason and disable action; full diagnostics use DEBUG.

DEBUG volume bounds
-------------------

Normal DEBUG logs cover request phases, scheduling decisions/reasons and cache changes.
They do not dump prompts, generated text, token arrays, grammars or memory descriptors.
Diagnostic events allow at most 16 fields, strings up to 256 characters and sequences
summarized by count and 8 items; nested objects retain only type/count information.

Successful KV transfers aggregate page counts, physical bytes, KV tokens and summed
page-task durations per sender/receiver using ``stats_interval`` in the logging config (default 10 s).
The first completion logs immediately; subsequent updates are rate limited and the
final partial window flushes on idle. Empty windows do not log. Request ID samples
are capped at 8. ``task_duration_ms_total`` sums page-task creation-to-completion
times, not request end-to-end latency. Receiver GPU copy times and explicitly enabled
NIXL telemetry remain aggregated. Forced termination may lose the pending window.

Shared TP cache and pause/resume events use the group leader. Rank-specific errors
remain visible. Recurring PD connection failures keep the first traceback, followed
by counts every 10 s for the same callsite, exception type and short reason. Different
reasons retain independent tracebacks; deduplication state is capped at 128 entries.
Request and transfer errors do not use this connection-failure deduplication policy.

Engine throughput windows
-------------------------

Text groups engine snapshots into throughput, request state and KV occupancy:

.. code-block:: text

   15:54:23.459 INFO [D/router] Engine stats | Throughput: prefill 0.0 tok/s, output 31.6 tok/s | Requests: running 1, waiting 0, paused 0, done 0.50/s | KV: active 5.2%, cached 0.0%

Idle transitions use ``Engine idle``. Single-DP KV occupancy is a percentage;
multiple local DPs use ordered lists, such as ``active [3.0%, 12.5%]``, without
averaging away imbalances. ``cached`` means reusable KV occupancy, not a prefix
cache hit rate. ``done`` counts completed internal requests per second.

JSON retains the ``engine.stats`` / ``engine.idle`` events and exact numeric fields
``prefill_tok_s``, ``output_tok_s`` and ``completed_req_s``, along with
``kv_capacity_tokens_per_dp``, ``window_s`` and raw window counts
``prefill_tokens``, ``output_tokens`` and ``completed_reqs``. Rates use actual
monotonic elapsed time; the default interval remains 10 seconds. In-flight
requests contribute tokens before they finish.

With MTP enabled, windows containing draft verification add a second text line:

.. code-block:: text

   15:54:23.459 INFO [D/router] MTP stats | Draft acceptance: 66.7% | Mean length: 2.33 | Draft tokens: accepted 4, verified 6 | Throughput: accepted 2.0 tok/s, verified draft 3.0 tok/s

Draft acceptance divides accepted draft tokens by the actual number of verified
draft tokens, including dynamic verification's selected widths. Both exclude the
mandatory target token. Mean length is ``1 + accepted draft tokens / request rounds``,
including that target token; rounds without draft verification are excluded.
These are verification results before stop/length clipping, whereas output throughput
counts committed outputs. Already synchronized CPU results feed the counters; logging
adds no GPU synchronization. JSON includes ``mtp_accepted_tokens``,
``mtp_verified_draft_tokens``, ``mtp_rounds``, ``mtp_acceptance_pct``,
``mtp_mean_acceptance_length``, ``mtp_accepted_tok_s`` and ``mtp_verified_draft_tok_s``
in the same engine event. Empty MTP windows use null acceptance/mean values and omit
the extra text line. Disabled MTP omits all these fields.

One progress-publishing TP leader writes each CPU shared counter row; the router
sums local DP rows. Pure multi-node TP still logs only on the primary router.
Prefill counts computed input tokens, excluding GPU/CPU/disk cache restoration
and received PD KV. Actual prompt recomputation after a pause counts again.
Generation counts committed outputs, including accepted MTP tokens, excluding
outputs after a stop condition and simulated termination tokens. Completions
count internal engine requests (choices/segments), with normal stop/length only.

Prefill nodes count the first generated token; decode nodes exclude that received
token and count subsequent local generation. Add P and D generation counts for
PD totals; do not add the master's end-to-end request summaries again. Short
requests completed between snapshots appear in the next window. Stable idle emits
one transition. Counters update after existing inference synchronization without
introducing GPU synchronization.

Service readiness is announced only after the bound HTTP /readiness endpoint succeeds.
INFO reports one throughput, queue and KV usage snapshot per node. Active KV and reusable
cache occupancy are separate. JSON retains raw window counts omitted from compact text.
Successful request summaries default to 1% sampling; P/D stage summaries are DEBUG only.
Failures bypass sampling; cancellations are DEBUG. Terminal reasons are preserved.

For HTTPS with required client certificates, set ``LIGHTLLM_READINESS_CLIENT_CERT``
to a PEM client certificate trusted by Hypercorn and ``LIGHTLLM_READINESS_CLIENT_KEY``
to its unencrypted PEM private key. The key setting can be omitted when the key is
in the certificate file. The probe continues to verify the server certificate;
use a dedicated client identity with client-authentication permission.
