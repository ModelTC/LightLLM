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

Explicit CLI values override environment variables. Unspecified values inherit the
environment. Both hyphenated and underscored CLI names are accepted. Configuration
is propagated to spawned workers and the HTTP subprocess through environment variables.

.. list-table::
   :header-rows: 1

   * - CLI
     - Environment variable
     - Default
   * - ``--log-level``
     - ``LIGHTLLM_LOG_LEVEL``
     - ``info``
   * - ``--log-format``
     - ``LIGHTLLM_LOG_FORMAT``
     - ``text``; alternatively ``json``
   * - ``--log-module-levels``
     - ``LIGHTLLM_LOG_MODULE_LEVELS``
     - ``{}``; JSON mapping of module prefixes to levels
   * - ``--log-request-sample-rate``
     - ``LIGHTLLM_LOG_REQUEST_SAMPLE_RATE``
     - ``0.01``; 1% of successful completions; failures are never sampled
   * - ``--log-stats-interval``
     - ``LIGHTLLM_LOG_STATS_INTERVAL``
     - ``10`` seconds; must be positive
   * - ``--log-http-level``
     - ``LIGHTLLM_LOG_HTTP_LEVEL``
     - ``warning``; Hypercorn error logger
   * - ``--enable-http-access-log``
     - ``LIGHTLLM_LOG_HTTP_ACCESS``
     - ``0``; set to ``1`` to enable
   * - Environment only
     - ``LIGHTLLM_LOG_HTTP_EXCLUDE_PATHS``
     - ``/health,/healthz,/readiness,/liveness,/metrics``; excludes these application paths and descendants
   * - Environment only
     - ``LIGHTLLM_LOG_DIR``
     - Unset; optional per-PID file output in addition to stdout
   * - Environment only
     - ``LIGHTLLM_LOG_MAX_BYTES``
     - ``104857600``; file rotation threshold
   * - Environment only
     - ``LIGHTLLM_LOG_BACKUP_COUNT``
     - ``3``; positive number of rotation backups per PID

Enable DEBUG only for scheduling:

.. code-block:: bash

   export LIGHTLLM_LOG_LEVEL=INFO
   export LIGHTLLM_LOG_MODULE_LEVELS='{"lightllm.server.router":"DEBUG"}'

Use JSON and sample 1% of request completion summaries:

.. code-block:: bash

   export LIGHTLLM_LOG_FORMAT=json
   export LIGHTLLM_LOG_REQUEST_SAMPLE_RATE=0.01

Sampling is deterministic by request ID and does not alter application RNG state.
Request summaries never enable prompt, output text, sampling parameter, or tensor dumps.
The existing explicit ``--detail_log`` diagnostic option remains available; its DEBUG
messages also require enabling DEBUG for the relevant module.

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
page-task durations per sender/receiver using ``--log-stats-interval`` (default 10 s).
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

``engine.stats`` / ``engine.idle`` display ``prefill_tok_s``, ``output_tok_s`` and
``completed_req_s``. JSON also retains ``window_s`` and raw window counts
``prefill_tokens``, ``output_tokens`` and ``completed_reqs``. Rates use actual
monotonic elapsed time; the default interval remains 10 seconds. In-flight
requests contribute tokens before they finish.

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
