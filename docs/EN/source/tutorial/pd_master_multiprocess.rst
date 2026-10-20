.. _tutorial/pd_master_multiprocess_en:

Multiprocess PD Master
======================

Purpose and scope
-----------------

Multiprocess mode starts multiple independent HTTP workers on one PD Master host. API parsing,
tokenization, P/D orchestration, token handling, and SSE serialization are distributed across CPU
processes while clients keep using one public port. This mode scales Master-side CPU work and does
not load additional GPU model copies.

Set ``--run_mode pd_master --httpserver_workers N`` with ``N > 1`` to enable it.
``--httpserver_workers 1`` keeps the existing single-process startup path.

Architecture
------------

.. code-block:: text

   Clients
     │ HTTP / SSE
     ▼
   HAProxy (public port)
     ├── /liveness, /readiness, /health: aggregate every worker
     └── other HTTP requests: leastconn forwarding
          │
          ├── PD Master worker 0 (private port and P/D connections)
          ├── PD Master worker 1 (private port and P/D connections)
          └── PD Master worker N-1
                 │
                 ├── RPyC ── request ID service
                 ├── RPyC ── P-node statistics service
                 └── RPyC ── Prompt Cache Tree service

   Prefill / Decode
     1. Read pd_master_multi_ports from /get_server_info on the public port
     2. Use each worker port as its local multiprocess pd_node_id
     3. Connect directly to /pd_register and /kv_move_status on every worker

HAProxy does not proxy P/D WebSockets. ``/pd_register`` and ``/kv_move_status`` return HTTP 426 on
the public port. P/D nodes must be able to reach every port in ``pd_master_multi_ports`` directly.
Each request remains owned by the worker that accepted it, and per-token data never crosses a shared
RPyC service.

Startup flow
------------

The launcher performs these steps:

1. Validate HAProxy and arguments, then establish the shared IPC name for this run.
2. Dynamically allocate ``pd_master_multi_ports``. Each worker uses its listening port as
   ``pd_node_id``.
3. Start the Metric Server, request ID service, P-node statistics service, and Prompt Cache Tree
   service.
4. Start N independent PD Master HTTP workers.
5. Generate the HAProxy configuration from the planned worker addresses and start the public
   endpoint. HAProxy continuously probes worker state.
6. Supervise all critical processes. If any one exits, stop the group and let the deployment system
   restart it.

Each backend process runs one Hypercorn worker. All workers inherit the same start arguments and
service name. The launcher passes only ``worker_index`` in the child environment; the worker uses
that index to read its listening port from the shared port table and connects to the same local
shared services. The IPC name and complete worker port table are not stored in ``StartArgs``.
Multiprocess workers receive ``config_server_host=None`` and ``config_server_port=None``;
they do not run registration loops and the launcher does not start a local config server.

P/D discovery
-------------

Without an explicitly configured external config server, Prefill and Decode first request:

.. code-block:: text

   GET http://<pd_master_ip>:<pd_master_port>/get_server_info

The response contains ``pd_master_multi_ports``. Single-worker mode returns ``[port]``;
single-worker mode uses the returned ``pd_node_id``. Multiprocess mode returns the complete
launcher-planned port list, and P/D uses each worker port as its ``pd_node_id`` to build
``(worker_port, host:worker_port)`` entries directly. Discovery makes only one public request,
needs no local registry, and cannot confuse workers because the public endpoint load-balanced the
mode probe.

Other deployments that explicitly set ``--config_server_host`` and ``--config_server_port`` keep
using the external config server's ``/registered_objects`` protocol. A single-host multiprocess PD
Master rejects those two arguments, so the discovery modes are not mixed.

Request IDs
-----------

Multiprocess mode starts one request ID RPyC service. Each HTTP worker client obtains 1024 group
request IDs at a time and generates IDs locally until its range is exhausted. The server aligns IDs
to ``MAX_BEST_OF = 8`` and allocates the half-open interval ``[start_id, end_id)`` while holding its
lock. IDs are unique across all workers within one service run.

This mode does not need config-server-backed large ranges and does not proxy a multimodal ID
allocator through the public endpoint. IDs are not guaranteed to remain unique across independent
deployments or service restarts.

Shared selection information
----------------------------

Each worker owns its P/D WebSockets, candidate nodes, and request state. Two separate RPyC components
hold only the heuristics that must be shared across workers:

* ``pd_node_stats_client.py`` / ``pd_node_stats_server.py`` manage pending Prefill characters and
  request counts.
* ``prompt_cache_tree_client.py`` / ``prompt_cache_tree_server.py`` manage the sampled prompt prefix
  tree.

The selector reads counter snapshots for its current P-node candidates and selects from that local
candidate set. ``add_dispatched_info`` atomically increments both counters after selection, and
``sub_dispatched_info`` decrements them later. The shared service stores data by ``connection_key``;
it does not maintain a worker node registry or choose candidates.

The cache-aware policy samples the prompt by ``sample_stride`` before calling
``prefix_match_sampled`` or ``insert_sampled``. Hit-rate windows and dynamic thresholds remain local
to each worker. Per-token messages, GPU token load, and request objects remain process-local.

Health and proxy behavior
-------------------------

HAProxy maintains separate probe pools for ``/liveness``, ``/readiness``, and ``/health``. A public
probe returns HTTP 200 only when every worker passes that probe; one failure or timeout returns HTTP
503. ``/healthz`` uses the aggregated ``/health`` state.

Ordinary requests use ``leastconn`` and are not removed from routing by those aggregate probe states.
Automatic retries are disabled for inference requests so a submitted non-idempotent request cannot
be replayed. SSE responses are streamed directly.

Deployment
----------

The entry point uses the native HAProxy bundled in the
`ray-haproxy <https://pypi.org/project/ray-haproxy/>`_ wheel. The dependency is pinned in
``requirements.txt``:

.. code-block:: bash

   pip install -r requirements.txt

.. code-block:: bash

   python -m lightllm.server.api_server \
     --run_mode pd_master \
     --model_dir /path/to/model \
     --host 10.0.0.10 \
     --port 8089 \
     --httpserver_workers 4 \
     --select_p_d_node_strategy cache_aware

``--port`` is the public endpoint. Before starting any worker, the launcher dynamically allocates
the complete internal port list from ``--httpserver_workers``. ``--host`` and every worker port must
be reachable from P/D nodes.

P/D still receives only the public endpoint:

.. code-block:: text

   --pd_master_ip 10.0.0.10 --pd_master_port 8089

Inspect the published topology and aggregate probes with:

.. code-block:: bash

   curl http://10.0.0.10:8089/get_server_info
   curl http://10.0.0.10:8089/liveness
   curl http://10.0.0.10:8089/readiness
   curl http://10.0.0.10:8089/health

End-to-end and performance tests
--------------------------------

The CPU-only end-to-end suite starts the real launcher, HAProxy, HTTP workers, and RPyC services.
Lightweight protocol peers replace GPU-backed P/D nodes. Coverage includes:

* discovery, regular generation, and SSE with one, two, and four workers;
* busy responses, client cancellation, unique request IDs, and shared metrics;
* all-worker liveness/readiness/health aggregation and private WebSocket isolation;
* group cleanup after a worker, shared service, or HAProxy exits;
* concurrent consistency of the Prompt Cache Tree, P-node statistics, and request ID services.

.. code-block:: bash

   PYTHONPATH=. pytest -q \
     unit_tests/server/test_pd_master_multiprocess_e2e.py \
     unit_tests/server/test_pd_master_launcher_integration.py \
     unit_tests/server/test_pd_master_topology_api.py \
     unit_tests/server/test_pd_master_req_ids.py \
     unit_tests/server/test_pd_node_stats.py \
     unit_tests/server/test_prompt_cache_tree_service.py

The high-load comparison defaults to 2,000 requests at concurrency 128 and compares one worker with
up to four workers:

.. code-block:: bash

   LIGHTLLM_RUN_PD_MASTER_PERF=1 \
   PYTHONPATH=. pytest -q -s \
     unit_tests/server/test_pd_master_multiprocess_e2e.py::test_multiprocess_pd_master_high_load_speedup

Tune the load with ``LIGHTLLM_PD_MASTER_PERF_REQUESTS``,
``LIGHTLLM_PD_MASTER_PERF_CONCURRENCY``, and ``LIGHTLLM_PD_MASTER_PERF_WORKERS``. A dedicated
``LIGHTLLM_PD_MASTER_MIN_SPEEDUP`` defaults to ``1.0`` and verifies that multiple workers outperform
one worker. A dedicated performance host can set a higher threshold; set it to ``0`` for reporting
only. Normal CI does not enable this performance case, avoiding failures caused by host load
variation. Results report requests per second, P50/P95 latency, service CPU-core usage, and
throughput speedup.

This benchmark exercises the complete HTTP, PD Master, and P/D WebSocket protocol path without
loading model weights. It measures Master control-plane scaling; GPU inference throughput still
requires a separate benchmark with real P/D deployments.
