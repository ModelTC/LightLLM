# GLM53 optimized image

The image contains `glm53_optim` on `support_ds4`: Replay8, FP8 KV, cache-aware
routing and actual cache measurements, ConfigServer TCPStore startup and GLM
packed-KV transfer fixes. DeepEP is rebuilt for SM90 / CUDA 13 / PyTorch 2.11 /
NVSHMEM 3.6.5 with the RC-QP layout fix from upstream PR 696. No package
installation, source patching, external dependency directory or HAProxy is
needed at startup.

## Start

Mount the same complete model (including MTP and tokenizer files) at `/model`.
Set `CONFIG_SERVER_HOST` to a reachable ConfigServer service DNS name; no P/D
machine IP list is needed. There must be exactly one ConfigServer for this
cluster. Run each command in its own container/process:

| Component | Count | Command | Port |
|---|---:|---|---:|
| ConfigServer | 1 | `glm53-pd config` | 60088 |
| PD Master | 4, one per GPU host/pod | `glm53-pd master` | 60011 each |
| Prefill | 2 hosts, 8 H100 each | `glm53-pd p0` / `glm53-pd p1` | 8019 each |
| Decode | 2 hosts, 8 H100 each | `glm53-pd d0` / `glm53-pd d1` | 8121 each |

For the ConfigServer container, leave `CONFIG_SERVER_HOST` unset so it binds
`0.0.0.0`. For every Master/P/D container, set it to the service DNS name,
for example `lightllm-config`. Expose 60088 and NCCL TCPStore ports 29501/29502
on that service. P/D pods must reach each other and all Masters over the cluster
network, with RDMA device access configured by the cluster runtime. Each Master
uses one HTTP worker; clients can use any Master or an existing Kubernetes
Service covering the four Masters.

```bash
# Common environment in each Master/P/D container:
export CONFIG_SERVER_HOST=lightllm-config
export CONFIG_SERVER_PORT=60088
export MODEL_DIR=/model
export RUN_DIR=/var/log/lightllm

# Execute ONE role, as appropriate for that container:
glm53-pd master
glm53-pd p0
glm53-pd p1
glm53-pd d0
glm53-pd d1
```

Start ConfigServer first, then the four Masters and the P/D groups. P0/P1 form
one TP16/DP16 group; D0/D1 form a second. Role names fill `node_rank=0/1` within
each group. Node discovery does not assign distributed ranks.

Logs go to stdout and `$RUN_DIR/logs/ROLE.log`. Mount `$RUN_DIR` to persistent
writable **local storage** on each GPU host; its disk cache defaults to
`$RUN_DIR/disk_cache/ROLE`. CPU cache needs 1024 GiB per P host and disk cache
needs up to 10240 GiB. RDMA, sufficient host RAM/shared memory and model/log/cache
mounts are deployment resources supplied by the operator; they are not image
contents. `MODEL_DIR` must use the same path on all roles so routing namespaces
match. Masters can mount just model/tokenizer metadata at that same path.

Inspect the exact command without starting or writing anything:

```bash
CONFIG_SERVER_HOST=lightllm-config glm53-pd p0 --dry-run
CONFIG_SERVER_HOST=lightllm-config glm53-pd master --dry-run
```

The measured configuration stays 1M context, P128/D256, MTP3, BF16 SSM,
Replay8/precompute/raw, `fp8kv_dsa`, P batch4096/chunk2048, D batch1024,
P memory .75 / D .85, D KV cap 2,800,000 per GPU, graph batch16
(P graphs disabled, D graphs enabled). Master resource waits are 200 seconds;
GPU KV transfer timeout is 600 seconds; Schema limit is 65536 for every role.

## Build and verify

Build from a clean public checkout, with no credentials, model files or local
runtime directories in the context:

```bash
git clone --single-branch -b glm53_optim https://github.com/ModelTC/LightLLM.git
cd LightLLM
revision=$(git rev-parse HEAD)
docker build --network host -f docker/Dockerfile.glm53_optim \
  --build-arg VCS_REF="$revision" -t "lightllm:glm53_optim-${revision:0:8}" .
```

The base image is pinned by digest. Runtime overlays use explicit versions and
`--no-deps`; NIXL's overlapping distributions are removed and reinstalled as a
family with SHA256 pins. Do not blanket-upgrade Torch/CUDA/NCCL/NVSHMEM. DeepEP's
pinned source, patch hashes, linked dependency versions and native binary hash
are recorded in `/opt/lightllm-build/deepep_rcqp.json`; the built wheel is kept
under `/opt/lightllm-build/wheels/`. The application revision is in the OCI label
and `/opt/lightllm-build/source_revision`.

```bash
# CPU validation: dependency/API checks, binary receipt and source revision.
docker run --rm --entrypoint /opt/conda/bin/python IMAGE \
  /lightllm/docker/validate_glm53_optim.py
# Adds Hopper native import checks; does not run model requests.
docker run --rm --gpus all --entrypoint /opt/conda/bin/python IMAGE \
  /lightllm/docker/validate_glm53_optim.py --gpu
```

Import checks do not substitute for a two-node DeepEP numerical check or a
complete four-node P/D inference check. Startup or HTTP 200 alone does not prove
that an entire streamed response completed.
