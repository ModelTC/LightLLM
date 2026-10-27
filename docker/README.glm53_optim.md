# GLM53 optimized overlay image

This Dockerfile updates the `support_ds4` release image with `glm53_optim` source,
ReplaySSM/FP8 KV, cache-aware routing and its actual-cache measurements. The
control plane uses a ConfigServer, four independent PD masters (one worker each)
and HAProxy. GPU roles are `p0`, `p1`, `d0`, `d1`; their group ranks are filled by
the launcher.

## Build

Use a clean checkout of `glm53_optim` from the public repository, with no registry
credentials, local dependency directories or model files in the build context.
The Dockerfile retains `.git` so the source can be inspected and updated later.
The base image is pinned by digest, and the source commit is checked during
build and recorded in an OCI label and `/opt/lightllm-build/source_revision`.

```bash
git clone --depth 1 -b glm53_optim https://github.com/ModelTC/LightLLM.git
cd LightLLM
revision=$(git rev-parse HEAD)
docker build --network host -f docker/Dockerfile.glm53_optim \
  --build-arg VCS_REF="$revision" -t "lightllm:glm53_optim-${revision:0:8}" .
```

`requirements.glm53_optim.txt` is an explicit overlay, installed with `--no-deps`.
It includes the required transitive DSL libraries. This preserves the base
CUDA 13/Torch 2.11/Triton 3.6 and compiled DeepEP, DeepGEMM, FlashMLA and LightMem.
`requirements.glm53_optim_extra.txt` adds the required CUTLASS cu12 companion and
the base NVSHMEM's missing CCCL headers, while keeping the CUDA 13.0 SDK version.
These extra wheels have official PyPI SHA256 pins for Linux x86_64 / Python 3.10.
If PyPI downloads are slow, pass
`--build-arg POST_UPGRADE_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`;
the hashes still verify that the downloaded wheels match the official artifacts.
The NIXL distributions are removed together and reinstalled after the overlay
upgrade. The old release wheels have overlapping file ownership, so an ordinary in-place
upgrade can delete files just installed by another wheel while reporting the
new versions as successfully installed.
Do not run a blanket dependency upgrade. The base image contains unrelated
vLLM/MCP packages with pre-existing metadata conflicts; `pip check` should be
compared with the base result, not treated as proof of LightLLM GPU correctness.

## Validate

```bash
# No GPU or model requests: versions, real native-library imports, XGrammar
# compile/match, GLM/Replay/cache-aware imports and API CLI options.
docker run --rm --entrypoint /opt/conda/bin/python IMAGE \
  /lightllm/docker/validate_glm53_optim.py

# On a Hopper machine with a CUDA 13 compatible driver: additionally import
# SGL's native FlashMLA/FlashAttention bindings. This is an API check, not a
# numerical kernel test or multi-node communication test.
docker run --rm --gpus all --entrypoint /opt/conda/bin/python IMAGE \
  /lightllm/docker/validate_glm53_optim.py --gpu
```

The build host does not need a GPU. Do not disable the NVIDIA entrypoint's CUDA
driver requirements to make a validation appear successful. Production use
still requires Hopper kernel and distributed P/D validation on the target hosts.

## Start the 2P2D deployment

The image provides `glm53-pd`, which invokes the existing launcher with
image-local source/dependencies and `/model` as the model mount. Logs default
to `/var/log/lightllm/logs/ROLE.log`; mount `/var/log/lightllm` to persistent host
storage. The proxy uses the base image's bundled `ray-haproxy` 2.8.25 binary,
the same binary exercised by the independent-master integration test. Review
all IP/port/model/cache mounts before starting.

```bash
# Inspect every control-plane command without starting anything.
docker run --rm -e PD_MASTER_IP=CPU_IP IMAGE glm53-pd cpu --dry-run

# CPU host: mount the model/tokenizer metadata needed by all four masters.
docker run --name glm53-control --network host --ipc host \
  -v /host/model_meta:/model:ro -v /host/logs/cpu:/var/log/lightllm \
  -e PD_MASTER_IP=CPU_IP IMAGE glm53-pd cpu

# One command on EACH GPU host; set ROLE to p0/p1/d0/d1 as appropriate.
# Configure RDMA device access/mounts for the host's container runtime as well.
docker run --name "glm53-$ROLE" --gpus all --network host --ipc host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /host/model:/model:ro -v /host/logs:/var/log/lightllm \
  -v /host/disk_cache:/mnt/vqa/lightllm_disk_cache \
  -e PD_MASTER_IP=CPU_IP -e P0_IP=P0_IP -e P1_IP=P1_IP \
  -e D0_IP=D0_IP -e D1_IP=D1_IP IMAGE glm53-pd "$ROLE"
```

The default topology remains 1M context, P128/D256, MTP3, BF16 SSM, Replay8 with
precompute/raw, FP8 KV and D KV cap 2,800,000 per card. `cpu` starts ports 60011
(HAProxy), 60088 (ConfigServer) and 60111–60114 (masters). Use a separate log
mount for each host. Persistent disk cache paths and RDMA access are host-specific
and must be supplied by the operator. See
[`README_glm53_optim_2p2d.md`](../test/start_scripts/glm53/README_glm53_optim_2p2d.md)
for the full topology and available overrides.
