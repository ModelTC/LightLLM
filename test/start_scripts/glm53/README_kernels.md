# GLM-5.3-Flash: FlashKDA + SM90 FP8 MegaMoE

This image combines the FlashKDA integration from PR #1733 and the SM90 FP8
MegaMoE integration from PR #1734 on `glm53_optim` commit `23e3a15f`.
It retains the branch's `LinearStateManager`, native/replay state, CPU/disk cache
and PD transfer interfaces. The optional packages are already compiled for SM90a
(H100/H200), Python 3.10, PyTorch 2.11.0 and CUDA 13.0.

## Keep the existing two-node EP16 groups

Keep the shared environment, ConfigServer, PD Master, and P/D commands. Add this
to P (and D if it also performs prefill):

```bash
--llm_prefill_att_backend auto auto
```

The first entry selects sparse attention; the second selects KDA. With BF16
activations and BF16 SSM state on Hopper, `auto` uses FlashKDA up to a maximum
query length of 8192 when there are at least 16 local KDA heads, or 2048 for fewer
heads. Longer queries use Triton. This is a per-forward policy based on the
largest query in the batch; `--batch_max_tokens` is not the selection criterion.
Missing or incompatible FlashKDA installations also fall back under `auto`.

To force either implementation for a comparison:

```bash
--llm_prefill_att_backend auto flashkda
--llm_prefill_att_backend auto triton
```

An explicit `flashkda` selection validates support and fails if the extension
cannot load. It does not silently fall back. `--llm_kv_type fp8kv_dsa` remains
supported: it controls sparse-attention KV storage. FlashKDA's Q/K/V are BF16,
and its recurrent state is BF16 or FP32; it does not implement FP8 recurrent
state or change decode's KDA/MTP implementation.

**Do not enable MegaMoE on `--nnodes 2 --tp 16 --dp 16`.** The SM90 FP8 kernel
supports a single NVLink node per EP group. Keep `--ep_moe_backend auto` and
leave `LIGHTLLM_ENABLE_SM90_FP8_MEGA_MOE` unset in the original topology.

## Enable both features with independent EP8 groups

Use one 8-GPU node per group. The two P hosts and two D hosts become four
independent groups, all using the existing ConfigServer and PD Master.
Change the following arguments in **every P and D command**:

| Argument | Existing EP16 | Independent EP8 |
|---|---|---|
| `--nnodes` | `2` | `1` |
| `--node_rank` | `0` or `1` | `0` on all four hosts |
| `--tp`, `--dp` | `16`, `16` | `8`, `8` |
| `--ep_moe_backend` | `auto` | `megamoe` |
| `--expert_dtype` | inferred | `fp8` |
| `--llm_prefill_att_backend` | `auto` | `auto auto` or `auto flashkda` |
| `--max_total_token_num` | P `2249892`, D `2800000` | omit initially; size from `--mem_fraction` |

Keep `--enable_ep_moe`, `--quant_type fp8w8a8-b128-deepgemm`,
`--linear_att_ssm_data_type bfloat16`, `--llm_kv_type fp8kv_dsa`, P native/MTP1,
D replay/MTP3, and the same NIXL page settings. Do not enable EPLB, RL, or
prefill/decode microbatch overlap with MegaMoE.

Use a distinct NCCL store port for each independent group:

| Host | Node rank | NCCL port | API port | Disk cache name |
|---|---:|---:|---:|---|
| P0 | 0 | 29501 | 8019 | `2p2d-p0` |
| P1 | 0 | 29503 | 8019 | `2p2d-p1` |
| D0 | 0 | 29502 | 8121 | unused |
| D1 | 0 | 29504 | 8121 | unused |

With ConfigServer NCCL initialization, expose ports 29501–29504 on the
ConfigServer to the GPU hosts. Preserve the existing NIXL network connectivity.
P's `--use_dp_split_mode_connect_pd_master` also works with single-node DP8.
The existing `PD_MASTER_MODE=2p2d` and cache-aware selector can be retained.

EP8 puts twice as many routed experts on each GPU as EP16. Reusing the old
fixed KV capacities can cause OOM. Check the capacity printed at startup and
the remaining GPU/pinned host memory before setting new explicit capacities.
`MAX_REQ_TOTAL_LEN=1048576` is a request limit, not proof that a particular GPU
and cache allocation can serve a one-million-token request. MegaMoE's
communication buffer is derived from `batch_max_tokens` for P and from
per-DP request/graph capacity multiplied by `mtp_step + 1` for D.

## Packaged launcher

The image contains `glm53-pd-kernels`. It reproduces the supplied P/D options
and shared environment, with these explicit feature/topology controls:

```bash
export MODEL_DIR=/example
export MTP_DRAFT_MODEL_DIR="$MODEL_DIR"
export CONFIG_SERVER_HOST=lightllm-config
export CONFIG_SERVER_PORT=60088
export PD_MASTER_PORT=60011
export PD_MASTER_MODE=2p2d
export MAX_REQ_TOTAL_LEN=1048576

# Original topology with automatic FlashKDA/Triton selection:
export PD_TOPOLOGY=ep16
export EP_MOE_BACKEND=auto
export KDA_PREFILL_BACKEND=auto

# Alternatively, use both features on independent 8-GPU groups:
export PD_TOPOLOGY=ep8
export EP_MOE_BACKEND=megamoe
export KDA_PREFILL_BACKEND=auto
```

Run exactly one role per corresponding pod:

```bash
exec glm53-pd-kernels config
exec glm53-pd-kernels master
exec glm53-pd-kernels p0
exec glm53-pd-kernels p1
exec glm53-pd-kernels d0
exec glm53-pd-kernels d1
```

These are six separate commands for six roles, not a sequence in one shell.
The role supplies node rank; K8s `NODE_RANK=1` must not be reused as the NCCL rank
of an independent EP8 group. P1 still gets its own disk-cache name.

Use `glm53-pd-kernels p0 --dry-run` to inspect arguments. Set
`KDA_PREFILL_BACKEND=flashkda` to force FlashKDA or `triton` for a baseline.
`P_MAX_TOTAL_TOKEN_NUM` and `D_MAX_TOTAL_TOKEN_NUM` override the KV capacity;
setting either to `0` omits the corresponding CLI option. EP8 omits both by
default. Other overrides are listed by `glm53-pd-kernels --help`.

The supplied legacy tuning environment is retained for compatibility.
Environment variable presence alone does not prove a kernel is active;
feature selection is controlled by the CLI above. Confirm these startup logs:

```text
KDA prefill attention backend: automatic FlashKDA/Triton.
Initialize DeepEP MoE buffers: ... mega_moe=True, mega_moe_mma_type=fp8xfp8 ...
```

With forced FlashKDA the first line ends in `flashkda.`. Automatic selection can
still use Triton for longer queries. MegaMoE disables the incompatible shared
expert overlap path internally.

## Build and verify

The dependency image starts from the previous published image's immutable
digest. Ordinary `deep_gemm`, DeepEP, NIXL and their version pins are preserved;
the MegaMoE provider is installed separately as `sgl_deep_gemm`.

| Component | Pinned revision |
|---|---|
| FlashKDA | `d7e3b3a9f448500630fa60fb5200f4f9c9b1fc72` |
| SGLang DeepGEMM | `dbbb0f4745af11c4c899ea7648888af687394ede` |
| Original image | `sha256:1c48fccdecef901df2a322227b53aae3ee6b17fff8102298ff6831b95a36da00` |

From a clean checkout of the integration branch:

```bash
python docker/prepare_glm53_kernels.py /tmp/glm53-kernel-inputs
docker build -f docker/Dockerfile.glm53_kernels \
  -t lightllm:glm53-kernels /tmp/glm53-kernel-inputs
docker build -f docker/Dockerfile.glm53_flashkda_megamoe \
  --build-arg KERNEL_IMAGE=lightllm:glm53-kernels \
  --build-arg VCS_REF="$(git rev-parse HEAD)" \
  -t lightllm:glm53-flashkda-megamoe .
docker run --rm --gpus all lightllm:glm53-flashkda-megamoe \
  python /lightllm/docker/validate_glm53_kernels.py --gpu
```

Use a regular clean clone for the final build so `.git` is self-contained.
The image stores source revision, dependency input hashes, native extension
hashes and third-party licenses in `/opt/lightllm-build`.
The validator checks installed dependencies and interfaces; numerical kernel
tests and actual model requests are separate acceptance checks.
