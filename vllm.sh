#!/usr/bin/env bash
set -euo pipefail

export BS=8
export ILEN=10000
export OLEN=500
export WARMUP_OLEN=10

export ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_DIR="$ROOT_DIR/../Qwen3-235B-A22B"
export PROFILE_DIR="$ROOT_DIR/profiles"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

rm -rf "$PROFILE_DIR/qwen3_static_decode"
mkdir -p "$PROFILE_DIR/qwen3_static_decode"


cd "$PROFILE_DIR"

env -u PYTHONPATH python - <<'PY'
import os
import time

from vllm import LLM
from vllm.config import ProfilerConfig


batch_size = int(os.environ["BS"])
input_len = int(os.environ["ILEN"])
model_dir = os.environ["MODEL_DIR"]
profile_dir = os.path.join(os.environ["PROFILE_DIR"], "qwen3_static_decode")

profiler_config = ProfilerConfig(
  profiler="torch",
  torch_profiler_dir=profile_dir,
  delay_iterations=0,
  max_iterations=0,
  ignore_frontend=True,
)

llm = LLM(
  model=model_dir,
  skip_tokenizer_init=True,
  tensor_parallel_size=8,
  dtype="bfloat16",
  max_model_len=32768,
  max_num_seqs=batch_size,
  max_num_batched_tokens=batch_size * input_len,
  enable_prefix_caching=False,
  enforce_eager=False,
  profiler_config=profiler_config,
)


def warmup_decode_dummy_run(worker, num_tokens: int, seq_len: int):
  worker.model_runner._dummy_run(
    num_tokens=num_tokens,
    uniform_decode=True,
    profile_seq_lens=seq_len,
  )
  return worker.rank


def profile_decode_dummy_run(worker, num_tokens: int, seq_len: int):
  if worker.rank == 0:
    worker.profile(True)
  worker.model_runner._dummy_run(
    num_tokens=num_tokens,
    uniform_decode=True,
    profile_seq_lens=seq_len,
  )
  if worker.rank == 0:
    worker.profile(False)
    return {"rank": worker.rank, "profiled": True}
  return {"rank": worker.rank, "profiled": False}


print("Warm up pure decode forward...")
print(
  f"Warmup replies: {llm.collective_rpc(warmup_decode_dummy_run, args=(batch_size, input_len))}"
)

print(f"Profiling pure decode forward into {profile_dir} ...")
print(
  f"Profile replies: {llm.collective_rpc(profile_decode_dummy_run, args=(batch_size, input_len))}"
)

# Give worker processes a moment to flush trace files.
time.sleep(5)
PY
