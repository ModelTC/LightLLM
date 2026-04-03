sudo env PYTHONPATH=/data/nvme0/chenjunyi/project/lightllm \
  /usr/local/cuda/bin/ncu \
    --target-processes all \
    --profile-from-start on \
    --launch-skip 0 \
    --launch-count 200 \
    --section LaunchStats \
    --section Occupancy \
    --section SchedulerStats \
    --section WarpStateStats \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    -f \
    -o /data/nvme0/chenjunyi/project/lightllm/ncu_stage_root \
    /data/nvme0/chenjunyi/miniconda3/envs/lightllm/bin/python \
    /data/nvme0/chenjunyi/project/lightllm/test/speculative/kernel/bench_decode.py \
    --warmup 50 --iters 100