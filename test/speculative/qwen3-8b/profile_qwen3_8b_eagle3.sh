export CUDA_VISIBLE_DEVICES=1,5
export LIGHTLLM_TRACE_DIR=/data/nvme0/chenjunyi/project/lightllm/trace
export LOADWORKER=18 
sudo /home/chenjunyi/nsight-systems-2026.2.1/bin/nsys profile \
    --cuda-memory-usage=true \
    --trace-fork-before-exec=true \
    --sample=none \
    --cpuctxsw=process-tree \
sudo -u chenjunyi bash -lc '
source /data/nvme0/chenjunyi/miniconda3/etc/profile.d/conda.sh
conda activate lightllm
python -m lightllm.server.api_server --port 8088 \
--tp 2 --max_total_token_num 200000 \
--model_dir /mtc/models/qwen3-8b \
--mtp_mode eagle3 \
--mtp_draft_model_dir /mtc/models/qwen3-8b-eagle3 \
--enable_profiling nvtx \
--mtp_step 12  \
--llm_decode_att_backend triton  \
--mtp_dynamic_verify \
--disable_cudagraph'
# if you want to enable microbatch overlap, you can uncomment the following lines
#--enable_prefill_microbatch_overlap \
#--enable_decode_microbatch_overlap \