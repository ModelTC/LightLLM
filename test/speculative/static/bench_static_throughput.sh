export PYTHONPATH=/data/nvme0/chenjunyi/project/lightllm:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=4,5
python test/speculative/benchmark_mtp_throughput.py \
    --model_dir /mtc/models/qwen3-8b \
    --mtp_draft_model_dir /mtc/models/qwen3-8b-eagle3 \
    --dataset /data/nvme0/chenjunyi/project/lightllm/datasets/gsm8k.json \
    --tp 2 \
    --nccl_port 56788 \
    --data_type bfloat16 \
    --mtp_mode eagle3 \
    --mtp_step 8 \
    --batch_sizes 1 \
    --output_lens 16 \
    --num_samples 100 \
    --llm_decode_att_backend triton \
    --max_total_token_num 200000 \
    --max_req_total_len 8192 \
    --skip_no_mtp