export LOADWORKER=18 
python -m lightllm.server.api_server --port 8090 \
--tp 2 --max_total_token_num 200000 \
--model_dir /data/chenjunyi/models/qwen3-8b \
--mtp_mode eagle3 \
--mtp_draft_model_dir /data/chenjunyi/models/qwen3-8b-eagle3 \
--mtp_step 12  \
--llm_decode_att_backend fa3
# if you want to enable microbatch overlap, you can uncomment the following lines
#--enable_prefill_microbatch_overlap \
#--enable_decode_microbatch_overlap \
