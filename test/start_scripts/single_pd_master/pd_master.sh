# pd_master for deepseek R1
# pd_master_ip: the ip of the pd master
# sh pd_master.sh <pd_master_ip>
export pd_master_ip=$1
python -m lightllm.server.api_server --model_dir /path/DeepSeek-R1 --run_mode "pd_master" --host $pd_master_ip --port 60011