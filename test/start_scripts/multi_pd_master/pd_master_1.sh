# pd_master 1
# host: the host of the pd master
# config_server_host: the host of the config server
# sh pd_master_1.sh <host> <config_server_host>
export host=$1
export config_server_host=$2
python -m lightllm.server.api_server --model_dir /path/DeepSeek-R1 --run_mode "pd_master" --host $host --port 60011 --config_server_host $config_server_host --config_server_port 60088
