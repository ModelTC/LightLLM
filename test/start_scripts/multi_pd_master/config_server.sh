# config_server
# config_server_host: the host of the config server   
# sh config_server.sh <config_server_host>
export config_server_host=$1
python -m lightllm.server.api_server --run_mode "config_server" --config_server_host $config_server_host --config_server_port 60088
