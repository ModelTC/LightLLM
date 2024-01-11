#!/bin/bash

DATASET_PATH="/your/date/path" # 你的数据集路径
FILE_PATH="/model/root" # 确保这里是所有模型的上级目录
HOST="0.0.0.0"
NCCL_PORT=28000
CUDA_LIST=(0 1 2 3)
PORT=8000
MAX_PORT=65535
NUM_PROMPTS=100
REQUEST_RATE=20

test_models() {
    local -a models=("${!1}")
    local -a modes=("${!2}")
    echo "models: ${models[@]}"
    echo "modes: ${modes[@]}"
    local model_num=${#models[@]}
    local loop_num=${#modes[@]}

    for model in "${models[@]}"; do
        local model_dir="${FILE_PATH}/${model}"
        # export CUDA_VISIBLE_DEVICES=${CUDA_LIST[i]}

        for ((i = 0; i <= loop_num; i++)); do
            local current_port=$PORT
            local current_nccl_port=$((NCCL_PORT+i))

            # 检查端口是否被占用
            while lsof -i:$current_nccl_port &>/dev/null || lsof -i:$current_port &>/dev/null; do
                current_nccl_port=$((current_nccl_port+1))
                current_port=$((current_port+1))
                if [ "$current_port" -gt "$MAX_PORT" ] || [ "$current_nccl_port" -gt "$MAX_PORT" ]; then
                    echo "No available ports found."
                    exit 1
                fi
            done

            echo "Start ${model_dir} on port ${current_port} with GPU ${CUDA_LIST[i]} and NCCL_PORT ${current_nccl_port} with mode ${modes[i]}"
            if [ "$i" -eq 0 ]; then
                nohup python -m lightllm.server.api_server --model_dir "${model_dir}" --host ${HOST} --port ${current_port} --tp 1 --trust_remote_code --nccl_port ${current_nccl_port} > server_output.log 2>&1 &
            else
                echo "idx:${i} with mode ${modes[i-1]}"
                nohup python -m lightllm.server.api_server --model_dir "${model_dir}" --mode "${modes[i-1]}" --host ${HOST} --port ${current_port} --tp 1 --trust_remote_code --nccl_port ${current_nccl_port} > server_output.log 2>&1 &
            fi
            local server_pid=$!

            # 等待服务器启动并监控输出
            echo "Waiting for server to start..."
            tail -f server_output.log | while read line; do
                echo "${line}"
                if [[ "${line}" == *"Uvicorn running on http://0.0.0.0"* ]]; then
                    echo "Server is ready. Starting the client..."
                    pkill -P $$ tail # 终止 tail 进程 继续执行后面的命令
                    break
                fi
            done

            # 启动接收端程序
            echo "Starting the client to send requests..."
            python test/benchmark_serving.py --tokenizer "${model_dir}" --dataset "${DATASET_PATH}" --num-prompts ${NUM_PROMPTS} --request-rate ${REQUEST_RATE} --port ${current_port} --model "${model}"
            echo "Client finished."

            # 接收端程序完成后，关闭服务器
            echo "Shutting down the server: pid=${server_pid}"
            kill "${server_pid}"
            sleep 1
            # 检查进程是否仍然存在
            if ps -p "${server_pid}" > /dev/null; then # 尝试获取特定 PID 的进程信息
                echo "The server is still running."
                kill -9 "${server_pid}"
            else
                echo "The server has been stopped."
            fi
        done
    done
}

# 示例调用
MODEL_ARRAY_LLAMA=("llama2-13b-chat")
MODE_ARRAY_LLAMA=("triton_int8weight" "triton_int4weight")
test_models MODEL_ARRAY_LLAMA[@] MODE_ARRAY_LLAMA[@]