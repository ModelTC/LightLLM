#### lightllm 

#### Launch service

~~~shell
python -m lightllm.server.api_server --model_dir /path/llama-7b --tp 1 --max_total_token_num 121060 --tokenizer_mode auto
~~~

#### Evaluation

~~~shell
python benchmark_serving.py --tokenizer /path/llama-7b --dataset /path/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 2000 --request-rate 200
~~~

#### vllm 

#### Launch service
~~~shell
python -m vllm.entrypoints.api_server --model /path/llama-7b --swap-space 16 --disable-log-requests --port 9009
~~~

#### Evaluation

~~~shell
python benchmark_serving_vllm.py --backend vllm --tokenizer /path/llama-7b --dataset /path/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 2000 --request-rate 200 --host 127.0.0.1 --port 9009
~~~