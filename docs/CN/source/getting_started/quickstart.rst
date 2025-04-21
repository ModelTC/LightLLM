.. _quickstart:

快速开始
==========

使用lightllm部署模型非常简单，最快只需要两个步骤：

1. 准备 Lightllm 所支持的模型的权重文件。
2. 使用命令行启动模型服务。
3. (可选) 对模型服务进行测试。

.. note::
    在继续这个教程之前，请确保你完成了 :ref:`安装指南 <installation>` .



1. 准备模型文件
-------------------------

下面的内容将会以 `Llama-2-7b-chat <https://huggingface.co/meta-llama/Llama-2-7b-chat>`_ 演示lightllm对大语言模型的支持。
下载模型的方法可以参考文章：`如何快速下载huggingface模型——全方法总结 <https://zhuanlan.zhihu.com/p/663712983>`_ 

下面是下载模型的实例代码：

(1) （可选）创建文件夹

.. code-block:: console

    $ mkdirs ~/models && cd ~/models
    
(2) 安装 ``huggingface_hub``

.. code-block:: console

    $ pip install -U huggingface_hub

(3) 下载模型文件

.. code-block:: console
    
    $ huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir Llama-2-7b-chat

.. tip::
    上面的下载模型的代码需要科学上网，并且需要花费一定的时间，你可以使用其它下载方式或者其它支持的模型作为替代。最新的支持的模型的列表请查看 `项目主页 <https://github.com/ModelTC/lightllm>`_ 。


2. 启动模型服务
-------------------------

下载完Llama-2-7b-chat模型以后，在终端使用下面的代码部署API服务：

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/Llama-2-7b-chat

.. note::
    上面代码中的 ``--model_dir`` 参数需要修改为你本机实际的模型路径。

单机H200部署 DeepSeek-R1 模型, 启动命令如下:

.. code-block:: console

    $ LOADWORKER=8 python -m lightllm.server.api_server --model_dir ~/models/DeepSeek-R1 --tp 8 --graph_max_batch_size 100

.. note::
    LOADWORKER 指定了模型加载的线程，可以提高模型加载的速度。--graph_max_batch_size 指定了要捕获的cudagraph的数量，将捕获从1到100的batch size的图。

双机H100部署 DeepSeek-R1 模型，启动命令如下：

.. code-block:: console

    $ # Node 0
    $ LOADWORKER=8 python -m lightllm.server.api_server --model_dir ~/models/DeepSeek-R1 --tp 16 --graph_max_batch_size 100 --nccl_host master_addr --nnodes 2 --node_rank 0
    $ # Node 1
    $ LOADWORKER=8 python -m lightllm.server.api_server --model_dir ~/models/DeepSeek-R1 --tp 16 --graph_max_batch_size 100 --nccl_host master_addr --nnodes 2 --node_rank 1

3. PD 分离启动模型服务
-------------------------
查找本机IP

.. code-block:: console

    $ hostname -i

运行MPS(可选, 有mps支持性能会好特别多，但是部分显卡和驱动环境开启mps会容易出现错误，建议升级驱动到较高版本，特别是H系列卡)

.. code-block:: console

    $ nvidia-cuda-mps-control -d 


运行pd_master服务

.. code-block:: console

    $ python -m lightllm.server.api_server \
    $ --model_dir /your/model/path \
    $ --run_mode "pd_master" \
    $ --host /your/host/ip \
    $ --port 60011

新建终端,运行prefill服务 

.. code-block:: console

    $ CUDA_VISIBLE_DEVICES=0,1 KV_TRANS_USE_P2P=1 LOADWORKER=1 python -m lightllm.server.api_server --model_dir /data/fengdahu/model/Qwen2-7B/ \
    $ --run_mode "prefill" \
    $ --host /your/host/ip \
    $ --port 8017 \
    $ --tp 2 \
    $ --nccl_port 2732 \
    $ --max_total_token_num 400000 \
    $ --tokenizer_mode fast \
    $ --pd_master_ip /your/host/ip \
    $ --pd_master_port 60011 \
    $ --max_req_total_len 16000 \
    $ --running_max_req_size 128 \
    $ --disable_cudagraph

新建终端,运行decoding服务

.. code-block:: console

    $ CUDA_VISIBLE_DEVICES=2,3 KV_TRANS_USE_P2P=1 LOADWORKER=10 python -m lightllm.server.api_server --model_dir /data/fengdahu/model/Qwen2-7B/ \
    $ --run_mode "decode" \
    $ --host /your/host/ip \
    $ --port 8118 \
    $ --nccl_port 12322 \
    $ --tp 2 \
    $ --max_total_token_num 400000 \
    $ --graph_max_len_in_batch 2048 \
    $ --graph_max_batch_size 16 \
    $ --tokenizer_mode fast \
    $ --pd_master_ip /your/host/ip \
    $ --pd_master_port 60011

.. note::
    prefill和decoding阶段的tp大小保持一致, 目前可以支持 prefill 和 decode 节点的数量是变化的，同时prefill 和 decode可以跨机部署。


4. （可选）测试模型服务
-------------------------

在新的终端，使用下面的指令对模型服务进行测试：

.. code-block:: console

    $ curl http://server_ip:server_port/generate \
    $      -H "Content-Type: application/json" \
    $      -d '{
    $            "inputs": "What is AI?",
    $            "parameters":{
    $              "max_new_tokens":17, 
    $              "frequency_penalty":1
    $            }
    $           }'


对于DeepSeek-R1模型，可以用如下脚本进行测试：

.. code-block:: console

    $ cd test
    $ python benchmark_client.py --num_clients 100 --input_num 2000 --tokenizer_path /nvme/DeepSeek-R1/ --url http://127.0.01:8000/generate_stream


3. PD 分离多PD_Master节点类型启动模型服务
-------------------------
查找本机IP

.. code-block:: console

    $ hostname -i

运行MPS(可选, 有mps支持性能会好特别多，但是部分显卡和驱动环境开启mps会容易出现错误，建议升级驱动到较高版本，特别是H系列卡)

.. code-block:: console

    $ nvidia-cuda-mps-control -d 


运行config_server服务
.. code-block:: console

$ python -m lightllm.server.api_server \
$ --run_mode "config_server" \
$ --config_server_host /your/host/ip \
$ --config_server_port 60088 \


运行pd_master服务, 在多pd_master节点模式下，可以开启多个pd_master服务，来实现负载均衡，单个pd_master因为python gil锁的原因
其并发性能存在上限。

.. code-block:: console

    $ python -m lightllm.server.api_server \
    $ --model_dir /your/model/path \
    $ --run_mode "pd_master" \
    $ --host /your/host/ip \
    $ --port 60011 \
    $ --config_server_host <config_server_host> \
    $ --config_server_port <config_server_port>

新建终端,运行prefill服务 

.. code-block:: console

    $ CUDA_VISIBLE_DEVICES=0,1 KV_TRANS_USE_P2P=1 LOADWORKER=1 python -m lightllm.server.api_server --model_dir /data/fengdahu/model/Qwen2-7B/ \
    $ --run_mode "prefill" \
    $ --host /your/host/ip \
    $ --port 8017 \
    $ --tp 2 \
    $ --nccl_port 2732 \
    $ --max_total_token_num 400000 \
    $ --tokenizer_mode fast \
    $ --max_req_total_len 16000 \
    $ --running_max_req_size 128 \
    $ --disable_cudagraph \
    $ --config_server_host <config_server_host> \
    $ --config_server_port <config_server_port>

新建终端,运行decoding服务

.. code-block:: console

    $ CUDA_VISIBLE_DEVICES=2,3 KV_TRANS_USE_P2P=1 LOADWORKER=10 python -m lightllm.server.api_server --model_dir /data/fengdahu/model/Qwen2-7B/ \
    $ --run_mode "decode" \
    $ --host /your/host/ip \
    $ --port 8118 \
    $ --nccl_port 12322 \
    $ --tp 2 \
    $ --max_total_token_num 400000 \
    $ --graph_max_len_in_batch 2048 \
    $ --graph_max_batch_size 16 \
    $ --tokenizer_mode fast \
    $ --config_server_host <config_server_host> \
    $ --config_server_port <config_server_port>

.. note::
    prefill和decoding阶段的tp大小保持一致, 目前可以支持 prefill 和 decode 节点的数量是变化的，同时prefill 和 decode可以跨机部署。


4. （可选）测试模型服务
-------------------------

在新的终端，使用下面的指令对模型服务进行测试, 在多pd_master模式下，每个pd_master都可以作为访问入口：

.. code-block:: console

    $ curl http://server_ip:server_port/generate \
    $      -H "Content-Type: application/json" \
    $      -d '{
    $            "inputs": "What is AI?",
    $            "parameters":{
    $              "max_new_tokens":17, 
    $              "frequency_penalty":1
    $            }
    $           }'


对于DeepSeek-R1模型，可以用如下脚本进行测试：

.. code-block:: console

    $ cd test
    $ python benchmark_client.py --num_clients 100 --input_num 2000 --tokenizer_path /nvme/DeepSeek-R1/ --url http://127.0.01:8000/generate_stream

