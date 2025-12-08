Benchmark 测试指南
==================

LightLLM 提供了全面的性能测试工具，包括服务端性能测试和静态推理性能测试。本文档将详细介绍如何使用这些工具进行性能评估。

服务端性能测试 (Service Benchmark)
-----------------------------------

服务端性能测试主要用于评估 LightLLM 在真实服务场景下的性能表现，包括吞吐量、延迟等关键指标。

QPS 测试 (benchmark_qps.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

QPS (Queries Per Second) 测试是评估服务端性能的核心工具，支持 LightLLM 和 OpenAI 兼容的 API 格式

**使用方法：**

.. code-block:: bash

    python test/benchmark/service/benchmark_qps.py \
        --url http://127.0.0.1:8000/generate_stream \
        --tokenizer_path /path/to/tokenizer \
        --num_clients 100 \
        --input_num 2000 \
        --input_qps 30.0 \
        --input_len 1024 \
        --output_len 128 \
        --server_api lightllm \
        --dump_file results.json

**主要参数说明：**

- ``--url``: 服务端地址，支持 LightLLM 和 OpenAI 格式
- ``--tokenizer_path``: 分词器路径
- ``--input_num``: 测试请求总数
- ``--input_qps``: 输入 QPS 限制
- ``--input_len``: 输入序列长度
- ``--output_len``: 输出序列长度
- ``--server_api``: 服务端 API 类型 (lightllm/openai)
- ``--data_path``: 自定义数据集路径
- ``--continuous_send``: 是否连续发送 (0/1)
- ``--force_terminate``: 强制终止模式 (0/1)

**输出指标：**

- Total QPS: 总体每秒查询数
- Sender QPS: 发送端 QPS
- Avg Input Length: 平均输入长度
- Avg Output Length: 平均输出长度
- Total Throughput: 总体吞吐量 (token/s)
- Input Throughput: 输入吞吐量 (token/s)
- Output Throughput: 输出吞吐量 (token/s)
- request_time P{25,50,75,90,95,99,100}: 请求延迟百分位数
- first_token_time P{25,50,75,90,95,99,100}: 首 token 延迟百分位数
- decode_token_time P{25,50,75,90,95,99,100}: 解码 token 延迟百分位数

固定并发测试 (benchmark_client.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

用于评估不同客户端并发数下的性能表现。

**使用方法：**

.. code-block:: bash

    python test/benchmark/service/benchmark_client.py \
        --url http://127.0.0.1:8000/generate_stream \
        --tokenizer_path /path/to/tokenizer \
        --num_clients 100 \
        --input_num 2000 \
        --input_len 1024 \
        --output_len 128 \
        --server_api lightllm

ShareGPT 数据集测试 (benchmark_sharegpt.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用 ShareGPT 真实对话数据进行性能测试。

**使用方法：**

.. code-block:: bash

    $ wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json


.. code-block:: bash

    python test/benchmark/service/benchmark_sharegpt.py \
        --dataset /path/to/sharegpt_dataset.json \
        --tokenizer /path/to/tokenizer \
        --num_prompts 1000 \
        --request_rate 10.0

**主要参数：**

- ``--dataset``: ShareGPT 格式数据集路径
- ``--tokenizer``: 分词器路径
- ``--num_prompts``: 测试提示数量
- ``--request_rate``: 请求速率 (requests/s)


Prompt Cache 测试
~~~~~~~~~~~~~~~~~

评估不同命中率下，prompt cache 的性能，通过调整 --first_input_len, --output_len --subsequent_input_len 来控制命中率。
每轮命中率 = (first_input_len + (output_len + subsequent_input_len) * (num_turns - 1)) / (first_input_len + (output_len + subsequent_input_len) * num_turns)
注意要根据最大token容量控制并发数和users数，确保能够放下所有请求，保障其实际命中率和自己预设的命中率一致。

.. code-block:: bash

    python test/benchmark/service/benchmark_prompt_cache.py \
        --model_url http://127.0.0.1:8000/generate_stream \
        --model_name model \
        --num_workers 10 \
        --first_input_len 512 \
        --subsequent_input_len 512 \
        --output_len 128 \
        --num_turns 10 \
        --num_users 10

参数说明：

- ``--model_url``: 服务地址
- ``--model_name``: 结果保存文件名
- ``--num_workers``: 并发数
- ``--first_input_len``: 第一轮输入长度
- ``--subsequent_input_len``: 后续轮输入长度
- ``--output_len``: 输出长度
- ``--num_turns``: 轮数
- ``--num_users``: 用户数

静态推理性能测试 (Static Inference Benchmark)
----------------------------------------------

静态推理测试用于评估模型在固定输入条件下的推理性能, 主要评估算子的优劣
模型推理测试 (model_infer.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**主要特性：**

- 支持 prefill 和 decode 阶段性能测试
- 支持 microbatch overlap 优化
- 支持多 GPU 并行推理
- 提供详细的吞吐量统计

**使用方法：**

.. code-block:: bash

    python test/benchmark/static_inference/test_model.py \
        --model_dir /path/to/model \
        --batch_size 32 \
        --input_len 1024 \
        --output_len 128 \
        --tp 2 \
        --data_type bf16

**主要参数：**

- ``--model_dir``: 模型路径
- ``--batch_size``: 批次大小
- ``--input_len``: 输入序列长度
- ``--output_len``: 输出序列长度
- ``--tp``: Tensor Parallel 并行度
- ``--data_type``: 数据类型 (bf16/fp16/fp32)
- ``--enable_prefill_microbatch_overlap``: 启用 prefill microbatch overlap，仅适用于DeepSeek模型的EP模式
- ``--enable_decode_microbatch_overlap``: 启用 decode microbatch overlap，仅适用于DeepSeek模型的EP模式
- ``--torch_profile``: 启用 torch profiler 进行性能分析

.. note::
    这里没有列举完整的启动参数，静态测试脚本也共享lightllm的启动参数，更多启动配置可以参考 :ref:`tutorial/api_server_args_zh` 。

**输出指标：**

- Prefill 阶段吞吐量 (tokens/s)
- Decode 阶段吞吐量 (tokens/s)
- 各阶段延迟统计

多结果预测性能测试 (model_infer_mtp.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

多结果预测静态性能测试，默认百分百接受率，用来评估多结果预测的极限性能。目前只支持DeepSeek 系列模型

**使用方法：**

.. code-block:: bash

    python test/benchmark/static_inference/test_model.py \
        --model_dir /path/to/main_model \
        --mtp_mode deepseekv3 \
        --mtp_step 1 \
        --mtp_draft_model_dir /path/to/draft_model \
        --batch_size 32 \
        --input_len 1024 \
        --output_len 128

参数说明：

- ``--model_dir``: 主模型路径
- ``--mtp_mode``: 指定多结果预测的模型，目前只支持deepseekv2/v3/r1
- ``--mtp_step``: 每次forward step产生的token 数量，默认为1
- ``--mtp_draft_model_dir``: 草稿模型路径

Vision Transformer 测试 (test_vit.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

用于测试 Vision Transformer 模型的性能。

**使用方法：**

.. code-block:: bash

    python test/benchmark/static_inference/test_vit.py \
        --model_dir ./InternVL2/InternVL2-8B/ \
        --batch_size 1 \
        --image_size 448 \
        --world_size 2