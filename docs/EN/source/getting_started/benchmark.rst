Benchmark Testing Guide
=======================

LightLLM provides multiple performance testing tools, including service performance testing and static inference performance testing. This document will detailedly introduce how to use these tools for performance evaluation.

Service Performance Testing (Service Benchmark)
-----------------------------------------------

Service performance testing is mainly used to evaluate LightLLM's performance in real service scenarios, including key metrics such as throughput and latency.

QPS Testing (benchmark_qps.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QPS (Queries Per Second) testing is the core tool for evaluating service performance, supporting LightLLM and OpenAI compatible API formats.

**Usage:**

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

**Main Parameter Description:**

- ``--url``: Service address, supports LightLLM and OpenAI formats
- ``--tokenizer_path``: Tokenizer path
- ``--input_num``: Total number of test requests
- ``--input_qps``: Input QPS limit
- ``--input_len``: Input sequence length
- ``--output_len``: Output sequence length
- ``--server_api``: Service API type (lightllm/openai)
- ``--data_path``: Custom dataset path
- ``--continuous_send``: Whether to send continuously (0/1)
- ``--force_terminate``: Force termination mode (0/1)

**Output Metrics:**

- Total QPS: Overall queries per second
- Sender QPS: Sender QPS
- Avg Input Length: Average input length
- Avg Output Length: Average output length
- Total Throughput: Overall throughput (token/s)
- Input Throughput: Input throughput (token/s)
- Output Throughput: Output throughput (token/s)
- request_time P{25,50,75,90,95,99,100}: Request latency percentiles
- first_token_time P{25,50,75,90,95,99,100}: First token latency percentiles
- decode_token_time P{25,50,75,90,95,99,100}: Decode token latency percentiles

Fixed Concurrency Testing (benchmark_client.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Used to evaluate performance under different client concurrency levels.

**Usage:**

.. code-block:: bash

    python test/benchmark/service/benchmark_client.py \
        --url http://127.0.0.1:8000/generate_stream \
        --tokenizer_path /path/to/tokenizer \
        --num_clients 100 \
        --input_num 2000 \
        --input_len 1024 \
        --output_len 128 \
        --server_api lightllm

ShareGPT Dataset Testing (benchmark_sharegpt.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Performance testing using ShareGPT real conversation data.

**Usage:**

.. code-block:: bash

    $ wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

.. code-block:: bash

    python test/benchmark/service/benchmark_sharegpt.py \
        --dataset /path/to/sharegpt_dataset.json \
        --tokenizer /path/to/tokenizer \
        --num_prompts 1000 \
        --request_rate 10.0

**Main Parameters:**

- ``--dataset``: ShareGPT format dataset path
- ``--tokenizer``: Tokenizer path
- ``--num_prompts``: Number of test prompts
- ``--request_rate``: Request rate (requests/s)

Prompt Cache Testing
~~~~~~~~~~~~~~~~~~~~

Evaluate prompt cache performance under different hit rates by adjusting --first_input_len, --output_len --subsequent_input_len to control hit rate.
Hit rate per round = (first_input_len + (output_len + subsequent_input_len) * (num_turns - 1)) / (first_input_len + (output_len + subsequent_input_len) * num_turns)
Note: Control concurrency and user numbers based on max_total_token_num to ensure all requests can fit, guaranteeing that the actual hit rate matches your preset hit rate.

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

Parameter Description:

- ``--model_url``: Service address
- ``--model_name``: Result save filename
- ``--num_workers``: Concurrency number
- ``--first_input_len``: First round input length
- ``--subsequent_input_len``: Subsequent round input length
- ``--output_len``: Output length
- ``--num_turns``: Number of rounds
- ``--num_users``: Number of users

Static Inference Performance Testing (Static Inference Benchmark)
------------------------------------------------------------------

Static inference testing is used to evaluate model inference performance under fixed input conditions, mainly evaluating operator quality.

Model Inference Testing (model_infer.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Main Features:**

- Supports prefill and decode stage performance testing
- Supports microbatch overlap optimization
- Supports multi-GPU parallel inference
- Provides detailed throughput statistics

**Usage:**

.. code-block:: bash

    python test/benchmark/static_inference/test_model.py \
        --model_dir /path/to/model \
        --batch_size 32 \
        --input_len 1024 \
        --output_len 128 \
        --tp 2 \
        --data_type bf16

**Main Parameters:**

- ``--model_dir``: Model path
- ``--batch_size``: Batch size
- ``--input_len``: Input sequence length
- ``--output_len``: Output sequence length
- ``--tp``: Tensor Parallel degree
- ``--data_type``: Data type (bf16/fp16/fp32)
- ``--enable_prefill_microbatch_overlap``: Enable prefill microbatch overlap, only applicable to DeepSeek model EP mode
- ``--enable_decode_microbatch_overlap``: Enable decode microbatch overlap, only applicable to DeepSeek model EP mode
- ``--torch_profile``: Enable torch profiler for performance analysis

.. note::
    Complete startup parameters are not listed here. Static testing scripts also share Lightllm's startup parameters. For more startup configurations, please refer to :ref:`tutorial/api_server_args_zh`.

**Output Metrics:**

- Prefill stage throughput (tokens/s)
- Decode stage throughput (tokens/s)
- Latency statistics for each stage

Multi-Token Prediction Performance Testing (model_infer_mtp.py)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Multi-token prediction static performance testing with 100% acceptance rate by default, used to evaluate the ultimate performance of multi-token prediction. Currently only supports DeepSeek series models.

**Usage:**

.. code-block:: bash

    python test/benchmark/static_inference/test_model.py \
        --model_dir /path/to/main_model \
        --mtp_mode deepseekv3 \
        --mtp_step 1 \
        --mtp_draft_model_dir /path/to/draft_model \
        --batch_size 32 \
        --input_len 1024 \
        --output_len 128

Parameter Description:

- ``--model_dir``: Main model path 