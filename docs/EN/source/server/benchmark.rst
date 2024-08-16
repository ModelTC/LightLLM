Benchmark
==================

After deploying the model, it is very important to evaluate the service performance. By adjusting the configuration based on the service performance, the graphics card resources can be better utilized.
In this article, we use the LLaMA-7B model to compare the performance of lightllm and vLLM==0.1.2 on an 80G A800 graphics card.
For the specific comparison method, please refer to the following steps:

1. Download datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json


2. Launching Server
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir /path/llama-7b --tp 1 --max_total_token_num 121060 --tokenizer_mode auto


3. Benchmark
^^^^^^^^^^^^^^^^

.. code-block:: console

   $ cd test
   $ python benchmark_serving.py --tokenizer /path/llama-7b --dataset /path/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 2000 --request-rate 200 


output:

.. code-block:: console
    
    read data set finish
    total tokens: 494250
    Total time: 111.37 s
    Throughput: 8.98 requests/s
    Average latency: 43.52 s
    Average latency per token: 0.15 s
    Average latency per output token: 0.73 s