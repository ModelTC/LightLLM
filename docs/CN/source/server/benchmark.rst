服务性能评测
==================

部署完模型以后，对服务性能进行评测是非常重要的，通过服务性能的表现调整配置从而更好地利用显卡资源。
本文中，我们使用 LLaMA-7B 模型，在80G的A800显卡上，比较了lightllm 和 vLLM==0.1.2 的性能。
具体比较方式参考以下步骤：

1. 下载数据集
^^^^^^^^^^^^^^

.. code-block:: console

    $ wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json


2. 开启模型服务
^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir /path/llama-7b --tp 1 --max_total_token_num 121060 --tokenizer_mode auto


3. 性能评测
^^^^^^^^^^^^^^^^

.. code-block:: console

   $ cd test
   $ python benchmark_serving.py --tokenizer /path/llama-7b --dataset /path/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 2000 --request-rate 200 


输出：

.. code-block:: console
    
    read data set finish
    total tokens: 494250
    Total time: 111.37 s
    Throughput: 8.98 requests/s
    Average latency: 43.52 s
    Average latency per token: 0.15 s
    Average latency per output token: 0.73 s