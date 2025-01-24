.. _Performance_Benchmark:

Performance
===========

Service Performance
-------------------

We compared the service performance of LightLLM and vLLM==0.1.2 on LLaMA-7B using an A800 with 80G GPU memory.

To begin, prepare the data as follows:

.. code-block:: shell

   wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

Launch the service:

.. code-block:: shell

   python -m lightllm.server.api_server --model_dir /path/llama-7b --tp 1 --max_total_token_num 121060 --tokenizer_mode auto

Evaluation:

.. code-block:: shell

   cd test
   python benchmark_serving.py --tokenizer /path/llama-7b --dataset /path/ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 2000 --request-rate 200

The performance comparison results are presented below:

+-------------------+-------------------+
| vLLM              | LightLLM          |
+===================+===================+
| Total time: 361.79| Total time: 188.85|
| Throughput: 5.53  | Throughput: 10.59 |
| requests/s        | requests/s        |
+-------------------+-------------------+

Static Inference Performance
----------------------------

For debugging, we offer static performance testing scripts for various models. For instance, you can evaluate the inference performance of the LLaMA model by:

.. code-block:: shell

   cd test/model
   python test_llama.py