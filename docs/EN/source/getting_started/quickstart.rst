.. _quickstart:

Quick Start
===========

Deploying a model with Lightllm is very straightforward and requires only two steps:

1. Prepare the weight file for a model supported by Lightllm.
2. Start the model service using the command line.
3. (Optional) Test the model service.

.. note::
    Before continuing with this tutorial, please ensure you have completed the :ref:`installation guide <installation>`.

1. Prepare the Model File
-------------------------

The following content will demonstrate Lightllm's support for large language models using `Llama-2-7b-chat <https://huggingface.co/meta-llama/Llama-2-7b-chat>`_. You can refer to the article: `How to Quickly Download Hugging Face Models — A Summary of Methods <https://zhuanlan.zhihu.com/p/663712983>`_ for methods to download models.

Here is an example of how to download the model:

(1) (Optional) Create a directory

.. code-block:: console

    $ mkdir -p ~/models && cd ~/models
    
(2) Install ``huggingface_hub``

.. code-block:: console

    $ pip install -U huggingface_hub

(3) Download the model file

.. code-block:: console
    
    $ huggingface-cli download meta-llama/Llama-2-7b-chat --local-dir Llama-2-7b-chat

.. tip::
    The above code for downloading the model requires a stable internet connection and may take some time. You can use alternative download methods or other supported models as substitutes. For the latest list of supported models, please refer to the `project homepage <https://github.com/ModelTC/lightllm>`_.


2. Start the Model Service
---------------------------

After downloading the Llama-2-7b-chat model, use the following command in the terminal to deploy the API service:

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/Llama-2-7b-chat

.. note::
    The ``--model_dir`` parameter in the above command should be changed to the actual path of your model on your machine. 

For the DeepSeek-R1 model on single H200, it can be launched with the following command:

.. code-block:: console

    $ LOADWORKER=8 python -m lightllm.server.api_server --model_dir ~/models/DeepSeek-R1 --tp 8 --graph_max_batch_size 100

.. note::
    LOADWORKER specifies the thread for model loading, which can enhance the speed of model loading. The --graph_max_batch_size parameter specifies the number of cudagraphs to be captured, which will capture graphs for batch sizes ranging from 1 to 100.

For the DeepSeek-R1 model on two H100, it can be launched with the following command:

.. code-block:: console

    $ # Node 0
    $ LOADWORKER=8 python -m lightllm.server.api_server --model_dir ~/models/DeepSeek-R1 --tp 16 --graph_max_batch_size 100 --nccl_host master_addr --nnodes 2 --node_rank 0
    $ # Node 1
    $ LOADWORKER=8 python -m lightllm.server.api_server --model_dir ~/models/DeepSeek-R1 --tp 16 --graph_max_batch_size 100 --nccl_host master_addr --nnodes 2 --node_rank 1


3. (Optional) Test the Model Service
--------------------------------------

In a new terminal, use the following command to test the model service:

.. code-block:: console

    $ curl http://localhost:8000/generate \
    $      -H "Content-Type: application/json" \
    $      -d '{
    $            "inputs": "What is AI?",
    $            "parameters":{
    $              "max_new_tokens":17, 
    $              "frequency_penalty":1
    $            }
    $           }'


For DeepSeek-R1 benchmark, use the following command to test the model service:

.. code-block:: console

    $ cd test
    $ python benchmark_client.py --num_clients 100 --input_num 2000 --tokenizer_path /nvme/DeepSeek-R1/ --url http://127.0.01:8000/generate_stream


