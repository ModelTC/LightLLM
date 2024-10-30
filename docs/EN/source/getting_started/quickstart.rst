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



