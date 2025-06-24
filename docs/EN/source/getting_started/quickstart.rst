.. _quickstart:

Quick Start
===========

Deploying models with Lightllm is very simple, requiring only two steps at minimum:

1. Prepare model weight files supported by Lightllm.
2. Use command line to start the model service.
3. (Optional) Test the model service.

.. note::
    Before continuing with this tutorial, please ensure you have completed the :ref:`Installation Guide <installation>`.

1. Prepare Model Files
----------------------

Download `Qwen3-8B <https://huggingface.co/Qwen/Qwen3-8B>`_ first.
Below is an example code for downloading the model:

(1) (Optional) Create folder

.. code-block:: console

    $ mkdirs ~/models && cd ~/models
    
(2) Install ``huggingface_hub``

.. code-block:: console

    $ pip install -U huggingface_hub

(3) Download model files

.. code-block:: console
    
    $ huggingface-cli download Qwen/Qwen3-8B --local-dir Qwen3-8B

2. Start Model Service
----------------------

After downloading the Qwen3-8B model, use the following code in the terminal to deploy the API service:

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/Qwen3-8B

.. note::
    The ``--model_dir`` parameter in the above code needs to be modified to your actual local model path.

3. Test Model Service
---------------------

.. code-block:: console

    $ curl http://127.0.0.1:8000/generate \
         -H "Content-Type: application/json" \
         -d '{
               "inputs": "What is AI?",
               "parameters":{
                 "max_new_tokens":17, 
                 "frequency_penalty":1
               }
              }'


