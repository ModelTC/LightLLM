Supported Models
================

lightllm supports most mainstream open source large language models and multimodal models, and will continue to expand the list of supported models. In later versions, lightllm will support more types of models (such as reward models).

.. note::

  Due to its lightweight design, Lightllm is highly extensible, which means that adding new model support is very simple. For more information, please refer to the **How to Add New Model Support** section.

-----

LLM
^^^^^^^^^^^^^^^^^^^^^^


.. list-table::
  :widths: 25 25 
  :header-rows: 1

  * - model
    - note
  * - `BLOOM <https://huggingface.co/bigscience/bloom>`_
    -  
  * - `LLaMA <https://github.com/facebookresearch/llama>`_
    -  
  * - `LLaMA V2 <https://huggingface.co/meta-llama>`_
    -   
  * - `StarCoder <https://github.com/bigcode-project/starcoder>`_
    -  
  * - `Qwen-7b <https://github.com/QwenLM/Qwen-7B>`_
    - :code:`--eos_id 151643 --trust_remote_code`
  * - `ChatGLM2-6b <https://github.com/THUDM/ChatGLM2-6B>`_
    - :code:`--trust_remote_code`
  * - `Baichuan-7b <https://github.com/baichuan-inc/Baichuan-7B>`_
    - :code:`--trust_remote_code`  
  * - `Baichuan-13b <https://github.com/baichuan-inc/Baichuan-13B>`_
    - :code:`--trust_remote_code`
  * - `Baichuan2-7b <https://github.com/baichuan-inc/Baichuan2>`_
    - :code:`--trust_remote_code`
  * - `Baichuan2-13b <https://github.com/baichuan-inc/Baichuan2>`_
    - :code:`--trust_remote_code`
  * - `InternLM-7b <https://github.com/InternLM/InternLM>`_
    - :code:`--trust_remote_code`
  * - `Yi-34b <https://huggingface.co/01-ai/Yi-34B>`_
    -   
  * - `Mixtral <https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1>`_
    -   
  * - `Stablelm <https://huggingface.co/stabilityai/stablelm-2-1_6b>`_
    - :code:`--trust_remote_code`
  * - `MiniCPM <https://huggingface.co/openbmb/MiniCPM-2B-sft-bf16>`_
    -   
  * - `Phi-3 <https://huggingface.co/collections/microsoft/phi-3-6626e15e9585a200d2d761e3>`_
    -  only supports Mini and Small.
  * - `CohereForAI <https://huggingface.co/CohereForAI/c4ai-command-r-plus>`_
    - :code:`--data_type bfloat16`
  * - `DeepSeek-V2-Lite <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite>`_ 
    - :code:`--data_type bfloat16`
  * - `DeepSeek-V2 <https://huggingface.co/deepseek-ai/DeepSeek-V2>`_ 
    - :code:`--data_type bfloat16`


VLM
^^^^^^^^^^^^^^^^^

.. list-table::
  :widths: 25 25 
  :header-rows: 1

  * - model
    - note
  * - `Qwen-VL <https://huggingface.co/Qwen/Qwen-VL>`_
    -  :code:`--trust_remote_code --enable_multimodal`
  * - `Qwen-VL-Chat <https://huggingface.co/Qwen/Qwen-VL-Chat>`_
    -  :code:`--trust_remote_code --enable_multimodal`
  * - `Llava-7b <https://huggingface.co/liuhaotian/llava-v1.5-7b>`_
    -  :code:`--enable_multimodal`
  * - `Llava-13b <https://huggingface.co/liuhaotian/llava-v1.5-13b>`_
    -  :code:`--enable_multimodal`
  * - `Google Gemma3 <https://huggingface.co/google/gemma-3-12b-it>`_
    -  :code:`--enable_multimodal`


Reward Model
^^^^^^^^^^^^^^^^^

.. list-table::
  :widths: 25 25 
  :header-rows: 1

  * - model
    - note
  * - `internLM-reward <https://huggingface.co/internlm/internlm2-1_8b-reward>`_
    -  :code:`--use_reward_model`

