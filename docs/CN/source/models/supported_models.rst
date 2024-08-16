支持的模型列表
================

lightllm 支持大多数的主流的开源大语言模型以及多模态模型，并且会不断扩充支持的模型列表。在后面的版本中，lightllm将会支持更多类型的模型（例如reward模型）。

.. note::

    由于lightllm 的轻量级设计，lightllm具有非常高的可扩展性，这意味着添加新的模型支持非常简单，具体方法请参考 **添加新模型** 一节`


-----

大语言模型
^^^^^^^^^^^^^^^^^^^^^^


.. list-table::
  :widths: 25 25 
  :header-rows: 1

  * - 模型
    - 备注
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


多模态模型
^^^^^^^^^^^^^^^^^

.. list-table::
  :widths: 25 25 
  :header-rows: 1

  * - 模型
    - 备注
  * - `Qwen-VL <https://huggingface.co/Qwen/Qwen-VL>`_
    -  :code:`--trust_remote_code --enable_multimodal`
  * - `Qwen-VL-Chat <https://huggingface.co/Qwen/Qwen-VL-Chat>`_
    -  :code:`--trust_remote_code --enable_multimodal`
  * - `Llava-7b <https://huggingface.co/liuhaotian/llava-v1.5-7b>`_
    -  :code:`--enable_multimodal`
  * - `Llava-13b <https://huggingface.co/liuhaotian/llava-v1.5-13b>`_
    -  :code:`--enable_multimodal`


Reward模型
^^^^^^^^^^^^^^^^^

.. list-table::
  :widths: 25 25 
  :header-rows: 1

  * - 模型
    - 备注
  * - `internLM-reward <https://huggingface.co/internlm/internlm2-1_8b-reward>`_
    -  :code:`--use_reward_model`

