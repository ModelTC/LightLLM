欢迎了解 Lightllm!
==================

.. figure:: ./assets/logos/lightllm-logo.png
  :width: 100%
  :align: center
  :alt: Lightllm
  :class: no-scaled-link

.. raw:: html

   <p style="text-align:center">
   <strong>一个轻量级、高性能的大语言模型服务框架
   </strong>
   </p>

   <p style="text-align:center">
   <script async defer src="https://buttons.github.io/buttons.js"></script>
   <a class="github-button" href="https://github.com/ModelTC/lightllm" data-show-count="true" data-size="large" aria-label="Star">Star</a>
   <a class="github-button" href="https://github.com/ModelTC/lightllm/subscription" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
   <a class="github-button" href="https://github.com/ModelTC/lightllm/fork" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
   </p>


Lightllm 是一个纯python开发的大语言模型推理和服务框架，具有轻量级设计、易扩展以及高性能等特点。
Lightllm 整合了众多的开源方案的优点，包括但不限于 FasterTransformer、TGI、vLLM、SGLang 和 FlashAttention。


**重要特性**:

* 多进程协同：输入文本编码、语言模型推理、视觉模型推理、输出解码等工作异步进行，大幅提高GPU利用率。
* 跨进程请求对象共享：通过共享内存，实现跨进程请求对象共享，降低进程间通信延迟。
* 高效的调度策略：带预测的峰值显存调度策略，最大化GPU显存利用率的同时，降低请求逐出。
* 高性能的推理后端：高效的算子实现，多种并行方式支持（张量并行，数据并行以及专家并行），动态kv缓存，丰富的量化支持（int8, fp8, int4），结构化输出以及多结果预测。

文档列表
-------------

.. toctree::
   :maxdepth: 1
   :caption: 快速入门

   安装指南 <getting_started/installation>
   快速开始 <getting_started/quickstart>
   性能评测 <getting_started/benchmark>

.. toctree::
   :maxdepth: 1
   :caption: 部署教程

   DeepSeek R1 部署 <tutorial/deepseek_deployment>
   多级缓存部署 <tutorial/multi_level_cache_deployment>
   多模态部署 <tutorial/multimodal>
   奖励模型部署 <tutorial/reward_model>
   OpenAI 接口使用 <tutorial/openai>
   工具调用（Function Calling） <tutorial/function_calling>
   思考解析（Reasoning Parser） <tutorial/reasoning_parser>
   APIServer 参数详解 <tutorial/api_server_args_zh>
   lightllm api介绍 <tutorial/api_param>
   
.. toctree::
   :maxdepth: 1
   :caption: 模型支持

   支持的模型列表 <models/supported_models>
   添加新模型 <models/add_new_model>
   
.. toctree::
   :maxdepth: 1
   :caption: 架构介绍

   架构介绍 <framework/framework>
   token attention介绍 <framework/token_attention>
   峰值显存调度器介绍 <framework/router>
   
.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
