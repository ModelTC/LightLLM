.. _lightllm:

LightLLM introduction
===========================

With the popularity of ChatGPT, large language model, abbreviated as LLM, has received increasing attention. The emergence of such models has greatly improved people's work efficiency. However, the key to further widespread adoption lies in how to deploy models with billons of parameters at low cost and high throughput. To improve the throughput of large model services and enable more interested researchers to quickly get involved, a lightweight LLM inference service framework called LightLLM has emerged. LightLLM introduces a more fine-grained kv cache management algorithm called TokenAttention and designs an Efficient Router scheduling implementation that works efficiently with TokenAttention. Through the interaction of TokenAttention and Efficient Router, LightLLM achieves higher throughput than vLLM and Text Generation Inference in most scenarios, with performance improvements of around 4 times in some cases. LightLLM is flexible, user-friendly, and efficient. Interested friends may want to click on the link below to try it out.

Project：https://github.com/ModelTC/lightllm

.. _challenge:

The challenge of LLM Serving
-------------------------------

Large language models have garnered significant attention from researchers due to their excellent performance. These models not only engage in everyday conversations with humans but also assist in completing various daily tasks, thereby enhancing productivity. However, despite the remarkable performance demonstrated by these models, deploying large-scale models to improve service performance poses the following challenges:

* **Severe fragmentation of memory**: Network weights ranging from tens to hundreds of gigabytes, as well as the constantly dynamic growing KV Cache during inference, easily leads to low memory utilization.
* **Low efficiency in request scheduling**: The length of requests dynamically changes over time, which can result in GPU idling or low utilization issues.
* **High difficulty in kernel customization**: Customizing kernels for networks is necessary to efficiently utilize memory and improve service throughput. However, it will require a significant amount of effort from researchers.

.. _solutions_and_problems:

Existing solutions and problems
-------------------------------------

To address the aforementioned challenges, many excellent LLM inference frameworks have emerged, such as FasterTransformer, Text-Generation-Inference (referred to as TGI), vLLM, etc. The core features and capability matrices of these frameworks are shown in the table below:

.. list-table:: Comparison of various frameworks
   :header-rows: 1

   * - Framework
     - NV Triton + FasterTransformer
     - TGI
     - vLLM
     - LightLLM
   * - core feature
     - 	Efficient kernel
     - `Continuous batch <https://github.com/huggingface/text-generation-inference/tree/main/router>`_, Token streaming
     - `PageAttention <https://vllm.ai/>`_
     - Tri-process asynchronous collaboration，:ref:`TokenAttention`，:ref:`Efficient_Router`
   * - Memory fragmentation
     - low
     - high
     - low
     - low
   * - Request scheduling efficiency
     - low
     - middle
     - middle
     - high
   * - Difficulty of kernel customization
     - high
     - middle
     - middle
     - low

These frameworks all have their own unique features. For example, FasterTransformer has excellent static inference performance but lacks robust service scheduling and is primarily developed in C++, resulting in high secondary development costs. TGI has excellent service interfaces and scheduling features such as Continuous Batch, but its inference performance, scheduling strategy, and memory management have some shortcomings. vLLM has excellent memory management but lacks efficiency in request scheduling, and its overall implementation details are more suitable for deploying small models.

Lightllm
----------------------

Therefore, to address these issues, we have developed a LLM deployment framework called LightLLM, which is based on the pure Python language. It enables researchers to easily deploy and customize lightweight models locally, allowing for rapid expansion of different models and integration of various excellent open-source features. The core features of LightLLM are as follows:

* 三进程异步协作：分词、模型推理、去分词异步进行，GPU利用率大幅提升。
* :ref:`TokenAttention`：实现token-wise的KV缓存内存管理机制，实现推理时内存零浪费。
* :ref:`Efficient_Router`：与Token Attention合作，精心管理每个Token的GPU内存，从而优化系统吞吐量。

* Tri-process asynchronous collaboration: tokenization, model inference, and detokenization are performed asynchronously, leading to a considerable improvement in GPU utilization.
* :ref:`TokenAttention`: implements token-wise's KV cache memory management mechanism, allowing for zero memory waste during inference.
* :ref:`Efficient_Router`: collaborates with Token Attention to meticulously manage the GPU memory of each token, thereby optimizing system throughput.

With the highly coordinated efficient kernels developed based on OpenAI Triton and service scheduling, LightLLM achieves excellent throughput performance

.. figure:: ../assets/lightllm/arch.png
  :width: 100%
  :align: center
  :alt: Lightllm
  :class: no-scaled-link



Performance
-----------

We conducted performance comparisons on the ShareGPT_Vicuna_unfiltered dataset using the current mainstream inference frameworks TGI, NV Triton + FasterTransformer, and vLLM. The results are shown in the graph below. It can be observed that LightLLM achieves higher throughput across different model sizes. TGI suffers from severe memory fragmentation, making it difficult to achieve high throughput. vLLM introduces PageAttention but due to its overall implementation details being more favorable for small model inference, its concurrent performance on large models is not very ideal (using default configurations). In contrast, LightLLM maintains robust performance across various model sizes and achieves around a 2-3x improvement over TGI and vLLM on large models (LLaMA-65B).

.. image:: ../assets/lightllm/Performance.png
   :alt: Efficient_Router1
   :align: center

TGI Compatibility & Ablation Analysis To further validate the effectiveness of TokenAttention and Router, we also integrated these features into TGI to address its memory fragmentation issue, as shown in the figure below (left). It can be observed that introducing TokenAttention and Router leads to more than a 4x performance improvement compared to the original TGI.

Improvement in case of mixed long and short requests：From the figure below (left), it can be noticed that the introduction of Router did not bring a more significant performance improvement, which is due to the fact that the difference in the question length of ShareGPT_Vicuna_unfiltered's dataset is not significant. For this reason, we constructed a collection of requests with a greater difference in the length, and verified the performance of our Efficient Router. The results are shown below (right). It can be seen that our Efficient Router can make better use of GPU resources, and can bring about nearly 50% performance improvement with requests that have large differences in question lengths.

.. image:: ../assets/lightllm/Performance2.png
   :alt: Efficient_Router1
   :align: center


The left figure shows the compatibility of LightLLM and TGI and the ablation analysis, and the right figure shows the enhancement of our Efficient Router with the long and short request

Future Work
----------------

* Support for more models
* router scheduling enhancements
* High-performance int8 int4 weight only support and int8 kv cache.
* Fully quantized models
* Mixed-precision models
* Sparsification

LightLLM is committed to enabling more people to participate, allowing flexible and efficient exploration of various LLM deployment and inference solutions. It also serves as a reference for hardware manufacturers to promote the development of the field. We hope that everyone can give it more stars, fork the project, and contribute. We believe that in the future, more technologies and solutions (such as TensorRT) will emerge, continuously reducing deployment costs and making AGI more accessible to ordinary households.