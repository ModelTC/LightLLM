.. _lightllm:

LightLLM 介绍
================

随着ChatGPT的流行，大语言模型(简称LLM)受到越来越多的关注。此类模式的出现，极大地提高了人们的工作效率。
然而，进一步广泛使用LLM的关键在于如何以低成本和高吞吐量地部署数十亿参数的模型。
为了提高大模型服务的吞吐量，让更多感兴趣的研究人员快速参与进来，
一种名为 LightLLM 的轻量级 LLM 推理服务框架应运而生。 
LightLLM 引入了一种更细粒度的kvCache管理算法，称为TokenAttention，
并设计了一个与TokenAttention高效配合的Efficient Router调度算法。
通过 TokenAttention 和 Efficient Router 的配合，
LightLLM 在大多数场景下实现了比 vLLM 和 Text Generation Inference 更高的吞吐量，
甚至在某些情况下性能提升了 4 倍左右。 LightLLM 灵活、用户友好且高效，
欢迎有兴趣的朋友进入 `项目主页 <https://github.com/ModelTC/lightllm>`_ 了解更多。


.. _challenge:

LLM 推理服务的挑战
------------------

大型语言模型由于其优异的性能而引起了研究人员的极大关注。
这些模型不仅可以与人类进行日常对话，还可以帮助完成各种日常任务，从而提高生产力。
然而，尽管这些模型表现出了出色的性能，但提高部署大规模模型的性能仍面临以下挑战：

* **内存碎片严重**：从几十到几百G不等的网络权重，以及推理过程中不断动态增长的KV Cache，容易导致大量的内存碎片，进而导致内存利用率低。
* **请求调度效率低**：请求的长度随时间动态变化，可能导致GPU空闲或利用率低的问题。
* **内核定制难度大**：为了高效利用内存、提高服务吞吐量，需要针对网络定制内核。然而，这需要研究人员付出大量的努力。


.. _solutions_and_problems:

现有的解决方案和存在问题
-----------------------------

为了应对上述挑战，许多优秀的LLM推理框架应运而生，
例如FasterTransformer、Text-Generation-Inference（简称TGI）、vLLM等。这些框架的核心特性和能力如下表所示：


.. list-table:: 各个框架对比
   :header-rows: 1

   * - 框架
     - NV Triton + FasterTransformer
     - TGI
     - vLLM
     - LightLLM
   * - 核心特征
     - 高效算子
     - `Continuous batch <https://github.com/huggingface/text-generation-inference/tree/main/router>`_, Token streaming
     - `PageAttention <https://vllm.ai/>`_
     - 三进程异步协同, `Token Attention <https://github.com/ModelTC/lightllm/blob/main/docs/TokenAttention.md>`_, Efficient Router
   * - 内存碎片
     - 少
     - 多
     - 少
     - 少
   * - 请求的调度效率
     - 低
     - 中
     - 中
     - 高
   * - 定制化算子的难度
     - 高
     - 中
     - 中
     - 低

这些框架都有自己独特的特点。
例如，FasterTransformer具有优异的静态推理性能，但缺乏健壮的服务调度，并且主要采用C++开发，导致二次开发成本较高。 
TGI具有优秀的服务接口和Continuation Batch等调度特性，但其推理性能、调度策略、内存管理等方面存在一些不足。 
vLLM具有出色的内存管理能力，但在请求调度方面缺乏效率，其整体实现细节更适合部署小型模型。


Lightllm
----------------------

因此，为了解决这些问题，我们开发了一个名为LightLLM的LLM部署框架，它是基于纯Python语言的。
它使研究人员能够在本地轻松部署和定制轻量级模型，从而可以快速扩展不同模型并集成各种优秀的开源功能。 
LightLLM的核心特点如下：

* 三进程异步协作：分词、模型推理、去分词异步进行，GPU利用率大幅提升。
* :ref:`TokenAttention`：实现token-wise的KV缓存内存管理机制，实现推理时内存零浪费。
* :ref:`Efficient_Router`：与Token Attention合作，精心管理每个Token的GPU内存，从而优化系统吞吐量。

凭借基于OpenAI Triton开发的高度协调的高效内核和服务调度，LightLLM实现了优异的吞吐性能。

.. figure:: ../assets/lightllm/arch.png
  :width: 100%
  :align: center
  :alt: Lightllm
  :class: no-scaled-link



LightLLM致力于让更多人参与进来，从而灵活高效地探索各种LLM部署和推理解决方案。也为硬件厂商推动该领域的发展提供参考。我们希望大家能够给它更多的star，fork这个项目，并做出贡献。我们相信未来将会出现更多的技术和解决方案（如TensorRT），不断降低部署成本，让AGI更容易走进普通家庭。