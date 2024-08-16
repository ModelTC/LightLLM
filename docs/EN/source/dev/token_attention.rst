.. _TokenAttention:

TokenAttention
=======================

Transformers form the basis of modern large language models. During autoregressive decoding, these models cache key-value tensors of context tokens into GPU memory to facilitate fast generation of the next token. However, these caches occupy significant GPU memory. The unpredictable nature of cache size, due to the variability in the length of each request, exacerbates the issue, resulting in significant memory fragmentation in the absence of a suitable memory management mechanism.

To alleviate this issue, PagedAttention was proposed to store the KV cache in non-contiguous memory spaces. It partitions the KV cache of each sequence into multiple blocks, with each block containing the keys and values for a fixed number of tokens. This approach effectively controls memory waste within the last block during attention computation. While PagedAttention alleviates memory fragmentation to some extent, it still leaves room for memory waste. Additionally, when handling multiple high-concurrency requests, the allocation and deallocation of memory blocks fall short of efficiency, leading to suboptimal memory utilization.

To address the above challenges, we introduce TokenAttention, an attention mechanism that manages key and value caching at the token level. Compared to PagedAttention, our TokenAttention not only minimizes memory fragmentation and enables efficient memory sharing but also facilitates efficient memory allocation and deallocation. It allows for more precise and fine-grained memory management, thus optimizing memory utilization.

.. list-table:: Feature Comparison
   :widths: 30 15 15
   :header-rows: 1

   * - Features
     - PagedAttention
     - TokenAttention
   * - Low memory fragmentation
     - ✓
     - ✓
   * - Efficient memory sharing
     - ✓
     - ✓
   * - Efficient memory allocation and deallocation
     - ✗
     - ✓
   * - Fine-grained memory management
     - ✗
     - ✓


The operation mechanism of TokenAttention is illustrated in the figure below:

.. figure:: ../assets/lightllm/token_attn.gif
  :width: 100%
  :align: center
  :alt: Lightllm
  :class: no-scaled-link


During model initialization, the KV cache is pre-allocated based on the user-set **max_total_token_num** and a Token Table is created to record the actual storage locations of input tokens.

When handling new requests, the system first checks for available contiguous space in the pre-allocated Token cache for storing the key-value (KV) cache. TokenAttention favors assigning contiguous graphics memory space for requests to minimize memory access during the inference process. Only when contiguous space is insufficient does it allocate non-contiguous graphics memory for the requests. Since memory management is conducted on a token-by-token basis, TokenAttention achieves nearly zero waste, yielding higher throughput compared to vllm.

We have implemented an efficient TokenAttention operator using OpenAI Triton. When provided with a query vector, this operator can efficiently retrieve the corresponding KV cache based on the Token Table and conduct the attention computation.

Upon completion of requests, the corresponding graphics memory can be quickly freed by deleting their records on the Token Table, which makes way for scheduling new requests. Given that TokenAttention pre-allocates all KV cache space during model initialization, it can efficiently release memory for completed requests and merge different batches of requests during dynamic scheduling, thereby effectively maximizing GPU utilization.

The specific steps are as follows:


1. During model initialization, the KV cache is pre-allocated based on the user-set max_total_token_num and a Token Table is created to record the actual storage locations of input tokens.
2. When handling new requests, the system first checks for available contiguous space in the pre-allocated Token cache for storing the key-value (KV) cache. TokenAttention favors assigning contiguous graphics memory space for requests to minimize memory access during the inference process. Only when contiguous space is insufficient does it allocate non-contiguous graphics memory for the requests. The allocated space is recorded in the Token Table for subsequent attention calculations.
3. For cache of newly generated tokens, it is only necessary to find unused space from the pre-allocated token cache and add the corresponding entry to the Token Table. Moreover, to efficiently allocate and release the Cache, we utilize the parallel computing capabilities of torch Tensor on the GPU to manage the state of the pre-allocated Token Cache. First, we define the states as follows:

    .. code-block:: python

        self.mem_state = torch.ones((size,), dtype=torch.bool, device="cuda")
        self._mem_cum_sum = torch.empty((size,), dtype=torch.int32, device="cuda")
        self.indexes = torch.arange(0, size, dtype=torch.long, device="cuda")
        self.can_use_mem_size = size


    The mem_state records the usage status of the cache, where 1 represents unused and 0 represents used. The _mem_cum_sum is used for the cumulative sum of mem_state which is used to efficiently identify and select unused space for cache allocation. The allocation process is as follows:
    
    .. code-block:: python

        torch.cumsum(self.mem_state, dim=0, dtype=torch.int32, out=self._mem_cum_sum)
        # 
        select_index = torch.logical_and(self._mem_cum_sum <= need_size, self.mem_state == 1)
        select_index = self.indexes[select_index]
        self.mem_state[select_index] = 0
        self.can_use_mem_size -= len(select_index)


    It can be observed that our cache state management is all done on the GPU, fully utilizing the parallel capabilities of torc, thereby allowing the system to efficiently allocate cache space for each request.

4. Upon completion of requests, the corresponding graphics memory can be quickly freed by deleting their records on the Token Table, which makes way for scheduling new requests.

    .. code-block:: python

        self.can_use_mem_size += free_index.shape[0]
        self.mem_state[free_index] = 1

5. Token Attention allows for zero wastage of GPU memory, due to its GPU memory management at the token level. It can accurately calculate how many new tokens the system can accommodate for computation. Therefore, when combined with a high-performance router to manage requests, it can continuously add new requests during the inference process, fully utilizing every piece of GPU memory and maximizing GPU utilization.

