NVFP4 KV 缓存
==============

``--llm_kv_type nvfp4kv`` 使用 vLLM/FlashInfer 的 NVFP4 量化配方：
K/V 使用 E2M1，每 token、每 KV head、每 16 个特征值共享一个动态 E4M3 scale。
全局 K/V scale 固定为 1，对齐 checkpoint 未提供全局校准 scale 时的行为。
此模式不需要校准导出，也不接受 ``--kv_quant_calibration_config_path``。

.. code-block:: bash

    python -m lightllm.server.api_server \
      --model_dir /path/to/model --tp 2 --dp 1 \
      --data_type bfloat16 --llm_kv_type nvfp4kv \
      --llm_prefill_att_backend auto --llm_decode_att_backend auto

Q 保持 BF16/FP16，prefill 与 decode 均读取量化后的 K/V。
请安装当前仓库 ``requirements.txt`` 中的依赖。在 H200 上，``auto`` 优先选择
FA3 prefill 和 FlashInfer decode；原生 FlashInfer 路径支持 head_dim 128/256。
其他 head_dim、MTP block decode、非因果 decode 和滑窗层使用 Triton 路径。
也可将两个 attention backend 参数显式设置为 ``triton``。
设置 ``FLASHINFER_DISABLE_JIT`` 时不启用额外的小 tile JIT 优化。

包含 scale 后，KV 存储为 BF16 的 9/32，相同 KV 内存可容纳约 3.56 倍 token；
此比例不包含 attention 临时空间和其他模型内存。

当前支持 head_dim 为 16、32、64、128、256 的 Llama/Qwen GQA/MHA/MQA，
以及 Qwen 混合模型的 full-attention 层。支持 CUDA Graph、前缀缓存、滑动窗口
和 MTP block decode；不支持 ALiBi、attention sink 和 MLA/DSA。
这是 LightLLM 中的配方实现，不代表 vLLM 引擎本身的执行性能。

精度验证使用 Qwen2.5-14B-Instruct、H200 TP2、MTP=0，完整 GSM8K 1,319 题。
FA3 prefill/FlashInfer 小 tile decode 路径同协议下仓库口径 BF16 为 92.19%，NVFP4 为 92.27%，
差值为 +0.08 个百分点，配对检验 p=1.000；严格匹配口径为 91.05% 和 90.75%，
差值为 -0.30 个百分点，配对检验 p=0.755，两个口径均未发现显著差异；
该结果不保证其他模型或数据集上的精度。
抢占重算后的长文本 greedy 输出仍可能发生 token 差异，BF16 对照也观察到此现象；
当前不保证切换 batch 或发生抢占后输出逐 token 一致。
