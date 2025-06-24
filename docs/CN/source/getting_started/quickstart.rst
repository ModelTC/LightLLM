.. _quickstart:

快速开始
==========

使用lightllm部署模型非常简单，最快只需要两个步骤：

1. 准备 Lightllm 所支持的模型的权重文件。
2. 使用命令行启动模型服务。
3. (可选) 对模型服务进行测试。

.. note::
    在继续这个教程之前，请确保你完成了 :ref:`安装指南 <installation>` .



1. 准备模型文件
-------------------------

下面的内容将会以 `Qwen3-8B <https://huggingface.co/Qwen/Qwen3-8B>`_ 演示lightllm对大语言模型的支持。
下载模型的方法可以参考文章：`如何快速下载huggingface模型——全方法总结 <https://zhuanlan.zhihu.com/p/663712983>`_ 

下面是下载模型的实例代码：

(1) （可选）创建文件夹

.. code-block:: console

    $ mkdirs ~/models && cd ~/models
    
(2) 安装 ``huggingface_hub``

.. code-block:: console

    $ pip install -U huggingface_hub

(3) 下载模型文件

.. code-block:: console
    
    $ huggingface-cli download Qwen/Qwen3-8B --local-dir Qwen3-8B


2. 启动模型服务
-------------------------

下载完Qwen3-8B模型以后，在终端使用下面的代码部署API服务：

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/Qwen3-8B

.. note::
    上面代码中的 ``--model_dir`` 参数需要修改为你本机实际的模型路径。

3. 测试模型服务
-------------------------

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