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

下面的内容将会以 `Qwen2-0.5B <https://huggingface.co/Qwen/Qwen2-0.5B>`_ 演示lightllm对大语言模型的支持。
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
    
    $ huggingface-cli download Qwen/Qwen2-0.5B --local-dir Qwen2-0.5

.. tip::
    上面的下载模型的代码需要科学上网，并且需要花费一定的时间，你可以使用其它下载方式或者其它支持的模型作为替代。最新的支持的模型的列表请查看 `项目主页 <https://github.com/ModelTC/lightllm>`_ 。


2. 启动模型服务
-------------------------

下载完Qwen2-0.5B模型以后，在终端使用下面的代码部署API服务：

.. code-block:: console

    $ python -m lightllm.server.api_server --model_dir ~/models/Qwen2-0.5B  \
    $                                       --host 0.0.0.0                  \
    $                                       --port 8080                     \
    $                                       --tp 1                          \
    $                                       --max_total_token_num 120000    \
    $                                       --trust_remote_code             \
    $                                       --eos_id 151643   

.. note::
    上面代码中的 ``--model_dir`` 参数需要修改为你本机实际的模型路径。 ``--eos_id 151643`` 是Qwen模型专属，其它模型请删除这个参数。


3. （可选）测试模型服务
-------------------------

在新的终端，使用下面的指令对模型服务进行测试：

.. code-block:: console

    $ curl http://localhost:8080/generate \
    $      -H "Content-Type: application/json" \
    $      -d '{
    $            "inputs": "What is AI?",
    $            "parameters":{
    $              "max_new_tokens":17, 
    $              "frequency_penalty":1
    $            }
    $           }'


